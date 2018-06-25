package de.tml.inference.dtm.vb

import scala.collection.concurrent.TrieMap

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.sum
import breeze.numerics.exp
import breeze.numerics.log
import breeze.optimize.DiffFunction
import breeze.optimize.FirstOrderMinimizer
import breeze.optimize.LBFGS
import de.tml.data.TimeAnnotatedCorpus
import de.tml.data.TimeDocument
import de.tml.data.TimeSlice
import de.tml.data.TimeSliceProvider

class LambdaOptimizer[TimeSliceType <: TimeSlice[TimeSliceType]]
(phi: IndexedSeq[DenseMatrix[Double]],
 zeta: DenseMatrix[Double],
 sigmaSquared: Double,
 nySquared: Double,
 timeSliceProvider: TimeSliceProvider[TimeSliceType],
 corpus: TimeAnnotatedCorpus[_, TimeSliceType]) {

  private[this] val sizeOfDictionary = corpus.wordList.numTokens
  private[this] val numTimeSlices = corpus.timeSliceProvider.getNumTimeSlices


  private[this] val phi_hat_k_t_cache = TrieMap[(Int, Int), Double]()

  /**
    * @param k Topic Index
    * @param t TimeSlicePositionIndex
    */
  def phi_hat_k_t(k: Int, t: Int): Double = {
    phi_hat_k_t_cache.getOrElseUpdate((k, t), {
      val timeSlice = timeSliceProvider.getTimeSliceAtPosition(t)
      val docsAtT: IndexedSeq[TimeDocument[TimeSliceType]] =
        corpus.getDocumentsByTimeSlice(timeSlice)

      val result: Double =
        docsAtT.map(doc => corpus.getDocumentId(doc))
            .map(docId => phi(docId))
            .map(
              (phi_d: DenseMatrix[Double]) => {
                val phi_dk: DenseVector[Double] = phi_d(::, k)
                sum(phi_dk)
              }
            ).sum

      result
    })
  }

  private[this] val phi_hat_k_t_byV_cache = TrieMap[(Int, Int), DenseVector[Double]]()

  /**
    * @param k Topic Index
    * @param t TimeSlicePositionIndex
    */
  def phi_hat_k_t_byV(k: Int, t: Int): DenseVector[Double] = {

    phi_hat_k_t_byV_cache.getOrElseUpdate((k, t), {
      val timeSlice = timeSliceProvider.getTimeSliceAtPosition(t)
      val docsAtT: IndexedSeq[TimeDocument[TimeSliceType]] =
        corpus.getDocumentsByTimeSlice(timeSlice)

      if (docsAtT.isEmpty) {
        DenseVector.zeros[Double](sizeOfDictionary)
      } else {
        val result: DenseVector[Double] =
          docsAtT
              .map(doc => (corpus.getDocumentId(doc), doc))
              .map { case (docId, doc) => (phi(docId), doc) }
              .map {
                case (phi_d: DenseMatrix[Double], doc: TimeDocument[TimeSliceType]) => {

                  val phi_dkv =
                    doc.allTokensVector.zip(phi_d(::, k).toScalaVector())
                        .foldRight(DenseVector.zeros[Double](sizeOfDictionary))(
                          (word_phi_dk: (Int, Double), bag: DenseVector[Double]) => {
                            val (word, phi_dk) = word_phi_dk

                            bag(word) = bag(word) + phi_dk
                            bag
                          }
                        )
                  phi_dkv
                }
              }.reduce(_ + _)

        result
      }
    })
  }

  /**
    *
    * @param lambda
    * @param topicIdx
    * @return Seq[timeSlices] -> dictioary
    */
  def computeLambdaDerivative(lambda: DenseMatrix[Double], topicIdx: Int)(
      kalman: KalmanFilterResult = KalmanFilter.computeKalmanFilter(lambda, sigmaSquared, nySquared)
  ): Seq[DenseVector[Double]] = {

    def t_minus_tminusone(sequence: Seq[DenseVector[Double]]): Seq[DenseVector[Double]] =
      sequence.drop(1).zip(sequence.dropRight(1)).map { case (t, t_minus_1) => t - t_minus_1 }

    val mback_diff = t_minus_tminusone(kalman.mktBackward)

    // lambda_t -> firstpart_of_derivative
    val first_summand: Seq[DenseVector[Double]] =
      kalman.diffMktBackward.map(
        (diffMktBackward_fixedLambda: Seq[DenseVector[Double]]) => {
          val diff_mback_diff = t_minus_tminusone(diffMktBackward_fixedLambda)
          val inner_sum = mback_diff.zip(diff_mback_diff).map { case (m, d) => m *:* d }
          val sum = inner_sum.reduce(_ + _)

          sum * -1d / sigmaSquared
        }
      )

    val expMktVkt: Seq[DenseVector[Double]] =
      kalman.mktBackward.zip(kalman.vktBackward)
          .map { case (mkt, vkt) => exp(mkt + vkt / 2d) }

    val second_summand: Seq[DenseVector[Double]] =
      kalman.diffMktBackward.map(
        (diffMktBackward_fixedLambda: Seq[DenseVector[Double]]) => {
          val inner_sum: Seq[DenseVector[Double]] =
            diffMktBackward_fixedLambda.zip(expMktVkt)
                .drop(1).zipWithIndex
                .map {
                  case ((diffmkt, sumExp), timeSlicePosition) => {
                    val first: DenseVector[Double] = phi_hat_k_t_byV(topicIdx, timeSlicePosition)

                    val phi_hat_k_t1 = phi_hat_k_t(topicIdx, timeSlicePosition)
                    val second: DenseVector[Double] = phi_hat_k_t1 / zeta(timeSlicePosition,
                      topicIdx) * sumExp

                    (first - second) *:* diffmkt
                  }
                }

          val result = inner_sum.reduce(_ + _)
          result
        }
      )

    val result = first_summand.zip(second_summand).map { case (f, s) => f + s }

    result
  }

  /**
    * Value of the ELBO given lambda, leaving out all the constant summands
    *
    * @param lambda
    * @param topicIdx
    * @return
    */
  def computePartialElbo(lambda: DenseMatrix[Double], topicIdx: Int)(
      kalman: KalmanFilterResult = KalmanFilter.computeKalmanFilter(lambda, sigmaSquared, nySquared)
  ): Double = {

    val first_summand: Double =
      sum(kalman.mktBackward.drop(1).zipWithIndex
          .map { case (mkt, tsIdx) => sum(mkt *:* phi_hat_k_t_byV(topicIdx, tsIdx)) }
      )

    val second_summand: Double =
      sum(
        kalman.mktBackward.zip(kalman.vktBackward).drop(1).zipWithIndex.map {
          case ((mkt, vkt), tsIdx) => {
            val logZeta = log(zeta(tsIdx, topicIdx))
            val zetaValue: Double = 1d / zeta(tsIdx, topicIdx)
            val sum1: Double = sum(exp(mkt + (vkt /:/ 2d)))
            phi_hat_k_t(topicIdx, tsIdx) * (1d - logZeta - zetaValue * sum1)
          }
        }
      )

    val third_summand: Double =
      (1 / (2d * sigmaSquared)) *
          sum(kalman.mktBackward.sliding(2).map(
            elements => {
              val mkt1 = elements.head
              val mkt2 = elements(1)

              val difference = mkt2 - mkt1
              sum(difference *:* difference)
            }
          ))

    first_summand + second_summand - third_summand
  }

  def stackedVectorAsMatrix(vector: DenseVector[Double]): DenseMatrix[Double] = {
    vector.asDenseMatrix.reshape(numTimeSlices, sizeOfDictionary)
  }

  def optimizeLambda(lambda: DenseMatrix[Double], topicIdx: Int): DenseMatrix[Double] = {
    val function = new DiffFunction[DenseVector[Double]] {
      /**
        * @param lambda timeSlices x dictionary
        * @return
        */
      override def calculate(lambda: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val lambdaAsMatrix: DenseMatrix[Double] =
          stackedVectorAsMatrix(lambda)

        val kalman = KalmanFilter.computeKalmanFilter(lambdaAsMatrix, sigmaSquared, nySquared)
        DtmVBUpdateRules.updateZetaForTopicIdx(zeta, kalman, topicIdx)

        val derivative: Seq[DenseVector[Double]] =
          computeLambdaDerivative(lambdaAsMatrix, topicIdx)()

        val derivativeAsVector: DenseVector[Double] =
          DenseMatrix.vertcat(derivative.map(_.asDenseMatrix): _*).toDenseVector


        val partialElbo = computePartialElbo(lambdaAsMatrix, topicIdx)()

        // use negative values, because we want to maximize ELBO, hence we minimize negative ELBO!
        (-partialElbo, -derivativeAsVector)
      }
    }

    val valueConvergence = FirstOrderMinimizer.functionValuesConverged[DenseVector[Double]](
      tolerance = 1e-3, relative = true, historyLength = 10
    )
    val iterationsCheck = FirstOrderMinimizer.maxIterationsReached[DenseVector[Double]](150)
    val lbfgs = new LBFGS[DenseVector[Double]](
      m = 6,
      convergenceCheck = iterationsCheck || valueConvergence || FirstOrderMinimizer.searchFailed
    )

    val optimum = lbfgs.minimize(function, lambda.toDenseVector)

    stackedVectorAsMatrix(optimum)
  }

}

object LambdaOptimizer {
  def apply[TimeSliceType <: TimeSlice[TimeSliceType]]
  (phi: IndexedSeq[DenseMatrix[Double]],
   zeta: DenseMatrix[Double],
   sigmaSquared: Double,
   nySquared: Double,
   timeSliceProvider: TimeSliceProvider[TimeSliceType],
   corpus: TimeAnnotatedCorpus[_, TimeSliceType]): LambdaOptimizer[TimeSliceType] =
    new LambdaOptimizer(phi, zeta, sigmaSquared, nySquared, timeSliceProvider, corpus)
}
