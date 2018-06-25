package de.tml.inference.dtm.vb

import scala.collection.concurrent.TrieMap

import breeze.linalg.Axis
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.sum
import breeze.numerics.digamma
import breeze.numerics.exp
import breeze.numerics.log
import com.typesafe.scalalogging.LazyLogging
import de.tml.data.TimeAnnotatedCorpus
import de.tml.data.TimeDocument
import de.tml.data.TimeSlice
import de.tml.data.TimeSliceProvider

object DtmVBUpdateRules extends LazyLogging {

  def updateLambda[TimeSliceType <: TimeSlice[TimeSliceType]]
  (phi: IndexedSeq[DenseMatrix[Double]],
   zeta: DenseMatrix[Double],
   sigmaSquared: Double,
   nySquared: Double,
   lambda: IndexedSeq[DenseMatrix[Double]],
   timeSliceProvider: TimeSliceProvider[TimeSliceType],
   corpus: TimeAnnotatedCorpus[_, TimeSliceType]): IndexedSeq[DenseMatrix[Double]] = {

    val lambdaOptimizer = LambdaOptimizer(phi, zeta, sigmaSquared, nySquared, timeSliceProvider,
      corpus)

    // k -> lamda_i -> v -> Double
    val newLambda: IndexedSeq[DenseMatrix[Double]] =
      lambda.zipWithIndex.par
          .map {
            case (l: DenseMatrix[Double], k: Int) => {
              logger.info(s"Optimizing topic $k")
              lambdaOptimizer.optimizeLambda(l, k)
            }
          }
          .toIndexedSeq

    newLambda
  }

  def updateGamma(alpha: DenseVector[Double],
                  phi: IndexedSeq[DenseMatrix[Double]],
                  gamma: DenseMatrix[Double]): Unit =

    phi.map(alpha.t + sum(_, Axis._0)).zipWithIndex.foreach {
      case (vect, docId) => gamma(docId, ::) := vect
    }

  def updateZeta(zeta: DenseMatrix[Double], kalmanFilter: Seq[KalmanFilterResult]): Unit = {
    kalmanFilter.zipWithIndex.foreach {
      case (kalman, topicIdx) => {
        updateZetaForTopicIdx(zeta, kalman, topicIdx)
      }
    }
  }

  def updateZetaForTopicIdx(zeta: DenseMatrix[Double], kalman: KalmanFilterResult,
                            topicIdx: Int): Unit = {
    kalman.mktBackward.zip(kalman.vktBackward).drop(1).zipWithIndex.foreach {
      case ((mkt: DenseVector[Double], vkt: DenseVector[Double]), timeIdx: Int) => {
        val vktHalf = vkt /:/ -2d
        val expSum = exp(mkt + vktHalf)
        val result: Double = sum(expSum)
        zeta(timeIdx to timeIdx, topicIdx to topicIdx) := result
      }
    }
  }

  def updatePhi[TimeSliceType <: TimeSlice[TimeSliceType]]
  (gamma: DenseMatrix[Double],
   zeta: DenseMatrix[Double],
   kalmanFilter: Seq[KalmanFilterResult],
   phi: IndexedSeq[DenseMatrix[Double]],
   documents: IndexedSeq[TimeDocument[TimeSliceType]],
   timeSliceProvider: TimeSliceProvider[TimeSliceType]): Unit = {

    val gamma_digamma = digamma(gamma)

    def computeTimeSliceDependentData(timeSlicePosition: Int):
    (Seq[DenseVector[Double]], DenseVector[Double], DenseVector[Double]) = {
      val mkt_timeSlice: Seq[DenseVector[Double]] =
        kalmanFilter.map(_.mktBackward(timeSlicePosition))
      val vkt_timeSlice = kalmanFilter.map(_.vktBackward(timeSlicePosition))

      val zeta_timeslice = zeta(timeSlicePosition, ::).t
      val neg_log_zeta: DenseVector[Double] = log(zeta_timeslice) *:* -1d

      val neg_one_over_zeta = -1d / zeta_timeslice
      val summation: Seq[Double] = mkt_timeSlice.zip(vkt_timeSlice).map {
        case (mkt: DenseVector[Double], vkt: DenseVector[Double]) => {
          val neg_vkt_half = vkt /:/ -2d
          val exp_sum = exp(mkt + neg_vkt_half)
          sum(exp_sum)
        }
      }
      val sum_over_v: DenseVector[Double] = DenseVector(summation: _*)
      val last_summand = neg_one_over_zeta *:* sum_over_v

      (mkt_timeSlice, neg_log_zeta, last_summand)
    }

    val timeSliceDependentData =
      new TrieMap[Int, (Seq[DenseVector[Double]], DenseVector[Double], DenseVector[Double])]()

    phi.zip(documents).zipWithIndex.par.foreach {
      case ((phi_d: DenseMatrix[Double], doc: TimeDocument[TimeSliceType]), docIndex) => {

        val timeSlicePosition: Int = timeSliceProvider.getTimeSlicePositionIdx(doc.timeSlice)
        val (mkt_timeSlice, neg_log_zeta, last_summand) =
          timeSliceDependentData
              .getOrElseUpdate(timeSlicePosition, computeTimeSliceDependentData(timeSlicePosition))

        val gamma_digamma_doc: DenseVector[Double] = gamma_digamma(docIndex, ::).t

        doc.allTokensVector.zipWithIndex.foreach {
          case (word, wordIndex) => {
            val mkt_timeslice_word: DenseVector[Double] =
              DenseVector(mkt_timeSlice.map(_ (word)): _*)

            val v = gamma_digamma_doc + mkt_timeslice_word + neg_log_zeta + last_summand
            val exponent = exp(v)

            val result = exponent /:/ sum(exponent)

            phi_d(wordIndex, ::) := result.t
          }
        }
      }
    }
  }

}

