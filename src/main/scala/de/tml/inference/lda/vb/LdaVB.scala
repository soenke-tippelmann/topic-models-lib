package de.tml.inference.lda.vb

import breeze.linalg.*
import breeze.linalg.Axis
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.sum
import breeze.numerics.digamma
import breeze.numerics.exp
import com.typesafe.scalalogging.LazyLogging
import de.tml.data.Corpus
import de.tml.data.Document
import org.scalactic.Requirements._
import org.scalactic.Tolerance._
import org.scalactic.TripleEquals._

class LdaVB[TokenType](val corpus: Corpus[TokenType], val numTopics: Int,
                       ensureInvariantsEnabled: Boolean = false)
                      (alpha: DenseVector[Double] = DenseVector.ones(numTopics),
                       eta: DenseVector[Double] = DenseVector.ones(corpus.wordList.numTokens))
    extends LazyLogging {

  final val EPS: Double = 1e-10

  logger.info("Setting up LdaVB")

  var gamma: DenseMatrix[Double] =
    DenseMatrix.rand[Double](corpus.numDocuments, numTopics) *:* 5d + 2d

  private val sizeOfDictionary = corpus.wordList.numTokens
  var lambda: DenseMatrix[Double] =
    DenseMatrix.rand[Double](numTopics, sizeOfDictionary) *:* 5d + 2d

  // breeze has no n-d matrix support
  var phi: IndexedSeq[DenseMatrix[Double]] = corpus.allDocuments
      .map(doc => DenseMatrix.rand[Double](doc.getTokenCount, numTopics) + 1e-12)
      .map(matrix => {
        val sum1 = sum(matrix, Axis._1)
        matrix(::, *) /:/ sum1
      })

  logger.info(s"dict: $sizeOfDictionary, numDocs: ${phi.length}, numTopics: $numTopics")

  ensureInvariants()

  logger.info("Done setting up LdaVB")

  def ensureInvariants(): Unit = {
    if (ensureInvariantsEnabled) {
      logger.info("Checking invariants.")

      requireState(breeze.linalg.all(gamma >:> 0d), "gamma contains 0d.")
      requireState(!gamma.data.exists(d => d.isNaN), "gamma contains NaN.")

      requireState(breeze.linalg.all(lambda >:> 0d), "lambda contains 0d.")
      requireState(!lambda.data.exists(d => d.isNaN), "lambda contains NaN.")


      phi.foreach(phi_d => {
        requireState(!phi_d.data.exists(d => d.isNaN), "phi_d contains NaN.")

        val sum_phi_d = sum(phi_d, Axis._1)
        requireState(sum_phi_d.length == phi_d.rows)

        requireState(sum_phi_d.data.forall(_ === 1.0 +- EPS),
          "phi_d contains invalid multinomials.")
      })

      logger.info("Done checking invariants.")
    }
  }

  def executeUpdateCycle(): Unit = {
    logger.info("Updating phi.")
    updatePhi()

    logger.info("Updating gamma.")
    updateGamma()

    logger.info("Updating lambda.")
    updateLambda()

    logger.info("Done updating variational parameters.")

    ensureInvariants()
  }

  private def updateGamma(): Unit = LdaVB_updateRules.updateGamma(alpha, phi, gamma)

  private def updateLambda(): Unit = LdaVB_updateRules
      .updateLambda(lambda, eta, phi, corpus.allDocuments, numTopics, sizeOfDictionary)

  private def updatePhi(): Unit = LdaVB_updateRules
      .updatePhi(gamma, lambda, phi, corpus.allDocuments)

  /**
    * Caution: Reuses the matricies used during inference, ie executing executeUpdateCycle will
    * change these. The correctness of the returned ResultCorpus is not guaranteed and in most
    * situations the contained results after continuing inference will be wrong and have to be
    * freshly exported after the iteration!
    *
    * @return
    */
  def exportUnreliableResult: LdaVBCorpusResult[TokenType] =
    new LdaVBCorpusResult[TokenType](corpus, gamma, lambda, numTopics)

  /**
    * Caution: Creates deep copies of gamma and lambda matricies.
    *
    * @return
    */
  def exportResult: LdaVBCorpusResult[TokenType] =
    new LdaVBCorpusResult[TokenType](corpus, gamma.copy, lambda.copy, numTopics)
}

object LdaVB_updateRules {
  def updateGamma(alpha: DenseVector[Double],
                  phi: IndexedSeq[DenseMatrix[Double]],
                  gamma: DenseMatrix[Double]): Unit =

    phi.map(alpha.t + sum(_, Axis._0)).zipWithIndex.foreach {
      case (vect, docId) => gamma(docId, ::) := vect
    }

  def updateLambda(lambda: DenseMatrix[Double],
                   eta: DenseVector[Double],
                   phi: IndexedSeq[DenseMatrix[Double]],
                   documents: IndexedSeq[Document],
                   numTopics: Int,
                   sizeOfDictionary: Int): Unit = {

    val sumOverPhi: DenseMatrix[Double] = DenseMatrix.zeros[Double](numTopics, sizeOfDictionary)

    phi.zip(documents).foreach {
      case (phi_d: DenseMatrix[Double], doc) => {
        doc.allTokensVector.zipWithIndex
            .foreach {
              case (word, index) => sumOverPhi(::, word) += phi_d(index, ::).t
            }
      }
    }

    lambda := sumOverPhi(*, ::) + eta
  }

  def updatePhi(gamma: DenseMatrix[Double],
                lambda: DenseMatrix[Double],
                phi: IndexedSeq[DenseMatrix[Double]],
                documents: IndexedSeq[Document]): Unit = {

    val gamma_digamma = digamma(gamma)
    val lambda_digamma = digamma(lambda)

    val lambda_sum_over_v = sum(lambda, Axis._1)
    val lambda_sum_over_v_digamma = digamma(lambda_sum_over_v).t

    phi.zip(documents).zipWithIndex.par.foreach {
      case ((phi_d: DenseMatrix[Double], doc), docIndex) => {
        doc.allTokensVector.zipWithIndex.foreach {
          case (word, wordIndex) => {
            val v = gamma_digamma(docIndex, ::) + lambda_digamma(::, word)
                .t - lambda_sum_over_v_digamma
            val digammaExp = exp(v)

            requireState(digammaExp.t.data.forall(!_.isInfinity))

            phi_d(wordIndex, ::) := digammaExp /:/ sum(digammaExp)
          }
        }
      }
    }
  }
}

object LdaVB extends LazyLogging {

  def runInference[TokenType]
  (corpus: Corpus[TokenType], numTopics: Int, numSweeps: Int)
  (
      postInitalizationHandler: LdaVBCorpusResult[TokenType] => Unit =
      (_: LdaVBCorpusResult[TokenType]) => (),
      preCylceHAndler: (Int, LdaVBCorpusResult[TokenType]) => Unit =
      (_: Int, _: LdaVBCorpusResult[TokenType]) => (),
      postCylceHandler: (Int, LdaVBCorpusResult[TokenType]) => Unit =
      (_: Int, _: LdaVBCorpusResult[TokenType]) => ()
  )
  : LdaVBCorpusResult[TokenType] = {

    val ldaVB = new LdaVB[TokenType](corpus, numTopics)()

    postInitalizationHandler(ldaVB.exportUnreliableResult)

    for (i <- Seq.range(0, numSweeps)) {
      logger.info(s"Inference round: $i")
      preCylceHAndler(i, ldaVB.exportUnreliableResult)
      ldaVB.executeUpdateCycle()
      postCylceHandler(i, ldaVB.exportUnreliableResult)
      logger.info(s"Done with inference round: $i")
    }

    ldaVB.exportUnreliableResult
  }
}


