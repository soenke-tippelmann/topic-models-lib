package de.tml.inference.dtm.vb

import scala.collection.immutable.Vector

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.sum
import breeze.numerics._

class DtmVBTest extends DtmTest {

  describe("Dtm Variational Bayes inference update Rules") {

    describe("given documents, timeslices and hyperparameters") {

      describe("update Gamma") {

        val phi: IndexedSeq[DenseMatrix[Double]] =
          Vector(
            DenseMatrix.create(3, numTopics, Array(.4, .2, .7, .5, .3, .1, .1, .5, .2)),
            DenseMatrix.create(2, numTopics, Array(.6, .7, .2, 0d, .2, .3)),
            DenseMatrix.create(4, numTopics, Array(.1, .1, .6, .2, .8, .5, .3, .3, .1, .4, .1, .5))
          )

        val gamma: DenseMatrix[Double] = DenseMatrix.ones(numDocuments, numTopics)

        // expected outcome
        val expectedGamma: DenseMatrix[Double] = DenseMatrix.zeros(numDocuments, numTopics)
        expectedGamma(0, ::) := DenseVector(2.3, 2.9, 3.8).t
        expectedGamma(1, ::) := DenseVector(2.3, 2.2, 3.5).t
        expectedGamma(2, ::) := DenseVector(2d, 3.9, 4.1).t

        DtmVBUpdateRules.updateGamma(alpha, phi, gamma)

        it("gamma should be transformed to expectedGamma") {
          breeze.linalg.all(abs(gamma - expectedGamma) <:< eps) shouldBe true
        }
      }

      describe("update zeta") {
        val kfr: Seq[KalmanFilterResult] =
          Seq.fill(3)(
            KalmanFilterResult(
              Seq.empty,
              Seq.fill(3)(DenseVector.rangeD(1d, 6d)), //include initial values of kalman sequence
              Seq.empty,
              Seq.fill(3)(DenseVector.ones(5)), //include initial values of kalman sequence
              Seq.empty, Seq.empty)
          )

        val zeta: DenseMatrix[Double] = DenseMatrix.ones(numTimeSlices, numTopics)

        // expected outcome
        val expectedSum = sum(exp(DenseVector.rangeD(1d, 6d) - DenseVector.fill(5)(.5)))

        val expectedZeta: DenseMatrix[Double] = DenseMatrix.zeros(numTimeSlices, numTopics)
        expectedZeta(0, ::) := DenseVector.fill(3)(expectedSum).t
        expectedZeta(1, ::) := DenseVector.fill(3)(expectedSum).t

        DtmVBUpdateRules.updateZeta(zeta, kfr)

        it("zeta should be transformed to expectedZeta") {
          breeze.linalg.all(abs(zeta - expectedZeta) <:< eps) shouldBe true
        }
      }

      describe("update phi") {
        val mktValTs1 = 1d
        val mktValTs2 = 2d

        val vktValTs1 = 1d
        val vktValTs2 = 2d

        val gammaVal = 1d

        val zetaValTs1 = 1d
        val zetaValTs2 = 2d

        val gamma: DenseMatrix[Double] = DenseMatrix.fill(numDocuments, numTopics)(gammaVal)

        val zeta: DenseMatrix[Double] = DenseMatrix.ones(numTimeSlices, numTopics)
        zeta(0, ::) := zetaValTs1
        zeta(1, ::) := zetaValTs2

        val kfr: Seq[KalmanFilterResult] =
          Seq.fill(3)(
            KalmanFilterResult(
              Seq.empty,
              Seq(DenseVector.fill(5)(.1d), DenseVector.fill(5)(mktValTs1),
                DenseVector.fill(5)(mktValTs2)), //include initial values of kalman sequence
              Seq.empty,
              Seq(DenseVector.fill(5)(.1d), DenseVector.fill(5)(vktValTs1),
                DenseVector.fill(5)(vktValTs2)), //include initial values of kalman sequence
              Seq.empty, Seq.empty)
          )

        /** Seq[Document] -> tokenCount x numTopics */
        val phi: IndexedSeq[DenseMatrix[Double]] =
          IndexedSeq(
            DenseMatrix.zeros[Double](3, numTopics),
            DenseMatrix.zeros[Double](2, numTopics),
            DenseMatrix.zeros[Double](4, numTopics)
          )

        val sumTs1 = numWordTokens * exp(mktValTs1 - vktValTs1 / 2)
        val sumTs2 = numWordTokens * exp(mktValTs2 - vktValTs2 / 2)

        val digamma_gamma: Double = digamma(gammaVal)

        val proportionalPhiTs1 =
          exp(digamma_gamma + mktValTs1 - log(zetaValTs1) - sumTs1)
        val valPhiTs1 = proportionalPhiTs1 / (numTopics * proportionalPhiTs1)

        val proportionalPhiTs2 =
          exp(digamma_gamma + mktValTs2 - log(zetaValTs2) - sumTs2)
        val valPhiTs2 = proportionalPhiTs2 / (numTopics * proportionalPhiTs2)

        val doc1Matrix: DenseMatrix[Double] = DenseMatrix.fill(3, numTopics)(valPhiTs1)
        val doc2Matrix: DenseMatrix[Double] = DenseMatrix.fill(2, numTopics)(valPhiTs1)
        val doc3Matrix: DenseMatrix[Double] = DenseMatrix.fill(4, numTopics)(valPhiTs2)

        val expectedPhi: IndexedSeq[DenseMatrix[Double]] = IndexedSeq(
          doc1Matrix, doc2Matrix, doc3Matrix
        )

        DtmVBUpdateRules.updatePhi(gamma, zeta, kfr, phi, documents, tsprovider)

        it("phi should be transformed to expectedPhi") {
          phi.zip(expectedPhi).foreach {
            case (actual, expected) =>
              breeze.linalg.all(abs(actual - expected) <:< eps) shouldBe true
          }
        }
      }

      describe("update lambda") {

        val zeta: DenseMatrix[Double] = DenseMatrix.rand(numTimeSlices, numTopics)
        /** Seq[Document] -> tokenCount x numTopics */
        val phi: IndexedSeq[DenseMatrix[Double]] =
          IndexedSeq(
            DenseMatrix.rand[Double](3, numTopics),
            DenseMatrix.rand[Double](2, numTopics),
            DenseMatrix.rand[Double](4, numTopics)
          )

        /** Seq[Topic] ->  numTimeSlices x sizeOfDictionary */
        val lambda: IndexedSeq[DenseMatrix[Double]] =
          IndexedSeq.fill(numTopics)(DenseMatrix.rand[Double](numTimeSlices, numWordTokens))

        val lambdaResult: IndexedSeq[DenseMatrix[Double]] = DtmVBUpdateRules
            .updateLambda(phi, zeta, sigmaSquared, nySquared, lambda, tsprovider, corpus)

        it("lambdaResult should differ from expectedLambda") {
          lambdaResult.zip(lambda).foreach {
            case (actual: DenseMatrix[Double], expected: DenseMatrix[Double]) =>
              val squaredDistance: Double = sum(actual *:* actual + expected *:* expected)
              squaredDistance should be > actual.data.length * eps
          }
        }
      }
    }
  }
}
