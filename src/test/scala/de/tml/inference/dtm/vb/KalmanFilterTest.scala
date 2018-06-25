package de.tml.inference.dtm.vb

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import org.scalatest.FunSpec
import org.scalatest.Matchers

class KalmanFilterTest extends FunSpec with Matchers {

  describe("KalmanFilter update rules") {

    describe("Given lambda and hyperparameters") {
      val numTopics = 3
      val numDocuments = 3
      val numWordTokens = 5
      val numTimeSlices = 2

      val sigmaSquared: Double = 2d
      val nySquared: Double = 3d

      val lambda: DenseMatrix[Double] = DenseMatrix.ones(numTimeSlices, numWordTokens)
      val lambda_t = 1d

      describe("a calculation of the kalman result") {

        val kalmanResult = KalmanFilter.computeKalmanFilter(lambda, sigmaSquared, nySquared)

        val initialVkt = 10d
        val ts1vkt = nySquared / (initialVkt + sigmaSquared + nySquared) * (initialVkt + sigmaSquared)
        val ts2vkt = nySquared / (ts1vkt + sigmaSquared + nySquared) * (ts1vkt + sigmaSquared)

        val expectedVktForward = Seq(
          DenseVector.fill(numWordTokens)(initialVkt), //initial
          DenseVector.fill(numWordTokens)(ts1vkt), // timeSlice 1
          DenseVector.fill(numWordTokens)(ts2vkt) // timeSlice 2
        )

        it("vkt should equal the expected vkt kalman result") {
          kalmanResult.vktForward should contain theSameElementsInOrderAs expectedVktForward
        }

        val ts2vktB = ts2vkt
        val squaredFactorTs1vktB = 1d - sigmaSquared / (ts1vkt + sigmaSquared)
        val ts1vktB = ts1vkt + squaredFactorTs1vktB * squaredFactorTs1vktB *
            (ts2vktB - ts1vkt - sigmaSquared)
        val squaredFactorInitialVktB = 1d - sigmaSquared / (initialVkt + sigmaSquared)
        val initialVktB = initialVkt +
            squaredFactorInitialVktB * squaredFactorInitialVktB *
                (ts1vktB - initialVkt - sigmaSquared)

        val expectedVktBackward = Seq(
          DenseVector.fill(numWordTokens)(initialVktB), //initial
          DenseVector.fill(numWordTokens)(ts1vktB), // timeSlice 1
          DenseVector.fill(numWordTokens)(ts2vktB) // timeSlice 2
        )

        it("vkt backward should equal the expected vkt backward kalman result") {
          kalmanResult.vktBackward should contain theSameElementsInOrderAs expectedVktBackward
        }

        val initialMkt = 0d
        val ts1MktFactor = nySquared / (initialVkt + sigmaSquared + nySquared)
        val ts1mkt = ts1MktFactor * initialMkt + (1 - ts1MktFactor) * lambda_t
        val ts2MktFactor = nySquared / (ts1vkt + sigmaSquared + nySquared)
        val ts2mkt = ts2MktFactor * ts1mkt + (1 - ts2MktFactor) * lambda_t

        val expectedMktForward = Seq(
          DenseVector.fill(numWordTokens)(initialMkt), //initial
          DenseVector.fill(numWordTokens)(ts1mkt), // timeSlice 1
          DenseVector.fill(numWordTokens)(ts2mkt) // timeSlice 2
        )

        it("mkt should equal the expected mkt kalman result") {
          kalmanResult.mktForward should contain theSameElementsInOrderAs expectedMktForward
        }

        val ts2mktB = ts2mkt
        val factorTs1mktB = sigmaSquared / (ts1vkt + sigmaSquared)
        val ts1mktB = factorTs1mktB * ts1mkt + (1 - factorTs1mktB) * ts2mktB
        val factorInitialmktB = sigmaSquared / (initialVkt + sigmaSquared)
        val initialMktB = factorInitialmktB * initialMkt + (1 - factorInitialmktB) * ts1mktB

        val expectedMktBackward = Seq(
          DenseVector.fill(numWordTokens)(initialMktB), //initial
          DenseVector.fill(numWordTokens)(ts1mktB), // timeSlice 1
          DenseVector.fill(numWordTokens)(ts2mktB) // timeSlice 2
        )

        it("mkt backward should equal the expected mkt kalman result") {
          kalmanResult.mktBackward should contain theSameElementsInOrderAs expectedMktBackward
        }

        val factorTs1DiffMkt = nySquared / (initialVkt + sigmaSquared + nySquared)
        val factorTs2DiffMkt = nySquared / (ts1vkt + sigmaSquared + nySquared)

        val initialDiffMktL1 = 0d
        val ts1DiffMktL1 = factorTs1DiffMkt * initialDiffMktL1 + (1 - factorTs1DiffMkt)
        val ts2DiffMktL1 = factorTs2DiffMkt * ts1DiffMktL1

        val initialDiffMktL2 = 0d
        val ts1DiffMktL2 = factorTs1DiffMkt * initialDiffMktL2
        val ts2DiffMktL2 = factorTs2DiffMkt * ts1DiffMktL2 + (1 - factorTs2DiffMkt)


        //lambda_t ~> mkt ~> vector
        val expectedDiffMkt = Seq(
          Seq(
            DenseVector.fill(numWordTokens)(initialDiffMktL1), //initial
            DenseVector.fill(numWordTokens)(ts1DiffMktL1), // timeSlice 1
            DenseVector.fill(numWordTokens)(ts2DiffMktL1) // timeSlice 2
          ),
          Seq(
            DenseVector.fill(numWordTokens)(initialDiffMktL2), //initial
            DenseVector.fill(numWordTokens)(ts1DiffMktL2), // timeSlice 1
            DenseVector.fill(numWordTokens)(ts2DiffMktL2) // timeSlice 2
          )
        )
        it("diffMkt should equal expected diffMkt kalman result") {
          kalmanResult.diffMktForward.zip(expectedDiffMkt).foreach {
            case (actual, expected) => actual should contain theSameElementsInOrderAs expected
          }
        }

        val factorTs1DiffMktB = sigmaSquared / (ts1vkt + sigmaSquared)
        val factorInitialDiffMktB = sigmaSquared / (initialVkt + sigmaSquared)

        val ts2DiffMktBL1 = ts2DiffMktL1
        val ts1DiffMktBL1 = factorTs1DiffMktB * ts1DiffMktL1 + (1 - factorTs1DiffMktB) * ts2DiffMktBL1
        val initialDiffMktBL1 = factorInitialDiffMktB * initialDiffMktL1 + (1 - factorInitialDiffMktB) * ts1DiffMktBL1

        val ts2DiffMktBL2 = ts2DiffMktL2
        val ts1DiffMktBL2 = factorTs1DiffMktB * ts1DiffMktL2 + (1 - factorTs1DiffMktB) * ts2DiffMktBL2
        val initialDiffMktBL2 = factorInitialDiffMktB * initialDiffMktL2 + (1 - factorInitialDiffMktB) * ts1DiffMktBL2

        //lambda_t ~> mkt ~> vector
        val expectedDiffMktB = Seq(
          Seq( // L1
            DenseVector.fill(numWordTokens)(initialDiffMktBL1), //initial
            DenseVector.fill(numWordTokens)(ts1DiffMktBL1), // timeSlice 1
            DenseVector.fill(numWordTokens)(ts2DiffMktBL1) // timeSlice 2
          ),
          Seq( // L2
            DenseVector.fill(numWordTokens)(initialDiffMktBL2), //initial
            DenseVector.fill(numWordTokens)(ts1DiffMktBL2), // timeSlice 1
            DenseVector.fill(numWordTokens)(ts2DiffMktBL2) // timeSlice 2
          )
        )
        it("diffMkt backward should equal expected diffMkt kalman result") {
          kalmanResult.diffMktBackward.zip(expectedDiffMktB).foreach {
            case (actual, expected) => actual should contain theSameElementsInOrderAs expected
          }
        }
      }
    }
  }
}
