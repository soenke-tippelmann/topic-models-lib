package de.tml.inference.dtm.vb

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics.abs
import breeze.numerics.exp

class LambdaOptimizerTest extends DtmTest {

  describe("LambdaOptimizer test") {

    val zeta: DenseMatrix[Double] = DenseMatrix.ones(numTimeSlices, numTopics)

    /** Seq[Document] -> tokenCount x numTopics */
    val phi: IndexedSeq[DenseMatrix[Double]] =
      IndexedSeq(
        DenseMatrix.ones[Double](3, numTopics),
        DenseMatrix.ones[Double](2, numTopics),
        DenseMatrix.ones[Double](4, numTopics)
      )

    val optimizer = LambdaOptimizer(phi, zeta, sigmaSquared, nySquared, tsprovider, corpus)

    describe("phi_hat_k_t") {
      val expectedPhiHat_K0_T0 = 5d
      val expectedPhiHat_K0_T1 = 4d

      it("should produce expected results") {
        optimizer.phi_hat_k_t(0, 0) shouldEqual expectedPhiHat_K0_T0
        optimizer.phi_hat_k_t(0, 1) shouldEqual expectedPhiHat_K0_T1
      }
    }

    describe("phi_hat_k_t_byV") {
      val expectedPhiHatByV_K0_T0 = DenseVector(1d, 1d, 1d, 2d, 0d)
      val expectedPhiHatByV_K0_T1 = DenseVector(1d, 0d, 1d, 1d, 1d)

      it("should produce expected results") {
        optimizer.phi_hat_k_t_byV(0, 0) shouldEqual expectedPhiHatByV_K0_T0
        optimizer.phi_hat_k_t_byV(0, 1) shouldEqual expectedPhiHatByV_K0_T1
      }
    }

    describe("given lambda and a kalman filter result") {
      /** Seq[Topic] ->  numTimeSlices x sizeOfDictionary */
      val lambda: IndexedSeq[DenseMatrix[Double]] =
        IndexedSeq.fill(numTopics)(DenseMatrix.ones[Double](numTimeSlices, numWordTokens))

      val kfr: Seq[KalmanFilterResult] =
        Seq.fill(3)(
          KalmanFilterResult(
            Seq.empty,
            //include initial values of kalman sequence, hence numTimeSlices+1
            Seq.fill(numTimeSlices + 1)(DenseVector.ones(numWordTokens)),
            Seq.empty,
            Seq.fill(numTimeSlices + 1)(DenseVector.ones(numWordTokens)),
            Seq.empty,
            Seq.fill(numTimeSlices + 1)(
              Seq.fill(numTimeSlices + 1)(DenseVector.ones(numWordTokens)))
          )
        )

      val topicIdx = 0

      describe("compute partial ELBO") {
        val expectedPartialElbo = 9d + (9d - 45d * exp(1.5d)) - 0d

        it("should produce expected results") {
          optimizer.computePartialElbo(lambda(topicIdx), topicIdx)(
            kfr(topicIdx)) shouldEqual expectedPartialElbo
        }
      }

      describe("compute lambda derivative") {
        val expectedDerivative = Seq(
          optimizer.phi_hat_k_t_byV(0, 0) + optimizer.phi_hat_k_t_byV(0, 1) - 9d * exp(1.5d),
          optimizer.phi_hat_k_t_byV(0, 0) + optimizer.phi_hat_k_t_byV(0, 1) - 9d * exp(1.5d)
        )

        it("should produce expected results") {
          optimizer.computeLambdaDerivative(lambda(topicIdx), topicIdx)(kfr(topicIdx))
              .zip(expectedDerivative).foreach {
            case (actual, expected) =>
              breeze.linalg.all(abs(actual - expected) <:< eps) shouldBe true
          }
        }
      }

    }
  }
}
