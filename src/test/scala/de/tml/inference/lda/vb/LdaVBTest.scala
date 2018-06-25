package de.tml.inference.lda.vb

import scala.collection.immutable.Vector

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Transpose
import breeze.numerics._
import de.tml.data.Document
import org.scalatest.FunSpec
import org.scalatest.Matchers

case class TestDocument(override val identifier: String,
                        tokens: Vector[Int]) extends Document(identifier) {
  override def getTokenCount: Int = tokens.length

  override def getTokenAt(index: Int): Int = tokens(index)
}

class LdaVBTest extends FunSpec with Matchers {
  val eps = 1e-10

  describe("Lda Variational Bayes inference update Rules") {

    describe("given documents and hyperparameters") {
      val numTopics = 3
      val numDocuments = 3
      val numWordTokens = 5

      val alpha: DenseVector[Double] = DenseVector(1d, 2d, 3d)
      val eta: DenseVector[Double] = DenseVector(1d, 2d, 3d, 4d, 5d)

      val documents = Vector(
        TestDocument("doc 1", Vector(1, 3, 3)),
        TestDocument("doc 2", Vector(0, 2)),
        TestDocument("doc 3", Vector(0, 3, 2, 4))
      )

      describe("given phi") {
        val phi: IndexedSeq[DenseMatrix[Double]] =
          Vector(
            DenseMatrix.create(3, numTopics, Array(.4, .2, .7, .5, .3, .1, .1, .5, .2)),
            DenseMatrix.create(2, numTopics, Array(.6, .7, .2, 0d, .2, .3)),
            DenseMatrix.create(4, numTopics, Array(.1, .1, .6, .2, .8, .5, .3, .3, .1, .4, .1, .5))
          )

        describe("updateGamma") {
          val gamma: DenseMatrix[Double] = DenseMatrix.ones(numDocuments, numTopics)

          // expected outcome
          val expectedGamma: DenseMatrix[Double] = DenseMatrix.zeros(numDocuments, numTopics)
          expectedGamma(0, ::) := DenseVector(2.3, 2.9, 3.8).t
          expectedGamma(1, ::) := DenseVector(2.3, 2.2, 3.5).t
          expectedGamma(2, ::) := DenseVector(2d, 3.9, 4.1).t

          // calculation
          LdaVB_updateRules.updateGamma(alpha, phi, gamma)

          it("gamma should be transformed to expectedGamma") {
            breeze.linalg.all(abs(gamma - expectedGamma) <:< eps) shouldBe true
          }
        }

        describe("updateLambda") {
          val lambda: DenseMatrix[Double] = DenseMatrix.ones(numTopics, numWordTokens)

          // expected outcome
          val expectedLambda: DenseMatrix[Double] = DenseMatrix.zeros(numTopics, numWordTokens)
          expectedLambda(0, ::) := DenseVector(1.7, 2.4, 4.3, 5d, 5.2).t
          expectedLambda(1, ::) := DenseVector(2d, 2.5, 3.3, 4.9, 5.3).t
          expectedLambda(2, ::) := DenseVector(1.3, 2.1, 3.4, 5.1, 5.5).t

          // calculation
          LdaVB_updateRules.updateLambda(lambda, eta, phi, documents, numTopics, numWordTokens)

          it("lambda should be transformed to expectedLambda") {
            breeze.linalg.all(abs(lambda - expectedLambda) <:< eps) shouldBe true
          }
        }
      }
      describe("update phi") {
        //given
        val gamma: DenseMatrix[Double] = DenseMatrix.ones(numDocuments, numTopics)
        gamma(0, ::) := DenseVector(1d, 2d, 3d).t
        gamma(1, ::) := DenseVector(4d, 5d, 6d).t
        gamma(2, ::) := DenseVector(7d, 8d, 9d).t

        val lambda: DenseMatrix[Double] = DenseMatrix.ones(numTopics, numWordTokens)
        lambda(0, ::) := DenseVector(1d, 2d, 3d, 4d, 5d).t
        lambda(1, ::) := DenseVector(6d, 7d, 8d, 9d, 10d).t
        lambda(2, ::) := DenseVector(11d, 12d, 13d, 14d, 15d).t


        val phi: IndexedSeq[DenseMatrix[Double]] =
          Vector(
            DenseMatrix.fill(3, numTopics)(1d / 3),
            DenseMatrix.fill(2, numTopics)(1d / 3),
            DenseMatrix.fill(4, numTopics)(1d / 3)
          )

        val lambda_k_sum = Vector(15d, 40d, 65d)

        def buildDocVector(values: (Double, Double, Int)*): Transpose[DenseVector[Double]] = {
          val numerator = values
              .map { case (gamma_dk, lambda_kw, topicIdx) =>
                digamma(gamma_dk) + digamma(lambda_kw) - digamma(lambda_k_sum(topicIdx))
              }
              .map(value => exp(value))

          val denominator = numerator.sum

          DenseVector(numerator.map(_ / denominator): _*).t
        }

        def buildPhiDoc(docIndex: Int): DenseMatrix[Double] = {
          val phi_doc: DenseMatrix[Double] =
            DenseMatrix.zeros(documents(docIndex).getTokenCount, numTopics)

          for ((token, tokenIndex) <- documents(docIndex).tokens.zipWithIndex) {
            phi_doc(tokenIndex, ::) :=
                buildDocVector(
                  Seq.range(0, 3).map(
                    topic => (gamma(docIndex, topic), lambda(topic, token), topic)
                  ): _*)
          }

          phi_doc
        }

        //expected outcome
        val expectedPhi: IndexedSeq[DenseMatrix[Double]] =
          Vector(Seq.range(0, 3).map(buildPhiDoc): _*)

        //calculation
        LdaVB_updateRules.updatePhi(gamma, lambda, phi, documents)

        it("phi should be trasformed to expectedPhi") {
          phi.zip(expectedPhi).foreach {
            case (phi_d, expected_phi_d) =>
              breeze.linalg.all(abs(phi_d - expected_phi_d) <:< eps) shouldBe true
          }
        }
      }
    }
  }
}
