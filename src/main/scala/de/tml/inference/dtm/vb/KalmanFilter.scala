package de.tml.inference.dtm.vb

import scala.annotation.tailrec

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

case class KalmanFilterResult
(
    mktForward: Seq[DenseVector[Double]],
    mktBackward: Seq[DenseVector[Double]],
    vktForward: Seq[DenseVector[Double]],
    vktBackward: Seq[DenseVector[Double]],

    /** lambda_t -> mkt_t -> DenseVector */
    diffMktForward: Seq[Seq[DenseVector[Double]]],
    diffMktBackward: Seq[Seq[DenseVector[Double]]]
)


object KalmanFilter {

  /**
    * @param lambda DenseMatrix of dimensions numTimeSlices x sizeOfDictionary
    */
  def computeKalmanFilter(lambda: DenseMatrix[Double], sigmaSquared: Double,
                          nySquared: Double): KalmanFilterResult = {

    val numTimeSlices = lambda.rows
    val sizeOfDictionary = lambda.cols

    val vktForward = computeVktForward(sigmaSquared, nySquared,
      DenseVector.ones[Double](sizeOfDictionary) *:* 10d,
      numTimeSlices)

    val mktForward = computeMktForward(sigmaSquared, nySquared,
      DenseVector.zeros[Double](sizeOfDictionary),
      vktForward, lambda)

    val vktBackward = computeVktBackward(sigmaSquared, vktForward)

    val mktBackward = computeMktBackward(sigmaSquared, mktForward, vktForward)

    val diffMktForward = computeDiffMktForward(sigmaSquared, nySquared,
      DenseVector.zeros[Double](sizeOfDictionary), vktForward, numTimeSlices)

    val diffMktBackward = diffMktForward.map(computeMktBackward(sigmaSquared, _, vktForward))

    KalmanFilterResult(mktForward, mktBackward, vktForward, vktBackward, diffMktForward,
      diffMktBackward)
  }

  def computeVktForward(sigmaSquared: Double, nySquared: Double, V0: DenseVector[Double],
                        numTimeSlices: Int): Seq[DenseVector[Double]] = {

    @tailrec
    def computeNext(VktPrev: Seq[DenseVector[Double]], itemsToGo: Int): Seq[DenseVector[Double]] = {
      if (itemsToGo <= 0) {
        VktPrev
      } else {
        val head: DenseVector[Double] = VktPrev.head
        val denominator = head + sigmaSquared + nySquared
        val factor1: DenseVector[Double] = nySquared / denominator

        val factor2: DenseVector[Double] = head + sigmaSquared

        val result: DenseVector[Double] = factor1 *:* factor2

        computeNext(result +: VktPrev, itemsToGo - 1)
      }
    }

    computeNext(Seq(V0), numTimeSlices).reverse
  }

  def computeVktBackward(sigmaSquared: Double, VktForward: Seq[DenseVector[Double]]):
  Seq[DenseVector[Double]] = {

    @tailrec
    def computeNext(VktNext: Seq[DenseVector[Double]], VktForward: Iterator[DenseVector[Double]]):
    Seq[DenseVector[Double]] = {

      if (VktForward.isEmpty) {
        VktNext
      } else {
        val vktForward = VktForward.next()
        val head: DenseVector[Double] = VktNext.head

        val expression = sigmaSquared / (vktForward + sigmaSquared) *:* -1d + 1d
        val factor1: DenseVector[Double] = expression *:* expression

        val factor2: DenseVector[Double] = head + (vktForward *:* -1d) + (-1d * sigmaSquared)

        val result: DenseVector[Double] = vktForward + (factor1 *:* factor2)

        computeNext(result +: VktNext, VktForward)
      }
    }

    val reversedForward = VktForward.reverse
    computeNext(Seq(reversedForward.head), reversedForward.drop(1).iterator)
  }

  def computeMktForward(sigmaSquared: Double, nySquared: Double, m0: DenseVector[Double],
                        Vkt: Seq[DenseVector[Double]], Lambda: DenseMatrix[Double]):
  Seq[DenseVector[Double]] = {

    @tailrec
    def computeNext(MktPrev: Seq[DenseVector[Double]],
                    vktIterator: Iterator[DenseVector[Double]],
                    lambdaIterator: Iterator[DenseVector[Double]]): Seq[DenseVector[Double]] = {
      if (vktIterator.isEmpty || lambdaIterator.isEmpty) {
        MktPrev
      } else {
        val head = MktPrev.head

        val lambdaT = lambdaIterator.next()

        val VktPrev: DenseVector[Double] = vktIterator.next()
        val denominator = VktPrev + sigmaSquared + nySquared
        val factor1: DenseVector[Double] = nySquared / denominator
        val factor2: DenseVector[Double] = -1d * factor1 + 1d

        val result: DenseVector[Double] = factor1 *:* head + factor2 *:* lambdaT

        computeNext(result +: MktPrev, vktIterator, lambdaIterator)
      }
    }

    val lambdaIterator: Iterator[DenseVector[Double]] =
      Seq.range(0, Lambda.rows).map(rowIdx => Lambda(rowIdx, ::).inner).iterator
    computeNext(Seq(m0), Vkt.iterator, lambdaIterator).reverse
  }

  def computeMktBackward(sigmaSquared: Double, MktForward: Seq[DenseVector[Double]],
                         VktForward: Seq[DenseVector[Double]]): Seq[DenseVector[Double]] = {

    @tailrec
    def computeNext(MktNext: Seq[DenseVector[Double]], MktForward: Iterator[DenseVector[Double]],
                    VktForward: Iterator[DenseVector[Double]]): Seq[DenseVector[Double]] = {

      if (MktForward.isEmpty) {
        MktNext
      } else {
        val head: DenseVector[Double] = MktNext.head

        val mktForward = MktForward.next()
        val vktForward = VktForward.next()

        val factor1 = sigmaSquared / (vktForward + sigmaSquared)
        val factor2 = factor1 *:* -1d + 1d

        val result: DenseVector[Double] = factor1 *:* mktForward + factor2 *:* head

        computeNext(result +: MktNext, MktForward, VktForward)
      }
    }

    val reversedForward = MktForward.reverse
    val vktReversedForward = VktForward.reverse
    computeNext(Seq(reversedForward.head), reversedForward.drop(1).iterator,
      vktReversedForward.drop(1).iterator)
  }

  def computeDiffMktForward(sigmaSquared: Double, nySquared: Double, diffm0: DenseVector[Double],
                            Vkt: Seq[DenseVector[Double]], numTimeSlices: Int):
  Seq[Seq[DenseVector[Double]]] = {

    @tailrec
    def computeNext(diffMktPrev: Seq[DenseVector[Double]],
                    vktIterator: Iterator[DenseVector[Double]],
                    lambdaTimeSlicePosition: Int
                   ): Seq[DenseVector[Double]] = {
      if (vktIterator.isEmpty) {
        diffMktPrev
      } else {
        val head = diffMktPrev.head

        val VktPrev: DenseVector[Double] = vktIterator.next()
        val denominator = VktPrev + sigmaSquared + nySquared
        val factor1: DenseVector[Double] = nySquared / denominator
        val factor2: DenseVector[Double] = -1d * factor1 + 1d

        val last_summand: DenseVector[Double] =
          if (lambdaTimeSlicePosition == 0) {
            factor2
          } else {
            DenseVector.zeros[Double](factor2.length)
          }

        val result: DenseVector[Double] = factor1 *:* head + last_summand

        computeNext(result +: diffMktPrev, vktIterator, lambdaTimeSlicePosition - 1)
      }
    }

    Seq.range(0, numTimeSlices).par
        .map(
          computeNext(Seq(diffm0), Vkt.dropRight(1).iterator, _).reverse
        )
        .seq
  }


}
