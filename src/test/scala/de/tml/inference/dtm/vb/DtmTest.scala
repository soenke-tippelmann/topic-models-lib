package de.tml.inference.dtm.vb

import breeze.linalg.DenseVector
import org.scalatest.FunSpec
import org.scalatest.Matchers

abstract class DtmTest extends FunSpec with Matchers {
  val eps = 1e-10

  val numTopics = 3
  val numDocuments = 3
  val numWordTokens = 5
  val numTimeSlices = 2

  val tsprovider = new TestTSProvider
  val timeSlice1: TestTimeSlice = tsprovider.createTimeSliceById("1")
  val timeSlice2: TestTimeSlice = tsprovider.createTimeSliceById("2")

  val alpha: DenseVector[Double] = DenseVector(1d, 2d, 3d)
  val eta: DenseVector[Double] = DenseVector(1d, 2d, 3d, 4d, 5d)

  val sigmaSquared: Double = 2d
  val nySquared: Double = 3d

  val documents = Vector(
    TestDocument("doc 1", timeSlice1, Vector(1, 3, 3)),
    TestDocument("doc 2", timeSlice1, Vector(0, 2)),
    TestDocument("doc 3", timeSlice2, Vector(0, 3, 2, 4))
  )

  val corpus = TestCorpus(documents, tsprovider, numWordTokens)
}
