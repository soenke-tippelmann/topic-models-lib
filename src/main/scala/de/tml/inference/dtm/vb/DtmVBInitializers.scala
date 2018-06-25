package de.tml.inference.dtm.vb

import breeze.linalg.*
import breeze.linalg.Axis
import breeze.linalg.DenseMatrix
import breeze.linalg.sum
import com.typesafe.scalalogging.LazyLogging
import de.tml.data.TimeAnnotatedCorpus
import de.tml.data.TimeSlice
import de.tml.inference.lda.vb.LdaVB

object DtmVBInitializers extends LazyLogging {

  def initializeState[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
  (corpus: TimeAnnotatedCorpus[TokenType, TimeSliceType], numTopics: Int): DtmVBInferenceState = {

    val numTimeSlices = corpus.timeSliceProvider.getNumTimeSlices
    val sizeOfDictionary = corpus.wordList.numTokens

    val initialPhi: IndexedSeq[DenseMatrix[Double]] =
      corpus.allDocuments
          .map(doc => DenseMatrix.rand[Double](doc.getTokenCount, numTopics) + 1e-12)
          .map(matrix => {
            val sum1 = sum(matrix, Axis._1)
            matrix(::, *) /:/ sum1
          })

    val initialLambda: IndexedSeq[DenseMatrix[Double]] =
      IndexedSeq
          .range(0, numTopics)
          .map(
            topicIdx => {
              val randomTopic = DenseMatrix.rand[Double](1, sizeOfDictionary)
              DenseMatrix.vertcat(
                Seq.fill(numTimeSlices)(randomTopic): _*
              )
            }
          )

    DtmVBInferenceState(
      gamma = DenseMatrix.rand[Double](corpus.numDocuments, numTopics) *:* 5d + 2d,
      phi = initialPhi,
      lambda = initialLambda,
      zeta = DenseMatrix.rand[Double](numTimeSlices, numTopics) *:* 5d + 2d
    )
  }

  def initializeStateLda[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]](numLdaSweeps:Int)
  (corpus: TimeAnnotatedCorpus[TokenType, TimeSliceType], numTopics: Int): DtmVBInferenceState = {

    val numTimeSlices = corpus.timeSliceProvider.getNumTimeSlices
    val sizeOfDictionary = corpus.wordList.numTokens

    val ldaResult = new LdaVB[TokenType](corpus, numTopics)()
    Seq.range(0, numLdaSweeps)
        .foreach(iteration => {
          logger.info(s"ldaRun: $iteration")
          ldaResult.executeUpdateCycle()
        })

    val initialPhi: IndexedSeq[DenseMatrix[Double]] =
      ldaResult.phi

    val initialLambda: IndexedSeq[DenseMatrix[Double]] =
      IndexedSeq.range(0, numTopics)
          .map(topicIdx => {
            val lambdaForTopic = ldaResult.lambda(topicIdx, ::).t
            // get expectation of dirichlet
            val normalizedLambdaForTopic = lambdaForTopic /:/ sum(lambdaForTopic)
            val lambdaRowMatrix = normalizedLambdaForTopic.asDenseMatrix
            DenseMatrix.vertcat(
              Seq.fill(numTimeSlices)(lambdaRowMatrix): _*
            )
          })

    val initialGamma: DenseMatrix[Double] = ldaResult.gamma(::, *) /:/ sum(ldaResult.gamma(*, ::))

    DtmVBInferenceState(
      gamma = initialGamma,
      phi = initialPhi,
      lambda = initialLambda,
      zeta = DenseMatrix.rand[Double](numTimeSlices, numTopics) *:* 5d + 2d
    )
  }
}
