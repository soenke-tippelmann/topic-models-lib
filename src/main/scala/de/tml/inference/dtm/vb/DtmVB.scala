package de.tml.inference.dtm.vb

import scala.language.postfixOps

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import com.typesafe.scalalogging.LazyLogging
import de.tml.data.TimeAnnotatedCorpus
import de.tml.data.TimeSlice

private[vb] case class DtmVBInferenceState
(
    /** numDocuments x numTopics */
    gamma: DenseMatrix[Double],

    /** Seq[Document] -> tokenCount x numTopics */
    phi: IndexedSeq[DenseMatrix[Double]],

    /** Seq[Topic] ->  numTimeSlices x sizeOfDictionary */
    var lambda: IndexedSeq[DenseMatrix[Double]],

    /** numTimeSlices x num Topics */
    zeta: DenseMatrix[Double]
)

class DtmVB[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
(
    val corpus: TimeAnnotatedCorpus[TokenType, TimeSliceType],
    val numTopics: Int,
    initializer: (TimeAnnotatedCorpus[TokenType, TimeSliceType], Int) =>
        DtmVBInferenceState = DtmVBInitializers.initializeState _
)(
    alpha: DenseVector[Double] = DenseVector.ones(numTopics),
    sigmaSquared: Double = .0005,
    nySquared: Double = .0005
) extends LazyLogging {

  private val numTimeSlices: Int = corpus.timeSliceProvider.getNumTimeSlices
  private val sizeOfDictionary: Int = corpus.wordList.numTokens

  private val currentState: DtmVBInferenceState = initializer(corpus, numTopics)

  logger.info(s"dict: $sizeOfDictionary, numDocs: ${currentState.phi.length}," +
      s" numTopics: $numTopics, numTimeslices: $numTimeSlices")

  corpus.allTimeSlices
      .map(ts => (ts, corpus.getDocumentsByTimeSlice(ts).length))
      .foreach(ts => logger.debug(ts.toString()))

  def executeUpdateCycle(): Unit = {
    logger.info("Updating gamma.")
    DtmVBUpdateRules.updateGamma(alpha, currentState.phi, currentState.gamma)

    logger.info("Calculating kalman equations.")
    val kalman: IndexedSeq[KalmanFilterResult] =
      currentState.lambda.map(KalmanFilter.computeKalmanFilter(_, sigmaSquared, nySquared))

    logger.info("Updating zeta.")
    DtmVBUpdateRules.updateZeta(currentState.zeta, kalman)

    logger.info("Update phi.")
    DtmVBUpdateRules.updatePhi(currentState.gamma, currentState.zeta, kalman, currentState.phi,
      corpus.allDocuments, corpus.timeSliceProvider)

    logger.info("Update lambda.")
    currentState.lambda = DtmVBUpdateRules
        .updateLambda(currentState.phi, currentState.zeta, sigmaSquared, nySquared,
          currentState.lambda, corpus.timeSliceProvider, corpus)
  }

  /**
    * Caution: Reuses the matricies used during inference, ie executing executeUpdateCycle will
    * change these. The correctness of the returned ResultCorpus is not guaranteed and in most
    * situations the contained results after continuing inference will be wrong and have to be
    * freshly exported after the iteration!
    *
    * @return
    */
  def exportUnreliableResult: DtmVBCorpusResult[TokenType, TimeSliceType] =
    new DtmVBCorpusResult[TokenType, TimeSliceType](corpus, currentState.gamma, currentState.lambda)

  /**
    * Caution: Creates deep copies of gamma and lambda matricies.
    *
    * @return
    */
  def exportResult: DtmVBCorpusResult[TokenType, TimeSliceType] =
    new DtmVBCorpusResult[TokenType, TimeSliceType](corpus, currentState.gamma.copy,
      currentState.lambda.map(_.copy))
}

object DtmVB extends LazyLogging {
  def runInference[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
  (corpus: TimeAnnotatedCorpus[TokenType, TimeSliceType], numTopics: Int, numSweeps: Int,
   initializer: (TimeAnnotatedCorpus[TokenType, TimeSliceType], Int) => DtmVBInferenceState =
   DtmVBInitializers.initializeState[TokenType, TimeSliceType] _)
  (
      postInitializationHandler: DtmVBCorpusResult[TokenType, TimeSliceType] => Unit =
      (_: DtmVBCorpusResult[TokenType, TimeSliceType]) => (),
      preCylceHandler: (Int, DtmVBCorpusResult[TokenType, TimeSliceType]) => Unit =
      (_: Int, _: DtmVBCorpusResult[TokenType, TimeSliceType]) => (),
      postCylceHandler: (Int, DtmVBCorpusResult[TokenType, TimeSliceType]) => Unit =
      (_: Int, _: DtmVBCorpusResult[TokenType, TimeSliceType]) => ()
  )
  : DtmVBCorpusResult[TokenType, TimeSliceType] = {

    val dtmVB = new DtmVB[TokenType, TimeSliceType](corpus, numTopics, initializer)()

    postInitializationHandler(dtmVB.exportUnreliableResult)

    for (i <- Seq.range(0, numSweeps)) {
      logger.info(s"Inference round: $i")
      preCylceHandler(i, dtmVB.exportUnreliableResult)
      dtmVB.executeUpdateCycle()
      postCylceHandler(i, dtmVB.exportUnreliableResult)
      logger.info(s"Done with inference round: $i")
    }

    dtmVB.exportUnreliableResult
  }
}
