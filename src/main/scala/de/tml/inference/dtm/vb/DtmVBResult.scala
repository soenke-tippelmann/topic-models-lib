package de.tml.inference.dtm.vb

import scala.collection.mutable.{Map => mMap}
import scala.collection.mutable.{HashMap => mHashMap}

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.argtopk
import breeze.linalg.sum
import breeze.numerics.exp
import de.tml.data.TimeAnnotatedCorpus
import de.tml.data.TimeDocument
import de.tml.data.TimeSlice
import de.tml.data.TimeSliceProvider
import de.tml.inference.result.TimeAnnotatedCorpusResult
import de.tml.inference.result.TimeDocumentResult
import de.tml.inference.result.TimeEvolvingTopic
import de.tml.inference.result.TimeTopic

class DtmVBCorpusResult[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
(override val corpus: TimeAnnotatedCorpus[TokenType, TimeSliceType], gamma: DenseMatrix[Double],
 lambda: IndexedSeq[DenseMatrix[Double]])
    extends TimeAnnotatedCorpusResult[TokenType, TimeSliceType,
        DtmVBDocumentResult[TokenType, TimeSliceType]](corpus) {

  private val documentResults: mMap[Int, DtmVBDocumentResult[TokenType, TimeSliceType]] =
    new mHashMap()

  override def getDocumentResultById(id: Int): DtmVBDocumentResult[TokenType, TimeSliceType] = {
    documentResults.getOrElseUpdate(id, {
      val data = (gamma(id, ::) /:/ sum(gamma(id, ::))).t.toScalaVector()

      val mixture: Map[Int, Double] =
        Map(allTopics.map(topic => (topic.index, data(topic.index))): _*)

      DtmVBDocumentResult(corpus.getDocumentById(id), mixture, allTopics)
    })
  }

  override val allTopics: IndexedSeq[DtmVBTopic[TokenType, TimeSliceType]] =
    lambda.zipWithIndex.map {
      case (lambdaTopic, index) =>
        DtmVBTopic[TokenType, TimeSliceType](index, lambdaTopic, corpus.timeSliceProvider)
    }
}

case class DtmVBDocumentResult[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
(override val document: TimeDocument[TimeSliceType],
 mixture: Map[Int, Double], topicsByIndex: IndexedSeq[DtmVBTopic[TokenType, TimeSliceType]])
    extends TimeDocumentResult[TokenType, TimeSliceType](document) {

  override def getTopicMixtureByTopicIndex: Map[Int, Double] = mixture

  override def getTopicMixture: Map[TimeEvolvingTopic[TokenType, TimeSliceType], Double] =
    Map(mixture.map { case (i, d) => (topicsByIndex(i), d) }.toSeq: _*)
}

case class DtmVBTopic[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
(override val index: Int, lambda: DenseMatrix[Double],
 timeSliceProvider: TimeSliceProvider[TimeSliceType])
    extends TimeEvolvingTopic[TokenType, TimeSliceType](index) {

  override def getTopicsIterator: Iterator[DtmVBTimeTopic[TokenType, TimeSliceType]] =
    timeSliceProvider.makeIterator
        .map(slice => (slice, timeSliceProvider.getTimeSlicePositionIdx(slice)))
        .map {
          case (slice, position) => DtmVBTimeTopic(index, slice, lambda(position, ::).inner)
        }
}

case class DtmVBTimeTopic[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
(override val index: Int, timeslice: TimeSliceType, lambda: DenseVector[Double])
    extends TimeTopic[TokenType, TimeSliceType](index, timeslice) {


  private lazy val lambdaExp = exp(lambda)
  lazy val normalizedLambda: Vector[Double] =
    (lambdaExp /:/ sum(lambdaExp)).toScalaVector()

  override def getProbabilityOfTokenByTokenId(tokenId: Int): Double = normalizedLambda(tokenId)

  override def getTopNTokenIds(n: Int): IndexedSeq[Int] = argtopk(lambda, n)

  override def getDistribution: Seq[Double] = normalizedLambda
}