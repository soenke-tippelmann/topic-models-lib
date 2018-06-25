package de.tml.inference.result

import de.tml.data.Document
import de.tml.data.TimeAnnotatedCorpus
import de.tml.data.TimeDocument
import de.tml.data.TimeSlice

abstract class TimeAnnotatedCorpusResult
[TokenType, TimeSliceType <: TimeSlice[TimeSliceType],
DocumentResultType <: TimeDocumentResult[TokenType, TimeSliceType]]
(val corpus: TimeAnnotatedCorpus[TokenType, TimeSliceType]) {
  def getDocumentResultById(id: Int): DocumentResultType

  val allTopics: IndexedSeq[TimeEvolvingTopic[TokenType, TimeSliceType]]

  def getResultsIterator: Iterator[DocumentResultType] =
    corpus.getDocumentsIterator.map(getDocumentResultByDocument)

  def getDocumentResultByIdentifier(identifier: String): DocumentResultType =
    getDocumentResultById(corpus.getDocumentIdByIdentifier(identifier))

  def getDocumentResultByDocument(document: Document): DocumentResultType =
    getDocumentResultByIdentifier(document.identifier)

  lazy val allDocumentResults: IndexedSeq[DocumentResultType] =
    corpus.allDocuments.map(getDocumentResultByDocument).toVector

  def getDocumentResultsForTimeSlice(time: TimeSliceType): IndexedSeq[DocumentResultType] =
    corpus.getDocumentsByTimeSlice(time).map(getDocumentResultByDocument)

  lazy val allDocumentsByTimeSlice: Iterator[(TimeSliceType, IndexedSeq[DocumentResultType])] =
    corpus.timeSliceProvider.makeIterator.map(t => (t, getDocumentResultsForTimeSlice(t)))
}

abstract class TimeDocumentResult[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
(val document: TimeDocument[TimeSliceType]) {
  def getTopicMixture: Map[TimeEvolvingTopic[TokenType, TimeSliceType], Double]

  def getTopicMixtureByTopicIndex: Map[Int, Double] =
    getTopicMixture.map { case (topic, probability) => (topic.index, probability) }
}

abstract class TimeEvolvingTopic[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
(val index: Int) {
  protected lazy val timeSliceToTopic = Map(getTopicsIterator.map(t => (t.timeSlice, t)).toSeq: _*)

  def getTopicForTimeSlice(time: TimeSliceType): TimeTopic[TokenType, TimeSliceType] =
    timeSliceToTopic(time)

  def getTopicsIterator: Iterator[TimeTopic[TokenType, TimeSliceType]]
}

abstract class TimeTopic[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
(override val index: Int, val timeSlice: TimeSliceType) extends Topic[TokenType](index)
