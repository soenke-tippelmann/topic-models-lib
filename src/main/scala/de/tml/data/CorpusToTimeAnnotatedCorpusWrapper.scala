package de.tml.data

class CorpusToTimeAnnotatedCorpusWrapper[TokenType]
(corpus: Corpus[TokenType])
    extends TimeAnnotatedCorpus[TokenType, CTTACTimeSlice] {

  override val timeSliceProvider: TimeSliceProvider[CTTACTimeSlice] = new CTTACTimeSliceProvider()
  private val timeSliceObject = timeSliceProvider.createTimeSliceById("default")

  override def getDocumentsIterator: Iterator[TimeDocument[CTTACTimeSlice]] =
    corpus.getDocumentsIterator.map(doc => new CTTACTDocument(doc, timeSliceObject))

  override val wordList: Wordlist[TokenType] = corpus.wordList
}

object CorpusToTimeAnnotatedCorpusWrapper {
  def apply[TokenType](corpus: Corpus[TokenType]): CorpusToTimeAnnotatedCorpusWrapper[TokenType] =
    new CorpusToTimeAnnotatedCorpusWrapper(corpus)
}

class CTTACTDocument(doc: Document, timeSlice: CTTACTimeSlice)
    extends TimeDocument[CTTACTimeSlice](doc.identifier, timeSlice) {
  override def getTokenCount: Int = doc.getTokenCount

  override def getTokenAt(index: Int): Int = doc.getTokenAt(index)
}

case class CTTACTimeSlice() extends TimeSlice[CTTACTimeSlice]("Default_Time_Slice") {
  override def compare(that: CTTACTimeSlice): Int = 0
}

class CTTACTimeSliceProvider extends TimeSliceProvider[CTTACTimeSlice] {
  override protected def makeTimeSliceForId(id: String): CTTACTimeSlice = CTTACTimeSlice()

  override def generatePrevTimeSliceId(id: String): String = s"${id}_PREV"

  override def generateNextTimeSliceId(id: String): String = s"${id}_NEXT"
}