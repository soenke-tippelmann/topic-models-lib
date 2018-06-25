package de.tml.inference.dtm.vb

import scala.collection.immutable.Vector

import de.tml.data.TimeAnnotatedCorpus
import de.tml.data.TimeDocument
import de.tml.data.TimeSlice
import de.tml.data.TimeSliceProvider
import de.tml.data.Wordlist

class TestTSProvider extends TimeSliceProvider[TestTimeSlice] {
  override protected def makeTimeSliceForId(id: String): TestTimeSlice = TestTimeSlice(id.toInt)

  override def generatePrevTimeSliceId(id: String): String = (id.toInt - 1).toString

  override def generateNextTimeSliceId(id: String): String = (id.toInt + 1).toString
}

case class TestTimeSlice(counter: Int) extends TimeSlice[TestTimeSlice](counter.toString) {
  override def compare(that: TestTimeSlice): Int = counter.compareTo(that.counter)
}

case class TestDocument(override val identifier: String, override val timeSlice: TestTimeSlice,
                        tokens: Vector[Int])
    extends TimeDocument(identifier, timeSlice) {

  override def getTokenCount: Int = tokens.length

  override def getTokenAt(index: Int): Int = tokens(index)
}

case class TestCorpus(documents: Vector[TestDocument], tsProvider: TestTSProvider, numTokens:Int)
    extends TimeAnnotatedCorpus[Int, TestTimeSlice] {
  override val timeSliceProvider: TimeSliceProvider[TestTimeSlice] = tsProvider

  override def getDocumentsIterator: Iterator[TimeDocument[TestTimeSlice]] = documents.iterator

  override val wordList: Wordlist[Int] = new Wordlist[Int] {
    override def getTokenById(id: Int): Int = id

    override val numTokens: Int = TestCorpus.this.numTokens
  }
}