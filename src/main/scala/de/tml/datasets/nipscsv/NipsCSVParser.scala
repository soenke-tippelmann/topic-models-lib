package de.tml.datasets.nipscsv

import java.io.File

import scala.collection.mutable

import com.typesafe.scalalogging.LazyLogging
import de.tml.data.TimeAnnotatedCorpus
import de.tml.data.TimeDocument
import de.tml.data.TimeSlice
import de.tml.data.TimeSliceProvider
import de.tml.data.Wordlist
import de.tml.datasets.DatasetsUtil

case class NipsWordList(words: Seq[String]) extends Wordlist[String] {
  override def getTokenById(id: Int): String = words(id)

  override val numTokens: Int = words.length
}

case class NipsCSVCorpus(documents: Map[String, TimeDocument[NipsCSVTimeSlice]],
                         wordlist: NipsWordList,
                         timeSliceProvider: TimeSliceProvider[NipsCSVTimeSlice])
    extends TimeAnnotatedCorpus[String, NipsCSVTimeSlice] {

  override val wordList: Wordlist[String] = wordlist

  override def getDocumentByIdentifier(identifier: String): TimeDocument[NipsCSVTimeSlice] =
    documents(identifier)

  override val getDocumentsIterator: Iterator[TimeDocument[NipsCSVTimeSlice]] =
    documents.valuesIterator
}

case class NipsCSVTimeSlice(year: Int, timesliceLength: Int)
    extends TimeSlice[NipsCSVTimeSlice](
      NipsCSVTimeSlice.getTimeSliceIdByYear(year, timesliceLength)) {

  override def compare(that: NipsCSVTimeSlice): Int = year.compareTo(that.year)
}

object NipsCSVTimeSlice {
  def getTimeSliceIdByYear(year: Int, timesliceLength: Int): String =
    (year - (year % timesliceLength)).toString
}

class NipsCSVTimeSliceProvider(timesliceLength: Int)
    extends TimeSliceProvider[NipsCSVTimeSlice] {

  override protected def makeTimeSliceForId(id: String): NipsCSVTimeSlice = {
    if (id.toInt % timesliceLength != 0) {
      throw new IllegalArgumentException(s"$id is not a vailid start of a TimeSlice!")
    }
    NipsCSVTimeSlice(id.toInt, timesliceLength)
  }

  override def generatePrevTimeSliceId(id: String): String = (id.toInt - timesliceLength).toString

  override def generateNextTimeSliceId(id: String): String = (id.toInt + timesliceLength).toString
}

case class NipsCSVDocument(override val identifier: String,
                           override val timeSlice: NipsCSVTimeSlice,
                           words: Array[Int])
    extends TimeDocument[NipsCSVTimeSlice](identifier, timeSlice) {

  private val wordsExpanded = words.zipWithIndex
      .flatMap { case (count, index) => Seq.fill(count)(index) }

  override def getTokenCount: Int = wordsExpanded.length

  override def getTokenAt(index: Int): Int = wordsExpanded(index)
}

object NipsCSVParser extends LazyLogging {

  private case class ProgressState(starttime: Long, previousElapsedTime: Long,
                                   previousLetter: Option[String])

  private def initProgressState: ProgressState = {
    val starttime = System.currentTimeMillis()
    ProgressState(starttime, starttime, None)
  }

  private def monitorProgress(prevState: ProgressState, word: String): ProgressState = {
    val firstLetterOfWord = word.charAt(0).toString

    if (prevState.previousLetter.forall(!word.startsWith(_))) {
      val currentTime = System.currentTimeMillis()

      val elapsedTime = currentTime - prevState.starttime
      val timeTaken = currentTime - prevState.previousElapsedTime

      logger.info(f"Current state: $word%20s, time: $elapsedTime%6d ($timeTaken%d)")

      ProgressState(prevState.starttime, currentTime, Some(firstLetterOfWord))
    } else {
      prevState
    }
  }

  def parseNipsCSV(datasetPath: File)
                  (timeSliceLength: Int = 1): (Array[String], Map[String, Array[Int]]) = {

    val bufferedSource = io.Source.fromFile(datasetPath)

    val lines: Iterator[String] = bufferedSource.getLines

    val firstLine: String = lines.next()
    val documentNames = firstLine.split(",").map(_.trim).drop(1).toSeq

    var progressState = initProgressState

    val (wordlistQ, docWordCountsQs): (mutable.Queue[String], Seq[mutable.Queue[Int]]) =
      lines.foldLeft(
        (mutable.Queue.empty[String], documentNames.map(_ => mutable.Queue.empty[Int]))
      )(
        (entry: (mutable.Queue[String], Seq[mutable.Queue[Int]]), nextLine: String) => {
          val columns: Array[String] = nextLine.split(",").map(_.trim)
          val word = columns(0).replace("\"", "")
          val counts = columns.drop(1).map(_.toInt).toSeq

          progressState = monitorProgress(progressState, word)

          entry match {
            case (wl, documents) =>
              (wl += word, documents.zip(counts).map { case (doc, c) => doc += c })
          }
        }
      )

    bufferedSource.close

    val wordlist: Array[String] = wordlistQ.toArray

    val docNameToWordcount: Seq[(String, Array[Int])] =
      documentNames.zip(docWordCountsQs.map(_.toArray))
    val docToWordcounts: Map[String, Array[Int]] = Map(docNameToWordcount: _*)

    (wordlist, docToWordcounts)
  }

  def nipsCSVData(dataset: File = DatasetsUtil.nipsCsv)(timeSliceLength: Int = 1): NipsCSVCorpus = {
    val (wordlist, docToWordcounts) = parseNipsCSV(dataset)(timeSliceLength)

    def getYearFromDocId(docId: String) = docId.split('_')(0).replace("\"", "")

    val timeSliceProvider = new NipsCSVTimeSliceProvider(timeSliceLength)

    // caution: this map has side effects and depends on timeSliceProvider
    val documents = docToWordcounts
        .map {
          case (docId, wordCounts) =>
            val year = getYearFromDocId(docId).toInt
            val timeSliceId = NipsCSVTimeSlice.getTimeSliceIdByYear(year, timeSliceLength)
            val timeSlice = timeSliceProvider.createTimeSliceById(timeSliceId)
            (docId, NipsCSVDocument(docId, timeSlice, wordCounts))
        }

    NipsCSVCorpus(documents, NipsWordList(wordlist), timeSliceProvider)
  }

}
