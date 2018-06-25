package de.tml.datasets.newsgroups

import java.io.File
import java.nio.charset.MalformedInputException

import scala.collection.mutable
import scala.util.Failure
import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import de.tml.data.Corpus
import de.tml.data.Document
import de.tml.data.Wordlist
import de.tml.datasets.DatasetsUtil

case class NewsgroupsWordList(words: IndexedSeq[String]) extends Wordlist[String] {
  override def getTokenById(id: Int): String = words(id)

  override val numTokens: Int = words.length
}

case class NewsgroupsCorpus(documents: IndexedSeq[Document],
                            wordlist: NewsgroupsWordList) extends Corpus[String] {
  override val wordList: Wordlist[String] = wordlist

  override lazy val allDocuments: IndexedSeq[Document] = documents

  override def getDocumentsIterator: Iterator[Document] = documents.iterator
}

case class NewsgroupsDocument(override val identifier: String, groupName: String,
                              words: IndexedSeq[Int]) extends Document(identifier) {
  override def getTokenCount: Int = words.length

  override def getTokenAt(index: Int): Int = words(index)
}

object NewsgroupsParser extends LazyLogging {

  def newsgroupsData(dataset: File = DatasetsUtil.newsgroups,
                     selectGroups: Option[Seq[String]] = None)
                    (minWordLength: Int = 3, minWordCount: Int = 10,
                     maxWordCountPercentage: Double = .9, stopwords: Set[String] = Set.empty,
                     removeEmailHeader: Boolean = false,
                     removeDigits: Boolean = false): NewsgroupsCorpus = {

    logger.info("Start reading files.")

    def stripDigits(filecontent: String): String = {
      if (removeDigits) {
        filecontent.replaceAll("[\\d]", "")
      } else {
        filecontent
      }
    }

    def stripEmailHeader(filecontent: String) = {
      if (removeEmailHeader) {
        val parts: Array[String] = filecontent.split("Subject:|Message-ID:|Lines:")
        if (parts.length < 4) {
          ""
        } else {
          (Seq(1) ++ Seq.range(3, parts.length)).map(parts(_)).mkString("\n")
        }
      } else {
        filecontent
      }
    }

    def tokenize(filecontent: String): Try[Seq[String]] = {

      val tokens: Try[Seq[String]] = Try {
        stripDigits(stripEmailHeader(filecontent))
            .replaceAll("[\\W_]", " ")
            .split("\\s+")
            .filter(!_.isEmpty)
            .toSeq
      }

      tokens match {
        case Failure(exception) =>
          if (!exception.isInstanceOf[MalformedInputException]) {
            throw exception
          }
        case _ =>
      }

      tokens
    }

    val allEntries: Seq[(String, String, Seq[String])] =
      DatasetsUtil
          .listFilesInTarArchive(dataset)
          .map {
            case (filepath, content) =>
              val parts = filepath.split("/")
              val group = parts(1)
              val filename = parts(2)
              (group, filename, content)
          }
          .filter(!_._3.isEmpty)
          .filter {
            case (group, filename, content) =>
              selectGroups.isEmpty || selectGroups.get.contains(group)
          }
          .map {
            case (group, filename, content) => (group, filename, tokenize(content))
          }
          .filter(_._3.isSuccess)
          .map {
            case (g, n, tokens) => Try {
              // Make sure no falsely encoded strings remain.
              //@todo: There must be a better solution to this.
              tokens.get.toSet
              (g, n, tokens.get)
            }
          }
          .filter(_.isSuccess).map(_.get)


    logger.info("Done reading files, starting to compose wordlist.")

    val wordToWordCount: mutable.Map[String, Int] =
      allEntries
          .flatMap(_._3)
          .foldLeft(mutable.Map[String, Int]())(
            (map, word) => {
              map.put(word, map.getOrElse(word, 0) + 1)
              map
            }
          )

    val numberOfmaxCountWordRemovals = (wordToWordCount.size * (1.0 - maxWordCountPercentage)).toInt
    val maxCountWordRemovals = wordToWordCount.toSeq.sortBy(_._2)
        .takeRight(numberOfmaxCountWordRemovals).map(_._1).toSet

    val allWords: Vector[String] = wordToWordCount
        .filter { case (_, count) => count >= minWordCount }
        .filter { case (word, _) => word.length >= minWordLength }
        .filter { case (word, _) => !maxCountWordRemovals.contains(word.toLowerCase) }
        .filter { case (word, _) => !stopwords.contains(word.toLowerCase) }
        .keySet.toVector.sorted

    val wordlist = NewsgroupsWordList(allWords)

    logger.info("Done composing wordlist, starting to create documents.")

    val documents = allEntries
        .map {
          case (group, identifier, tokens) =>
            val translatedTokens = tokens.filter(wordlist.hasToken).map(wordlist.getIdForToken)
                .toVector
            val name = s"${group}_$identifier"
            NewsgroupsDocument(name, group, translatedTokens)
        }
        .filter(d => d.getTokenCount > 0)
        .toVector

    logger.info("Done creating documents.")

    NewsgroupsCorpus(documents, wordlist)
  }

}
