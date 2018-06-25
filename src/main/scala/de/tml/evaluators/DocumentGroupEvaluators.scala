package de.tml.evaluators

import java.io.PrintStream

import de.tml.data.TimeSlice
import de.tml.inference.result.TimeAnnotatedCorpusResult
import de.tml.inference.result.TimeDocumentResult

object DocumentGroupEvaluators {

  def peakingDocumentsPerGroup[TokenType, TimeSliceType <: TimeSlice[TimeSliceType],
  DocumentResultType <: TimeDocumentResult[TokenType, TimeSliceType]]
  (groupExtractor: String => String, topicProbabilityThreshold: Double,
   outputStream: PrintStream = System.out)
  (result: TimeAnnotatedCorpusResult[TokenType, TimeSliceType, DocumentResultType]): Unit = {

    val docsAsGroupAndHighestTopic = result.corpus.allDocuments
        .map(d => {
          val group = groupExtractor(d.identifier)
          val topicsRankedByProbabilityDescending = result.getDocumentResultByDocument(d)
              .getTopicMixtureByTopicIndex.toVector.sortBy(_._2).reverse
          val highestTopic = topicsRankedByProbabilityDescending.head._1
          val probability = topicsRankedByProbabilityDescending.head._2

          (group, highestTopic, probability)
        })

    val numDocumentsPerGroup =
      docsAsGroupAndHighestTopic
          .groupBy(_._1)
          .mapValues(_.size)

    val grouping: Seq[(String, Map[Int, Int])] =
      docsAsGroupAndHighestTopic
          .filter {
            case (group, highestTopic, probability) => probability > topicProbabilityThreshold
          }
          .map {
            case (group, highestTopic, probability) => (group, highestTopic)
          }
          .groupBy(_._1)
          .mapValues(
            (sequence: IndexedSeq[(String, Int)]) =>
              sequence.map(_._2).groupBy(identity).mapValues(_.size)
          )
          .toSeq
          .sortBy(_._1)


    grouping.foreach {
      case (group, countings) =>
        val overallCount = countings.values.sum
        val docsInGroupCount = numDocumentsPerGroup(group)
        outputStream.println(s"Group: $group ($overallCount / $docsInGroupCount)")
        countings.toVector
            .sortBy(_._1)
            .foreach {
              case (topic, count) => outputStream.println(s"   Topic: $topic: $count")
            }
    }
  }

}
