package de.tml.evaluators

import java.io.PrintStream

import de.tml.data.TimeSlice
import de.tml.inference.result.TimeAnnotatedCorpusResult
import de.tml.inference.result.TimeDocumentResult

object TimeTopicEvaluators {

  def printTopics[TokenType, TimeSliceType <: TimeSlice[TimeSliceType],
  DocumentResultType <: TimeDocumentResult[TokenType, TimeSliceType]]
  (numTopTokens: Int = 50, showProbabilities: Boolean = true, seperator: String = "\n",
   padding: Boolean = true, outputStream: PrintStream = System.out)
  (result: TimeAnnotatedCorpusResult[TokenType, TimeSliceType, DocumentResultType]): Unit = {

    result.allTopics
        .seq
        .sortBy(topic => topic.index)
        .foreach(
          topic => {
            outputStream.println(s"Topic: ${topic.index}")
            topic.getTopicsIterator
                .map(timeTopic => {
                  (timeTopic.timeSlice.identifier,
                      timeTopic.getTopNTokenIds(numTopTokens)
                          .map(tokenId => (result.corpus.wordList.getTokenById(tokenId),
                              timeTopic.getProbabilityOfTokenByTokenId(tokenId)))
                          .toList
                          .sortBy(_._2)
                          .reverse
                          .map {
                            case (token, prob) =>
                              val paddedToken =
                                (if (padding) " " * (20 - token.toString.length) else "") + token.toString
                              s"$paddedToken" + (if (showProbabilities) s"\t\t $prob" else "")
                          }
                          .mkString(seperator))
                })
                .toList
                .sortBy(_._1)
                .foreach(outputStream.println)
          }
        )
  }

}
