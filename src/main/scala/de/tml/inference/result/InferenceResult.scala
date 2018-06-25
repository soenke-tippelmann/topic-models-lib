package de.tml.inference.result

import de.tml.data.Corpus
import de.tml.data.Document
import de.tml.data.Wordlist

abstract class CorpusResult[TokenType, DocumentResultType <: DocumentResult[TokenType]]
(val corpus: Corpus[TokenType]) {
  def getDocumentResultById(id: Int): DocumentResultType

  val allTopics: IndexedSeq[Topic[TokenType]]

  val resultsIterator: Iterator[DocumentResultType] =
    corpus.getDocumentsIterator.map(getDocumentResultByDocument)

  def getDocumentResultByIdentifier(identifier: String): DocumentResultType =
    getDocumentResultById(corpus.getDocumentIdByIdentifier(identifier))

  def getDocumentResultByDocument(document: Document): DocumentResultType =
    getDocumentResultByIdentifier(document.identifier)

  lazy val allDocumentResults: IndexedSeq[DocumentResultType] =
    corpus.allDocuments.map(getDocumentResultByDocument)
}

abstract class DocumentResult[TokenType](document: Document) {
  def getTopicMixture: Map[Topic[TokenType], Double]
}

abstract class Topic[TokenType](val index: Int) {
  def getProbabilityOfTokenByTokenId(tokenId: Int): Double

  def getTopNTokenIds(n: Int): IndexedSeq[Int]

  def getDistribution: Seq[Double]
}

class TopicDecorator[TokenType](topic: Topic[TokenType], wordlist: Wordlist[TokenType]) {
  def getProbabilityOfToken(token: TokenType): Double =
    topic.getProbabilityOfTokenByTokenId(wordlist.getIdForToken(token))

  def getTopNTokens(n: Int): IndexedSeq[TokenType] =
    topic.getTopNTokenIds(n).map(wordlist.getTokenById)
}