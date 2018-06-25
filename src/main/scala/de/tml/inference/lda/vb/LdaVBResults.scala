package de.tml.inference.lda.vb

import scala.collection.mutable

import breeze.linalg.DenseMatrix
import breeze.linalg.argtopk
import breeze.linalg.sum
import de.tml.data.Corpus
import de.tml.data.Document
import de.tml.inference.result.CorpusResult
import de.tml.inference.result.DocumentResult
import de.tml.inference.result.Topic

class LdaVBCorpusResult[TokenType](override val corpus: Corpus[TokenType],
                                   gamma: DenseMatrix[Double], lambda: DenseMatrix[Double],
                                   numTopics: Int)
    extends CorpusResult[TokenType, LdaVBDocumentResult[TokenType]](corpus) {

  override lazy val allTopics: IndexedSeq[LdaVBTopic[TokenType]] =
    Seq.range(0, numTopics).toIndexedSeq.map(new LdaVBTopic[TokenType](_, lambda))

  private val documentResults: mutable.Map[Int, LdaVBDocumentResult[TokenType]] =
    new mutable.HashMap()

  override def getDocumentResultById(id: Int): LdaVBDocumentResult[TokenType] =

    documentResults.getOrElseUpdate(id, {
      val data = (gamma(id, ::) /:/ sum(gamma(id, ::))).t.toScalaVector()

      val mixture: Map[LdaVBTopic[TokenType], Double] =
        Map(allTopics.map(topic => (topic, data(topic.index))): _*)

      LdaVBDocumentResult(corpus.getDocumentById(id), mixture)
    })
}

case class LdaVBDocumentResult[TokenType](document: Document,
                                          mixture: Map[LdaVBTopic[TokenType], Double])
    extends DocumentResult[TokenType](document) {

  // ugly workaround caused by invariant key type for scala maps.
  override def getTopicMixture: Map[Topic[TokenType], Double] = Map(mixture.toSeq: _*)
}

class LdaVBTopic[TokenType](override val index: Int, lambda: DenseMatrix[Double])
    extends Topic[TokenType](index) {

  lazy val normalizedLambda: Vector[Double] =
    (lambda(index, ::) /:/ sum(lambda(index, ::))).t.toScalaVector()

  override def getProbabilityOfTokenByTokenId(tokenId: Int): Double = normalizedLambda(tokenId)

  override def getTopNTokenIds(n: Int): IndexedSeq[Int] = argtopk(lambda(index, ::).t, n)

  override def getDistribution: Seq[Double] = normalizedLambda
}