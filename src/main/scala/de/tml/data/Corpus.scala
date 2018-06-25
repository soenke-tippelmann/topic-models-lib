package de.tml.data

abstract class Wordlist[TokenType] {
  def getTokenById(id: Int): TokenType

  val numTokens: Int

  private lazy val tokenToId =
    Map(Seq.range(0, numTokens).map(index => (getTokenById(index), index)): _*)

  def getIdForToken(token: TokenType): Int = tokenToId(token)

  def hasToken(token: TokenType): Boolean = tokenToId.keySet.contains(token)
}

abstract class Corpus[TokenType] {
  val wordList: Wordlist[TokenType]

  def getDocumentsIterator: Iterator[Document]

  lazy val allDocuments: IndexedSeq[Document] = getDocumentsIterator.toVector

  lazy val numDocuments: Int = allDocuments.length

  protected lazy val docIdentToId: Map[String, Int] = Map(
    allDocuments.zipWithIndex.map { case (d, index) => (d.identifier, index) }: _*)

  def getDocumentIdByIdentifier(identifier: String): Int = docIdentToId(identifier)

  def getDocumentId(document: Document): Int = getDocumentIdByIdentifier(document.identifier)

  def getDocumentById(id: Int): Document = allDocuments(id)

  def getDocumentByIdentifier(identifier: String): Document =
    allDocuments(getDocumentIdByIdentifier(identifier))
}


abstract class Document(val identifier: String) {
  def getTokenCount: Int

  def getTokenAt(index: Int): Int

  def getAllTokens: Iterable[Int] = Seq.range(0, getTokenCount).map(getTokenAt)

  lazy val allTokensVector: Vector[Int] = getAllTokens.toVector
}
