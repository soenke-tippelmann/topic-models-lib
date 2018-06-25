package de.tml.data

import scala.collection.mutable.{Map => mMap}
import scala.util.Try

abstract class TimeAnnotatedCorpus[TokenType, TimeSliceType <: TimeSlice[TimeSliceType]]
    extends Corpus[TokenType] {

  val timeSliceProvider: TimeSliceProvider[TimeSliceType]

  override def getDocumentsIterator: Iterator[TimeDocument[TimeSliceType]]

  override lazy val allDocuments: IndexedSeq[TimeDocument[TimeSliceType]] =
    getDocumentsIterator.toVector

  override def getDocumentById(id: Int): TimeDocument[TimeSliceType] = allDocuments(id)

  override def getDocumentByIdentifier(identifier: String): TimeDocument[TimeSliceType] =
    allDocuments(getDocumentIdByIdentifier(identifier))

  lazy val allTimeSlices: IndexedSeq[TimeSliceType] = allDocuments.map(_.timeSlice).distinct.sorted

  protected lazy val timeSliceToDocuments:
    Map[TimeSliceType, IndexedSeq[TimeDocument[TimeSliceType]]] =
    allDocuments.map(d => (d.timeSlice, d)).groupBy(_._1).mapValues(_.map(_._2))
        .withDefaultValue(IndexedSeq.empty[TimeDocument[TimeSliceType]])

  def getDocumentsByTimeSlice(timeSlice: TimeSliceType): IndexedSeq[TimeDocument[TimeSliceType]] =
    timeSliceToDocuments(timeSlice)
}

abstract class TimeSlice[SubType <: TimeSlice[SubType]](val identifier: String)
    extends Ordered[SubType]

abstract class TimeSliceProvider[TimeSliceType <: TimeSlice[TimeSliceType]] {

  val allTimeSlices: mMap[String, TimeSliceType] = mMap.empty

  def getNumTimeSlices: Int = allTimeSlices.size

  private lazy val timeSliceToPosition: Map[TimeSliceType, Int] =
    Map(makeIterator.zipWithIndex.toSeq: _*)

  /**
    * This method should only be called after all TimeSlices are created, otherwise wrong
    * results will be returned in later calls
    *
    * @todo Add boolean flag that prevents further creations after querying has started
    */
  def getTimeSlicePositionIdx(timeSlice: TimeSliceType): Int = timeSliceToPosition(timeSlice)

  private lazy val allTimeSlicesIndexed = makeIterator.toIndexedSeq

  /**
    * This method should only be called after all TimeSlices are created, otherwise wrong
    * results will be returned in later calls
    *
    * @todo Add boolean flag that prevents further creations after querying has started
    */
  def getTimeSliceAtPosition(timeSlicePosition: Int): TimeSliceType =
    allTimeSlicesIndexed(timeSlicePosition)

  protected def makeTimeSliceForId(id: String): TimeSliceType

  def generatePrevTimeSliceId(id: String): String

  def generateNextTimeSliceId(id: String): String

  def getPrevTimeSlice(slice: TimeSliceType): Option[TimeSliceType] =
    getTimeSliceById(generatePrevTimeSliceId(slice.identifier))

  def getNextTimeSlice(slice: TimeSliceType): Option[TimeSliceType] =
    getTimeSliceById(generateNextTimeSliceId(slice.identifier))

  val timeSliceTypeOrdering: Ordering[TimeSliceType] =
    (x: TimeSliceType, y: TimeSliceType) => x.compareTo(y)

  def getMinTimeSlice: Try[TimeSliceType] = Try(allTimeSlices.values.min(timeSliceTypeOrdering))

  def getMaxTimeSlice: Try[TimeSliceType] = Try(allTimeSlices.values.max(timeSliceTypeOrdering))

  def makeIterator: Iterator[TimeSliceType] =
    if (allTimeSlices.isEmpty) {
      Iterator.empty
    } else {
      new Iterator[TimeSliceType] {
        private var nextElement: Option[TimeSliceType] = Some(getMinTimeSlice.get)

        override def hasNext: Boolean = nextElement.isDefined

        override def next(): TimeSliceType = {
          if (!hasNext) throw new UnsupportedOperationException("No next element available.")
          val currentElement = nextElement.get
          nextElement = getNextTimeSlice(nextElement.get)
          currentElement
        }
      }
    }

  def getTimeSliceById(id: String): Option[TimeSliceType] = allTimeSlices.get(id)

  /**
    * Caution: This operation is not stack safe!
    *
    * @param id
    * @return
    */
  def createTimeSliceById(id: String): TimeSliceType = {
    if (allTimeSlices.get(id).isDefined) {
      allTimeSlices(id)
    } else {
      val newTimeSlice: TimeSliceType = makeTimeSliceForId(id)

      // produce missing timeslices to have a range available
      if (allTimeSlices.nonEmpty) {
        val minSlice = getMinTimeSlice.get
        lazy val maxSlice = getMaxTimeSlice.get

        if (newTimeSlice < minSlice) {
          createTimeSliceById(generateNextTimeSliceId(newTimeSlice.identifier))
        } else if (newTimeSlice > maxSlice) {
          createTimeSliceById(generatePrevTimeSliceId(newTimeSlice.identifier))
        }
      }

      allTimeSlices.put(id, newTimeSlice)
      newTimeSlice
    }
  }
}

abstract class TimeDocument[TimeSliceType <: TimeSlice[TimeSliceType]]
(override val identifier: String, val timeSlice: TimeSliceType)
    extends Document(identifier)
