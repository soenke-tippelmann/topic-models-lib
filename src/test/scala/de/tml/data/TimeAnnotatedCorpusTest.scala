package de.tml.data

import org.scalatest.path.FunSpec
import org.scalatest.Matchers

case class TestTimeSlice(time: Int) extends TimeSlice[TestTimeSlice](time.toString) {
  override def compare(that: TestTimeSlice): Int = time.compareTo(that.time)
}

class TestTimeSliceProvider extends TimeSliceProvider[TestTimeSlice] {
  override protected def makeTimeSliceForId(id: String): TestTimeSlice = TestTimeSlice(id.toInt)

  override def generatePrevTimeSliceId(id: String): String = (id.toInt - 1).toString

  override def generateNextTimeSliceId(id: String): String = (id.toInt + 1).toString
}

class TimeAnnotatedCorpusTest extends FunSpec with Matchers {

  describe("A TimeSliceProvider") {
    val timeSliceProvider: TimeSliceProvider[TestTimeSlice] = new TestTimeSliceProvider

    it("should be empty at first") {
      timeSliceProvider.makeIterator shouldEqual Iterator.empty
    }

    it("should not have min and max values") {
      timeSliceProvider.getMinTimeSlice.isFailure shouldBe true
      timeSliceProvider.getMaxTimeSlice.isFailure shouldBe true
    }

    describe("If we add year 2010") {
      val slice2010 = timeSliceProvider.createTimeSliceById("2010")

      it("slice should have the year 2010") {
        slice2010.time shouldEqual 2010
      }

      it("contains only one element") {
        val slicesList = timeSliceProvider.makeIterator.toList

        slicesList.size shouldBe 1
        slicesList should contain(slice2010)
      }

      it("minimum = maximum = slice2010") {
        // should exist, otherwise we get an exception right here
        val min = timeSliceProvider.getMinTimeSlice.get
        val max = timeSliceProvider.getMaxTimeSlice.get

        min shouldEqual slice2010
        max shouldEqual slice2010
      }

      describe("If we further add year 2003") {

        val slice2003 = timeSliceProvider.createTimeSliceById("2003")

        it("the produced years should be the sequence from 2003 to 2010") {
          val includedYears = timeSliceProvider.makeIterator.map(_.time).toList
          includedYears shouldEqual Seq.range(2003, 2011)
        }

        it("minimum should be slice2003") {
          timeSliceProvider.getMinTimeSlice.get shouldEqual slice2003
        }

        it("maximum should be slice2010") {
          timeSliceProvider.getMaxTimeSlice.get shouldEqual slice2010
        }

        describe("If we further add year 2015") {

          val slice2015 = timeSliceProvider.createTimeSliceById("2015")

          it("the produced years should be the sequence from 2003 to 2015") {
            val includedYears = timeSliceProvider.makeIterator.map(_.time).toList
            includedYears shouldEqual Seq.range(2003, 2016)
          }

          it("minimum should be slice2003") {
            timeSliceProvider.getMinTimeSlice.get shouldEqual slice2003
          }

          it("maximum should be slice2015") {
            timeSliceProvider.getMaxTimeSlice.get shouldEqual slice2015
          }
        }
      }
    }
  }
}
