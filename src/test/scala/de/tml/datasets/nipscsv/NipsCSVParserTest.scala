package de.tml.datasets.nipscsv

import java.io.File

import de.tml.datasets.DatasetsUtil
import org.scalatest.FunSpec
import org.scalatest.Matchers

class NipsCSVParserTest extends FunSpec with Matchers {

  describe("NipsCSVParser") {
    val dataset: File = DatasetsUtil.nipsCsv

    val givenCSV: Iterator[String] = io.Source.fromFile(dataset).getLines()

    val orderOfDocuments: Array[String] = givenCSV.next().split(",").drop(1)

    describe("Regenerating the CSV") {

      val (wordlist, docToWordcount) = NipsCSVParser.parseNipsCSV(dataset)()

      val wordCountLines: Array[StringBuilder] = orderOfDocuments.map(docToWordcount(_))
          .foldLeft(wordlist.map(s => new StringBuilder("\"" + s + "\"")))(
            (result: Array[StringBuilder], column: Array[Int]) => {
              result.zip(column).map { case (r, c) => r ++= "," ++= c.toString }
            }
          )

      val csv: Iterator[String] = wordCountLines.map(_.toString).iterator

      it("") {
        csv.zip(givenCSV).foreach {
          case (csvLine, givenCsvLine) => csvLine shouldEqual givenCsvLine
        }
      }
    }
  }
}
