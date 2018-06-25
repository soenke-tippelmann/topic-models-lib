package de.tml.examples

import de.tml.datasets.DatasetsUtil
import de.tml.datasets.nipscsv.NipsCSVParser
import de.tml.datasets.nipscsv.NipsCSVTimeSlice
import de.tml.evaluators.DocumentGroupEvaluators
import de.tml.evaluators.TimeTopicEvaluators
import de.tml.inference.dtm.vb.DtmVB
import de.tml.inference.dtm.vb.DtmVBCorpusResult
import de.tml.inference.dtm.vb.DtmVBInitializers

object DtmExample {
  def main(args: Array[String]): Unit = {
    val timeCorpus = NipsCSVParser.nipsCSVData(DatasetsUtil.nipsCsv)(timeSliceLength = 5)

    def handlerFunc(currentCycle: Int,
                    result: DtmVBCorpusResult[String, NipsCSVTimeSlice]): Unit = {
      if (currentCycle % 5 == 0) {
        TimeTopicEvaluators.printTopics()(result)
        DocumentGroupEvaluators
            .peakingDocumentsPerGroup(identifier => identifier.split("_")(0), .7)(result)
      }
    }

    val result = DtmVB
        .runInference(timeCorpus, 8, 200,
          initializer = DtmVBInitializers.initializeStateLda[String, NipsCSVTimeSlice](300)
        )(postInitializationHandler = handlerFunc(0, _), postCylceHandler = handlerFunc)

    println("Final result")
    handlerFunc(0, result)

    TimeTopicEvaluators
        .printTopics(showProbabilities = false, seperator = ", ", padding = false)(result)
  }
}
