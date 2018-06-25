package de.tml.examples

import de.tml.datasets.DatasetsUtil
import de.tml.datasets.newsgroups.NewsgroupsParser
import de.tml.inference.lda.vb.LdaVB

object LdaExample {
  def main(args: Array[String]): Unit = {
        val corpus = NewsgroupsParser.newsgroupsData(
          DatasetsUtil.newsgroups,
          selectGroups = Some(
            Seq("comp.graphics", "rec.sport.baseball", "sci.crypt", "sci.med", "talk.religion.misc"))
        )(minWordCount = 15, maxWordCountPercentage = .95, stopwords = DatasetsUtil.enStopwordsSet,
          removeDigits = true, removeEmailHeader = true)

    val result = LdaVB.runInference(corpus, 8, 200)()

    //@todo add some evaluation logic
  }
}
