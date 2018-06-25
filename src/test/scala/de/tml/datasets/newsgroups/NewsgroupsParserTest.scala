package de.tml.datasets.newsgroups

import de.tml.datasets.DatasetsUtil
import org.scalatest.FunSpec
import org.scalatest.Matchers

class NewsgroupsParserTest extends FunSpec with Matchers {

  describe("NewsgroupsParser") {
    val dataset = DatasetsUtil.newsgroups

    val stopwordsPath: String = DatasetsUtil.getUrlAsFile(DatasetsUtil.EN_STOPWORDS_PATH, "txt")
        .getAbsolutePath

    val newsgroupsCorpus = NewsgroupsParser
        .newsgroupsData(dataset)(stopwords = DatasetsUtil.enStopwordsSet)

    it("") {
      newsgroupsCorpus.allDocuments.size shouldBe 16146

      val docsByGroups: Map[String, IndexedSeq[NewsgroupsDocument]] =
        newsgroupsCorpus.allDocuments.map(_.asInstanceOf[NewsgroupsDocument]).groupBy(_.groupName)

      docsByGroups.keySet.size shouldBe 20
    }
  }

}
