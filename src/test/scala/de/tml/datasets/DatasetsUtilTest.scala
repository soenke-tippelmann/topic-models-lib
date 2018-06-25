package de.tml.datasets

import org.scalatest.FunSpec
import org.scalatest.Matchers

class DatasetsUtilTest extends FunSpec with Matchers {

  describe("DatasetsUtil") {
    describe("getRandomTmpFile") {
      it("should return a non-existent file") {
        val tmpfile = DatasetsUtil.getRandomTmpFile("csv")
        tmpfile.exists() shouldBe false
      }
    }

    describe("given a url and a filename") {
      val url = "https://www.startpage.com/"
      val extension = "html"
      val tmpfile = DatasetsUtil.getRandomTmpFile(extension)

      describe("downloadUrlToFile") {
        it("should write the content to the file") {
          DatasetsUtil.downloadUrlToFile(url, tmpfile)

          tmpfile.exists() shouldBe true

          tmpfile.delete()
        }
      }

      describe("getUrlAsFile") {
        it("should download the file once") {
          DatasetsUtil.localCopyForUrlExists(url, extension) shouldBe false
          val createdFile = DatasetsUtil.getUrlAsFile(url, extension)
          DatasetsUtil.localCopyForUrlExists(url, extension) shouldBe true
          val secondFile = DatasetsUtil.getUrlAsFile(url, extension)
          createdFile shouldEqual secondFile
          createdFile.delete()
        }
      }
    }

    describe("listFilesInTarArchive") {

      val file = DatasetsUtil.getUrlAsFile(DatasetsUtil.NEWSGROUPS_TAR_GZ_URL, "tar.gz")
      val newsgroupsContent = DatasetsUtil.listFilesInTarArchive(file)

      it("should create a sequence of (filename, content) tuples") {
        newsgroupsContent.size shouldBe 19997
      }
    }
  }
}
