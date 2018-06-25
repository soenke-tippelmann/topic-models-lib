package de.tml.datasets

import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.net.URL
import java.nio.file.Paths
import java.util.zip.GZIPInputStream

import scala.annotation.tailrec
import scala.io.Source
import scala.sys.process._

import org.apache.commons.compress.archivers.ArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream

object DatasetsUtil {

  lazy val CACHE_DIRECTORY: String = getCacheDirectory

  val NIPS_CSV_URL: String = "https://archive.ics.uci.edu/ml/machine-learning-databases/00371/" +
      "NIPS_1987-2015.csv"
  def nipsCsv: File = getUrlAsFile(DatasetsUtil.NIPS_CSV_URL, "csv")

  val NEWSGROUPS_TAR_GZ_URL: String = "https://archive.ics.uci.edu/ml/machine-learning-databases/" +
      "20newsgroups-mld/20_newsgroups.tar.gz"
  def newsgroups: File = getUrlAsFile(DatasetsUtil.NEWSGROUPS_TAR_GZ_URL, "tar.gz")

  val EN_STOPWORDS_PATH: String = "https://gist.githubusercontent.com/sebleier/554280/raw/" +
      "7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
  def enStopwords: File = getUrlAsFile(DatasetsUtil.EN_STOPWORDS_PATH, "txt")
  def enStopwordsSet: Set[String] = Source.fromFile(enStopwords).getLines().map(_.trim).toSet

  private def getCacheDirectory: String = {
    val osTmpDir = System.getProperty("java.io.tmpdir")
    val cacheDirName = "topic_models_lib_faamoaJuoquerohkeiXe"
    val cacheDir = Paths.get(osTmpDir, cacheDirName).toFile
    cacheDir.mkdir()
    cacheDir.getAbsolutePath
  }

  def getTmpFileFromFilename(filename: String, extension: String): File = {
    Paths.get(CACHE_DIRECTORY, filename + '.' + extension).toFile
  }

  @tailrec
  def getRandomTmpFile(extension: String): File = {
    val r = new scala.util.Random()
    val filename = r.alphanumeric.filter(_.isLetter).take(20).mkString
    val file = getTmpFileFromFilename(filename, extension)

    if (file.exists()) {
      getRandomTmpFile(extension)
    } else {
      file
    }
  }

  def getLocalPathFromUrl(url: String, extension: String): File = {
    val filename = url.hashCode.toString
    getTmpFileFromFilename(filename, extension)
  }

  def downloadUrlToFile(url: String, file: File): Unit = {
    new URL(url) #> file !!
  }

  def localCopyForUrlExists(url: String, extension: String): Boolean = {
    getLocalPathFromUrl(url, extension).exists()
  }

  def getUrlAsFile(url: String, extension: String): File = {
    val localFile = getLocalPathFromUrl(url, extension)

    if (!localCopyForUrlExists(url: String, extension: String)) {
      downloadUrlToFile(url, localFile)
    }

    localFile
  }

  /**
    * @return a sequence of (filename, content) tuples
    */
  def listFilesInTarArchive(file: File): Seq[(String, String)] = {

    val tarInputStream: TarArchiveInputStream = new TarArchiveInputStream(
      new GZIPInputStream(new FileInputStream(file)))

    def readFile(element: ArchiveEntry): (String, String) = {
      val fileName = element.getName

      val BUFFER_MAX_SIZE = 1024
      val buffer = new Array[Byte](BUFFER_MAX_SIZE)
      val baos = new ByteArrayOutputStream()
      val outputStream = new BufferedOutputStream(baos, BUFFER_MAX_SIZE)

      def readStream(): Unit = {
        val count = tarInputStream.read(buffer, 0, BUFFER_MAX_SIZE)

        if (count != -1) {
          outputStream.write(buffer, 0, count)
          readStream()
        }
      }
      readStream()

      (fileName, baos.toString("UTF-8"))
    }

    @tailrec
    def processElement(entry: Option[ArchiveEntry],
                       result: Seq[(String, String)] = Seq.empty): Seq[(String, String)] = {
      entry match {
        case None => result
        case Some(element) => {
          val newResult = if (!element.isDirectory) {
            result :+ readFile(element)
          } else {
            result
          }

          processElement(Option(tarInputStream.getNextEntry), newResult)
        }
      }
    }

    processElement(Option(tarInputStream.getNextEntry))
  }
}
