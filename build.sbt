
lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "de.topic-models-lib",
      scalaVersion := "2.12.6",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "topic-models-lib",
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % Test,
    libraryDependencies += "org.apache.commons" % "commons-compress" % "1.16.1",
    libraryDependencies += "org.scalanlp" %% "breeze" % "1.0-RC2",
    libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.5",
    libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.0",
    libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3"
  )

  Test / parallelExecution := false
