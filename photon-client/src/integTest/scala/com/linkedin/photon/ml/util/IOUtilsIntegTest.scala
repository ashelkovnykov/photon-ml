/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.util

import org.apache.hadoop.fs.{FileSystem, Path}
import org.joda.time.{DateTime, DateTimeUtils}
import org.testng.Assert._
import org.testng.annotations.{AfterClass, BeforeClass, DataProvider, Test}

import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

/**
 * This class tests [[IOUtils]].
 */
class IOUtilsIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  private val input = getClass.getClassLoader.getResource("IOUtilsTest/input").getPath
  private val baseDir: Path = new Path(input, "daily")
  private val path1: Path = new Path(baseDir, "2016/01/01")
  private val path2: Path = new Path(baseDir, "2016/02/01")
  private val path3: Path = new Path(baseDir, "2016/03/01")
  private val today = "2016-04-01"

  @BeforeClass
  def setup() {
    DateTimeUtils.setCurrentMillisFixed(DateTime.parse(today).getMillis)
  }

  @AfterClass
  def teardown() {
    DateTimeUtils.setCurrentMillisSystem()
  }

  @DataProvider
  def inputPathDataStringProvider(): Array[Array[Any]] = {
    Array(
      Array(DateRange.fromDateString("20160101-20160401"), Seq(path1, path2, path3)),
      Array(DateRange.fromDateString("20160101-20160301"), Seq(path1, path2, path3)),
      Array(DateRange.fromDateString("20160101-20160201"), Seq(path1, path2)),
      Array(DateRange.fromDateString("20160101-20160102"), Seq(path1)),
      Array(DateRange.fromDateString("20160101-20160101"), Seq(path1)),
      Array(DaysRange.fromDaysString("95-1").toDateRange(), Seq(path1, path2, path3)),
      Array(DaysRange.fromDaysString("60-1").toDateRange(), Seq(path2, path3)),
      Array(DaysRange.fromDaysString("45-1").toDateRange(), Seq(path3)))
  }

  /**
   * Test filtering input paths that are within a given date range.
   *
   * @param dateRange The date range to restrict data to
   * @param expectedPaths The expected files
   */
  @Test(dataProvider = "inputPathDataStringProvider")
  def testGetInputPathsWithinDateRange(
      dateRange: DateRange,
      expectedPaths: Seq[Path]): Unit = sparkTest("testGetInputPathsWithinDateRange") {

    assertEquals(
      IOUtils.getInputPathsWithinDateRange(Set(baseDir), dateRange, sc.hadoopConfiguration, errorOnMissing = false),
      expectedPaths)
  }

  /**
   * Test that an empty set of input paths resulting from date range filtering will throw an error.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetInputPathsWithinDateRangeEmpty(): Unit = sparkTest("testGetInputPathsWithinDateRangeEmpty") {

    IOUtils.getInputPathsWithinDateRange(
      Set(baseDir),
      DateRange.fromDateString("19551105-19551106"),
      sc.hadoopConfiguration,
      errorOnMissing = true)
  }

  /**
   * Test whether an directory existing can be correctly determined.
   */
  @Test
  def testIsDirExisting(): Unit = sparkTest("testIsDirExisting") {

    val dir = new Path(getTmpDir)
    val hadoopConfiguration = sc.hadoopConfiguration

    Utils.deleteHDFSDir(dir, hadoopConfiguration)
    assertFalse(IOUtils.isDirExisting(dir, hadoopConfiguration))
    Utils.createHDFSDir(dir, hadoopConfiguration)
    assertTrue(IOUtils.isDirExisting(dir, hadoopConfiguration))
  }

  /**
   * Test preparing an output directory to receive files.
   */
  @Test
  def testProcessOutputDir(): Unit = sparkTest("testProcessOutputDir") {

    val hadoopConfiguration = sc.hadoopConfiguration

    // Case 1: When the output directory already exists and deleteOutputDirIfExists is true
    val dir1 = new Path(getTmpDir)
    IOUtils.processOutputDir(dir1, deleteOutputDirIfExists = true, hadoopConfiguration)
    assertFalse(IOUtils.isDirExisting(dir1, hadoopConfiguration))

    // Case 2: When the output directory already exists and deleteOutputDirIfExists is false
    val dir2 = new Path(getTmpDir)
    try {
      IOUtils.processOutputDir(dir2, deleteOutputDirIfExists = false, hadoopConfiguration)
    } catch {
      case e: Exception => assertTrue(e.isInstanceOf[IllegalArgumentException])
    } finally {
      assertTrue(IOUtils.isDirExisting(dir2, hadoopConfiguration))
    }

    // Case 3: When the output directory does not exist and deleteOutputDirIfExists is true
    val dir3 = new Path(getTmpDir)
    Utils.deleteHDFSDir(dir3, hadoopConfiguration)
    IOUtils.processOutputDir(dir3, deleteOutputDirIfExists = true, hadoopConfiguration)
    assertFalse(IOUtils.isDirExisting(dir3, hadoopConfiguration))

    // Case 4: When the output directory does not exist and deleteOutputDirIfExists is false
    val dir4 = new Path(getTmpDir)
    Utils.deleteHDFSDir(dir4, hadoopConfiguration)
    IOUtils.processOutputDir(dir4, deleteOutputDirIfExists = false, hadoopConfiguration)
    assertFalse(IOUtils.isDirExisting(dir4, hadoopConfiguration))
  }

  /**
   * Test writing to an HDFS file once.
   */
  @Test
  def testWriteFileOnce(): Unit = sparkTest("testWriteFileOnce") {

    val configuration = sc.hadoopConfiguration
    val fs = FileSystem.get(configuration)
    val filePath = new Path(getTmpDir, "wfo")
    val msg = "text"

    val res = IOUtils.writeToFile(configuration, filePath, Seq(msg))
    assertTrue(res.isSuccess)
    assertTrue(fs.exists(filePath))
    assertFalse(fs.exists(filePath.suffix(IOUtils.TMP_SUFFIX)))
    assertFalse(fs.exists(filePath.suffix(IOUtils.BACKUP_SUFFIX)))

    val fileText = IOUtils.readStringsFromHDFS(filePath, configuration)
    assertEquals(fileText.length, 1)
    assertEquals(fileText.head, msg)

    fs.delete(filePath, true)
  }

  /**
   * Test writing to an HDFS file repeatedly.
   */
  @Test(dependsOnMethods = Array("testWriteFileOnce"))
  def testWriteFileRepeated(): Unit = sparkTest("testWriteFileRepeated") {

    val configuration = sc.hadoopConfiguration
    val fs = FileSystem.get(configuration)
    val filePath = new Path(getTmpDir, "wfr")
    val tmpFilePath = filePath.suffix(IOUtils.TMP_SUFFIX)
    val backupFilePath = filePath.suffix(IOUtils.BACKUP_SUFFIX)
    val msg1 = "text1"
    val msg2 = "text2"

    val res1 = IOUtils.writeToFile(configuration, filePath, Seq(msg1))
    assertTrue(res1.isSuccess)
    assertTrue(fs.exists(filePath))
    assertFalse(fs.exists(tmpFilePath))
    assertFalse(fs.exists(backupFilePath))

    val file1Text = IOUtils.readStringsFromHDFS(filePath, configuration)
    assertEquals(file1Text.length, 1)
    assertEquals(file1Text.head, msg1)

    val res2 = IOUtils.writeToFile(configuration, filePath, Seq(msg2))
    assertTrue(res2.isSuccess)
    assertTrue(fs.exists(filePath))
    assertFalse(fs.exists(tmpFilePath))
    assertTrue(fs.exists(backupFilePath))

    val file2Text = IOUtils.readStringsFromHDFS(filePath, configuration)
    assertEquals(file2Text.length, 1)
    assertEquals(file2Text.head, msg2)

    val backupText = IOUtils.readStringsFromHDFS(backupFilePath, configuration)
    assertEquals(backupText.length, 1)
    assertEquals(backupText.head, msg1)

    fs.delete(filePath, true)
    fs.delete(backupFilePath, true)
  }
}
