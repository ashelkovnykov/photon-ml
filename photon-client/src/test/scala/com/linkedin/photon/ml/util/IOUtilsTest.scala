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

import java.io.File

import scala.collection.mutable.ArrayBuffer

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.TestTemplateWithTmpDir

/**
 * This class tests [[IOUtils]] can correctly read and write primitive/ASCII strings and international/UTF-8 strings.
 */
class IOUtilsTest extends TestTemplateWithTmpDir {

  @DataProvider
  def dataProvider(): Array[Array[Any]] = {
    Array(
      Array("ASCII", "Test string"),
      Array("UTF-8", "テスト"), // "Test" in Japanese katakana
      Array("UTF-8", "飛行場") // "Airport" in Japanese
    )
  }

  /**
   * Test that it's possible to read and write from HDFS.
   *
   * @param dir Test directory name
   * @param testString Text to read/write
   */
  @Test(dataProvider = "dataProvider")
  def testReadAndWrite(dir: String, testString: String): Unit = {

    val tmpDir = new Path(getTmpDir, dir)
    val conf = new Configuration()

    IOUtils.writeStringsToHDFS(List(testString).iterator, tmpDir, conf, forceOverwrite = true)
    val strings = IOUtils.readStringsFromHDFS(tmpDir, conf)

    Assert.assertEquals(strings, ArrayBuffer(testString))

    new File("/tmp/test1").delete
  }
}
