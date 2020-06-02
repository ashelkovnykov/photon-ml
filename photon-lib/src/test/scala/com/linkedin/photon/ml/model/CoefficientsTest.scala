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
package com.linkedin.photon.ml.model

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.CommonTestUtils

/**
 * Unit tests for Coefficients.
 */
class CoefficientsTest {

  import CoefficientsTest._

  @DataProvider(name = "invalidVectorProvider")
  def makeInvalidVectors(): Array[Array[Vector[Double]]] =
    Array(
      Array(dense(1,2,3), dense(1,2)),
      Array(sparse(2)(1,3)(0,2), sparse(3)(4,5)(0,2))
    )

  @Test(dataProvider = "invalidVectorProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testPreconditions(means: Vector[Double], variances: Vector[Double]): Unit =
    new Coefficients(means, Some(variances))

  @Test
  def testComputeScore(): Unit =
    for { v1 <- List(dense(1,0,3,0), sparse(4)(0,2)(1,3))
          v2 <- List(dense(-1,0,0,1), sparse(4)(0,3)(-1,1)) } {
      assertEquals(Coefficients(v1).computeScore(v2), v1.dot(v2), CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    }

  @Test
  def testEquals(): Unit = {

    val denseMeans1 = dense(1, 0, 2, 0)
    val denseMeans2 = dense(1, 0, 2)
    val denseMeans3 = dense(1, 0, 3, 0)

    val denseVariances1 = dense(0, 5, 0, 7)
    val denseVariances2 = dense(0, 5, 0)
    val denseVariances3 = dense(0, 5, 0, 9)

    val sparseMeans1 = sparse(4)(0, 2)(1, 2)

    val sparseVariances = sparse(4)(1, 3)(5, 7)

    val denseCoefficientsNoVariances1 = Coefficients(denseMeans1)
    val denseCoefficientsNoVariances2 = Coefficients(denseMeans2)
    val denseCoefficientsNoVariances3 = Coefficients(denseMeans3)

    val denseCoefficientsDenseVariances1 = Coefficients(denseMeans1, Some(denseVariances1))
    val denseCoefficientsDenseVariances2 = Coefficients(denseMeans2, Some(denseVariances2))
    val denseCoefficientsDenseVariances3 = Coefficients(denseMeans1, Some(denseVariances3))

    val denseCoefficientsSparseVariances = Coefficients(denseMeans1, Some(sparseVariances))

    val sparseCoefficientsNoVariances = Coefficients(sparseMeans1)

    assertTrue(denseCoefficientsNoVariances1 == denseCoefficientsNoVariances1)
    assertTrue(denseCoefficientsNoVariances2 == denseCoefficientsNoVariances2)
    assertTrue(denseCoefficientsNoVariances3 == denseCoefficientsNoVariances3)

    assertTrue(denseCoefficientsDenseVariances1 == denseCoefficientsDenseVariances1)
    assertTrue(denseCoefficientsDenseVariances2 == denseCoefficientsDenseVariances2)
    assertTrue(denseCoefficientsDenseVariances3 == denseCoefficientsDenseVariances3)

    assertTrue(denseCoefficientsSparseVariances == denseCoefficientsSparseVariances)

    assertTrue(sparseCoefficientsNoVariances == sparseCoefficientsNoVariances)

    // Means are not of same class
    assertFalse(denseCoefficientsNoVariances1 == sparseCoefficientsNoVariances)
    // Means are not of same length
    assertFalse(denseCoefficientsNoVariances1 == denseCoefficientsNoVariances2)
    // Means have different values
    assertFalse(denseCoefficientsNoVariances1 == denseCoefficientsNoVariances3)

    // One missing variance
    assertFalse(denseCoefficientsNoVariances1 == denseCoefficientsDenseVariances1)
    // Variances are not of same class
    assertFalse(denseCoefficientsDenseVariances1 == denseCoefficientsSparseVariances)
    // Variances are not of same length
    assertFalse(denseCoefficientsDenseVariances1 == denseCoefficientsDenseVariances2)
    // Variance values not the same
    assertFalse(denseCoefficientsDenseVariances1 == denseCoefficientsDenseVariances3)
  }
}

object CoefficientsTest {

  /**
   * Helper method to create [[DenseVector]] objects.
   *
   * @param values Ordered [[DenseVector]] values
   * @return New [[DenseVector]] object containing input values
   */
  def dense(values: Double*): DenseVector[Double] = new DenseVector[Double](Array[Double](values: _*))

  /**
   * Helper method to create [[SparseVector]] objects.
   *
   * @param length Length of new [[SparseVector]]
   * @param indices Indices of values in new [[SparseVector]]
   * @param nnz Ordered values (corresponding to indices) for new [[SparseVector]]
   * @return New [[SparseVector]] object containing input values at input indices
   */
  def sparse(length: Int)(indices: Int*)(nnz: Double*): SparseVector[Double] =
    new SparseVector[Double](Array[Int](indices: _*), Array[Double](nnz: _*), length)
}
