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
package com.linkedin.photon.ml.optimization.game

import org.mockito.Mockito._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.optimization.{OptimizerType, RegularizationContext}
import com.linkedin.photon.ml.util.DoubleRange

class CoordinateOptimizationConfigurationTest {

  import CoordinateOptimizationConfigurationTest._

  @DataProvider
  def invalidInputFixedEffect(): Array[Array[Any]] = Array(
    Array(0, 1E-4, 1D, None, None, 1D),
    Array(1, -0.01, 1D, None, None, 1D),
    Array(1, 1E-4, -1D, None, None, 1D),
    Array(1, 1E-4, 1D, Some(DoubleRange(0D, 1.1)), None, 1D),
    Array(1, 1E-4, 1D, None, Some(DoubleRange(-1D, 10D)), 1D),
    Array(1, 1E-4, 1D, None, None, -1D),
    Array(1, 1E-4, 1D, None, None, 0D),
    Array(1, 1E-4, 1D, None, None, 2D))

  @DataProvider
  def invalidInputRandomEffect(): Array[Array[Any]] = Array(
    Array(0, 1E-4, 1D, None, None),
    Array(1, -0.01, 1D, None, None),
    Array(1, 1E-4, -1D, None, None),
    Array(1, 1E-4, 1D, Some(DoubleRange(0D, 1.1)), None),
    Array(1, 1E-4, 1D, None, Some(DoubleRange(-1D, 10D))))

  /**
   * Test that [[FixedEffectOptimizationConfiguration]] will reject invalid input.
   *
   * @param maxIterations The upper limit on the number of optimization iterations to perform
   * @param tolerance The relative tolerance limit for optimization
   * @param regularizationWeight The regularization weight
   * @param regularizationWeightRange The regularization weight range
   * @param elasticNetParamRange Elastic net alpha range
   * @param downSamplingRate The down-sampling rate
   */
  @Test(dataProvider = "invalidInputFixedEffect", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testFixedEffectOptConfigSetupWithInvalidInput(
      maxIterations: Int,
      tolerance: Double,
      regularizationWeight: Double,
      elasticNetParamRange: Option[DoubleRange],
      regularizationWeightRange: Option[DoubleRange],
      downSamplingRate: Double): Unit = {

    val mockRegularizationContext = mock(classOf[RegularizationContext])

    FixedEffectOptimizationConfiguration(
      OPTIMIZER_TYPE,
      maxIterations,
      tolerance,
      mockRegularizationContext,
      regularizationWeight,
      regularizationWeightRange,
      elasticNetParamRange,
      downSamplingRate)
  }

  /**
   * Test that [[FixedEffectOptimizationConfiguration]] will reject invalid input.
   *
   * @param maxIterations The upper limit on the number of optimization iterations to perform
   * @param tolerance The relative tolerance limit for optimization
   * @param regularizationWeight The regularization weight
   * @param regularizationWeightRange The regularization weight range
   * @param elasticNetParamRange Elastic net alpha range
   */
  @Test(dataProvider = "invalidInputRandomEffect", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testRandomEffectOptConfigSetupWithInvalidInput(
    maxIterations: Int,
    tolerance: Double,
    regularizationWeight: Double,
    elasticNetParamRange: Option[DoubleRange],
    regularizationWeightRange: Option[DoubleRange]): Unit = {

    val mockRegularizationContext = mock(classOf[RegularizationContext])

    RandomEffectOptimizationConfiguration(
      OPTIMIZER_TYPE,
      maxIterations,
      tolerance,
      mockRegularizationContext,
      regularizationWeight,
      regularizationWeightRange,
      elasticNetParamRange)
  }
}

object CoordinateOptimizationConfigurationTest {

  private val OPTIMIZER_TYPE = OptimizerType.LBFGS
}
