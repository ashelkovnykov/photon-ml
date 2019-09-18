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

import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.optimization._

/**
 * Configuration for a GLM coordinate.
 *
 * @param optimizerType The type of optimizer to use during training
 * @param maximumIterations The upper limit on the number of optimization iterations to perform
 * @param tolerance The relative tolerance limit for optimization
 * @param regularizationContext Regularization context
 */
protected[ml] abstract class CoordinateOptimizationConfiguration(
    val optimizerType: OptimizerType,
    val maximumIterations: Int,
    val tolerance: Double,
    val regularizationContext: RegularizationContext)
  extends Serializable {

  require(0 < maximumIterations, s"Less than 1 specified for maximumIterations (specified: $maximumIterations")
  require(0.0d <= tolerance, s"Specified negative tolerance for optimizer: $tolerance")
}

/**
 * Configuration for a [[com.linkedin.photon.ml.algorithm.FixedEffectCoordinate]].
 *
 * @param optimizerType The type of optimizer to use during training
 * @param maximumIterations The upper limit on the number of optimization iterations to perform
 * @param tolerance The relative tolerance limit for optimization
 * @param regularizationContext Regularization context
 * @param downSamplingRate Down-sampling rate
 */
case class FixedEffectOptimizationConfiguration(
    override val optimizerType: OptimizerType,
    override val maximumIterations: Int,
    override val tolerance: Double,
    override val regularizationContext: RegularizationContext = NoRegularizationContext,
    downSamplingRate: Double = 1D)
  extends CoordinateOptimizationConfiguration(
    optimizerType,
    maximumIterations,
    tolerance,
    regularizationContext) {

  require(downSamplingRate > 0.0 && downSamplingRate <= 1.0, s"Unexpected downSamplingRate: $downSamplingRate")
}

/**
 * Configuration for a [[com.linkedin.photon.ml.algorithm.RandomEffectCoordinate]].
 *
 * @param optimizerType The type of optimizer to use during training
 * @param maximumIterations The upper limit on the number of optimization iterations to perform
 * @param tolerance The relative tolerance limit for optimization
 * @param regularizationContext Regularization context
 */
case class RandomEffectOptimizationConfiguration(
    override val optimizerType: OptimizerType,
    override val maximumIterations: Int,
    override val tolerance: Double,
    override val regularizationContext: RegularizationContext = NoRegularizationContext)
  extends CoordinateOptimizationConfiguration(
    optimizerType,
    maximumIterations,
    tolerance,
    regularizationContext)

object CoordinateOptimizationConfiguration {

  /**
   * Parameter extractor helper method.
   *
   * @param config An existing [[CoordinateOptimizationConfiguration]]
   * @return A ([[OptimizerType]], maximum iterations, relative tolerance, [[RegularizationContext]],
   *         regularization weight) tuple, each tuple value coming from the input [[CoordinateOptimizationConfiguration]]
   */
  def unapply(config: CoordinateOptimizationConfiguration): Option[(OptimizerType, Int, Double, RegularizationContext)] =
    Some((
      config.optimizerType,
      config.maximumIterations,
      config.tolerance,
      config.regularizationContext))
}