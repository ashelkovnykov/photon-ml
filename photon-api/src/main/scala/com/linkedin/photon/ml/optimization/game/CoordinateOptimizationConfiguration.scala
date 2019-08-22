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
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Generic trait for a configuration to define a coordinate.
 */
sealed trait CoordinateOptimizationConfiguration

/**
 * Configuration for a GLM coordinate.
 *
 * @param optimizerType The type of optimizer to use during training
 * @param maximumIterations The upper limit on the number of optimization iterations to perform
 * @param tolerance The relative tolerance limit for optimization
 * @param regularizationContext Regularization context
 * @param regularizationWeight Regularization weight
 * @param regularizationWeightRange Regularization weight range
 * @param elasticNetParamRange Elastic net alpha range
 */
protected[ml] abstract class GLMOptimizationConfiguration(
    val optimizerType: OptimizerType,
    val maximumIterations: Int,
    val tolerance: Double,
    val regularizationContext: RegularizationContext,
    val regularizationWeight: Double,
    val regularizationWeightRange: Option[DoubleRange] = None,
    val elasticNetParamRange: Option[DoubleRange] = None)
  extends CoordinateOptimizationConfiguration
    with Serializable {

  require(0 < maximumIterations, s"Less than 1 specified for maximumIterations (specified: $maximumIterations")
  require(0.0d <= tolerance, s"Specified negative tolerance for optimizer: $tolerance")
  require(0 <= regularizationWeight, s"Negative regularization weight: $regularizationWeight")

  regularizationWeightRange.foreach { case DoubleRange(start, _) =>
    require(start > 0.0, "Regularization weight ranges must be positive")
  }

  elasticNetParamRange.foreach { case DoubleRange(start, end) =>
    require(start >= 0.0 && end <= 1.0, "Elastic net alpha ranges must lie within [0, 1]")
  }
}

object GLMOptimizationConfiguration {

  /**
   * Parameter extractor helper method.
   *
   * @param config An existing [[GLMOptimizationConfiguration]]
   * @return A ([[OptimizerType]], maximum iterations, relative tolerance, [[RegularizationContext]],
   *         regularization weight) tuple, each tuple value coming from the input [[GLMOptimizationConfiguration]]
   */
  def unapply(
      config: GLMOptimizationConfiguration): Option[(OptimizerType, Int, Double, RegularizationContext, Double)] =
    Some((
      config.optimizerType,
      config.maximumIterations,
      config.tolerance,
      config.regularizationContext,
      config.regularizationWeight))
}

/**
 * Configuration for a [[com.linkedin.photon.ml.algorithm.FixedEffectCoordinate]].
 *
 * @param optimizerType The type of optimizer to use during training
 * @param maximumIterations The upper limit on the number of optimization iterations to perform
 * @param tolerance The relative tolerance limit for optimization
 * @param regularizationContext Regularization context
 * @param regularizationWeight Regularization weight
 * @param regularizationWeightRange Regularization weight range
 * @param elasticNetParamRange Elastic net alpha range
 * @param downSamplingRate Down-sampling rate
 */
case class FixedEffectOptimizationConfiguration(
    override val optimizerType: OptimizerType,
    override val maximumIterations: Int,
    override val tolerance: Double,
    override val regularizationContext: RegularizationContext = NoRegularizationContext,
    override val regularizationWeight: Double = 0D,
    override val regularizationWeightRange: Option[DoubleRange] = None,
    override val elasticNetParamRange: Option[DoubleRange] = None,
    downSamplingRate: Double = 1D)
  extends GLMOptimizationConfiguration(
    optimizerType,
    maximumIterations,
    tolerance,
    regularizationContext,
    regularizationWeight,
    regularizationWeightRange,
    elasticNetParamRange) {

  require(downSamplingRate > 0.0 && downSamplingRate <= 1.0, s"Unexpected downSamplingRate: $downSamplingRate")
}

/**
 * Configuration for a [[com.linkedin.photon.ml.algorithm.RandomEffectCoordinate]].
 *
 * @param optimizerType The type of optimizer to use during training
 * @param maximumIterations The upper limit on the number of optimization iterations to perform
 * @param tolerance The relative tolerance limit for optimization
 * @param regularizationContext Regularization context
 * @param regularizationWeight Regularization weight
 * @param regularizationWeightRange Regularization weight range
 * @param elasticNetParamRange Elastic net alpha range
 */
case class RandomEffectOptimizationConfiguration(
    override val optimizerType: OptimizerType,
    override val maximumIterations: Int,
    override val tolerance: Double,
    override val regularizationContext: RegularizationContext = NoRegularizationContext,
    override val regularizationWeight: Double = 0D,
    override val regularizationWeightRange: Option[DoubleRange] = None,
    override val elasticNetParamRange: Option[DoubleRange] = None)
  extends GLMOptimizationConfiguration(
    optimizerType,
    maximumIterations,
    tolerance,
    regularizationContext,
    regularizationWeight,
    regularizationWeightRange,
    elasticNetParamRange)
