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
package com.linkedin.photon.ml

import com.linkedin.photon.ml.optimization.game._
import com.linkedin.photon.ml.data.{CoordinateDataConfiguration, FixedEffectDataConfiguration, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.optimization.RegularizationType
import com.linkedin.photon.ml.util.DoubleRange

/**
 *
 * @param dataConfiguration Coordinate dataset definition
 * @param optimizationConfiguration Coordinate optimization problem definition
 * @param regularizationWeights Regularization weights
 * @param regularizationWeightRange
 * @param elasticNetParamRange
 */
abstract class CoordinateConfiguration(
    val dataConfiguration: CoordinateDataConfiguration,
    val optimizationConfiguration: CoordinateOptimizationConfiguration,
    val regularizationWeights: Set[Double],
    val regularizationWeightRange: Option[DoubleRange] = None,
    val elasticNetParamRange: Option[DoubleRange] = None) {

  require(regularizationWeights.nonEmpty, "At least one regularization weight required.")

  // Slightly different object requirements depending on regularization type
  optimizationConfiguration.regularizationContext match {
    case RegularizationType.NONE =>
      require(regularizationWeights.isEmpty, "Regularization disabled but regularization weights provided")
      require(regularizationWeightRange.isEmpty, "Regularization weight range set, but regularization not enabled")
      require(elasticNetParamRange.isEmpty, "Elastic net alpha range set, but elastic net regularization not enabled")

    case RegularizationType.ELASTIC_NET =>
      require(regularizationWeights.nonEmpty, "Regularization enabled but no regularization weights provided")
      regularizationWeightRange.foreach { case DoubleRange(start, _) =>
        require(start > 0.0, "Regularization weight ranges must be positive and non-zero")
      }
      elasticNetParamRange.foreach { case DoubleRange(start, end) =>
        require(start >= 0.0 && end <= 1.0, "Elastic net alpha ranges must lie within [0, 1]")
      }

    case _ =>
      require(regularizationWeights.nonEmpty, "Regularization enabled but no regularization weights provided")
      regularizationWeightRange.foreach { case DoubleRange(start, _) =>
        require(start > 0.0, "Regularization weight ranges must be positive and non-zero")
      }
      require(elasticNetParamRange.isEmpty, "Elastic net alpha range set, but elastic net regularization not enabled")
  }

  regularizationWeights.foreach { regularizationWeight =>
    require(0 <= regularizationWeight, s"Regularization weights cannot be negative")
  }
}

/**
 * Definition of a fixed effect problem coordinate.
 *
 * @param dataConfiguration Coordinate dataset definition
 * @param optimizationConfiguration Coordinate optimization problem definition
 * @param regularizationWeights Regularization weights
 * @param regularizationWeightRange
 * @param elasticNetParamRange
 */
case class FixedEffectCoordinateConfiguration(
    override val dataConfiguration: FixedEffectDataConfiguration,
    override val optimizationConfiguration: FixedEffectOptimizationConfiguration,
    override val regularizationWeights: Set[Double],
    override val regularizationWeightRange: Option[DoubleRange],
    override val elasticNetParamRange: Option[DoubleRange])
  extends CoordinateConfiguration(
    dataConfiguration,
    optimizationConfiguration,
    regularizationWeights,
    regularizationWeightRange,
    elasticNetParamRange)

/**
 * Definition of a random effect problem coordinate.
 *
 * @param dataConfiguration Coordinate dataset definition
 * @param optimizationConfiguration Coordinate optimization problem definition
 * @param regularizationWeights Regularization weights
 * @param regularizationWeightRange
 * @param elasticNetParamRange
 */
case class RandomEffectCoordinateConfiguration private (
    override val dataConfiguration: RandomEffectDataConfiguration,
    override val optimizationConfiguration: RandomEffectOptimizationConfiguration,
    override val regularizationWeights: Set[Double],
    override val regularizationWeightRange: Option[DoubleRange],
    override val elasticNetParamRange: Option[DoubleRange])
  extends CoordinateConfiguration(
    dataConfiguration,
    optimizationConfiguration,
    regularizationWeights,
    regularizationWeightRange,
    elasticNetParamRange)
