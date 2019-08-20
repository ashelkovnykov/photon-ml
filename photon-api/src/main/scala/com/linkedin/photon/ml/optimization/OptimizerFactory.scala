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
package com.linkedin.photon.ml.optimization

import com.linkedin.photon.ml.function.{DiffFunction, ObjectiveFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * Creates instances of Optimizer according to the user-requested optimizer type and regularization. The factory
 * methods in this class do not enforce the runtime rules for compatibility between user-selected loss functions and
 * optimizers: mixing incompatible optimizers and objective functions will result in a runtime error.
 */
protected[ml] object OptimizerFactory {

  /**
   * Creates an optimizer.
   *
   * @tparam T
   * @param config The Optimizer configuration
   * @param objectiveFunction
   * @param normalizationContext The normalization context
   * @param regularizationContext The regularization context
   * @param regularizationWeight The regularization weight
   * @return A new [[Optimizer]]
   */
  def build[T <: ObjectiveFunction](
      config: OptimizerConfig,
      objectiveFunction: T,
      normalizationContext: BroadcastWrapper[NormalizationContext],
      regularizationContext: RegularizationContext,
      regularizationWeight: Double = 0): Optimizer[T] =

    objectiveFunction match {
      case _: TwiceDiffFunction =>
        buildTwiceDiffFunction(
          config,
          objectiveFunction.asInstanceOf[T with TwiceDiffFunction],
          normalizationContext,
          regularizationContext,
          regularizationWeight)

      case _: DiffFunction =>
        buildDiffFunction(
          config,
          objectiveFunction.asInstanceOf[T with DiffFunction],
          normalizationContext,
          regularizationContext,
          regularizationWeight)

      case _ =>
        throw new IllegalArgumentException(
          s"OptimizerFactory cannot build Optimizer of with type ${objectiveFunction.getClass}")
    }

  private def buildDiffFunction[T <: DiffFunction](
      config: OptimizerConfig,
      diffFunction: T,
      normalizationContext: BroadcastWrapper[NormalizationContext],
      regularizationContext: RegularizationContext,
      regularizationWeight: Double = 0): Optimizer[T] = {

    (config.optimizerType, regularizationContext.regularizationType) match {
      case (OptimizerType.LBFGS, RegularizationType.L1 | RegularizationType.ELASTIC_NET) =>
        new OWLQN[T](
          diffFunction,
          config.tolerance,
          config.maximumIterations,
          normalizationContext,
          regularizationContext.getL1RegularizationWeight(regularizationWeight),
          constraintMapOpt = config.constraintMap)

      case (OptimizerType.LBFGS, RegularizationType.L2 | RegularizationType.NONE) =>
        new LBFGS(
          diffFunction,
          config.tolerance,
          config.maximumIterations,
          normalizationContext,
          constraintMapOpt = config.constraintMap)

      case (OptimizerType.LBFGS, regType) =>
        throw new IllegalArgumentException(s"Incompatible regularization selected: $regType")

      case (optType, _) =>
        throw new IllegalArgumentException(s"Incompatible optimizer selected: $optType")
    }
  }

  private def buildTwiceDiffFunction[T <: TwiceDiffFunction](
    config: OptimizerConfig,
    twiceDiffFunction: T,
    normalizationContext: BroadcastWrapper[NormalizationContext],
    regularizationContext: RegularizationContext,
    regularizationWeight: Double = 0): Optimizer[T] =

    (config.optimizerType, regularizationContext.regularizationType) match {
      case (OptimizerType.LBFGS, RegularizationType.L1 | RegularizationType.ELASTIC_NET) =>
        new OWLQN[T](
          twiceDiffFunction,
          config.tolerance,
          config.maximumIterations,
          normalizationContext,
          regularizationContext.getL1RegularizationWeight(regularizationWeight),
          constraintMapOpt = config.constraintMap)

      case (OptimizerType.LBFGS, RegularizationType.L2 | RegularizationType.NONE) =>
        new LBFGS(
          twiceDiffFunction,
          config.tolerance,
          config.maximumIterations,
          normalizationContext,
          constraintMapOpt = config.constraintMap)

      case (OptimizerType.TRON, RegularizationType.L2 | RegularizationType.NONE) =>
        new TRON(
          twiceDiffFunction,
          config.tolerance,
          config.maximumIterations,
          normalizationContext,
          constraintMapOpt = config.constraintMap)

      case (OptimizerType.TRON, RegularizationType.L1 | RegularizationType.ELASTIC_NET) =>
        throw new IllegalArgumentException("TRON optimizer incompatible with L1 regularization")

      case (OptimizerType.LBFGS | OptimizerType.TRON, regType) =>
        throw new IllegalArgumentException(s"Incompatible regularization selected: $regType")

      case (optType, _) =>
        throw new IllegalArgumentException(s"Incompatible optimizer selected: $optType")
    }
}
