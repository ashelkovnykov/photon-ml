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

import breeze.linalg.Vector

import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.{BroadcastWrapper, ConvergenceReason, GradientConverged}

/**
 * Common base class for the Photon ML optimization problem solvers.
 *
 * @tparam Function Generic type of the differentiable objective function to be optimized
 * @param objectiveFunction
 * @param relTolerance The relative tolerance used to gauge improvement between iterations.
 *                     Separate absolute tolerances for the loss function and the gradient are set from this relative
 *                     tolerance.
 * @param maxNumIterations The max number of iterations to perform
 * @param normalizationContext The normalization context
 */
abstract class DiffOptimizer[+Function <: DiffFunction](
    override protected val objectiveFunction: Function,
    override protected val relTolerance: Double,
    override protected val maxNumIterations: Int,
    override val normalizationContext: BroadcastWrapper[NormalizationContext])
  extends Optimizer[Function](objectiveFunction, relTolerance, maxNumIterations, normalizationContext) {

  protected var absoluteGradientTolerance: Double = Double.MaxValue

  /**
   * Clear the optimizer inner state.
   */
  override protected def clearOptimizerInnerState(): Unit = {

    absoluteGradientTolerance = Double.MaxValue

    super.clearOptimizerInnerState()
  }

  /**
   * Initialize the context of the optimizer (e.g., the history of LBFGS; the trust region size of TRON; etc.).
   *
   * @param initialState
   * @param data The training data
   */
  override protected def init(initialState: OptimizerState, data: objectiveFunction.Data): Unit = {

    super.init(initialState, data)

    zeroState match {
      case state: DiffOptimizerState => absoluteGradientTolerance = state.gradientNorm * relTolerance
      case state => throw new IllegalArgumentException(s"Unexpected zero-state type: ${state.getClass.getName}")
    }
  }

  /**
   * Calculate the Optimizer state given some data.
   *
   * @note involves a calculation over the whole dataset, so can be expensive.
   *
   * @param coefficients The model coefficients
   * @param data The training data
   * @return The current optimizer state
   */
  override protected def calculateState(
      coefficients: Vector[Double],
      data: objectiveFunction.Data): DiffOptimizerState = {

    val convertedCoefficients = objectiveFunction.convertFromVector(coefficients)
    val (value, gradient) = objectiveFunction.calculate(data, convertedCoefficients, normalizationContext)

    objectiveFunction.cleanupCoefficients(convertedCoefficients)

    DiffOptimizerState(iter = 0, coefficients, value, gradient)
  }

  /**
   * Get the optimizer convergence reason.
   *
   * @note It is not strictly necessary to check both the convergence of the loss function and the convergence of the
   *       gradient, from a correctness point of view. All we need in the end is convergence of the loss function to
   *       its optimum value. However, it can be useful to have a stopping criterion based on the gradient norm as
   *       the gradient can "misbehave" around the optimum of the loss function (oscillations, numerical issues...).
   *
   * @return The convergence reason
   */
  override protected def getConvergenceReason(
      currentState: OptimizerState,
      previousState: OptimizerState): Option[ConvergenceReason] = currentState match {

    case cState: DiffOptimizerState =>
      if (cState.gradientNorm <= absoluteGradientTolerance) {
        Some(GradientConverged)
      } else {
        super.getConvergenceReason(currentState, previousState)
      }

    case _ =>
      super.getConvergenceReason(currentState, previousState)
  }
}
