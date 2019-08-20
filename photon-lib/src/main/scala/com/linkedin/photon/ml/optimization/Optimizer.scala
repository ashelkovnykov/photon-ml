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
import breeze.numerics.abs

import com.linkedin.photon.ml.function.ObjectiveFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util._

/**
 * Common base class for the Photon ML optimization problem solvers.
 *
 * @tparam Function Generic type of the objective function to be optimized
 * @param objectiveFunction
 * @param relTolerance The relative tolerance used to gauge improvement between iterations.
 *                     Separate absolute tolerances for the loss function and the gradient are set from this relative
 *                     tolerance.
 * @param maxNumIterations The max number of iterations to perform
 * @param normalizationContext The normalization context
 */
abstract class Optimizer[+Function <: ObjectiveFunction](
    protected val objectiveFunction: Function,
    protected val relTolerance: Double,
    protected val maxNumIterations: Int,
    val normalizationContext: BroadcastWrapper[NormalizationContext])
  extends Serializable
  with Logging {

  protected var zeroState: OptimizerState = _
  protected var absoluteLossTolerance: Double = Double.MaxValue

  /**
   * Clear the optimizer inner state.
   */
  protected def clearOptimizerInnerState(): Unit = {

    zeroState = OptimizerState(iter = 0, Vector[Double](0D), loss = 0D)
    absoluteLossTolerance = Double.MaxValue
  }

  /**
   * Initialize the context of the optimizer (e.g., the history of LBFGS; the trust region size of TRON; etc.).
   *
   * @param initialState
   * @param data The training data
   */
  protected def init(initialState: OptimizerState, data: objectiveFunction.Data): Unit = {

    val zeroCoefficients = VectorUtils.zeroOfSameType(initialState.coefficients)

    zeroState = calculateState(zeroCoefficients, data)
    absoluteLossTolerance = zeroState.loss * relTolerance
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
  protected def calculateState(
      coefficients: Vector[Double],
      data: objectiveFunction.Data): OptimizerState = {

    val convertedCoefficients = objectiveFunction.convertFromVector(coefficients)
    val value = objectiveFunction.value(data, convertedCoefficients, normalizationContext)

    objectiveFunction.cleanupCoefficients(convertedCoefficients)

    OptimizerState(iter = 0, coefficients, value)
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
  protected def getConvergenceReason(
      currentState: OptimizerState,
      previousState: OptimizerState): Option[ConvergenceReason] =
    if (currentState.iter >= maxNumIterations) {
      Some(MaxIterations)
    } else if (currentState.iter == previousState.iter) {
      Some(ObjectiveNotImproving)
    } else if (abs(currentState.loss - previousState.loss) <= absoluteLossTolerance) {
      Some(FunctionValuesConverged)
    } else {
      None
    }

  /**
   * Run one iteration of the optimizer given the current state.
   *
   * @param data The training data
   * @return The updated state of the optimizer
   */
  protected def runOneIteration(data: objectiveFunction.Data): Option[OptimizerState]

  /**
   * Solve the provided convex optimization problem.
   *
   * @param initialCoefficients
   * @param data The training data
   * @return Optimized model coefficients and corresponding objective function's value
   */
  protected[ml] def optimize(
      initialCoefficients: Vector[Double],
      data: objectiveFunction.Data): (Vector[Double], OptimizationStatesTracker) = {

    clearOptimizerInnerState()

    val normalizedInitialCoefficients = normalizationContext.value.modelToTransformedSpace(initialCoefficients)
    // TODO: For cold start, we call calculateState with the same arguments twice
    val initialState = calculateState(normalizedInitialCoefficients, data)
    init(initialState, data)

    val optimizationStatesTracker = new OptimizationStatesTracker(System.currentTimeMillis())
    optimizationStatesTracker.append(initialState)

    var currentState = initialState
    var prevState = initialState
    var convergenceReason: Option[ConvergenceReason] = None

    do {
      prevState = currentState
      currentState = runOneIteration(data).getOrElse(currentState)
      convergenceReason = getConvergenceReason(currentState, prevState)

      optimizationStatesTracker.append(currentState)
    } while (convergenceReason.isEmpty)

    (currentState.coefficients, optimizationStatesTracker)
  }
}
