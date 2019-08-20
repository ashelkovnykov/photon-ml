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

/*
 * @note This code is heavily influenced by the SPARK LIBLINEAR TRON implementation,
 * though not an exact copy. It also subject to the LIBLINEAR project's license
 * and copyright notice:
 *
 * Copyright (c) 2007-2015 The LIBLINEAR Project.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither name of copyright holders nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.linkedin.photon.ml.optimization

import breeze.linalg.{Vector, norm}

import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.{BroadcastWrapper, Logging, VectorUtils}

/**
 * This class used to solve an optimization problem using trust region Newton method (TRON).
 * Reference 1: [[http://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf]]
 * Reference 2:
 *   [[http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/distributed-liblinear/spark/running_spark_liblinear.html]]
 *
 * @param objectiveFunction
 * @param relTolerance The tolerance threshold for improvement between iterations as a percentage of the initial loss
 * @param maxNumIterations The cut-off for number of optimization iterations to perform.
 * @param normalizationContext The normalization context
 * @param maxNumImprovementFailures The maximum number of successive times that the objective may fail to improve. For
 *                                  most optimizers using line search, like L-BFGS, the improvement failure is not
 *                                  supposed to happen, because any improvement failure should be captured during the
 *                                  line search step. Here we are trying to capture the improvement failure after the
 *                                  gradient step. As a result, any improvement failure in this case results from some
 *                                  bug and we should not tolerate it. However, for optimizers like TRON occasional
 *                                  improvement failure is acceptable.
 * @param constraintMapOpt (Optional) The map of constraints on the feature coefficients
 */
class TRON[Function <: TwiceDiffFunction](
    override protected val objectiveFunction: Function,
    override protected val relTolerance: Double,
    override protected val maxNumIterations: Int,
    override val normalizationContext: BroadcastWrapper[NormalizationContext],
    maxNumImprovementFailures: Int = TRON.DEFAULT_MAX_NUM_FAILURE,
    constraintMapOpt: Option[Map[Int, (Double, Double)]] = TRON.DEFAULT_CONSTRAINT_MAP)
  extends DiffOptimizer[Function](
    objectiveFunction,
    relTolerance,
    maxNumIterations,
    normalizationContext) {

  import TRON._

  /**
   * The size of the trust region.
   */
  private var delta = Double.MaxValue
  private var iter = 0
  private var prevCoef = Vector[Double](0D)
  private var prevLoss = Double.MaxValue
  private var prevGradient = Vector[Double](0D)

  /**
   * Reset the delta.
   */
  override protected def clearOptimizerInnerState(): Unit = {

    super.clearOptimizerInnerState()

    delta = Double.MaxValue
    iter = 0
    prevCoef = Vector[Double](0D)
    prevLoss = Double.MaxValue
    prevGradient = Vector[Double](0D)
  }

  /**
   * Initialize the trust region size.
   *
   * @param initialState
   * @param data The training data
   */
  override protected def init(initialState: OptimizerState, data: objectiveFunction.Data): Unit = {

    super.init(initialState, data)

    zeroState match {
      case state: DiffOptimizerState =>
        delta = state.gradientNorm
        iter = state.iter
        prevCoef = state.coefficients
        prevLoss = state.loss
        prevGradient = state.gradient

      case state =>
        throw new IllegalArgumentException(s"Unexpected initial-state type: ${state.getClass.getName}")
    }
  }

  /**
   * Run one iteration of the optimizer given the current state.
   *
   * @param data The training data
   * @return The updated state of the optimizer
   */
  override protected def runOneIteration(data: objectiveFunction.Data): Option[DiffOptimizerState] = {

    val convertedPrevCoefficients = objectiveFunction.convertFromVector(prevCoef)

    var improved = false
    var numImprovementFailure = 0
    var finalState: Option[DiffOptimizerState] = None

    do {
      // Retry the TRON optimization with the shrunken trust region boundary (delta) until either:
      // 1. The function value is improved
      // 2. The maximum number of improvement failures reached.
      val (cgIter, step, residual) = TRON.truncatedConjugateGradientMethod(
        objectiveFunction, prevGradient, normalizationContext, delta)(
        data, convertedPrevCoefficients)

      val updatedCoefficients = prevCoef + step
      val convertedUpdatedCoefficients = objectiveFunction.convertFromVector(updatedCoefficients)
      val gs = prevGradient.dot(step)
      // Compute the predicted reduction
      val predictedReduction = -0.5 * (gs - step.dot(residual))
      // Function value
      val (updatedFunctionValue, updatedFunctionGradient) = objectiveFunction.calculate(
        data,
        convertedUpdatedCoefficients,
        normalizationContext)

      objectiveFunction.cleanupCoefficients(convertedUpdatedCoefficients)

      // Compute the actual reduction.
      val actualReduction = prevLoss - updatedFunctionValue
      val stepNorm = norm(step, 2)

      // On the first iteration, adjust the initial step bound.
      if (iter == 0) {
        delta = math.min(delta, stepNorm)
      }

      // Compute prediction alpha*stepNorm of the step.
      val alpha = if (updatedFunctionValue - prevLoss - gs <= 0) {
          SIGMA_3
        } else {
          math.max(SIGMA_1, -0.5 * (gs / (updatedFunctionValue - prevLoss - gs)))
        }

      // Update the trust region bound according to the ratio of actual to predicted reduction.
      if (actualReduction < ETA_0 * predictedReduction) {
        delta = math.min(math.max(alpha, SIGMA_1) * stepNorm, SIGMA_2 * delta)
      } else if (actualReduction < ETA_1 * predictedReduction) {
        delta = math.max(SIGMA_1 * delta, math.min(alpha * stepNorm, SIGMA_2 * delta))
      } else if (actualReduction < ETA_2 * predictedReduction) {
        delta = math.max(SIGMA_1 * delta, math.min(alpha * stepNorm, SIGMA_3 * delta))
      } else {
        delta = math.max(delta, math.min(alpha * stepNorm, SIGMA_3 * delta))
      }
      val gradientNorm = norm(updatedFunctionGradient, 2)
      val residualNorm = norm(residual, 2)

      if (logger.isDebugEnabled) {
        logger.debug(
          f"iter $iter%3d act $actualReduction%5.3e pre $predictedReduction%5.3e delta $delta%5.3e " +
            f"f $updatedFunctionValue%5.3e |residual| $residualNorm%5.3e |g| $gradientNorm%5.3e CG $cgIter%3d")
      }

      if (actualReduction > ETA_0 * predictedReduction) {
        // if the actual function value reduction is greater than eta0 times the predicted function value reduction,
        // we accept the updated coefficients and move forward with the updated optimizer state
        val coefficients = updatedCoefficients

        improved = true
        // project coefficients into constrained space, if any, after the optimization step
        val projectedCoefficients = OptimizationUtils.projectCoefficientsToSubspace(coefficients, constraintMapOpt)

        finalState = Some(DiffOptimizerState(iter, projectedCoefficients, updatedFunctionValue, updatedFunctionGradient))

        iter = iter + 1
        prevCoef = projectedCoefficients
        prevLoss = updatedFunctionValue
        prevGradient = updatedFunctionGradient

      } else {
        // otherwise, the updated coefficients will not be accepted, and the old state will be returned along with
        // warning messages
        logger.warn(s"actual objective function value reduction is smaller than predicted " +
          s"(actualReduction = $actualReduction < eta0 = $ETA_0 * predictedReduction = $predictedReduction)")
        if (updatedFunctionValue < -1.0e+32) {
          logger.warn("updated function value < -1.0e+32")
        }
        if (actualReduction <= 0) {
          logger.warn("actual reduction of function value <= 0")
        }
        if (math.abs(actualReduction) <= 1.0e-12 && math.abs(predictedReduction) <= 1.0e-12) {
          logger.warn("both actual reduction and predicted reduction of function value are too small")
        }

        numImprovementFailure += 1
      }
    } while (!improved && numImprovementFailure < maxNumImprovementFailures)

    objectiveFunction.cleanupCoefficients(convertedPrevCoefficients)

    finalState
  }
}

object TRON extends Logging {

  // Initialize the hyperparameters for TRON (see Reference 2 for more details).
  private val ETA_0 = 1e-4
  private val ETA_1 = 0.25
  private val ETA_2 = 0.75
  private val SIGMA_1 = 0.25
  private val SIGMA_2 = 0.5
  private val SIGMA_3 = 4.0

  val DEFAULT_CONSTRAINT_MAP: Option[Map[Int, (Double, Double)]] = None
  val DEFAULT_MAX_NUM_FAILURE = 5
  val DEFAULT_TOLERANCE = 1.0E-5
  val DEFAULT_MAX_ITER = 15
  // The maximum number of iterations used in the conjugate gradient update. Larger value will lead to more accurate
  // solution but also longer running time.
  val MAX_CG_ITERATIONS: Int = 20

  /**
   * Run the truncated conjugate gradient (CG) method as a subroutine of TRON.
   * For details and notations of the following code, please see Algorithm 2
   * (Conjugate gradient procedure for approximately solving the trust region sub-problem)
   * in page 6 of the following paper: [[http://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf]].
   *
   * @param objectiveFunction The objective function
   * @param coefficients The model coefficients
   * @param gradient Gradient of the objective function
   * @param truncationBoundary The truncation boundary of truncatedConjugateGradientMethod.
   *                           In the case of Tron, this corresponds to the trust region size (delta).
   * @param data The training data
   * @return Tuple3(number of CG iterations, solution, residual)
   */
  private def truncatedConjugateGradientMethod(
      objectiveFunction: TwiceDiffFunction,
      gradient: Vector[Double],
      normalizationContext: BroadcastWrapper[NormalizationContext],
      truncationBoundary: Double)
      (data: objectiveFunction.Data,
      coefficients: objectiveFunction.Coefficients): (Int, Vector[Double], Vector[Double]) = {

    val step = VectorUtils.zeroOfSameType(gradient)
    val residual = gradient * -1.0
    val direction = residual.copy
    val conjugateGradientConvergenceTolerance = 0.1 * norm(gradient, 2)
    var iteration = 0
    var done = false
    var rTr = residual.dot(residual)
    while (iteration < MAX_CG_ITERATIONS && !done) {
      if (norm(residual, 2) <= conjugateGradientConvergenceTolerance) {
        done = true
      } else {
        iteration += 1
        // Compute the hessianVector
        val convertedDirection = objectiveFunction.convertFromVector(direction)
        val Hd = objectiveFunction.hessianVector(data, coefficients, convertedDirection, normalizationContext)
        var alpha = rTr / direction.dot(Hd)

        objectiveFunction.cleanupCoefficients(convertedDirection)

        step += direction * alpha
        if (norm(step, 2) > truncationBoundary) {
          logger.debug(s"cg reaches truncation boundary after $iteration iterations")
          // Solve equation (13) of Algorithm 2
          alpha = -alpha
          step += direction * alpha
          val std = step.dot(direction)
          val sts = step.dot(step)
          val dtd = direction.dot(direction)
          val dsq = truncationBoundary * truncationBoundary
          val rad = math.sqrt(std * std + dtd * (dsq - sts))
          if (std >= 0) {
            alpha = (dsq - sts) / (std + rad)
          } else {
            alpha = (rad - std) / dtd
          }
          step += direction * alpha
          alpha = -alpha
          residual += Hd * alpha
          done = true
        } else {
          // Find the new conjugate gradient direction
          alpha = -alpha
          residual += Hd * alpha
          val rnewTrnew = residual.dot(residual)
          val beta = rnewTrnew / rTr
          direction := direction * beta + residual
          rTr = rnewTrnew
        }
      }
    }

    (iteration, step, residual)
  }
}
