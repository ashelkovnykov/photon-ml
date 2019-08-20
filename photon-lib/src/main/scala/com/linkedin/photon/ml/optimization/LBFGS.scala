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
import breeze.optimize.FirstOrderMinimizer.State
import breeze.optimize.LBFGS.ApproximateInverseHessian
import breeze.optimize.{DiffFunction => BreezeDiffFunction, LBFGS => BreezeLBFGS}

import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * Class used to solve an optimization problem using Limited-memory BFGS (LBFGS).
 * Reference: [[http://en.wikipedia.org/wiki/Limited-memory_BFGS]].
 *
 * @param objectiveFunction
 * @param relTolerance The tolerance threshold for improvement between iterations as a percentage of the initial loss
 * @param maxNumIterations The cut-off for number of optimization iterations to perform.
 * @param normalizationContext The normalization context
 * @param numCorrections The number of corrections (tracked historical states of position and gradient) used by the
 *                       LBFGS algorithm to update its inverse Hessian matrix estimate. Small values of are inaccurate;
 *                       large values of result in excessive computing time.
 *                       Restriction:  numCorrections > 0
 *                       Recommended:  3 < numCorrections < 11
 * @param constraintMapOpt (Optional) The map of constraints on the feature coefficients
 */
class LBFGS[+Function <: DiffFunction](
    override protected val objectiveFunction: Function,
    override protected val relTolerance: Double,
    override protected val maxNumIterations: Int,
    override val normalizationContext: BroadcastWrapper[NormalizationContext],
    numCorrections: Int = LBFGS.DEFAULT_NUM_CORRECTIONS,
    constraintMapOpt: Option[Map[Int, (Double, Double)]] = LBFGS.DEFAULT_CONSTRAINT_MAP)
  extends DiffOptimizer[Function](
    objectiveFunction,
    relTolerance,
    maxNumIterations,
    normalizationContext) {

  import LBFGS._

  protected var breezeStates: Iterator[BreezeState] = Seq[BreezeState]().toIterator

  protected def initBreezeOptimizer(initialCoefficients: Vector[Double], data: objectiveFunction.Data): Unit = {

    val breezeDiffFunction = new BreezeDiffFunction[Vector[Double]]() {
      // Calculating the gradient and value of the objective function
      def calculate(coefficients: Vector[Double]): (Double, Vector[Double]) = {
        val convertedCoefficients = objectiveFunction.convertFromVector(coefficients)
        val result = objectiveFunction.calculate(data, convertedCoefficients, normalizationContext)

        objectiveFunction.cleanupCoefficients(convertedCoefficients)
        result
      }
    }

    breezeStates = new BreezeLBFGS[Vector[Double]](maxNumIterations, numCorrections, relTolerance).iterations(breezeDiffFunction, initialCoefficients)
    breezeStates.next()
  }

  /**
   * Just reset the whole BreezeOptimization instance.
   */
  override protected def clearOptimizerInnerState(): Unit = {

    breezeStates = Seq[BreezeState]().toIterator

    super.clearOptimizerInnerState()
  }

  /**
   * Initialize breeze optimization engine.
   *
   * @param initialState
   * @param data The training data
   */
  override protected def init(initialState: OptimizerState, data: objectiveFunction.Data): Unit = {

    super.init(initialState, data)

    initBreezeOptimizer(initialState.coefficients, data)
  }

  /**
   * Run one iteration of the optimizer given the current state.
   *
   * @param data The training data
   * @return The updated state of the optimizer
   */
  override protected def runOneIteration(data: objectiveFunction.Data): Option[OptimizerState] =
    if (breezeStates.hasNext) {
      Some(breezeToPhoton(breezeStates.next(), constraintMapOpt))
    } else {
      None
    }
}

object LBFGS {

  type BreezeState = State[Vector[Double], _, ApproximateInverseHessian[Vector[Double]]]

  val DEFAULT_CONSTRAINT_MAP: Option[Map[Int, (Double, Double)]] = None
  val DEFAULT_NUM_CORRECTIONS = 10

  def breezeToPhoton(breezeState: BreezeState, constraintMapOpt: Option[Map[Int, (Double, Double)]]): DiffOptimizerState =
    DiffOptimizerState(
      breezeState.iter,
      OptimizationUtils.projectCoefficientsToSubspace(breezeState.x, constraintMapOpt),
      breezeState.adjustedValue,
      breezeState.adjustedGradient)
}
