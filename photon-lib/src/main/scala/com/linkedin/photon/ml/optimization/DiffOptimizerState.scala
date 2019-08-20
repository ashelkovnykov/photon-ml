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

import java.text.MessageFormat

import breeze.linalg.{Vector, norm}

/**
 * Similar to [[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.FirstOrderMinimizer\$State breeze.
 *   optimize.FirstOrderMinimizer.State]]
 *
 * This class tracks the information about the optimizer, including the iteration, coefficients, objective function
 * value, and objective function gradient Euclidean norm.
 *
 * @param iter The current iteration number
 * @param coefficients The optimized coefficients for the current iteration
 * @param loss The objective function value for the coefficients
 * @param gradientNorm The Euclidean norm of the objective function gradient at the coefficients
 */
class DiffOptimizerState(
    override val iter: Int,
    override val coefficients: Vector[Double],
    override val loss: Double,
    val gradient: Vector[Double],
    val gradientNorm: Double)
  extends OptimizerState(iter, coefficients, loss) {

  import DiffOptimizerState._

  override def summaryAxis: String = SUMMARY_AXIS

  override def toSummaryString: String =
    MessageFormat.format(STRING_TEMPLATE, super.toSummaryString, f"$gradientNorm%15.2e")
}

object DiffOptimizerState {

  private val STRING_TEMPLATE = "{0}{1}"
  private val GRADIENT_NORM = "|Gradient|"
  val SUMMARY_AXIS: String = String.format("%s%15s", OptimizerState.SUMMARY_AXIS, GRADIENT_NORM)

  def apply(iter: Int, coefficients: Vector[Double], loss: Double, gradient: Vector[Double]): DiffOptimizerState =
    new DiffOptimizerState(iter, coefficients, loss, gradient, norm(gradient, 2))

  def unapply(arg: DiffOptimizerState): Option[(Int, Vector[Double], Double, Double)] =
    Some(arg.iter, arg.coefficients, arg.loss, arg.gradientNorm)
}
