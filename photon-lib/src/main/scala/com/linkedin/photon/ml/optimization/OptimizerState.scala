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

import breeze.linalg.Vector

import com.linkedin.photon.ml.util.Summarizable

/**
 * Similar to [[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.FirstOrderMinimizer\$State breeze.
 *   optimize.FirstOrderMinimizer.State]]
 *
 * This class tracks the information about the optimizer, including the iteration, coefficients, objective function and
 * value.
 *
 * @param iter The current iteration number
 * @param coefficients The optimized coefficients for the current iteration
 * @param loss The objective function value for the coefficients
 */
protected[optimization] class OptimizerState(
    val iter: Int,
    val coefficients: Vector[Double],
    val loss: Double)
  extends Summarizable {

  import OptimizerState._

  def summaryAxis: String = SUMMARY_AXIS

  override def toSummaryString: String = MessageFormat.format(STRING_TEMPLATE, f"$iter%10d", f"$loss%25.8f")
}

object OptimizerState {

  private val STRING_TEMPLATE = "{0}{1}"
  private val ITER = "Iter"
  private val LOSS = "Loss Value"
  val SUMMARY_AXIS: String = String.format("%10s%25s", ITER, LOSS)

  def apply(iter: Int, coefficients: Vector[Double], loss: Double): OptimizerState =
    new OptimizerState(iter, coefficients, loss)

  def unapply(arg: OptimizerState): Option[(Int, Vector[Double], Double)] =
    Some(arg.iter, arg.coefficients, arg.loss)
}