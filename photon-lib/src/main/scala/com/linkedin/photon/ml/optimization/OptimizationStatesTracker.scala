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

import scala.collection.mutable

import com.linkedin.photon.ml.util.{ConvergenceReason, Summarizable}

/**
 * Class to track the history of an optimizer's states and wall-clock time elapsed per iteration.
 *
 * @param maxNumStates The maximum number of states to track. This is used to prevent the OptimizationHistoryTracker
 *                     from using too much memory to track the history of the states.
 * @note  DO NOT USE this class outside of Photon-ML. It is intended as an internal utility, and is likely to be
 *        changed or removed in future releases.
 */
protected[ml] class OptimizationStatesTracker(private val startTime: Long, maxNumStates: Int = 100)
  extends Serializable
    with Summarizable {

  import OptimizationStatesTracker._

  private val states = mutable.Queue[OptimizerState]()
  private val times = mutable.Queue[Long]()

  private var convergeReason: Option[ConvergenceReason] = None

  states.sizeHint(maxNumStates)
  times.sizeHint(maxNumStates)

  /**
   * Getter method for [[convergeReason]].
   *
   * @return Value of [[convergeReason]]
   */
  def convergenceReason: Option[ConvergenceReason] = convergeReason

  /**
   * Setter method for [[convergeReason]].
   *
   * @param value New value of [[convergeReason]]
   */
  protected[optimization] def convergenceReason_=(value: Option[ConvergenceReason]): Unit = convergeReason = value

  /**
   * Add the most recent state to the list of tracked states. If the limit of cached states is reached, remove the
   * oldest state.
   *
   * @param state The most recent state
   */
  def append(state: OptimizerState): Unit = {

    states.enqueue(state)
    times.enqueue(System.currentTimeMillis() - startTime)

    while (times.length > maxNumStates) {
      states.dequeue
      times.dequeue
    }
  }

  /**
   * Get the sequence of recorded states as an Array.
   *
   * @return The recorded states
   */
  def trackedStates: Array[OptimizerState] = states.toArray

  /**
   * Get the sequence of times between states as an Array.
   *
   * @return The times between states
   */
  def trackedStateTimes: Array[Long] = times.toArray

  /**
   *
   * @return
   */
  override def toSummaryString: String = {

    val stringBuilder = new StringBuilder

    val convergenceReasonStr = convergeReason match {
      case Some(reason) => reason.summary
      case None => "Optimizer has not converged properly, please check the log for more information"
    }
    val statesIterator = states.iterator
    val timesIterator = times.iterator

    stringBuilder ++= s"Convergence reason: $convergenceReasonStr\n"
    stringBuilder ++= f"$TIME%10s" ++= states.head.summaryAxis ++= "\n"
    while (statesIterator.hasNext) {
      stringBuilder ++= f"${timesIterator.next() * 0.001}%10.3f" ++= statesIterator.next().toSummaryString ++= "\n"
    }
    stringBuilder ++= "\n"

    stringBuilder.result()
  }
}

object OptimizationStatesTracker {

  val TIME = "Time (s)"
}
