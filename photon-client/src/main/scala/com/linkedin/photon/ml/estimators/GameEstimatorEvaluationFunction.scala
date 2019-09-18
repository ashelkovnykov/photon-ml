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
package com.linkedin.photon.ml.estimators

import scala.collection.mutable
import scala.math.{exp, log}

import breeze.linalg.DenseVector
import org.apache.spark.sql.DataFrame

import com.linkedin.photon.ml.{CoordinateConfiguration, FixedEffectCoordinateConfiguration, RandomEffectCoordinateConfiguration}
import com.linkedin.photon.ml.Types.CoordinateId
import com.linkedin.photon.ml.estimators.GameEstimator.GameResult
import com.linkedin.photon.ml.hyperparameter.{EvaluationFunction, VectorRescaling}
import com.linkedin.photon.ml.optimization.RegularizationType
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Evaluation function implementation for GAME.
 *
 * An evaluation function is the integration point between the hyperparameter tuning module and an estimator, or any
 * system that can unpack a vector of values and produce a real evaluation.
 * @param estimator The estimator for GAME model
 * @param configs The initial configuration supplied by the user
 * @param data Training data
 * @param validationData Validation data
 * @param isOptMax a Boolean indicates that the problem is a maximization (true) or minimization (false).
 */
class GameEstimatorEvaluationFunction(
    estimator: GameEstimator,
    configs: Map[CoordinateId, CoordinateConfiguration],
    data: DataFrame,
    validationData: DataFrame,
    isOptMax: Boolean)
  extends EvaluationFunction[GameResult] {

  import GameEstimatorEvaluationFunction._

  // CoordinateOptimizationConfigurations sorted in order by coordinate ID name
  private val configsSeq = configs.toSeq.sortBy(_._1)

  // Pull the hyperparameter ranges from the optimization configuration
  protected[estimators] val ranges: Seq[DoubleRange] = configsSeq.flatMap { case (_, config) =>

    val regularizationWeightRange = config
      .regularizationWeightRange
      .getOrElse(DEFAULT_REG_WEIGHT_RANGE)
      .transform(log)

    val elasticNetParamRange = config
      .elasticNetParamRange
      .getOrElse(DEFAULT_REG_ALPHA_RANGE)

    config.optimizationConfiguration.regularizationContext.regularizationType match {
      case RegularizationType.ELASTIC_NET => Seq(regularizationWeightRange, elasticNetParamRange)
      case RegularizationType.L1 | RegularizationType.L2 => Seq(regularizationWeightRange)
      case RegularizationType.NONE => Seq()
    }
  }

  // Number of parameters in the base configuration
  val numParams: Int = ranges.length

  /**
   * Performs the evaluation.
   *
   * @param candidate The candidate vector of hyperparameter with values in [0, 1]
   * @return A tuple of the evaluated value and the original output from the inner estimator
   */
  override def apply(candidate: DenseVector[Double]): (Double, GameResult) = {

    val candidateScaled = VectorRescaling.scaleBackward(candidate, ranges)

    val newConfigurations = vectorToConfigurations(candidateScaled)

    estimator.setCoordinateConfigurations(newConfigurations)
    val model = estimator.fit(data, Some(validationData)).head
    val (_, _, Some(evaluations)) = model

    // If this is a maximization problem, flip signs of evaluation values
    val direction = if (isOptMax) -1 else 1

    // Assumes model selection evaluator is in "head" position
    (direction * evaluations.primaryEvaluation, model)
  }

  /**
   * Vectorize and scale a [[Seq]] of prior observations.
   *
   * @param observations Prior observations in estimator output form
   * @return Prior observations as tuples of (vector representation of the original estimator output, evaluated value)
   */
  override def convertObservations(observations: Seq[GameResult]): Seq[(DenseVector[Double], Double)] =
    observations.map { observation =>
      val candidate = vectorizeParams(observation)
      val candidateScaled = VectorRescaling.scaleForward(candidate, ranges)

      // If this is a maximization problem, flip signs of evaluation values
      val direction = if (isOptMax) -1 else 1

      val value = direction * getEvaluationValue(observation)

      (candidateScaled, value)
    }

  /**
   * Extracts a vector representation from the hyperparameters associated with the original estimator output.
   *
   * @param gameResult The original estimator output
   * @return A vector representation of hyperparameters for a [[GameResult]]
   */
  override def vectorizeParams(gameResult: GameResult): DenseVector[Double] =
    configurationToVector(gameResult._2)

  /**
   * Extracts the evaluated value from the original estimator output.
   *
   * @param gameResult The original estimator output
   * @return The evaluated value
   */
  override def getEvaluationValue(gameResult: GameResult): Double = gameResult match {
    case (_, _, Some(evaluations)) =>
      evaluations.primaryEvaluation

    case _ => throw new IllegalArgumentException(
      s"Can't extract evaluation value from a GAME result with no evaluations: $gameResult")
  }

  /**
   * Extracts a vector representation from the hyperparameters associated with the original estimator output.
   *
   * @param configuration The GAME optimization configuration containing parameters
   * @return A vector representation of hyperparameters for a [[GameOptimizationConfiguration]]
   */
  protected[ml] def configurationToVector(lambdaMap: Map[CoordinateId, Double]): DenseVector[Double] = {

    // Input configurations must contain the exact same coordinates as the base configuration
    require(
      configsSeq.size == lambdaMap.size,
      s"Configuration dimension mismatch; ${configsSeq.size} != ${lambdaMap.size}")
    configsSeq.foreach { case (coordinateId, _) =>
      require(lambdaMap.contains(coordinateId), s"Missing regularization weight for coordinate '$coordinateId'")
    }

    val parameterArray = configsSeq
      .flatMap { case (coordinateId, configuration) =>

        val regContext = configuration.optimizationConfiguration.regularizationContext

        regContext.regularizationType match {
          case RegularizationType.ELASTIC_NET => Seq(log(lambdaMap(coordinateId)), regContext.alpha)
          case RegularizationType.L1 | RegularizationType.L2 => Seq(log(lambdaMap(coordinateId)))
          case RegularizationType.NONE => Seq()
        }
      }
      .toArray

    require(parameterArray.length == numParams, s"Dimension mismatch; $numParams != ${parameterArray.length}")

    DenseVector(parameterArray)
  }

  /**
   * Unpacks the regularization weights from the hyperparameter vector, and returns an equivalent GAME optimization
   * configuration.
   *
   * @param hyperParameters The hyperparameter vector
   * @return The equivalent GAME optimization configuration
   */
  protected[ml] def vectorToConfigurations(
      hyperParameters: DenseVector[Double]): Map[CoordinateId, CoordinateConfiguration] = {

    require(hyperParameters.length == numParams, s"Dimension mismatch; $numParams != ${hyperParameters.length}")

    val paramValues = mutable.Queue(hyperParameters.toArray: _*)

    configsSeq
      .map { case (coordinateId, coordinateConfig) =>

        val regularizationContext = coordinateConfig.optimizationConfiguration.regularizationContext

        val newCoordinateConfig = regularizationContext.regularizationType match {
          case RegularizationType.ELASTIC_NET =>
            val regularizationWeight = exp(paramValues.dequeue())
            val elasticNetAlpha = paramValues.dequeue()
            val newRegularizationContext = regularizationContext.copy(elasticNetParam = Some(elasticNetAlpha))

            updateRegularizationWeight(coordinateConfig, regularizationWeight) match {
              case fixedConfig: FixedEffectCoordinateConfiguration =>
                fixedConfig.copy(
                  optimizationConfiguration = fixedConfig
                    .optimizationConfiguration
                    .copy(regularizationContext = newRegularizationContext))

              case randomConfig: RandomEffectCoordinateConfiguration =>
                randomConfig.copy(
                  optimizationConfiguration = randomConfig
                    .optimizationConfiguration
                    .copy(regularizationContext = newRegularizationContext))
            }

          case RegularizationType.L1 | RegularizationType.L2 =>
            updateRegularizationWeight(coordinateConfig, exp(paramValues.dequeue()))

          case RegularizationType.NONE =>
            coordinateConfig
        }

        (coordinateId, newCoordinateConfig)
      }
      .toMap
  }

  private def updateRegularizationWeight(
      configuration: CoordinateConfiguration,
      newRegWeight: Double): CoordinateConfiguration = configuration match {

    case fixedConfig: FixedEffectCoordinateConfiguration =>
      fixedConfig.copy(regularizationWeights = Set(newRegWeight))

    case randomConfig: RandomEffectCoordinateConfiguration =>
      randomConfig.copy(regularizationWeights = Set(newRegWeight))

    case other =>
      throw new IllegalArgumentException(s"Unknown coordinate configuration type: ${other.getClass}")
  }
}

object GameEstimatorEvaluationFunction {
  val DEFAULT_REG_WEIGHT_RANGE = DoubleRange(1e-4, 1e4)
  val DEFAULT_REG_ALPHA_RANGE = DoubleRange(0.0, 1.0)
}
