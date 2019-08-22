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

import scala.math.log

import java.util.Random

import breeze.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.estimators.GameEstimator.GameOptimizationConfiguration
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game._
import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Unit tests for [[GameEstimatorEvaluationFunction]].
 */
class GameEstimatorEvaluationFunctionTest {

  import GameEstimatorEvaluationFunction._
  import GameEstimatorEvaluationFunctionTest._

  /**
   * Test that hyperparameter ranges are correctly constructed from a [[GameOptimizationConfiguration]].
   */
  @Test
  def testRanges(): Unit = {
    val configuration: GameOptimizationConfiguration = Map(
      ("a", FIXED_EFFECT_OPTIMIZATION_CONFIG.copy(regularizationWeightRange = Some(DoubleRange(0.01, 100.0)))),
      ("b", RANDOM_EFFECT_OPTIMIZATION_CONFIG),
      ("c", RANDOM_EFFECT_OPTIMIZATION_CONFIG.copy(
        regularizationContext = ElasticNetRegularizationContext(REGULARIZATION_ALPHAS(0)),
        regularizationWeight = REGULARIZATION_WEIGHTS(2),
        elasticNetParamRange = Some(DoubleRange(0.0, 0.5)))))

    val evaluationFunction = new GameEstimatorEvaluationFunction(
      MOCK_ESTIMATOR,
      configuration,
      MOCK_DATA,
      MOCK_DATA,
      IS_MAX_OPTIMAL)

    assertEquals(evaluationFunction.ranges, Seq(
      DoubleRange(0.01, 100.0).transform(log),
      DEFAULT_REG_WEIGHT_RANGE.transform(log),
      DEFAULT_REG_WEIGHT_RANGE.transform(log),
      DoubleRange(0.0, 0.5)))
  }

  /**
   * Test that a [[GameOptimizationConfiguration]] can be correctly constructed from a hyperparameter vector.
   */
  @Test
  def testVectorToConfiguration(): Unit = {

    val configuration: GameOptimizationConfiguration = Map(
      ("a", FIXED_EFFECT_OPTIMIZATION_CONFIG),
      ("b", RANDOM_EFFECT_OPTIMIZATION_CONFIG),
      ("c", RANDOM_EFFECT_OPTIMIZATION_CONFIG.copy(
        regularizationContext = ElasticNetRegularizationContext(REGULARIZATION_ALPHAS(0)),
        regularizationWeight = REGULARIZATION_WEIGHTS(2))))

    val evaluationFunction = new GameEstimatorEvaluationFunction(
      MOCK_ESTIMATOR,
      configuration,
      MOCK_DATA,
      MOCK_DATA,
      IS_MAX_OPTIMAL)
    val hypers = DenseVector(
      log(REGULARIZATION_WEIGHTS(3)),
      log(REGULARIZATION_WEIGHTS(4)),
      log(REGULARIZATION_WEIGHTS(5)),
      REGULARIZATION_ALPHAS(1))
    val newConfiguration = evaluationFunction.vectorToConfiguration(hypers)

    assertEquals(
      newConfiguration("a").asInstanceOf[FixedEffectOptimizationConfiguration].regularizationWeight,
      REGULARIZATION_WEIGHTS(3),
      TOLERANCE)
    assertEquals(
      newConfiguration("b").asInstanceOf[RandomEffectOptimizationConfiguration].regularizationWeight,
      REGULARIZATION_WEIGHTS(4),
      TOLERANCE)
    assertEquals(
      newConfiguration("c").asInstanceOf[RandomEffectOptimizationConfiguration].regularizationWeight,
      REGULARIZATION_WEIGHTS(5),
      TOLERANCE)
    assertEquals(
      newConfiguration("c").asInstanceOf[RandomEffectOptimizationConfiguration]
        .regularizationContext
        .elasticNetParam
        .get,
      REGULARIZATION_ALPHAS(1),
      TOLERANCE)
  }

  @DataProvider
  def invalidVectorProvider(): Array[Array[Any]] = {

    val configuration: GameOptimizationConfiguration = Map(
      ("a", FIXED_EFFECT_OPTIMIZATION_CONFIG),
      ("b", RANDOM_EFFECT_OPTIMIZATION_CONFIG))

    Array(
      Array(configuration, DenseVector(log(REGULARIZATION_WEIGHTS(0)))),
      Array(
        configuration,
        DenseVector(log(REGULARIZATION_WEIGHTS(0)), log(REGULARIZATION_WEIGHTS(1)), log(REGULARIZATION_WEIGHTS(2)))))
  }

  /**
   * Test that errors caused by invalid vectors will be caught when attempting to construct a
   * [[GameOptimizationConfiguration]].
   *
   * @param config The base configuration to use as a template
   * @param hypers The hyperparameter vector
   */
  @Test(dataProvider = "invalidVectorProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidVectorToConfiguration(config: GameOptimizationConfiguration, hypers: DenseVector[Double]): Unit = {

    val evaluationFunction = new GameEstimatorEvaluationFunction(
      MOCK_ESTIMATOR,
      config,
      MOCK_DATA,
      MOCK_DATA,
      IS_MAX_OPTIMAL)

    evaluationFunction.vectorToConfiguration(hypers)
  }

  @DataProvider
  def configurationProvider: Array[Array[Any]] =
    Array(
      Array(
        Map(("a", FIXED_EFFECT_OPTIMIZATION_CONFIG)),
        DenseVector(log(REGULARIZATION_WEIGHTS(0)))),
      Array(
        Map(("b", RANDOM_EFFECT_OPTIMIZATION_CONFIG)),
        DenseVector(log(REGULARIZATION_WEIGHTS(1)))),
      Array(
        Map(
          ("a", FIXED_EFFECT_OPTIMIZATION_CONFIG),
          ("b", RANDOM_EFFECT_OPTIMIZATION_CONFIG)),
        DenseVector(log(REGULARIZATION_WEIGHTS(0)), log(REGULARIZATION_WEIGHTS(1)))),
      Array(
        Map(
          ("a",
            FIXED_EFFECT_OPTIMIZATION_CONFIG.copy(
              regularizationContext = ElasticNetRegularizationContext(REGULARIZATION_ALPHAS(0)))),
          ("b", RANDOM_EFFECT_OPTIMIZATION_CONFIG)),
        DenseVector(log(REGULARIZATION_WEIGHTS(0)), REGULARIZATION_ALPHAS(0), log(REGULARIZATION_WEIGHTS(1)))))

  /**
   * Test that a [[GameOptimizationConfiguration]] can be correctly converted to a hyperparameter vector.
   *
   * @param config The base configuration to use as a template
   * @param expected The expected hyperparameter vector for the configuration
   */
  @Test(dataProvider = "configurationProvider")
  def testConfigurationToVector(config: GameOptimizationConfiguration, expected: DenseVector[Double]): Unit = {

    val evaluationFunction = new GameEstimatorEvaluationFunction(
      MOCK_ESTIMATOR,
      config,
      MOCK_DATA,
      MOCK_DATA,
      IS_MAX_OPTIMAL)
    val result = evaluationFunction.configurationToVector(config)

    assertEquals(result.length, expected.length)
    assertIterableEqualsWithTolerance(result.toArray, expected.toArray, TOLERANCE)
  }

  @DataProvider
  def invalidConfigurationProvider: Array[Array[Any]] = {

    val configuration1: GameOptimizationConfiguration = Map(
      ("a", FIXED_EFFECT_OPTIMIZATION_CONFIG))
    val configuration2: GameOptimizationConfiguration = Map(
      ("a", FIXED_EFFECT_OPTIMIZATION_CONFIG),
      ("b", RANDOM_EFFECT_OPTIMIZATION_CONFIG))
    val configuration3: GameOptimizationConfiguration = Map(
      ("a", FIXED_EFFECT_OPTIMIZATION_CONFIG),
      ("c", RANDOM_EFFECT_OPTIMIZATION_CONFIG))
    val configuration4: GameOptimizationConfiguration = Map(
      ("a", RANDOM_EFFECT_OPTIMIZATION_CONFIG),
      ("c", RANDOM_EFFECT_OPTIMIZATION_CONFIG))

    Array(
      // Configuration dimension size mismatch
      Array(configuration1, configuration2),
      Array(configuration2, configuration1),

      // Configuration coordinate mismatch
      Array(configuration2, configuration3),

      // Configuration coordinate type mismatch
      Array(configuration3, configuration4))
  }

  /**
   * Test that [[GameOptimizationConfiguration]] instances which do not match the template configuration will be caught
   * and throw error when attempting to construct a hyperparameter vector.
   *
   * @param baseConfig The base configuration to use as a template
   * @param vectorConfig A dissimilar configuration from which to attempt to construct a hyperparameter vector
   */
  @Test(dataProvider = "invalidConfigurationProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testConfigurationToVector(
    baseConfig: GameOptimizationConfiguration,
    vectorConfig: GameOptimizationConfiguration): Unit = {

    val evaluationFunction = new GameEstimatorEvaluationFunction(
      MOCK_ESTIMATOR,
      baseConfig,
      MOCK_DATA,
      MOCK_DATA,
      IS_MAX_OPTIMAL)
    evaluationFunction.configurationToVector(vectorConfig)
  }
}

object GameEstimatorEvaluationFunctionTest {

  private val MOCK_REGULARIZATION_CONTEXT = L2RegularizationContext
  private val MOCK_ESTIMATOR = mock(classOf[GameEstimator])
  private val MOCK_DATA = mock(classOf[DataFrame])

  private val RANDOM = new Random(1)
  private val REGULARIZATION_SIZE = 6
  private val REGULARIZATION_WEIGHTS = Array.fill[Double](REGULARIZATION_SIZE) { RANDOM.nextDouble }
  private val REGULARIZATION_ALPHAS = Array.fill[Double](REGULARIZATION_SIZE) { RANDOM.nextDouble }
  private val FIXED_EFFECT_OPTIMIZATION_CONFIG = FixedEffectOptimizationConfiguration(
    OptimizerType.LBFGS,
    maximumIterations = 1,
    tolerance = 0.1,
    MOCK_REGULARIZATION_CONTEXT,
    REGULARIZATION_WEIGHTS(0))
  private val RANDOM_EFFECT_OPTIMIZATION_CONFIG = RandomEffectOptimizationConfiguration(
    OptimizerType.LBFGS,
    maximumIterations = 1,
    tolerance = 0.1,
    MOCK_REGULARIZATION_CONTEXT,
    REGULARIZATION_WEIGHTS(1))
  private val TOLERANCE = MathConst.EPSILON
  private val IS_MAX_OPTIMAL = true
}
