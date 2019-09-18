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
import scala.language.existentials

import org.apache.commons.cli.MissingArgumentException
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel
import org.slf4j.Logger

import com.linkedin.photon.ml.{CoordinateConfiguration, FixedEffectCoordinateConfiguration, RandomEffectCoordinateConfiguration, TaskType}
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types.{CoordinateId, FeatureShardId, UniqueSampleId}
import com.linkedin.photon.ml.algorithm._
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.function.glm._
import com.linkedin.photon.ml.model.{GameModel, RandomEffectModel}
import com.linkedin.photon.ml.normalization._
import com.linkedin.photon.ml.optimization.{DistributedOptimizationProblem, VarianceComputationType}
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.sampling.{DownSampler, DownSamplerHelper}
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.util._

/**
 * Estimator implementation for GAME models.
 *
 * @param sc The spark context for the application
 * @param logger The logger instance for the application
 */
class GameEstimator(val sc: SparkContext, implicit val logger: Logger) extends PhotonParams {

  import GameEstimator._

  // 2 types that make the code more readable
  type SingleNodeLossFunctionConstructor = PointwiseLossFunction => SingleNodeGLMLossFunction
  type DistributedLossFunctionConstructor = PointwiseLossFunction => DistributedGLMLossFunction

  private implicit val parent: Identifiable = this

  override val uid: String = Identifiable.randomUID(GAME_ESTIMATOR_PREFIX)

  //
  // Parameters
  //

  val trainingTask: Param[TaskType] = ParamUtils.createParam(
    "training task",
    "The type of training task to perform.")

  val inputColumnNames: Param[InputColumnsNames] = ParamUtils.createParam[InputColumnsNames](
    "input column names",
    "A map of custom column names which replace the default column names of expected fields in the Avro input.")

  val coordinateConfigurations: Param[Map[CoordinateId, CoordinateConfiguration]] =
    ParamUtils.createParam[Map[CoordinateId, CoordinateConfiguration]](
      "coordinate configurations",
      "A map of coordinate names to configurations.",
      PhotonParamValidators.nonEmpty[TraversableOnce, (CoordinateId, CoordinateConfiguration)])

  val coordinateUpdateSequence: Param[Seq[CoordinateId]] = ParamUtils.createParam(
    "coordinate update sequence",
    "The order in which coordinates are updated by the descent algorithm. It is recommended to order coordinates by " +
      "their stability (i.e. by looking at the variance of the feature distribution [or correlation with labels] for " +
      "each coordinate).",
    PhotonParamValidators.nonEmpty[Seq, CoordinateId])

  val coordinateDescentIterations: Param[Int] = ParamUtils.createParam(
    "coordinate descent iterations",
    "The number of coordinate descent iterations (one iteration is one full traversal of the update sequence).",
    ParamValidators.gt[Int](0.0))

  val coordinateNormalizationContexts: Param[Map[CoordinateId, NormalizationContext]] =
    ParamUtils.createParam[Map[CoordinateId, NormalizationContext]](
      "normalization contexts",
      "The normalization contexts for each coordinate. The type of normalization should be the same for each " +
        "coordinate, but the shifts and factors are different for each shard.",
      PhotonParamValidators.nonEmpty[TraversableOnce, (CoordinateId, NormalizationContext)])

  val initialModel: Param[GameModel] = ParamUtils.createParam(
    "initial model",
    "Prior model to use as a starting point for training.")

  val partialRetrainLockedCoordinates: Param[Set[CoordinateId]] = ParamUtils.createParam(
    "partial retrain locked coordinates",
    "The set of coordinates present in the pre-trained model to reuse during partial retraining.")

  val varianceComputationType: Param[VarianceComputationType] = ParamUtils.createParam[VarianceComputationType](
    "variance computation type",
    "Whether to compute coefficient variances and, if so, how.")

  val treeAggregateDepth: Param[Int] = ParamUtils.createParam[Int](
    "tree aggregate depth",
    "Suggested depth for tree aggregation.",
    ParamValidators.gt[Int](0.0))

  val validationEvaluators: Param[Seq[EvaluatorType]] = ParamUtils.createParam(
    "validation evaluators",
    "A list of evaluators used to validate computed scores (Note: the first evaluator in the list is the one used " +
      "for model selection)",
    PhotonParamValidators.nonEmpty[Seq, EvaluatorType])

  val ignoreThresholdForNewModels: Param[Boolean] = ParamUtils.createParam[Boolean](
    "ignore threshold for new models",
    "Flag to ignore the random effect samples lower bound when encountering a random effect ID without an existing " +
      "model during warm-start training.")

  val useWarmStart: Param[Boolean] = ParamUtils.createParam[Boolean](
    "use warm start",
    "Whether to train the current model with coefficients initialized by the previous model.")

  //
  // Initialize object
  //

  setDefaultParams()

  //
  // Parameter setters
  //

  def setTrainingTask(value: TaskType): this.type = set(trainingTask, value)

  def setInputColumnNames(value: InputColumnsNames): this.type = set(inputColumnNames, value)

  def setCoordinateConfigurations(value: Map[CoordinateId, CoordinateConfiguration]): this.type =
    set(coordinateConfigurations, value)

  def setCoordinateUpdateSequence(value: Seq[CoordinateId]): this.type = set(coordinateUpdateSequence, value)

  def setCoordinateDescentIterations(value: Int): this.type = set(coordinateDescentIterations, value)

  def setCoordinateNormalizationContexts(value: Map[CoordinateId, NormalizationContext]): this.type =
    set(coordinateNormalizationContexts, value)

  def setInitialModel(value: GameModel): this.type = set(initialModel, value)

  def setPartialRetrainLockedCoordinates(value: Set[CoordinateId]): this.type =
    set(partialRetrainLockedCoordinates, value)

  def setVarianceComputation(value: VarianceComputationType): this.type = set(varianceComputationType, value)

  def setTreeAggregateDepth(value: Int): this.type = set(treeAggregateDepth, value)

  def setValidationEvaluators(value: Seq[EvaluatorType]): this.type = set(validationEvaluators, value)

  def setIgnoreThresholdForNewModels(value: Boolean): this.type = set(ignoreThresholdForNewModels, value)

  def setUseWarmStart(value: Boolean): this.type = set(useWarmStart, value)

  //
  // Params trait extensions
  //

  override def copy(extra: ParamMap): GameEstimator = {

    val copy = new GameEstimator(sc, logger)

    extractParamMap(extra).toSeq.foreach { paramPair =>
      copy.set(copy.getParam(paramPair.param.name), paramPair.value)
    }

    copy
  }

  //
  // PhotonParams trait extensions
  //

  /**
   * Set the default parameters.
   */
  override protected def setDefaultParams(): Unit = {

    setDefault(inputColumnNames, InputColumnsNames())
    setDefault(coordinateDescentIterations, 1)
    setDefault(partialRetrainLockedCoordinates, Set.empty[CoordinateId])
    setDefault(varianceComputationType, VarianceComputationType.NONE)
    setDefault(treeAggregateDepth, DEFAULT_TREE_AGGREGATE_DEPTH)
    setDefault(ignoreThresholdForNewModels, false)
    setDefault(useWarmStart, true)
  }

  /**
   * Check that all required parameters have been set and validate interactions between parameters.
   *
   * @note In Spark, interactions between parameters are checked by
   *       [[org.apache.spark.ml.PipelineStage.transformSchema()]]. Since we do not use the Spark pipeline API in
   *       Photon-ML, we need to have this function to check the interactions between parameters.
   * @throws MissingArgumentException if a required parameter is missing
   * @throws IllegalArgumentException if a required parameter is missing or a validation check fails
   * @param paramMap The parameters to validate
   */
  override def validateParams(paramMap: ParamMap = extractParamMap): Unit = {

    // Just need to check that the training task has been explicitly set
    getRequiredParam(trainingTask)

    val updateSequence = getRequiredParam(coordinateUpdateSequence)
    val configs = getRequiredParam(coordinateConfigurations)
    val initialModelOpt = get(initialModel)
    val retrainModelCoordsOpt = get(partialRetrainLockedCoordinates)
    val normalizationContextsOpt = get(coordinateNormalizationContexts)
    val ignoreThreshold = getOrDefault(ignoreThresholdForNewModels)

    val numUniqueCoordinates = updateSequence.toSet.size

    // Cannot have coordinates repeat in the update sequence
    require(
      numUniqueCoordinates == updateSequence.size,
      "One or more coordinates are repeated in the update sequence.")

    // Warm-start must be enabled to ignore threshold
    require(
      !ignoreThreshold || initialModelOpt.isDefined,
      "'Ignore threshold for new models' flag set but no initial model provided for warm-start")

    // Partial retraining and warm-start training require an initial GAME model to be provided as input
    val coordinatesToTrain = (initialModelOpt, retrainModelCoordsOpt) match {
      case (Some(initModel), Some(retrainModelCoords)) =>

        val newCoordinates = updateSequence.filterNot(retrainModelCoords.contains)

        // Locked coordinates cannot be empty
        require(
          retrainModelCoords.nonEmpty,
          "Set of locked coordinates is empty.")

        // No point in training if every coordinate is being reused
        require(
          newCoordinates.nonEmpty,
          "All coordinates in the update sequence are re-used from the initial model: no new coordinates to train.")

        // All locked coordinates must be used by the update sequence
        require(
          retrainModelCoords.forall(updateSequence.contains),
          "One or more locked coordinates for partial retraining are missing from the update sequence.")

        // All locked coordinates must be present in the initial model
        require(
          retrainModelCoords.forall(initModel.toMap.contains),
          "One or more locked coordinates for partial retraining are missing from the initial model.")

        newCoordinates

      case (Some(_), None) | (None, None) =>
        updateSequence

      case (None, Some(_)) =>
        throw new IllegalArgumentException("Partial retraining enabled, but no base model provided.")
    }

    // All coordinates in update sequence which are being trained need a configuration
    val missingCoords = coordinatesToTrain.filterNot(configs.contains)
    require(
      missingCoords.isEmpty,
      s"Coordinates '${missingCoords.mkString(", ")}' are missing from the coordinate configurations.")

    // All coordinates in the update sequence which are not being trained should not have a configuration
    val badLockedCoords = configs.keys.toSet.intersect(retrainModelCoordsOpt.getOrElse(Set()))
    require(
      badLockedCoords.isEmpty,
      s"Locked coordinates '${badLockedCoords.mkString(", ")}' are present in the optimization configurations.")

    // If normalization is enabled, all non-locked coordinates must have a NormalizationContext
    coordinatesToTrain.foreach { coordinate =>
      require(
        normalizationContextsOpt.forall(normalizationContexts => normalizationContexts.contains(coordinate)),
        s"Coordinate $coordinate in the update sequence is missing normalization context")
    }
  }

  //
  // GameEstimator functions
  //

  /**
   * Fits a GAME model to the training dataset, once per configuration.
   *
   * @param data The training set
   * @param validationData Optional validation set for per-iteration validation
   * @return A set of (trained GAME model, optional evaluation results, GAME model configuration) tuples, one for each
   *         configuration
   */
  def fit(data: DataFrame, validationData: Option[DataFrame]): Seq[GameResult] = {

    // Verify valid GameEstimator settings
    validateParams()

    val configurations = getRequiredParam(coordinateConfigurations)

    // Group additional columns to include in GameDatum
    val randomEffectIdCols: Set[String] = configurations
      .flatMap { case (_, config) =>
        config.dataConfiguration match {
          case reConfig: RandomEffectDataConfiguration => Some(reConfig.randomEffectType)
          case _ => None
        }
      }
      .toSet
    val evaluatorCols = get(validationEvaluators).map(MultiEvaluatorType.getMultiEvaluatorIdTags).getOrElse(Set())
    val additionalCols = randomEffectIdCols ++ evaluatorCols

    // Gather the names of the feature shards used by the coordinates
    val featureShards = configurations
      .map { case (_, config) =>
        config.dataConfiguration.featureShardId
      }
      .toSet

    // Transform the GAME training data set into fixed and random effect specific datasets
    val gameDataset = Timed("Process training data from raw DataFrame to RDD of samples") {
      prepareGameDataset(data, featureShards, additionalCols)
    }
    val coordinates = Timed("") {
      prepareCoordinates(configurations, gameDataset)
    }
    val regularizationWeightMaps = Timed("") {
      prepareRegularizationWeights(configurations)
    }

    // Transform the GAME validation data set into fixed and random effect specific datasets
    val validationDatasetAndEvaluationSuiteOpt = Timed("Prepare validation data, if any") {
      prepareValidationDatasetAndEvaluators(
        validationData,
        featureShards,
        additionalCols)
    }

    val coordinateDescent = new CoordinateDescent(
      getRequiredParam(coordinateUpdateSequence),
      getOrDefault(coordinateDescentIterations),
      validationDatasetAndEvaluationSuiteOpt,
      getOrDefault(partialRetrainLockedCoordinates),
      logger)

    // Train GAME models on training data
    val results = Timed("Training models:") {
      var prevGameModel: Option[GameModel] = if (getOrDefault(useWarmStart)) {
        get(initialModel)
      } else {
        None
      }

      regularizationWeightMaps.map { lambdaMap =>
        val newCoordinates = coordinates.map { case (coordinateId, coordinate) =>
          val newCoordinate: C forSome { type C <: Coordinate[_] } = if (lambdaMap.contains(coordinateId)) {
            coordinate.updateRegularizationWeight(lambdaMap(coordinateId))
          } else {
            coordinate
          }

          (coordinateId, newCoordinate)
        }

        val (gameModel, evaluations) = coordinateDescent.run(newCoordinates, prevGameModel.map(_.toMap))

        if (getOrDefault(useWarmStart)) prevGameModel = Some(gameModel)

        (gameModel, lambdaMap, evaluations)
      }
    }

    // Purge the raw GAME data, training data, and validation data in reverse order of definition
    validationDatasetAndEvaluationSuiteOpt.map { case (validationDataset, evaluationSuite) =>
      validationDataset.unpersist()
      evaluationSuite.unpersistRDD()
    }
    coordinates.foreach { case (_, coordinate) =>
      coordinate match {
        case rddLike: RDDLike => rddLike.unpersistRDD()
        case _ =>
      }
      coordinate match {
        case broadcastLike: BroadcastLike => broadcastLike.unpersistBroadcast()
        case _ =>
      }
    }
    gameDataset.unpersist()

    // Return the trained models, along with validation information (if any), and model configuration
    results
  }

  /**
   * Construct a [[RDD]] of data processed into GAME format from a raw [[DataFrame]].
   *
   * @param data The raw [[DataFrame]]
   * @param featureShards The IDs of the feature shards to keep
   * @param additionalCols The names of fields containing information necessary for random effects or evaluation
   * @return A [[RDD]] of data processed into GAME format
   */
  protected def prepareGameDataset(
      data: DataFrame,
      featureShards: Set[FeatureShardId],
      additionalCols: Set[String]): RDD[(UniqueSampleId, GameDatum)] =
    GameConverters
      .getGameDatasetFromDataFrame(
        data,
        featureShards,
        additionalCols,
        isResponseRequired = true,
        getOrDefault(inputColumnNames))
      .partitionBy(new LongHashPartitioner(data.rdd.getNumPartitions))
      .setName("GAME training data")
      .persist(StorageLevel.DISK_ONLY)

  /**
   *
   * @param configurations
   * @param gameDataset
   * @return
   */
  protected def prepareCoordinates(
      configurations: Map[CoordinateId, CoordinateConfiguration],
      gameDataset: RDD[(UniqueSampleId, GameDatum)]): Map[CoordinateId, C forSome { type C <: Coordinate[_] }] = {

    logger.info("Model configuration:")
    configurations.foreach { case (coordinateId, coordinateConfig) =>
      logger.info(s"coordinate '$coordinateId':\n\n$coordinateConfig\n")
    }

    val task = getRequiredParam(trainingTask)
    val normalizationContexts = get(coordinateNormalizationContexts).getOrElse(Map())
    val varianceType = getOrDefault(varianceComputationType)
    val treeAggDepth = getOrDefault(treeAggregateDepth)
    val (glmConstructor, pointwiseLossFunction) = task match {
      case TaskType.LOGISTIC_REGRESSION => (LogisticRegressionModel.apply _, LogisticLossFunction)
      case TaskType.LINEAR_REGRESSION => (LinearRegressionModel.apply _, SquaredLossFunction)
      case TaskType.POISSON_REGRESSION => (PoissonRegressionModel.apply _, PoissonLossFunction)
      case _ => throw new Exception("Need to specify a valid loss function")
    }
    val downSamplerFactory = DownSamplerHelper.buildFactory(task)
    val lockedCoordinates = get(partialRetrainLockedCoordinates).getOrElse(Set())

    configurations.map { case (coordinateId, config) =>
      config match {
        case fixedEffectConfig: FixedEffectCoordinateConfiguration =>

          val fixedEffectOptConfig = fixedEffectConfig.optimizationConfiguration

          val fixedEffectDataset = FixedEffectDataset(gameDataset, fixedEffectConfig.dataConfiguration.featureShardId)
            .setName(s"Fixed Effect Dataset: $coordinateId")
            .persistRDD(StorageLevel.DISK_ONLY)
          val lossFunction = DistributedGLMLossFunction(fixedEffectOptConfig, pointwiseLossFunction, treeAggDepth)
          val downSamplerOpt = if (DownSampler.isValidDownSamplingRate(fixedEffectOptConfig.downSamplingRate)) {
            Some(downSamplerFactory(fixedEffectOptConfig.downSamplingRate))
          } else {
            None
          }
          val normalizationContextBroadcast =
            PhotonBroadcast(gameDataset.sparkContext.broadcast(normalizationContexts(coordinateId)))

          val fixedEffectCoordinate = if (lockedCoordinates.contains(coordinateId)) {
            new FixedEffectModelCoordinate(fixedEffectDataset)
          } else {
            new FixedEffectCoordinate(
              fixedEffectDataset,
              DistributedOptimizationProblem(
                fixedEffectOptConfig,
                lossFunction,
                downSamplerOpt,
                glmConstructor,
                normalizationContextBroadcast,
                varianceType))
          }

          // Eval this only in debug mode, because the "toSummaryString" call can be very expensive
          if (logger.isDebugEnabled) {
            logger.debug(
              s"Summary of fixed effect dataset for '$coordinateId':\n\n${fixedEffectDataset.toSummaryString}\n")
          }

          (coordinateId, fixedEffectCoordinate)

        case randomEffectConfig: RandomEffectCoordinateConfiguration =>

          val randomEffectDataConfig = randomEffectConfig.dataConfiguration
          val randomEffectOptConfig = randomEffectConfig.optimizationConfiguration

          val rePartitioner = RandomEffectDatasetPartitioner.fromGameDataset(gameDataset, randomEffectDataConfig)
          val existingModelKeysRddOpt = if (getOrDefault(ignoreThresholdForNewModels)) {
            getRequiredParam(initialModel).getModel(coordinateId).map {
              case rem: RandomEffectModel =>
                rem.modelsRDD.partitionBy(rePartitioner).keys

              case other =>
                throw new IllegalArgumentException(
                  s"Model type mismatch: expected Random Effect Model but found '${other.getClass}'")
            }
          } else {
            None
          }
          val randomEffectDataset = RandomEffectDataset(
            gameDataset,
            randomEffectDataConfig,
            rePartitioner,
            existingModelKeysRddOpt,
            StorageLevel.DISK_ONLY)
          randomEffectDataset.setName(s"Random Effect Data Set: $coordinateId")
          val lossFunction = SingleNodeGLMLossFunction(randomEffectOptConfig, pointwiseLossFunction)

          val randomEffectCoordinate = if (lockedCoordinates.contains(coordinateId)) {
            new RandomEffectModelCoordinate(randomEffectDataset)
          } else {
            RandomEffectCoordinate(
              randomEffectDataset,
              randomEffectOptConfig,
              lossFunction,
              glmConstructor,
              normalizationContexts(coordinateId),
              varianceType)
          }

          if (logger.isDebugEnabled) {
            // Eval this only in debug mode, because the call to "toSummaryString" can be very expensive
            logger.debug(
              s"Summary of random effect dataset with coordinate ID $coordinateId:\n" +
                s"${randomEffectDataset.toSummaryString}\n")
          }

          (coordinateId, randomEffectCoordinate)
      }
    }
  }

  protected def prepareRegularizationWeights(
      configurations: Map[CoordinateId, CoordinateConfiguration]): Seq[Map[CoordinateId, Double]] =

    configurations.foldLeft(Seq(Map[CoordinateId, Double]())){ case (abcdefg, coordinateConfig) =>

      val result = mutable.Seq[Map[CoordinateId, Double]]()
      val coordinateId = coordinateConfig._1
      val regularizationWeights = coordinateConfig._2.regularizationWeights

      abcdefg.foreach { lambdaMap =>
        regularizationWeights.foreach { lambda =>
          result ++ (lambdaMap + (coordinateId -> lambda))
        }
      }

      result.toSeq
    }

  /**
   * Optionally construct an [[RDD]] of validation data samples, and an [[EvaluationSuite]] to compute evaluation metrics
   * over the validation data.
   *
   * @param dataOpt Optional [[DataFrame]] of validation data
   * @param featureShards The feature shard columns to import from the [[DataFrame]]
   * @param additionalCols A set of additional columns whose values should be maintained for validation evaluation
   * @return An optional ([[RDD]] of validation data, validation metric [[EvaluationSuite]]) tuple
   */
  protected def prepareValidationDatasetAndEvaluators(
      dataOpt: Option[DataFrame],
      featureShards: Set[FeatureShardId],
      additionalCols: Set[String]): Option[(RDD[(UniqueSampleId, GameDatum)], EvaluationSuite)] =

    dataOpt.map { data =>
      val partitioner = new LongHashPartitioner(data.rdd.partitions.length)
      val gameDataset = Timed("Convert validation data from raw DataFrame to processed RDD of GAME data") {
        GameConverters
          .getGameDatasetFromDataFrame(
            data,
            featureShards,
            additionalCols,
            isResponseRequired = true,
            getOrDefault(inputColumnNames))
          .partitionBy(partitioner)
          .setName("Validation Game dataset")
          .persist(StorageLevel.DISK_ONLY)
      }
      val evaluationSuite = Timed("Prepare validation metric evaluators") {
        prepareValidationEvaluators(gameDataset)
      }

      (gameDataset, evaluationSuite)
    }

  /**
   * Construct the validation [[EvaluationSuite]].
   *
   * @param gameDataset An [[RDD]] of validation data samples
   * @return [[EvaluationSuite]] containing one or more validation metric [[Evaluator]] objects
   */
  protected def prepareValidationEvaluators(gameDataset: RDD[(UniqueSampleId, GameDatum)]): EvaluationSuite = {

    val validatingLabelsAndOffsetsAndWeights = gameDataset.mapValues { gameData =>
      (gameData.response, gameData.offset, gameData.weight)
    }
    val evaluators = get(validationEvaluators)
      .map(_.map(EvaluatorFactory.buildEvaluator(_, gameDataset)))
      .getOrElse(Seq(getDefaultEvaluator(getRequiredParam(trainingTask))))
    val evaluationSuite = EvaluationSuite(evaluators, validatingLabelsAndOffsetsAndWeights)
      .setName(s"Evaluation: validation data labels, offsets, and weights")
      .persistRDD(StorageLevel.MEMORY_AND_DISK)

    if (logger.isDebugEnabled) {

      val randomScores = gameDataset.mapValues(_ => math.random).persist()

      evaluationSuite
        .evaluate(randomScores)
        .evaluations
        .foreach { case (evaluator, evaluation) =>
          logger.debug(s"Random guessing baseline for evaluation metric '${evaluator.name}': $evaluation")
        }

      randomScores.unpersist()
    }

    evaluationSuite
  }
}

object GameEstimator {

  //
  // Types
  //

  type GameResult = (GameModel, Map[CoordinateId, Double], Option[EvaluationResults])

  //
  // Constants
  //

  private val GAME_ESTIMATOR_PREFIX = "GameEstimator"

  val DEFAULT_TREE_AGGREGATE_DEPTH = 1

  //
  // Functions
  //

  /**
   *
   * @param task
   * @return
   */
  def getDefaultEvaluator(task: TaskType): Evaluator = task match {
    case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => AreaUnderROCCurveEvaluator
    case TaskType.LINEAR_REGRESSION => RMSEEvaluator
    case TaskType.POISSON_REGRESSION => PoissonLossEvaluator
    case _ => throw new UnsupportedOperationException(s"$task is not a valid GAME training task")
  }
}
