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
package com.linkedin.photon.ml.algorithm

import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.{DatumScoringModel, RandomEffectModel}
import com.linkedin.photon.ml.optimization.game.RandomEffectOptimizationProblem
import com.linkedin.photon.ml.optimization.OptimizationTracker

/**
 * The optimization problem coordinate for a random effect model.
 *
 * @tparam Objective The type of objective function used to solve individual random effect optimization problems
 * @param dataset The training dataset
 * @param optimizationProblem The random effect optimization problem
 */
protected[ml] class RandomEffectCoordinate[Objective <: SingleNodeObjectiveFunction](
    override protected val  dataset: RandomEffectDataset,
    protected val optimizationProblem: RandomEffectOptimizationProblem)
  extends Coordinate[RandomEffectDataset](dataset) {

  //
  // Coordinate functions
  //

  /**
   * Update the coordinate with a new [[RandomEffectDataset]].
   *
   * @param dataset The updated [[RandomEffectDataset]]
   * @return A new coordinate with the updated [[RandomEffectDataset]]
   */
  override protected[algorithm] def updateCoordinateWithDataset(
      dataset: RandomEffectDataset): RandomEffectCoordinate[Objective] =
    new RandomEffectCoordinate(dataset, optimizationProblem)


  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(): (DatumScoringModel, Option[OptimizationTracker]) =
    (RandomEffectCoordinate.trainModel(dataset, optimizationProblem), None)

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(
      model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker]) =
    throw new IllegalArgumentException("Cannot train with initial model yet")

  /**
   * Compute scores for the coordinate data using a given model.
   *
   * @param model The input model
   * @return The dataset scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = model match {

    case randomEffectModel: RandomEffectModel =>
      RandomEffectCoordinate.score(dataset, randomEffectModel)

    case _ =>
      throw new UnsupportedOperationException(
        s"Scoring with model of type ${model.getClass} in ${this.getClass} is not supported")
  }

  //
  // RDDLike Functions
  //

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = optimizationProblem.sparkContext

  /**
   * Assign a given name to the [[optimizationProblem]] [[RDD]].
   *
   * @param name The parent name for all [[RDD]] objects in this class
   * @return This object with the name of the [[optimizationProblem]] [[RDD]] assigned
   */
  override def setName(name: String): RandomEffectCoordinate[Objective] = {

    optimizationProblem.setName(name)

    this
  }

  /**
   * Set the persistence storage level of the [[optimizationProblem]] [[RDD]].
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of the [[optimizationProblem]] [[RDD]] set
   */
  override def persistRDD(storageLevel: StorageLevel): RandomEffectCoordinate[Objective] = {

    optimizationProblem.persistRDD(storageLevel)

    this
  }

  /**
   * Mark the [[optimizationProblem]] [[RDD]] as unused, and asynchronously remove all blocks for it from memory and
   * disk.
   *
   * @return This object with the [[optimizationProblem]] [[RDD]] unpersisted
   */
  override def unpersistRDD(): RandomEffectCoordinate[Objective] = {

    optimizationProblem.unpersistRDD()

    this
  }

  /**
   * Materialize the [[optimizationProblem]] [[RDD]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be
   * evaluated).
   *
   * @return This object with the [[optimizationProblem]] [[RDD]] materialized
   */
  override def materialize(): RandomEffectCoordinate[Objective] = {

    optimizationProblem.materialize()

    this
  }
}

object RandomEffectCoordinate {

  /**
   * Train a new [[RandomEffectModel]] (i.e. run model optimization for each entity).
   *
   * @param randomEffectDataset The training dataset
   * @param randomEffectOptimizationProblem The per-entity optimization problems
   * @return A (new [[RandomEffectModel]], optional optimization stats) tuple
   */
  protected[algorithm] def trainModel(
      randomEffectDataset: RandomEffectDataset,
      randomEffectOptimizationProblem: RandomEffectOptimizationProblem): RandomEffectModel = {

    // All 3 RDDs involved in these joins use the same partitioner
    val dataAndOptimizationProblems = randomEffectDataset
      .activeData
      .join(randomEffectOptimizationProblem.optimizationProblems)

    // Left join the models to data and optimization problems for cases where we have a prior model but no new data
    val newModels = dataAndOptimizationProblems
      .mapValues { case (dataset, optimizationProblem) =>
        optimizationProblem.run(dataset)
      }

    new RandomEffectModel(
      newModels,
      randomEffectDataset.randomEffectType,
      randomEffectDataset.featureShardId)
  }

  /**
   * Score a [[RandomEffectDataset]] using a given [[RandomEffectModel]].
   *
   * For information about the differences between active and passive data, see the [[RandomEffectDataset]]
   * documentation.
   *
   * @note The score is the raw dot product of the model coefficients and the feature values - it does not go through a
   *       non-linear link function.
   * @param randomEffectDataset The [[RandomEffectDataset]] to score
   * @param randomEffectModel The [[RandomEffectModel]] with which to score
   * @return The computed scores
   */
  protected[algorithm] def score(
      randomEffectDataset: RandomEffectDataset,
      randomEffectModel: RandomEffectModel): CoordinateDataScores = {

    // Active data and models use the same partitioner, but scores need to use GameDatum partitioner
    val activeScores = randomEffectDataset
      .activeData
      .join(randomEffectModel.modelsRDD)
      .flatMap { case (_, (dataset, model)) =>
        val xgbDataArray = dataset.map(_._2)
        val matrix = new DMatrix(xgbDataArray.iterator)
        val rawScores = model.predict(matrix)

        dataset
          .zip(rawScores)
          .map { case ((uid, _), scoreArray) =>
            (uid, scoreArray(0).toDouble)
          }
      }
      .partitionBy(randomEffectDataset.uniqueIdPartitioner)
      .setName("Active scores")
      .persist(StorageLevel.DISK_ONLY)

    new CoordinateDataScores(activeScores)
  }
}
