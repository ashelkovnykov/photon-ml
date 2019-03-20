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
package com.linkedin.photon.ml.data

import scala.util.hashing.byteswap64

import ml.dmlc.xgboost4j.{LabeledPoint => XGBPoint}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkContext}

import com.linkedin.photon.ml.Types.{FeatureShardId, REId, REType, UniqueSampleId}
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}

/**
 * Dataset implementation for random effect data.
 *
 * All of the training data for a single random effect must fit into on Spark partition. The size limit of a single
 * Spark partition is 2 GB. If the size of (samples * features) exceeds the maximum size of a single Spark partition,
 * the data is split into two sections: active and passive data.
 *
 *   activeData + passiveData = full data set
 *
 * Active data is used for both training and scoring (to determine residuals for partial score). Passive data is used
 * only for scoring. In the vast majority of cases, all data is active data.
 *
 * @param activeData Per-entity datasets used to train per-entity models and to compute residuals
 * @param passiveData Per-entity datasets used only to compute residuals
 * @param activeUniqueIdToRandomEffectIds Map of unique sample id to random effect id for active data samples
 * @param randomEffectType The random effect type (e.g. "memberId")
 * @param featureShardId The ID of the data feature shard used by this dataset
 */
protected[ml] class RandomEffectDataset(
    val activeData: RDD[(REId, Array[(UniqueSampleId, XGBPoint)])],
    val passiveData: RDD[(UniqueSampleId, (REId, XGBPoint))],
    val activeUniqueIdToRandomEffectIds: RDD[(UniqueSampleId, REId)],
    val randomEffectType: REType,
    val featureShardId: FeatureShardId)
  extends Dataset[RandomEffectDataset]
    with BroadcastLike
    with RDDLike {

  lazy val passiveDataREIds: Broadcast[Set[REId]] = SparkSession
    .builder()
    .getOrCreate()
    .sparkContext
    .broadcast(passiveData.map(_._2._1).distinct().collect().toSet)
  val randomEffectIdPartitioner: Partitioner = activeData.partitioner.get
  val uniqueIdPartitioner: Partitioner = passiveData.partitioner.get

  //
  // RandomEffectDataset functions
  //

  /**
   * Update the dataset.
   *
   * @param updatedActiveData Updated active data
   * @param updatedPassiveData Updated passive data
   * @return A new updated dataset
   */
  def update(
    updatedActiveData: RDD[(REId, Array[(UniqueSampleId, XGBPoint)])],
    updatedPassiveData: RDD[(UniqueSampleId, (REId, XGBPoint))]): RandomEffectDataset =

    new RandomEffectDataset(
      updatedActiveData,
      updatedPassiveData,
      activeUniqueIdToRandomEffectIds,
      randomEffectType,
      featureShardId)

  //
  // Dataset functions
  //

  /**
   * Add residual scores to the data offsets.
   *
   * @param scores The residual scores
   * @return The dataset with updated offsets
   */
  override def addScoresToOffsets(scores: CoordinateDataScores): RandomEffectDataset = {

    // Add scores to active data offsets
    val scoresGroupedByRandomEffectId = scores
      .scoresRdd
      .join(activeUniqueIdToRandomEffectIds, uniqueIdPartitioner)
      .map { case (uniqueId, (score, reId)) => (reId, (uniqueId, score)) }
      .groupByKey(randomEffectIdPartitioner)
      .mapValues(_.toArray.sortBy(_._1))

    val updatedActiveData = activeData
      .join(scoresGroupedByRandomEffectId)
      .mapValues { case (localData, localScores) =>
        localData
          .zip(localScores)
          .map { case ((dataId, xgbPoint), (residualScoreId, residualScore)) =>

            require(residualScoreId == dataId, s"residual score Id ($residualScoreId) and data Id ($dataId) don't match!")

            val offset = xgbPoint.baseMargin

            (dataId, xgbPoint.copy(baseMargin = offset + residualScore.toFloat))
          }
      }

    update(updatedActiveData, passiveData)
  }

  //
  // BroadcastLike Functions
  //

  /**
   * Asynchronously delete cached copies of [[passiveDataREIds]] on all executors.
   *
   * @return This [[RandomEffectDataset]] with [[passiveDataREIds]] unpersisted
   */
  override protected[ml] def unpersistBroadcast(): RandomEffectDataset = {

    passiveDataREIds.unpersist()

    this
  }

  //
  // RDDLike Functions
  //

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = activeData.sparkContext

  /**
   * Assign a given name to [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the names [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]]
   *         assigned
   */
  override def setName(name: String): RandomEffectDataset = {

    activeData.setName(s"$name - Active Data")
    passiveData.setName(s"$name - Passive Data")
    activeUniqueIdToRandomEffectIds.setName(s"$name - UID to REID")

    this
  }

  /**
   * Set the storage level of [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]], and persist
   * their values across the cluster the first time they are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[activeData]], [[activeUniqueIdToRandomEffectIds]], and
   *         [[passiveData]] set
   */
  override def persistRDD(storageLevel: StorageLevel): RandomEffectDataset = {

    if (!activeData.getStorageLevel.isValid) activeData.persist(storageLevel)
    if (!passiveData.getStorageLevel.isValid) passiveData.persist(storageLevel)
    if (!activeUniqueIdToRandomEffectIds.getStorageLevel.isValid) activeUniqueIdToRandomEffectIds.persist(storageLevel)

    this
  }

  /**
   * Mark [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]] as non-persistent, and remove all
   * blocks for them from memory and disk.
   *
   * @return This object with [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]] marked
   *         non-persistent
   */
  override def unpersistRDD(): RandomEffectDataset = {

    if (activeData.getStorageLevel.isValid) activeData.unpersist()
    if (passiveData.getStorageLevel.isValid) passiveData.unpersist()
    if (activeUniqueIdToRandomEffectIds.getStorageLevel.isValid) activeUniqueIdToRandomEffectIds.unpersist()

    this
  }

  /**
   * Materialize [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]] (Spark [[RDD]]s are lazy
   * evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]] materialized
   */
  override def materialize(): RandomEffectDataset = {

    activeData.count()
    passiveData.count()
    activeUniqueIdToRandomEffectIds.count()

    this
  }

  //
  // Summarizable Functions
  //

  /**
   * Build a human-readable summary for [[RandomEffectDataset]].
   *
   * @return A summary of the object in string representation
   */
  override def toSummaryString: String = {
    ""
  }
}

object RandomEffectDataset {

  /**
   * Build a new [[RandomEffectDataset]] from the raw data using the given configuration.
   *
   * @param gameDataset The [[RDD]] of [[GameDatum]] used to generate the random effect dataset
   * @param randomEffectDataConfiguration The data configuration for the random effect dataset
   * @param randomEffectPartitioner The per-entity partitioner used to split the grouped active data
   * @param existingModelKeysRddOpt Optional set of entities that have existing models
   * @return A new [[RandomEffectDataset]]
   */
  def apply(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: RandomEffectDatasetPartitioner,
      existingModelKeysRddOpt: Option[RDD[REId]],
      storageLevel: StorageLevel): RandomEffectDataset = {

    val uniqueIdPartitioner = gameDataset.partitioner.get

    //
    // Generate RDDs
    //

    val activeData = generateActiveData(
      gameDataset,
      randomEffectDataConfiguration,
      randomEffectPartitioner,
      existingModelKeysRddOpt)
    activeData.persist(storageLevel).count

    val uniqueIdToRandomEffectIds = generateIdMap(activeData, uniqueIdPartitioner)
    uniqueIdToRandomEffectIds.persist(storageLevel).count

    val passiveData = generatePassiveData(gameDataset)
    passiveData.persist(storageLevel).count

    //
    // Return new dataset
    //

    new RandomEffectDataset(
      activeData,
      passiveData,
      uniqueIdToRandomEffectIds,
      randomEffectDataConfiguration.randomEffectType,
      randomEffectDataConfiguration.featureShardId)
  }

  /**
   * Generate active data.
   *
   * @param gameDataset The input dataset
   * @param randomEffectDataConfiguration The random effect data configuration
   * @param randomEffectPartitioner A random effect partitioner
   * @param existingModelKeysRddOpt An optional set of entities which have existing models
   * @return The active dataset
   */
  protected[data] def generateActiveData(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner,
      existingModelKeysRddOpt: Option[RDD[REId]]): RDD[(REId, Array[(UniqueSampleId, XGBPoint)])] = {

    val randomEffectType = randomEffectDataConfiguration.randomEffectType
    val featureShardId = randomEffectDataConfiguration.featureShardId

    val keyedRandomEffectDataset = gameDataset.map { case (uniqueId, gameData) =>
      val randomEffectId = gameData.idTagToValueMap(randomEffectType)
      val labeledPoint = gameData.generateXGBoostLabeledPoint(featureShardId)

      (randomEffectId, (uniqueId, labeledPoint))
    }

    val groupedRandomEffectDataset = randomEffectDataConfiguration
      .numActiveDataPointsUpperBound
      .map { activeDataUpperBound =>
        groupDataByKeyAndSample(
          keyedRandomEffectDataset,
          randomEffectPartitioner,
          activeDataUpperBound,
          randomEffectType)
      }
      .getOrElse(keyedRandomEffectDataset.groupByKey(randomEffectPartitioner))

    randomEffectDataConfiguration
      .numActiveDataPointsLowerBound
      .map { activeDataLowerBound =>
        existingModelKeysRddOpt match {
          case Some(existingModelKeysRdd) =>
            groupedRandomEffectDataset
              .zipPartitions(existingModelKeysRdd, preservesPartitioning = true) { (dataIt, existingKeysIt) =>

                val lookupTable = existingKeysIt.toSet

                dataIt.filter { case (key, data) =>
                  (data.size >= activeDataLowerBound) || !lookupTable.contains(key)
                }
              }

          case None =>
            groupedRandomEffectDataset.filter { case (_, data) =>
              data.size >= activeDataLowerBound
            }
        }
      }
      .getOrElse(groupedRandomEffectDataset)
      .mapValues(data => data.toArray.sortBy(_._1))
  }

  /**
   * Generate a dataset grouped by random effect ID and limited to a maximum number of samples selected via reservoir
   * sampling.
   *
   * The 'Min Heap' reservoir sampling algorithm is used for two reasons:
   * 1. The exact sampling must be reproducible so that [[RDD]] partitions can be recovered
   * 2. The linear algorithm is non-trivial to combine in a distributed manner
   *
   * @param rawKeyedDataset The raw dataset, with samples keyed by random effect ID
   * @param partitioner The partitioner
   * @param sampleCap The sample cap
   * @param randomEffectType The type of random effect
   * @return An [[RDD]] of data grouped by individual ID
   */
  private def groupDataByKeyAndSample(
      rawKeyedDataset: RDD[(REId, (UniqueSampleId, XGBPoint))],
      partitioner: Partitioner,
      sampleCap: Int,
      randomEffectType: REType): RDD[(REId, Iterable[(UniqueSampleId, XGBPoint)])] = {

    // Helper class for defining a constant ordering between data samples (necessary for RDD re-computation)
    case class ComparableLabeledPointWithId(comparableKey: Int, uniqueId: UniqueSampleId, labeledPoint: XGBPoint)
      extends Comparable[ComparableLabeledPointWithId] {

      override def compareTo(comparableLabeledPointWithId: ComparableLabeledPointWithId): Int = {
        if (comparableKey - comparableLabeledPointWithId.comparableKey > 0) {
          1
        } else {
          -1
        }
      }
    }

    val createCombiner =
      (comparableLabeledPointWithId: ComparableLabeledPointWithId) => {
        new MinHeapWithFixedCapacity[ComparableLabeledPointWithId](sampleCap) += comparableLabeledPointWithId
      }

    val mergeValue = (
        minHeapWithFixedCapacity: MinHeapWithFixedCapacity[ComparableLabeledPointWithId],
        comparableLabeledPointWithId: ComparableLabeledPointWithId) => {
      minHeapWithFixedCapacity += comparableLabeledPointWithId
    }

    val mergeCombiners = (
        minHeapWithFixedCapacity1: MinHeapWithFixedCapacity[ComparableLabeledPointWithId],
        minHeapWithFixedCapacity2: MinHeapWithFixedCapacity[ComparableLabeledPointWithId]) => {
      minHeapWithFixedCapacity1 ++= minHeapWithFixedCapacity2
    }

    // The reservoir sampling algorithm is fault tolerant, assuming that the uniqueId for a sample is recovered after
    // node failure. We attempt to maximize the likelihood of successful recovery through RDD replication, however there
    // is a non-zero possibility of massive failure. If this becomes an issue, we may need to resort to check-pointing
    // the raw data RDD after uniqueId assignment.
    rawKeyedDataset
      .mapValues { case (uniqueId, labeledPoint) =>
        val comparableKey = (byteswap64(randomEffectType.hashCode) ^ byteswap64(uniqueId)).hashCode()
        ComparableLabeledPointWithId(comparableKey, uniqueId, labeledPoint)
      }
      .combineByKey[MinHeapWithFixedCapacity[ComparableLabeledPointWithId]](
        createCombiner,
        mergeValue,
        mergeCombiners,
        partitioner)
      .mapValues { minHeapWithFixedCapacity =>
        val count = minHeapWithFixedCapacity.getCount
        val data = minHeapWithFixedCapacity.getData
        val weightMultiplierOpt = if (count > sampleCap) Some(1D * count / sampleCap) else None

        data.map { case ComparableLabeledPointWithId(_, uniqueId, xgbPoint) =>
          val w = xgbPoint.weight
          (uniqueId, xgbPoint.copy(weight = weightMultiplierOpt.map(_.toFloat * w).getOrElse(w)))
        }
      }
  }

  /**
   * Generate a map of unique sample id to random effect id for active data samples.
   *
   * @param activeData The active dataset
   * @param partitioner The [[Partitioner]] to use for the [[RDD]] of unique sample ID to random effect ID
   * @return A map of unique sample id to random effect id for active data samples
   */
  protected[data] def generateIdMap(
      activeData: RDD[(REId, Array[(UniqueSampleId, XGBPoint)])],
      partitioner: Partitioner): RDD[(UniqueSampleId, REId)] =
    activeData
      .flatMap { case (individualId, data) =>
        data.map(abc => (abc._1, individualId))
      }
      .partitionBy(partitioner)

  /**
   * Generate passive dataset.
   *
   * @param gameDataset The raw input dataset
   * @return The passive dataset
   */
  protected[data] def generatePassiveData(gameDataset: RDD[(UniqueSampleId, GameDatum)]): RDD[(UniqueSampleId, (REId, XGBPoint))] =
    gameDataset.sparkContext.emptyRDD[(UniqueSampleId, (REId, XGBPoint))]
}
