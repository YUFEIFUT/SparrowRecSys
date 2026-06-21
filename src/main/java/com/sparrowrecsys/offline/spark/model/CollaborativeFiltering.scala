package com.sparrowrecsys.offline.spark.model

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
 * ALS 协同过滤——训练侧。
 *
 * 职责（只做"训练相关"的事，保持单一职责）：
 *   读数据 → 转换数据 → 训练模型 → 评估模型 → 保存模型到 modeldata/alsModel
 *   外加一个"为指定已知用户快速生成Top-10"的小演示。
 *
 * 模型保存后，下游的隐向量导出 / 每用户Top-N推荐导出，交给 [[AlsModelExporter]]：
 * 它直接加载这里保存的模型，避免每生成一种产物都得重新训练一遍。
 */
object CollaborativeFiltering {

  // 模型保存目录名（保存在 classpath 的 /webroot/modeldata/ 下，AlsModelExporter 从同一位置加载）
  val MODEL_DIR_NAME = "alsModel"

  // 控制是否执行交叉验证调参（设为false可跳过耗时的交叉验证，只做一次训练）
  val ENABLE_CROSS_VALIDATION = false

  def main(args: Array[String]): Unit = {
    // 为了去掉讨厌的报错
    sys.props("hadoop.home.dir") = "D:\\dev_software\\hadoop"

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("collaborativeFiltering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    import spark.implicits._

    // ==================== 读数据 + 转换数据 ====================
    val ratingResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    // 类型转换UDF：CSV读入后默认全是String，需转成Int/Double
    val toInt = udf[Int, String](_.toInt)
    val toFloat = udf[Double, String](_.toFloat)
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingResourcesPath.getPath)
      .withColumn("userIdInt", toInt(col("userId")))
      .withColumn("movieIdInt", toInt(col("movieId")))
      .withColumn("ratingFloat", toFloat(col("rating")))

    val Array(training, test) = ratingSamples.randomSplit(Array(0.8, 0.2), seed = 42)

    // ==================== 训练模型 ====================
    // ALS（交替最小二乘法）：把"用户-物品评分矩阵"分解为用户隐因子和物品隐因子
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userIdInt")
      .setItemCol("movieIdInt")
      .setRatingCol("ratingFloat")

    var t0 = System.currentTimeMillis()
    val model = als.fit(training)
    println(s"[计时] ALS模型训练(fit): ${System.currentTimeMillis() - t0}ms")

    // ==================== 评估模型 ====================
    // coldStartStrategy=drop：丢弃测试集中训练集没出现过的user/item，避免预测出 NaN
    model.setColdStartStrategy("drop")
    val predictions = model.transform(test)

    // 看一眼模型学到的隐因子（各前10条）
    model.itemFactors.show(10, truncate = false)
    model.userFactors.show(10, truncate = false)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("ratingFloat")
      .setPredictionCol("prediction")
    t0 = System.currentTimeMillis()
    val rmse = evaluator.evaluate(predictions)
    println(s"[计时] 评估器evaluate(predictions): ${System.currentTimeMillis() - t0}ms")
    println(s"Root-mean-square error = $rmse")

    // ==================== 保存模型 ====================
    // ALSModel.save 输出的是一个"自包含目录"（内部含 metadata/ 与 data/ 的parquet），不是零散文件。
    // 保存到 /webroot/modeldata/alsModel；overwrite() 允许重复运行覆盖旧模型。
    val modelPath = this.getClass.getResource("/webroot/modeldata/").getPath + MODEL_DIR_NAME
    t0 = System.currentTimeMillis()
    model.write.overwrite().save(modelPath)
    println(s"[计时] 保存模型: ${System.currentTimeMillis() - t0}ms")
    println(s"模型已保存到目录: $modelPath")

    // ==================== 为指定的几个已知用户快速生成Top-10电影推荐（演示） ====================
    // 直接构造已知用户id的小DataFrame，用 recommendForUserSubset 推荐，
    // 避免 recommendForAllUsers 的全量计算、也避免 distinct().limit() 的 shuffle。
    // 注意：构造的DataFrame列名必须与 ALS 的 userCol 一致（userIdInt）。
    t0 = System.currentTimeMillis()
    val knownUsers = Seq(10, 20, 30).toDF(als.getUserCol)
    val knownUserRecs = model.recommendForUserSubset(knownUsers, 10)
    println(s"[计时] recommendForUserSubset(3个用户): ${System.currentTimeMillis() - t0}ms")
    println("指定用户的Top-10电影推荐：")
    knownUserRecs.show(truncate = false)

    // ==================== 可选：交叉验证调参 ====================
    if (ENABLE_CROSS_VALIDATION) {
      println("开始交叉验证调参（这可能需要较长时间）...")
      // 仅搜索 regParam=0.01（实际场景可加多个候选值做网格搜索）
      val paramGrid = new ParamGridBuilder()
        .addGrid(als.regParam, Array(0.01))
        .build()

      // 10折交叉验证：数据分10份，轮流9份训练1份验证，取10次评估的平均，比单次划分更可靠
      val cv = new CrossValidator()
        .setEstimator(als)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(10) // Use 3+ in practice
      // 注意：交叉验证应在训练集上做，不能用测试集（否则数据泄漏）
      val cvModel = cv.fit(training)
      val avgMetrics = cvModel.avgMetrics

      paramGrid.zip(avgMetrics).foreach { case (params, metric) =>
        println(s"参数: $params -> 平均RMSE: $metric")
      }
      println(s"最优平均RMSE: ${avgMetrics.min}")
    } else {
      println("跳过交叉验证调参（ENABLE_CROSS_VALIDATION = false）")
      println(s"当前模型RMSE: $rmse")
    }

    spark.stop()
  }
}
