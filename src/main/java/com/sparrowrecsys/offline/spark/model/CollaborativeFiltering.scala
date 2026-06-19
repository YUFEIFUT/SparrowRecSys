package com.sparrowrecsys.offline.spark.model

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object CollaborativeFiltering {

  def main(args: Array[String]): Unit = {
    // 为了去掉讨厌的报错
    sys.props("hadoop.home.dir") = "D:\\dev_software\\hadoop"

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("collaborativeFiltering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val ratingResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    // 定义类型转换UDF：将字符串转为Int和Double，因为CSV读入后默认全是String类型
    val toInt = udf[Int, String]( _.toInt)
    val toFloat = udf[Double, String]( _.toFloat)
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingResourcesPath.getPath)
      .withColumn("userIdInt", toInt(col("userId")))
      .withColumn("movieIdInt", toInt(col("movieId")))
      .withColumn("ratingFloat", toFloat(col("rating")))

    val Array(training, test) = ratingSamples.randomSplit(Array(0.8, 0.2))

    // ==================== 构建ALS协同过滤模型 ====================
    // ALS（交替最小二乘法）：通过矩阵分解将用户-物品评分矩阵分解为用户隐因子和物品隐因子
    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      // 最大迭代次数
      .setMaxIter(5)
      // 正则化参数，防止过拟合
      .setRegParam(0.01)
      .setUserCol("userIdInt")
      .setItemCol("movieIdInt")
      .setRatingCol("ratingFloat")

    val model = als.fit(training)

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    // ==================== 模型评估 ====================
    // ALS 模型通过学习得到每个用户和每部电影的隐因子向量。但测试集中可能出现训练集中从未出现过的用户或电影，
    // 模型没有学过它们的隐因子，自然无法预测评分，结果就是 NaN（Not a Number，非数字）。
    model.setColdStartStrategy("drop")
    // 用训练好的模型对测试集进行预测
    val predictions = model.transform(test)

    // 查看模型学到的物品隐因子和用户隐因子（各展示前10条）
    model.itemFactors.show(10, truncate = false)
    model.userFactors.show(10, truncate = false)

    // 使用RMSE（均方根误差）评估模型预测精度
    // RMSE越小，说明预测评分与真实评分的偏差越小
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("ratingFloat")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    // ==================== 生成推荐结果 ====================
    // 为每个用户生成Top-10电影推荐
    // Generate top 10 movie recommendations for each user
    val userRecs = model.recommendForAllUsers(10)
    // 为每部电影推荐Top-10用户（即哪些用户最可能喜欢该电影）
    // Generate top 10 user recommendations for each movie
    val movieRecs = model.recommendForAllItems(10)

    // Generate top 10 movie recommendations for a specified set of users
    // 为指定的3个用户子集生成Top-10推荐（适用于线上单用户/少量用户的推荐场景）
    val users = ratingSamples.select(als.getUserCol).distinct().limit(3)
    val userSubsetRecs = model.recommendForUserSubset(users, 10)

    // 为指定的3部电影子集生成Top-10用户推荐
    // Generate top 10 user recommendations for a specified set of movies
    val movies = ratingSamples.select(als.getItemCol).distinct().limit(3)
    val movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    // $example off$
    userRecs.show(false)
    movieRecs.show(false)
    userSubsetRecs.show(false)
    movieSubSetRecs.show(false)

    // ==================== 交叉验证调参 ====================
    // 构建参数网格，这里仅搜索regParam=0.01这一个值（实际场景可添加多个候选值进行网格搜索）
    val paramGrid = new ParamGridBuilder()
      .addGrid(als.regParam, Array(0.01))
      .build()

    // 使用10折交叉验证评估模型的泛化能力
    // 原理：将数据分为10份，轮流用其中9份训练、1份验证，最终取10次评估指标的平均值
    // 交叉验证比单次train/test split更可靠，能有效避免因数据划分偶然性导致的评估偏差
    val cv = new CrossValidator()
      // 待评估的模型
      .setEstimator(als)
      // 评估器（RMSE）
      .setEvaluator(evaluator)
      // 参数网格
      .setEstimatorParamMaps(paramGrid)
      // 折数，实际生产环境建议至少3折
      .setNumFolds(10)  // Use 3+ in practice
    val cvModel = cv.fit(test)
    // 获取每组参数对应的平均评估指标
    val avgMetrics = cvModel.avgMetrics

    // 打印每组参数及其对应的平均RMSE，选择最优参数组合
    paramGrid.zip(avgMetrics).foreach { case (params, metric) =>
      println(s"参数: $params -> 平均RMSE: $metric")
    }

    // 找到最优参数
    val bestMetric = avgMetrics.min
    println(s"最优平均RMSE: $bestMetric")

    spark.stop()
  }
}