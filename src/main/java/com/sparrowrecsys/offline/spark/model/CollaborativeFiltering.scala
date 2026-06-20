package com.sparrowrecsys.offline.spark.model

import java.io.{BufferedWriter, File, FileWriter}
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object CollaborativeFiltering {

  // 这个可以不用加，主要是加上那个log4j的配置文件就行了
  //  Logger.getRootLogger.setLevel(Level.WARN)

  // 控制是否执行耗时的推荐结果生成操作
  // 设为false可以只保存embedding，跳过recommendForAllUsers等操作
  val GENERATE_RECOMMENDATIONS = false

  // 控制是否执行交叉验证调参
  // 设为false可以跳过耗时的交叉验证，只进行一次模型训练
  val ENABLE_CROSS_VALIDATION = false

  // 注意：这个要运行很久很久
  def main(args: Array[String]): Unit = {
    val totalStart = System.currentTimeMillis()
    // 为了去掉讨厌的报错
    sys.props("hadoop.home.dir") = "D:\\dev_software\\hadoop"

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("collaborativeFiltering")
      .set("spark.submit.deployMode", "client")

    var t0 = System.currentTimeMillis()
    val spark = SparkSession.builder.config(conf).getOrCreate()
    println(s"[计时] SparkSession创建: ${System.currentTimeMillis() - t0}ms")

    import spark.implicits._
    t0 = System.currentTimeMillis()
    val ratingResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    // 定义类型转换UDF：将字符串转为Int和Double，因为CSV读入后默认全是String类型
    val toInt = udf[Int, String]( _.toInt)
    val toFloat = udf[Double, String]( _.toFloat)
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingResourcesPath.getPath)
      .withColumn("userIdInt", toInt(col("userId")))
      .withColumn("movieIdInt", toInt(col("movieId")))
      .withColumn("ratingFloat", toFloat(col("rating")))

    val Array(training, test) = ratingSamples.randomSplit(Array(0.8, 0.2), seed = 42)
    println(s"[计时] CSV读取+类型转换+数据集划分: ${System.currentTimeMillis() - t0}ms")

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

    t0 = System.currentTimeMillis()
    val model = als.fit(training)
    println(s"[计时] ALS模型训练(fit): ${System.currentTimeMillis() - t0}ms")

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    // ==================== 模型评估 ====================
    // ALS 模型通过学习得到每个用户和每部电影的隐因子向量。但测试集中可能出现训练集中从未出现过的用户或电影，
    // 模型没有学过它们的隐因子，自然无法预测评分，结果就是 NaN（Not a Number，非数字）。
    model.setColdStartStrategy("drop")
    // 用训练好的模型对测试集进行预测
    // 注意：transform() 是惰性的，此时不会真正计算，只是构建了计算图
    val predictions = model.transform(test)

    // 查看模型学到的物品隐因子和用户隐因子（各展示前10条）
    // 这两步很快，因为 itemFactors/userFactors 是训练结束时就已经算好的
    t0 = System.currentTimeMillis()
    model.itemFactors.show(10, truncate = false)
    model.userFactors.show(10, truncate = false)
    println(s"[计时] 展示item/userFactors: ${System.currentTimeMillis() - t0}ms")

    // 使用RMSE（均方根误差）评估模型预测精度
    // RMSE越小，说明预测评分与真实评分的偏差越小
    // ⚠️ evaluate() 是一个 action 操作，会触发 predictions 的惰性求值
    // 这里需要为测试集的每条样本做 user_factor × item_factor 矩阵乘法，是整个流程中最耗时的一步
    println("开始计算预测值（这一步较耗时，因为要为测试集所有样本做矩阵乘法）...")
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("ratingFloat")
      .setPredictionCol("prediction")
    t0 = System.currentTimeMillis()
    val rmse = evaluator.evaluate(predictions)
    println(s"[计时] 评估器evaluate(predictions): ${System.currentTimeMillis() - t0}ms")
    println(s"RMSE计算完成: $rmse")
    println(s"Root-mean-square error = $rmse")

    // ==================== 保存用户和物品隐向量到CSV文件 ====================
    // 这些隐向量可以用于线上服务，格式与SparrowRecSys的embedding加载格式兼容
    // 通过classpath资源定位输出目录，避免依赖运行时工作目录的硬编码相对/绝对路径
    // （写法参考 Embedding.trainItem2vec 中的保存方式）
    val outputFolderPath = this.getClass.getResource("/webroot/sampledata/").getPath

    // 保存物品隐向量（电影embedding）
    // 格式：id:emb1 emb2 ... embN（与 Embedding 写入文件的格式保持一致）
    t0 = System.currentTimeMillis()
    val itemEmbFile = new File(outputFolderPath + "alsItemEmbeddings.csv")
    var itemBw: BufferedWriter = null
    try {
      itemBw = new BufferedWriter(new FileWriter(itemEmbFile))
      val itemFactors = model.itemFactors.collect()
      for (row <- itemFactors) {
        val id = row.getAs[Int]("id")
        val features = row.getAs[Seq[Float]]("features")
        itemBw.write(id + ":" + features.mkString(" ") + "\n")
      }
    } finally {
      if (itemBw != null) itemBw.close()
    }

    // 保存用户隐向量（用户embedding）
    val userEmbFile = new File(outputFolderPath + "alsUserEmbeddings.csv")
    var userBw: BufferedWriter = null
    try {
      userBw = new BufferedWriter(new FileWriter(userEmbFile))
      val userFactors = model.userFactors.collect()
      for (row <- userFactors) {
        val id = row.getAs[Int]("id")
        val features = row.getAs[Seq[Float]]("features")
        userBw.write(id + ":" + features.mkString(" ") + "\n")
      }
    } finally {
      if (userBw != null) userBw.close()
    }
    println(s"[计时] 保存item+user隐向量到文件: ${System.currentTimeMillis() - t0}ms")

    println(s"物品隐向量已保存到: ${itemEmbFile.getPath}")
    println(s"用户隐向量已保存到: ${userEmbFile.getPath}")

    // ==================== 生成推荐结果 ====================
    // 根据配置决定是否执行耗时的推荐结果生成操作
    if (GENERATE_RECOMMENDATIONS) {
      println("开始生成推荐结果（这可能需要较长时间）...")
      t0 = System.currentTimeMillis()
      
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
      println(s"[计时] 全量推荐(recommendForAllUsers/AllItems + subset): ${System.currentTimeMillis() - t0}ms")
    } else {
      println("跳过推荐结果生成（GENERATE_RECOMMENDATIONS = false）")
    }

    // ==================== 针对已知id的快速推荐 ====================
    // recommendForAllUsers/AllItems 要为全量用户/物品打分，非常耗时；
    // 而 ratingSamples.select(...).distinct().limit(3) 又会因 distinct 触发 shuffle。
    // 既然我们已经知道数据集中存在的几个用户id和电影id，直接构造一个小DataFrame，
    // 用 recommendForUserSubset / recommendForItemSubset 推荐即可，避免全量计算和shuffle，速度快得多。
    // 注意：构造的DataFrame列名必须与 ALS 设置的 userCol / itemCol 一致（userIdInt / movieIdInt）。

    // 为指定的几个已知用户生成Top-10电影推荐
    t0 = System.currentTimeMillis()
    val knownUsers = Seq(10, 20, 30).toDF(als.getUserCol)
    val knownUserRecs = model.recommendForUserSubset(knownUsers, 10)
    println(s"[计时] recommendForUserSubset(3个用户): ${System.currentTimeMillis() - t0}ms")
    println("指定用户的Top-10电影推荐：")
    knownUserRecs.show(truncate = false)

    // 为指定的几部已知电影生成Top-10用户推荐
    t0 = System.currentTimeMillis()
    val knownMovies = Seq(1, 2, 3).toDF(als.getItemCol)
    val knownMovieRecs = model.recommendForItemSubset(knownMovies, 10)
    println(s"[计时] recommendForItemSubset(3部电影): ${System.currentTimeMillis() - t0}ms")
    println("指定电影的Top-10用户推荐：")
    knownMovieRecs.show(truncate = false)

    // ==================== 交叉验证调参 ====================
    if (ENABLE_CROSS_VALIDATION) {
      println("开始交叉验证调参（这可能需要较长时间）...")
      t0 = System.currentTimeMillis()
      
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
      // 注意：交叉验证应该在训练集上进行，不能用测试集（否则是数据泄漏）
      val cvModel = cv.fit(training)
      println(s"[计时] 10折交叉验证fit: ${System.currentTimeMillis() - t0}ms")
      // 获取每组参数对应的平均评估指标
      val avgMetrics = cvModel.avgMetrics

      // 打印每组参数及其对应的平均RMSE，选择最优参数组合
      paramGrid.zip(avgMetrics).foreach { case (params, metric) =>
        println(s"参数: $params -> 平均RMSE: $metric")
      }

      // 找到最优参数
      val bestMetric = avgMetrics.min
      println(s"最优平均RMSE: $bestMetric")
    } else {
      println("跳过交叉验证调参（ENABLE_CROSS_VALIDATION = false）")
      println(s"当前模型RMSE: $rmse")
    }

    println(s"[计时] 总耗时: ${System.currentTimeMillis() - totalStart}ms")
    spark.stop()
  }
}