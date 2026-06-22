package com.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions.{format_number, _}
import org.apache.spark.sql.types.{DecimalType, FloatType, IntegerType, LongType}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import redis.clients.jedis.Jedis
import redis.clients.jedis.params.SetParams

import scala.collection.immutable.ListMap
import scala.collection.{JavaConversions, mutable}

object FeatureEngForRecModel {

  val NUMBER_PRECISION = 2
  val redisEndpoint = "localhost"
  val redisPort = 6379

  def addSampleLabel(ratingSamples:DataFrame): DataFrame ={
    ratingSamples.show(10, truncate = false)
    ratingSamples.printSchema()
    val sampleCount = ratingSamples.count()
    // 对每种rating，看看这个rating的评分次数占总的比例
    // 实际数据集里 rating 的取值可能是 0.5、1.0、1.5 ... 5.0 这些，分组后就能看到每档评分有多少人，大致是正态分布还是偏高偏低。
    // 这一步只是观察数据分布，和后面真正生成 label 的逻辑无关。
    ratingSamples.groupBy(col("rating")).count().orderBy(col("rating"))
      .withColumn("percentage", col("count")/sampleCount).show(100,truncate = false)

    ratingSamples.withColumn("label", when(col("rating") >= 3.5, 1).otherwise(0))
  }

  def addMovieFeatures(movieSamples:DataFrame, ratingSamples:DataFrame): DataFrame ={

    // ========================================
    // 第一步：将电影基本信息（标题、类型）关联到评分数据
    // ========================================
    // 通过 movieId 做左连接，把 movieSamples 中的电影元数据拼到每条评分记录上
    // 左连接保证评分数据不会丢（即使某部电影在 movieSamples 中缺失，评分行仍保留，对应列为 null）
    //add movie basic features
    val samplesWithMovies1 = ratingSamples.join(movieSamples, Seq("movieId"), "left")
    //add release year
    // ========================================
    // 第二步：从电影标题中提取上映年份
    // ========================================
    // 电影标题格式类似 "Toy Story (1995)"，最后6个字符是 "(1995)"
    // UDF：截取标题末尾的年份字符串，转为整数
    // 若标题为空或长度不足6，返回默认值 1990
    val extractReleaseYearUdf = udf({(title: String) => {
      if (null == title || title.trim.length < 6) {
        1990 // default value
      }
      else {
        val yearString = title.trim.substring(title.length - 5, title.length - 1)
        yearString.toInt
      }
    }})

    //add title
    // ========================================
    // 第三步：从电影标题中去掉年份部分，只保留纯片名
    // ========================================
    // UDF：截取标题最后6个字符之前的部分作为纯片名
    // 例如 "Toy Story (1995)" → "Toy Story"
    val extractTitleUdf = udf({(title: String) => {title.trim.substring(0, title.trim.length - 6).trim}})

    // 在 DataFrame 上连续操作：
    // 1) 用 extractReleaseYearUdf 新增 releaseYear 列（如 1995）
    // 2) 用 extractTitleUdf 覆写 title 列为纯片名
    // 3) 随即 drop 掉 title 列（当前模型用不上文本标题，后续可接 NLP 处理）
    // 不是同一个东西，samplesWithMovies1 完全不受影响。
    // Spark 的 DataFrame 是不可变（immutable）的，每次 .withColumn() / .drop() 都生成一个全新的 DataFrame，原始的不变。
    // 类比的话，就像 Java 的 String：DataFrame 也是同样的道理——每次转换操作都是返回新对象，不是在原对象上修改。
    val samplesWithMovies2 = samplesWithMovies1.withColumn("releaseYear", extractReleaseYearUdf(col("title")))
      // 这一步完全就是画蛇添足
      .withColumn("title", extractTitleUdf(col("title")))
      .drop("title")  //title is useless currently

    //split genres
    val samplesWithMovies3 = samplesWithMovies2.withColumn("movieGenre1",split(col("genres"),"\\|").getItem(0))
      .withColumn("movieGenre2",split(col("genres"),"\\|").getItem(1))
      .withColumn("movieGenre3",split(col("genres"),"\\|").getItem(2))

    //add rating features
    val movieRatingFeatures = samplesWithMovies3.groupBy(col("movieId"))
      .agg(
        count(lit(1)).as("movieRatingCount"),
        // 平均评分，保留指定位小数
        format_number(avg(col("rating")), NUMBER_PRECISION).as("movieAvgRating"),
        stddev(col("rating")).as("movieRatingStddev")
      )
      // null 值填充为 0
      .na.fill(0)
      // stddev 对只有1条评分的电影无法计算标准差，结果为 null
      // 必须先 na.fill(0) 将 null 转为数值 0，再 format_number 格式化
      // 顺序不能反：format_number 处理 null 会产生字符串 "null"，后续无法再填数值
      .withColumn("movieRatingStddev",format_number(col("movieRatingStddev"), NUMBER_PRECISION))


    // ========================================
    // 第六步：把电影评分统计特征关联回主数据
    // ========================================
    // 通过 movieId 左连接，把每部电影的评分统计特征拼到每条样本上
    // 这样每条样本都有了：电影基本信息 + 上映年份 + 类型 + 评分统计特征
    //join movie rating features
    val samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, Seq("movieId"), "left")
    samplesWithMovies4.printSchema()
    samplesWithMovies4.show(10, truncate = false)

    samplesWithMovies4
  }

  // ========================================
  // UDF：从用户的观影类型列表中提取偏好类型并按频次降序排列
  // ========================================
  // 输入：一个用户看过的所有电影类型的 Seq，如 ["Action|Adventure", "Action|Comedy", "Drama"]
  // 输出：去重且按出现频次降序排列的类型 Seq，如 ["Action", "Comedy", "Adventure", "Drama"]
  val extractGenres: UserDefinedFunction = udf { (genreArray: Seq[String]) => {
    // 用可变Map统计每种类型出现的次数
    val genreMap = mutable.Map[String, Int]()
    genreArray.foreach((element:String) => {
      // 每个元素可能包含多个类型，用"|"分割，如 "Action|Adventure" → ["Action","Adventure"]
      val genres = element.split("\\|")
      genres.foreach((oneGenre:String) => {
        // 将每种类型的计数+1，不存在则初始化为0再+1
        genreMap(oneGenre) = genreMap.getOrElse[Int](oneGenre, 0)  + 1
      })
    })
    // 按频次降序排序，用 ListMap 保持排序顺序
    val sortedGenres = ListMap(genreMap.toSeq.sortWith(_._2 > _._2):_*)
    // 只返回类型名的序列（丢弃计数），如 ["Action", "Comedy", "Drama"]
    sortedGenres.keys.toSeq
  }}

  // ========================================
  // 为每条评分样本添加用户维度的历史行为特征
  // ========================================
  // 核心思路：基于滑动窗口，只看该用户"当前时间戳之前"最近100条记录
  // 这样避免了数据泄露（不用未来信息预测过去）
  def addUserFeatures(ratingSamples:DataFrame): DataFrame ={
    val samplesWithUserFeatures = ratingSamples
      // --- 用户正向历史记录 ---
      // 收集该用户历史中 label=1（喜欢）的电影ID列表
      // label!=1 的记录用 null 占位，collect_list 会自动过滤 null
      // 窗口：按userId分区，按timestamp排序，取当前行之前最近100条
      // rowsBetween(-100, -1) 表示不包含当前行，只看历史
      .withColumn("userPositiveHistory", collect_list(when(col("label") === 1, col("movieId")).otherwise(lit(null)))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      // --- 反转列表，让最近喜欢的电影排在前面 ---
      .withColumn("userPositiveHistory", reverse(col("userPositiveHistory")))
      // --- 取最近喜欢的5部电影作为独立特征列 ---
      // 这些特征可以和当前电影做交叉，用于推荐模型（如：最近喜欢过这部电影的人是否喜欢当前电影）
      .withColumn("userRatedMovie1",col("userPositiveHistory").getItem(0))
      .withColumn("userRatedMovie2",col("userPositiveHistory").getItem(1))
      .withColumn("userRatedMovie3",col("userPositiveHistory").getItem(2))
      .withColumn("userRatedMovie4",col("userPositiveHistory").getItem(3))
      .withColumn("userRatedMovie5",col("userPositiveHistory").getItem(4))
      // --- 用户评分次数（活跃度特征）---
      // 统计该用户在当前行之前最近100条历史中有多少条评分记录
      .withColumn("userRatingCount", count(lit(1))
        .over(Window.partitionBy("userId")
          // 这个 100 是一个超参数，代表"最多参考最近100条历史"。如果某用户只有30条记录，那就只用这30条，不会报错。
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userAvgReleaseYear", avg(col("releaseYear"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)).cast(IntegerType))
      .withColumn("userReleaseYearStddev", stddev(col("releaseYear"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userAvgRating", format_number(avg(col("rating"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)), NUMBER_PRECISION))
      .withColumn("userRatingStddev", stddev(col("rating"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userGenres", extractGenres(collect_list(when(col("label") === 1, col("genres")).otherwise(lit(null)))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1))))
      // --- 统一填充null为0（和之前movieRatingStddev同样的道理，计算产生的null）---
      .na.fill(0)
      // --- 格式化标准差列（同理必须在 na.fill 之后）---
      .withColumn("userRatingStddev",format_number(col("userRatingStddev"), NUMBER_PRECISION))
      .withColumn("userReleaseYearStddev",format_number(col("userReleaseYearStddev"), NUMBER_PRECISION))
      // --- 拆分用户偏好类型为独立特征列 ---
      // 取排序后的前5个偏好类型，和电影侧的 genre1/2/3 对称
      // 最喜欢的类型
      .withColumn("userGenre1",col("userGenres").getItem(0))
      // 第2喜欢
      .withColumn("userGenre2",col("userGenres").getItem(1))
      .withColumn("userGenre3",col("userGenres").getItem(2))
      .withColumn("userGenre4",col("userGenres").getItem(3))
      .withColumn("userGenre5",col("userGenres").getItem(4))
      // --- 清理中间列 ---
      // genres：原始类型字符串，已被拆分为 genre1-5，不再需要
      // userGenres：中间的 Seq 类型列，已被拆分为 genre1-5，不再需要
      // userPositiveHistory：中间的列表列，已被拆分为 ratedMovie1-5，不再需要
      .drop("genres", "userGenres", "userPositiveHistory")
      // --- 过滤冷启动用户 ---
      // 评分次数<=1的用户没有足够的历史行为，特征无意义，直接过滤掉
      .filter(col("userRatingCount") > 1)

    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(100, truncate = false)

    samplesWithUserFeatures
  }

  def extractAndSaveMovieFeaturesToRedis(samples:DataFrame): DataFrame = {
    // ========================================
    // 第一步：提取每部电影的最新一条样本（包含最新的电影特征）
    // ========================================
    // 按 movieId 分区，按 timestamp 降序排列，取每部电影最新的一条记录
    // 和用户侧逻辑完全对称
    val movieLatestSamples = samples.withColumn("movieRowNum", row_number()
      .over(Window.partitionBy("movieId")
        .orderBy(col("timestamp").desc)))
      .filter(col("movieRowNum") === 1)
      .select("movieId","releaseYear", "movieGenre1","movieGenre2","movieGenre3","movieRatingCount",
        "movieAvgRating", "movieRatingStddev")
      .na.fill("")

    movieLatestSamples.printSchema()
    movieLatestSamples.show(100, truncate = false)

    val movieFeaturePrefix = "mf:"

    val redisClient = new Jedis(redisEndpoint, redisPort)
    val params = SetParams.setParams()
    //set ttl to 24hs * 30
    params.ex(60 * 60 * 24 * 30)
    val sampleArray = movieLatestSamples.collect()
    println("total movie size:" + sampleArray.length)
    var insertedMovieNumber = 0
    val movieCount = sampleArray.length
    for (sample <- sampleArray){
      val movieKey = movieFeaturePrefix + sample.getAs[String]("movieId")
      val valueMap = mutable.Map[String, String]()
      valueMap("movieGenre1") = sample.getAs[String]("movieGenre1")
      valueMap("movieGenre2") = sample.getAs[String]("movieGenre2")
      valueMap("movieGenre3") = sample.getAs[String]("movieGenre3")
      valueMap("movieRatingCount") = sample.getAs[Long]("movieRatingCount").toString
      valueMap("releaseYear") = sample.getAs[Int]("releaseYear").toString
      valueMap("movieAvgRating") = sample.getAs[String]("movieAvgRating")
      valueMap("movieRatingStddev") = sample.getAs[String]("movieRatingStddev")

      redisClient.hset(movieKey, JavaConversions.mapAsJavaMap(valueMap))
      insertedMovieNumber += 1
      if (insertedMovieNumber % 100 ==0){
        println(insertedMovieNumber + "/" + movieCount + "...")
      }
    }

    redisClient.close()
    movieLatestSamples
  }

  def splitAndSaveTrainingTestSamples(samples:DataFrame, savePath:String)={
    //generate a smaller sample set for demo
    // 从完整样本中随机抽取10%作为小样本集，用于快速调试和演示
    val smallSamples = samples.sample(0.1)

    //split training and test set by 8:2
    val Array(training, test) = smallSamples.randomSplit(Array(0.8, 0.2))

    val sampleResourcesPath = this.getClass.getResource(savePath)
    training.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/trainingSamples")
    test.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/testSamples")
  }

  /*
  和前面 randomSplit 版本的核心区别：
    randomSplit（随机划分）              按时间戳划分（本函数）
    ─────────────────────              ─────────────────────
    训练集和测试集随机混杂               训练集 = 早期数据，测试集 = 近期数据
    测试集里可能出现未来的数据            严格用过去预测未来
    评估结果偏乐观（有数据泄露风险）       评估结果更接近真实场景

    按时间划分才是推荐系统的正确评估方式——你不能用用户未来的评分去预测他过去的行为，
    这和 addUserFeatures 中 rowsBetween(-100, -1) 的防泄露设计一脉相承。
   */
  def splitAndSaveTrainingTestSamplesByTimeStamp(samples:DataFrame, savePath:String)={
    //generate a smaller sample set for demo
    // 从完整样本中随机抽取10%作为小样本集，用于快速调试
    // 同时将 timestamp 列转为 Long 类型，确保后续能做数值比较和分位数计算
    val smallSamples = samples.sample(0.1).withColumn("timestampLong", col("timestamp").cast(LongType))

    // 计算 timestampLong 的 0.8 分位数（第80百分位点）
    // 参数含义：数据列名、分位数数组[0.8]、允许的相对误差0.05（近似计算，牺牲精度换速度）
    // 返回值是 Array[Double]，取第一个元素作为训练/测试的时间分界点
    val quantile = smallSamples.stat.approxQuantile("timestampLong", Array(0.8), 0.05)
    val splitTimestamp = quantile.apply(0)

    // 按时间分界点划分训练集和测试集：
    // - 训练集：时间戳 <= 分界点（较早的80%数据）
    // - 测试集：时间戳 > 分界点（较晚的20%数据）
    // 划分后删除临时的 timestampLong 列，恢复原始列结构
    val training = smallSamples.where(col("timestampLong") <= splitTimestamp).drop("timestampLong")
    val test = smallSamples.where(col("timestampLong") > splitTimestamp).drop("timestampLong")

    val sampleResourcesPath = this.getClass.getResource(savePath)
    training.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/trainingSamples")
    test.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/testSamples")
  }


  /*
  整个函数做了两件事：
  第一步：从百万条样本中，只保留每个用户最新的一条
       ┌─────────────────────────────────────────────┐
       │  userId=1, rating=3.5, timestamp=1112486027  │ ← 保留这条（最新）
       │  userId=1, rating=4.0, timestamp=1094785740  │ ← 丢弃
       │  userId=2, rating=5.0, timestamp=1200000000  │ ← 保留这条（最新）
       │  ...                                         │
       └─────────────────────────────────────────────┘

第二步：将每个用户的特征以 Hash 结构写入 Redis
       Key:   "uf:1"
       Value: {
         "userAvgRating":      "3.87",
         "userGenre1":         "Action",
         "userRatedMovie1":    "296",
         "userRatingCount":    "156",
         ...
       }
       TTL:   30天后自动过期
   */
  def extractAndSaveUserFeaturesToRedis(samples:DataFrame): DataFrame = {
    // ========================================
    // 第一步：提取每个用户的最新一条样本（包含最新的用户特征）
    // ========================================
    // 按 userId 分区，按 timestamp 降序排列，用 row_number 给每行编号
    // 每个用户的最新一条记录编号为 1
    val userLatestSamples = samples.withColumn("userRowNum", row_number()
      .over(Window.partitionBy("userId")
        .orderBy(col("timestamp").desc)))
      .filter(col("userRowNum") === 1)
      // 只选取用户特征相关列（去掉 movieId、rating、label 等电影侧和标签列）
      .select("userId","userRatedMovie1", "userRatedMovie2","userRatedMovie3","userRatedMovie4","userRatedMovie5",
        "userRatingCount", "userAvgReleaseYear", "userReleaseYearStddev", "userAvgRating", "userRatingStddev",
        "userGenre1", "userGenre2","userGenre3","userGenre4","userGenre5")
      // 空值填充为空字符串，避免写入 Redis 时出现 null
      .na.fill("")

    // 打印 schema 和前100条数据，确认用户特征结构正确
    userLatestSamples.printSchema()
    userLatestSamples.show(100, truncate = false)

    // ========================================
    // 第二步：将用户特征逐条写入 Redis
    // ========================================
    // Redis 中的 key 格式为 "uf:userId"，value 为 Hash 结构（字段名 → 字段值）
    // 例如：HGETALL "uf:1" → {userAvgRating: "3.87", userGenre1: "Action", ...}
    val userFeaturePrefix = "uf:"

    val redisClient = new Jedis(redisEndpoint, redisPort)
    val params = SetParams.setParams()
    //set ttl to 24hs * 30
    params.ex(60 * 60 * 24 * 30)
    // 将 DataFrame 收集到 Driver 端转为数组（因为要逐条写入 Redis）
    // 注意：用户量大时这一步可能 OOM，生产环境建议用 foreachPartition 分批处理
    val sampleArray = userLatestSamples.collect()
    println("total user size:" + sampleArray.length)
    var insertedUserNumber = 0
    val userCount = sampleArray.length
    for (sample <- sampleArray){
      val userKey = userFeaturePrefix + sample.getAs[String]("userId")
      val valueMap = mutable.Map[String, String]()
      valueMap("userRatedMovie1") = sample.getAs[String]("userRatedMovie1")
      valueMap("userRatedMovie2") = sample.getAs[String]("userRatedMovie2")
      valueMap("userRatedMovie3") = sample.getAs[String]("userRatedMovie3")
      valueMap("userRatedMovie4") = sample.getAs[String]("userRatedMovie4")
      valueMap("userRatedMovie5") = sample.getAs[String]("userRatedMovie5")
      valueMap("userGenre1") = sample.getAs[String]("userGenre1")
      valueMap("userGenre2") = sample.getAs[String]("userGenre2")
      valueMap("userGenre3") = sample.getAs[String]("userGenre3")
      valueMap("userGenre4") = sample.getAs[String]("userGenre4")
      valueMap("userGenre5") = sample.getAs[String]("userGenre5")
      // 用户行为统计特征（不同列的原始类型不同，统一转为 String 存入 Redis）
      valueMap("userRatingCount") = sample.getAs[Long]("userRatingCount").toString
      valueMap("userAvgReleaseYear") = sample.getAs[Int]("userAvgReleaseYear").toString
      valueMap("userReleaseYearStddev") = sample.getAs[String]("userReleaseYearStddev")
      valueMap("userAvgRating") = sample.getAs[String]("userAvgRating")
      valueMap("userRatingStddev") = sample.getAs[String]("userRatingStddev")

      redisClient.hset(userKey, JavaConversions.mapAsJavaMap(valueMap))
      insertedUserNumber += 1
      // 进度打印：每写入100个用户输出一次进度
      if (insertedUserNumber % 100 ==0){
        println(insertedUserNumber + "/" + userCount + "...")
      }
    }

    redisClient.close()
    // 返回最新用户特征的 DataFrame（调用方可能还需要用）
    userLatestSamples
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("featureEngineering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)

    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)

    val ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    ratingSamplesWithLabel.show(10, truncate = false)

    val samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    val samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)


    //save samples as csv format
    splitAndSaveTrainingTestSamples(samplesWithUserFeatures, "/webroot/sampledata")

    //save user features and item features to redis for online inference
    //extractAndSaveUserFeaturesToRedis(samplesWithUserFeatures)
    //extractAndSaveMovieFeaturesToRedis(samplesWithUserFeatures)
    spark.close()
  }

}
