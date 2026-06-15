package com.sparrowrecsys.offline.spark.embedding

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}
import redis.clients.jedis.Jedis
import redis.clients.jedis.params.SetParams

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

object Embedding {

  val redisEndpoint = "localhost"
  val redisPort = 6379

  /**
   * 处理用户评分数据，生成每个用户的物品（电影）行为序列，用于后续Item2Vec等Embedding训练。
   *
   * 处理流程：
   * 1. 读取用户评分CSV数据
   * 2. 过滤掉低分评分（< 3.5），只保留用户感兴趣的电影
   * 3. 按用户分组，将每个用户的评分记录按时间戳排序
   * 4. 将每个用户的电影观看序列拼接为字符串，最终返回RDD[Seq[String]]
   *
   * @param sparkSession      Spark会话
   * @param rawSampleDataPath 评分数据的资源路径
   * @return 每个用户对应的电影ID序列（按时间排序）
   */
  def processItemSequence(sparkSession: SparkSession, rawSampleDataPath: String): RDD[Seq[String]] ={

    // 读取评分数据CSV文件
    val ratingsResourcesPath = this.getClass.getResource(rawSampleDataPath)
    val ratingSamples = sparkSession.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)

    // 自定义UDF：将每个用户的多条评分记录按时间戳排序，提取电影ID列表
    val sortUdf: UserDefinedFunction = udf((rows: Seq[Row]) => {
      rows.map { case Row(movieId: String, timestamp: String) => (movieId, timestamp) }
        .sortBy { case (_, timestamp) => timestamp }
        .map { case (movieId, _) => movieId }
    })

    ratingSamples.printSchema()

    // 过滤评分>=3.5的记录，按userId分组，收集电影ID并按时间排序，最后用空格拼接成字符串
    val userSeq = ratingSamples
      .where(col("rating") >= 3.5)
      .groupBy("userId")
      .agg(sortUdf(collect_list(struct("movieId", "timestamp"))) as "movieIds")
      .withColumn("movieIdStr", array_join(col("movieIds"), " "))

    userSeq.select("userId", "movieIdStr").show(10, truncate = false)
    // 将字符串形式的电影序列转为Seq[String]，作为Word2Vec的输入
    userSeq.select("movieIdStr").rdd.map(r => r.getAs[String]("movieIdStr").split(" ").toSeq)
  }

  /**
   * 基于用户历史行为生成用户Embedding向量，用于用户兴趣表示和相似用户推荐。
   *
   * 核心思想：将用户评分过的所有电影的Embedding向量进行平均，作为用户的兴趣向量表示。
   * 这种方法能够捕捉用户的整体偏好，适用于用户画像、相似用户发现等场景。
   *
   * 处理流程：
   * 1. 读取用户评分数据（CSV格式，包含userId、movieId、rating、timestamp等字段）
   * 2. 将评分数据按用户ID进行分组
   * 3. 对于每个用户，遍历其评分过的所有电影，获取电影的Embedding向量
   * 4. 将所有电影的Embedding向量进行累加，最后除以电影数量得到平均值，即用户Embedding
   * 5. 将用户Embedding写入文件，格式为：userId:emb1 emb2 ... embN
   * 6. 可选：将用户Embedding写入Redis，设置24小时过期时间，供线上服务使用
   *
   * 注意事项：
   * - 用户的Embedding质量取决于其历史行为数量，行为越少的用户Embedding质量越差
   * - 该方法对所有电影的Embedding进行等权平均，未考虑评分高低的差异
   * - 如果需要更精细的用户表示，可以考虑加权平均（如按评分加权）
   *
   * @param sparkSession       Spark会话
   * @param rawSampleDataPath  评分数据的资源路径（相对于classpath）
   * @param word2VecModel      训练好的Word2Vec模型，用于获取电影的Embedding向量
   * @param embLength          Embedding向量维度
   * @param embOutputFilename  输出文件名，保存用户Embedding到本地文件
   * @param saveToRedis        是否保存到Redis，供线上服务使用
   * @param redisKeyPrefix     Redis key前缀，格式为：prefix:userId
   */
  def generateUserEmb(sparkSession: SparkSession, rawSampleDataPath: String, word2VecModel: Word2VecModel, embLength:Int, embOutputFilename:String, saveToRedis:Boolean, redisKeyPrefix:String): Unit ={
    val ratingsResourcesPath = this.getClass.getResource(rawSampleDataPath)
    val ratingSamples = sparkSession.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    ratingSamples.show(10, false)

    val userEmbeddings = new ArrayBuffer[(String, Array[Float])]()

    ratingSamples.collect().groupBy(_.getAs[String]("userId"))
      .foreach(user => {
        val userId = user._1
        var userEmb = new Array[Float](embLength)

        var movieCount = 0
        // 使用foldRight遍历用户的电影评分记录，累加每个电影的Embedding向量
        userEmb = user._2.foldRight[Array[Float]](userEmb)((row, newEmb) => {
          val movieId = row.getAs[String]("movieId")
          val movieEmb = word2VecModel.getVectors.get(movieId)
          movieCount += 1
          if(movieEmb.isDefined){
            // 向量逐元素相加：将当前电影的Embedding累加到用户Embedding中
            newEmb.zip(movieEmb.get).map { case (x, y) => x + y }
          }else{
            newEmb
          }
        }).map((x: Float) => x / movieCount) // 计算平均值：总和除以电影数量得到用户Embedding
        userEmbeddings.append((userId,userEmb))
      })



    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(embFolderPath.getPath + embOutputFilename)
    val bw = new BufferedWriter(new FileWriter(file))

    for (userEmb <- userEmbeddings) {
      bw.write(userEmb._1 + ":" + userEmb._2.mkString(" ") + "\n")
    }
    bw.close()

    if (saveToRedis) {
      val redisClient = new Jedis(redisEndpoint, redisPort)
      val params = SetParams.setParams()
      //set ttl to 24hs
      params.ex(60 * 60 * 24)

      for (userEmb <- userEmbeddings) {
        redisClient.set(redisKeyPrefix + ":" + userEmb._1, userEmb._2.mkString(" "), params)
      }
      redisClient.close()
    }
  }

  /**
   * 基于用户行为序列训练Item2Vec模型，生成电影Embedding向量。
   *
   * 处理流程：
   * 1. 使用Word2Vec对输入的电影序列进行训练，得到每个电影的Embedding向量
   * 2. 打印指定电影的相似电影，用于验证Embedding质量
   * 3. 将所有电影的Embedding写入文件（格式：movieId:emb1 emb2 ...）
   * 4. 可选：将Embedding写入Redis，供线上服务使用
   * 5. 使用LSH对Embedding建立索引，支持近似最近邻检索
   *
   * @param sparkSession      Spark会话
   * @param samples           用户电影行为序列（每个元素为一个用户的电影ID列表）
   * @param embLength         Embedding向量维度
   * @param embOutputFilename 输出文件名
   * @param saveToRedis       是否保存到Redis
   * @param redisKeyPrefix    Redis key前缀
   * @return 训练好的Word2Vec模型
   */
  def trainItem2vec(sparkSession: SparkSession, samples : RDD[Seq[String]], embLength:Int, embOutputFilename:String, saveToRedis:Boolean, redisKeyPrefix:String): Word2VecModel = {
    // 配置Word2Vec参数：向量维度、滑动窗口大小、迭代次数
    val word2vec = new Word2Vec()
      .setVectorSize(embLength)
      .setWindowSize(5)
      .setNumIterations(10)

    // 训练Word2Vec模型
    val model = word2vec.fit(samples)

    // 验证：找出与电影"158"最相似的20部电影，打印余弦相似度
    val synonyms = model.findSynonyms("158", 20)
    for ((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    // 将Embedding写入文件，格式为 movieId:emb1 emb2 ...
    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(embFolderPath.getPath + embOutputFilename)
    val bw = new BufferedWriter(new FileWriter(file))
    for (movieId <- model.getVectors.keys) {
      bw.write(movieId + ":" + model.getVectors(movieId).mkString(" ") + "\n")
    }
    bw.close()

    // 可选：将Embedding写入Redis，设置24小时过期时间
    if (saveToRedis) {
      val redisClient = new Jedis(redisEndpoint, redisPort)
      val params = SetParams.setParams()
      //set ttl to 24hs
      params.ex(60 * 60 * 24)
      for (movieId <- model.getVectors.keys) {
        redisClient.set(redisKeyPrefix + ":" + movieId, model.getVectors(movieId).mkString(" "), params)
      }
      redisClient.close()
    }

    // 对Embedding建立LSH索引，支持近似最近邻检索（用于线上相似电影推荐）
    embeddingLSH(sparkSession, model.getVectors)
    model
  }

  def oneRandomWalk(transitionMatrix : mutable.Map[String, mutable.Map[String, Double]], itemDistribution : mutable.Map[String, Double], sampleLength:Int): Seq[String] ={
    val sample = mutable.ListBuffer[String]()

    //pick the first element
    val randomDouble = Random.nextDouble()
    var firstItem = ""
    var accumulateProb:Double = 0D
    breakable { for ((item, prob) <- itemDistribution) {
      accumulateProb += prob
      if (accumulateProb >= randomDouble){
        firstItem = item
        break
      }
    }}

    sample.append(firstItem)
    var curElement = firstItem

    breakable { for(_ <- 1 until sampleLength) {
      if (!itemDistribution.contains(curElement) || !transitionMatrix.contains(curElement)){
        break
      }

      val probDistribution = transitionMatrix(curElement)
      val randomDouble = Random.nextDouble()
      var accumulateProb: Double = 0D
      breakable { for ((item, prob) <- probDistribution) {
        accumulateProb += prob
        if (accumulateProb >= randomDouble){
          curElement = item
          break
        }
      }}
      sample.append(curElement)
    }}
    Seq(sample.toList : _*)
  }

  def randomWalk(transitionMatrix : mutable.Map[String, mutable.Map[String, Double]], itemDistribution : mutable.Map[String, Double], sampleCount:Int, sampleLength:Int): Seq[Seq[String]] ={
    val samples = mutable.ListBuffer[Seq[String]]()
    for(_ <- 1 to sampleCount) {
      samples.append(oneRandomWalk(transitionMatrix, itemDistribution, sampleLength))
    }
    Seq(samples.toList : _*)
  }

  def generateTransitionMatrix(samples : RDD[Seq[String]]): (mutable.Map[String, mutable.Map[String, Double]], mutable.Map[String, Double]) ={
    val pairSamples = samples.flatMap[(String, String)]( sample => {
      var pairSeq = Seq[(String,String)]()
      var previousItem:String = null
      sample.foreach((element:String) => {
        if(previousItem != null){
          pairSeq = pairSeq :+ (previousItem, element)
        }
        previousItem = element
      })
      pairSeq
    })

    val pairCountMap = pairSamples.countByValue()
    var pairTotalCount = 0L
    val transitionCountMatrix = mutable.Map[String, mutable.Map[String, Long]]()
    val itemCountMap = mutable.Map[String, Long]()

    pairCountMap.foreach( pair => {
      val pairItems = pair._1
      val count = pair._2

      if(!transitionCountMatrix.contains(pairItems._1)){
        transitionCountMatrix(pairItems._1) = mutable.Map[String, Long]()
      }

      transitionCountMatrix(pairItems._1)(pairItems._2) = count
      itemCountMap(pairItems._1) = itemCountMap.getOrElse[Long](pairItems._1, 0) + count
      pairTotalCount = pairTotalCount + count
    })

    val transitionMatrix = mutable.Map[String, mutable.Map[String, Double]]()
    val itemDistribution = mutable.Map[String, Double]()

    transitionCountMatrix foreach {
      case (itemAId, transitionMap) =>
        transitionMatrix(itemAId) = mutable.Map[String, Double]()
        transitionMap foreach { case (itemBId, transitionCount) => transitionMatrix(itemAId)(itemBId) = transitionCount.toDouble / itemCountMap(itemAId) }
    }

    itemCountMap foreach { case (itemId, itemCount) => itemDistribution(itemId) = itemCount.toDouble / pairTotalCount }
    (transitionMatrix, itemDistribution)
  }

  /**
   * 使用局部敏感哈希（LSH）对电影Embedding建立索引，支持近似最近邻检索。
   *
   * LSH核心原理：让相似的向量落入同一个"桶"，查询时只需在桶内搜索，将时间复杂度从O(n)降至O(1)
   *
   * 关键参数说明：
   *
   *   - BucketLength（桶长度）：桶的"松紧度"，是距离阈值，非容量！
   *     值越小 → 桶越紧，只有距离小于BucketLength的向量才会同桶，精度高但召回率可能低
   *     值越大 → 桶越松，更多向量会同桶，召回率高但计算量增加
   *     此处设为0.1，表示投影后差值小于0.1的向量才会被分到同一桶
   *
   *   - NumHashTables（哈希表数量）：使用的独立哈希函数数量（筛子数量）
   *     值越小 → 计算快，但假阳性率高
   *     值越大 → 假阳性率指数级下降（总体假阳性率 = p的numHashTables次方），但计算开销增加
   *     此处设为3，表示向量需通过3个独立哈希函数的考验
   *
   * 多桶策略：Spark MLlib默认使用"AND"策略，即向量必须在ALL哈希表中都落入同一个桶，才会被视为候选相似对。
   *   AND策略优点：最大程度减少候选集，提高计算效率
   *   AND策略缺点：可能漏掉部分相似点（可通过增加NumHashTables弥补）
   *   对比OR策略：在任一哈希表同桶即可，召回率高但候选集大
   *
   * 处理流程：
   *
   *   1. 将电影Embedding从Map转换为Spark DataFrame
   *   2. 使用BucketedRandomProjectionLSH建立哈希桶模型，将高维向量映射到哈希桶
   *   3. 每个向量得到NumHashTables个桶ID（如[[-2.0], [14.0], [8.0]]）
   *   4. 打印分桶结果，验证模型是否正常工作
   *   5. 使用示例向量进行近似最近邻查询，演示如何找到相似电影
   *
   * @param spark        Spark会话
   * @param movieEmbMap  电影Embedding映射（movieId -> 向量）
   */
  def embeddingLSH(spark:SparkSession, movieEmbMap:Map[String, Array[Float]]): Unit ={

    // 将Map转换为DataFrame，列名为movieId和emb
    val movieEmbSeq = movieEmbMap.toSeq.map(item => (item._1, Vectors.dense(item._2.map(f => f.toDouble))))
    val movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")

    // 配置LSH模型：
    // - setBucketLength(0.1): 桶长度为0.1，控制分桶精度
    // - setNumHashTables(3): 使用3个哈希表，降低假阳性率
    // - setInputCol("emb"): 指定输入向量列
    // - setOutputCol("bucketId"): 指定输出桶ID列
    val bucketProjectionLSH = new BucketedRandomProjectionLSH()
      .setBucketLength(0.1)
      .setNumHashTables(3)
      .setInputCol("emb")
      .setOutputCol("bucketId")

    // 训练LSH模型，为每个电影生成哈希桶ID（每个向量得到NumHashTables个桶ID）
    val bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    val embBucketResult = bucketModel.transform(movieEmbDF)
    println("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    println("movieId, emb, bucketId data result:")
    embBucketResult.show(10, truncate = false)

    // 演示：用一个示例Embedding向量查找5个最近邻电影
    // approxNearestNeighbors内部使用AND策略：向量必须在所有哈希表中同桶才会成为候选
    // 然后对候选集计算真实距离，返回Top-K最近邻
    println("Approximately searching for 5 nearest neighbors of the sample embedding:")
    val sampleEmb = Vectors.dense(0.795,0.583,1.120,0.850,0.174,-0.839,-0.0633,0.249,0.673,-0.237)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(truncate = false)
  }

  def graphEmb(samples : RDD[Seq[String]], sparkSession: SparkSession, embLength:Int, embOutputFilename:String, saveToRedis:Boolean, redisKeyPrefix:String): Word2VecModel ={
    val transitionMatrixAndItemDis = generateTransitionMatrix(samples)

    println(transitionMatrixAndItemDis._1.size)
    println(transitionMatrixAndItemDis._2.size)

    val sampleCount = 20000
    val sampleLength = 10
    val newSamples = randomWalk(transitionMatrixAndItemDis._1, transitionMatrixAndItemDis._2, sampleCount, sampleLength)

    val rddSamples = sparkSession.sparkContext.parallelize(newSamples)
    trainItem2vec(sparkSession, rddSamples, embLength, embOutputFilename, saveToRedis, redisKeyPrefix)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    // 为了去掉讨厌的报错
    sys.props("hadoop.home.dir") = "D:\\dev_software\\hadoop"

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    val rawSampleDataPath = "/webroot/sampledata/ratings.csv"
    val embLength = 10

    val samples = processItemSequence(spark, rawSampleDataPath)
    val model = trainItem2vec(spark, samples, embLength, "jeff2_item2vecEmb.csv", saveToRedis = false, "i2vEmb")
    //graphEmb(samples, spark, embLength, "itemGraphEmb.csv", saveToRedis = true, "graphEmb")
    //generateUserEmb(spark, rawSampleDataPath, model, embLength, "userEmb.csv", saveToRedis = false, "uEmb")
  }
}
