package com.sparrowrecsys.offline.spark.model

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.spark.SparkConf
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.sql.{Row, SparkSession}
import redis.clients.jedis.Jedis
import redis.clients.jedis.params.SetParams

/**
 * ALS 模型产物导出器——应用侧。
 *
 * 它【加载】 [[CollaborativeFiltering]] 训练并保存好的模型，然后导出线上需要的产物，
 * 好处是：每生成一种产物（隐向量 / 每用户Top-N推荐）都不必重新训练一遍模型。
 *
 * 提供两个独立方法，main 里按需选择性调用：
 *   - [[saveEmbeddings]] ：保存用户/物品隐向量（落盘 + 可选Redis）
 *   - [[saveUserRecs]]   ：方案A，每用户Top-N推荐结果（落盘 + 可选Redis）
 *
 * 数据格式约定（必须与线上读取端对齐）：
 *   隐向量文件 alsItemEmbeddings.csv / alsUserEmbeddings.csv：每行  id:e1 e2 ... eN
 *   隐向量Redis：alsI2vEmb:id / alsUEmb:id  ->  e1 e2 ... eN
 *   推荐文件 userRecs.csv：每行  userId:m1,m2,m3
 *   推荐Redis：rec:userId  ->  m1,m2,m3
 */
object AlsModelExporter {

  val redisEndpoint = "localhost"
  val redisPort = 6379

  // Redis写入统一的TTL：24小时
  private val REDIS_TTL_SECONDS = 60 * 60 * 24

  /** /webroot/modeldata/ 目录（导出文件都落在这里，模型也从这里加载） */
  private def modelDataFolder: String = this.getClass.getResource("/webroot/modeldata/").getPath

  /** 加载离线训练并保存的 ALS 模型，避免重复训练。目录由 CollaborativeFiltering 保存。 */
  def loadModel(): ALSModel = {
    val modelPath = modelDataFolder + CollaborativeFiltering.MODEL_DIR_NAME
    println(s"加载ALS模型: $modelPath")
    ALSModel.load(modelPath)
  }

  /**
   * 保存用户和物品隐向量：落盘为 CSV，并可选写入 Redis。
   * 格式：id:e1 e2 ... eN（与 Item2vec 的 Embedding 输出格式一致）。
   *
   * @param model       已加载的 ALS 模型
   * @param saveToRedis 是否同时写入 Redis
   */
  def saveEmbeddings(model: ALSModel, saveToRedis: Boolean): Unit = {
    val itemFactors = model.itemFactors.collect()
    val userFactors = model.userFactors.collect()

    // 落盘
    writeEmbFile(modelDataFolder + "alsItemEmbeddings.csv", itemFactors)
    writeEmbFile(modelDataFolder + "alsUserEmbeddings.csv", userFactors)
    println(s"隐向量已保存到: ${modelDataFolder}alsItemEmbeddings.csv 和 alsUserEmbeddings.csv")

    // 可选：写 Redis（物品前缀 alsI2vEmb，用户前缀 alsUEmb，与 Item2vec 的 i2vEmb/uEmb 区分开避免覆盖）
    if (saveToRedis) {
      val redisClient = new Jedis(redisEndpoint, redisPort)
      val params = SetParams.setParams().ex(REDIS_TTL_SECONDS)
      for (row <- itemFactors) {
        redisClient.set("alsI2vEmb:" + row.getAs[Int]("id"), row.getAs[Seq[Float]]("features").mkString(" "), params)
      }
      for (row <- userFactors) {
        redisClient.set("alsUEmb:" + row.getAs[Int]("id"), row.getAs[Seq[Float]]("features").mkString(" "), params)
      }
      redisClient.close()
      println("隐向量已写入Redis（前缀 alsI2vEmb: / alsUEmb:）")
    }
  }

  /** 把一组隐因子 Row（含 id:Int、features:Seq[Float]）按 "id:e1 e2 ... eN" 写入文件 */
  private def writeEmbFile(filePath: String, rows: Array[Row]): Unit = {
    var bw: BufferedWriter = null
    try {
      bw = new BufferedWriter(new FileWriter(new File(filePath)))
      for (row <- rows) {
        val id = row.getAs[Int]("id")
        val features = row.getAs[Seq[Float]]("features")
        bw.write(id + ":" + features.mkString(" ") + "\n")
      }
    } finally {
      if (bw != null) bw.close()
    }
  }

  /**
   * 方案A：为每个用户离线算好 Top-N 推荐，落盘为 CSV，并可选写入 Redis。
   * 供线上 RecForYouProcess.retrievalByAlsOffline 作为"离线ALS召回"这一路读取。
   * 格式：文件每行 userId:m1,m2,m3 ；Redis  rec:userId -> m1,m2,m3 （movieId按推荐分降序、逗号分隔）。
   *
   * @param model       已加载的 ALS 模型
   * @param topN        每个用户推荐多少部
   * @param saveToRedis 是否同时写入 Redis
   */
  def saveUserRecs(model: ALSModel, topN: Int, saveToRedis: Boolean): Unit = {
    // recommendForAllUsers 返回 [userIdInt, recommendations: array<struct<movieIdInt, rating>>]，已按推荐分降序
    // 计时：recommendForAllUsers 是惰性的，真正的全量打分计算被 collect() 触发，故把两步合在一起计时才准确
    val recStart = System.currentTimeMillis()
    val userRecRows = model.recommendForAllUsers(topN).collect()
    val recCost = System.currentTimeMillis() - recStart
    val userCount = userRecRows.length
    println(s"[计时] recommendForAllUsers+collect 总耗时: ${recCost}ms，用户数: $userCount，" +
      f"平均每用户: ${recCost.toDouble / math.max(userCount, 1)}%.3fms")

    // 把每个用户的推荐movieId列表拼成 "m1,m2,m3"（文件和Redis复用同一份字符串）
    val userRecPairs = userRecRows.map { row =>
      val userId = row.getAs[Int]("userIdInt")
      val recs = row.getAs[Seq[Row]]("recommendations")
      val movieIdStr = recs.map(_.getAs[Int]("movieIdInt")).mkString(",")
      (userId, movieIdStr)
    }

    // 落盘
    val userRecFile = new File(modelDataFolder + "userRecs.csv")
    var bw: BufferedWriter = null
    try {
      bw = new BufferedWriter(new FileWriter(userRecFile))
      for ((userId, movieIdStr) <- userRecPairs) {
        bw.write(userId + ":" + movieIdStr + "\n")
      }
    } finally {
      if (bw != null) bw.close()
    }
    println(s"每用户Top-N推荐已保存到: ${userRecFile.getPath}")

    // 可选：写 Redis
    if (saveToRedis) {
      val redisClient = new Jedis(redisEndpoint, redisPort)
      val params = SetParams.setParams().ex(REDIS_TTL_SECONDS)
      for ((userId, movieIdStr) <- userRecPairs) {
        redisClient.set("rec:" + userId, movieIdStr, params)
      }
      redisClient.close()
      println("每用户Top-N推荐已写入Redis（key前缀 rec:）")
    }
  }

  def main(args: Array[String]): Unit = {
    // 为了去掉讨厌的报错
    sys.props("hadoop.home.dir") = "D:\\dev_software\\hadoop"

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("alsModelExporter")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    // 加载已训练好的模型（无需重新训练）
    val model = loadModel()

    // 按需选择性调用（不需要的注释掉即可）：
//    saveEmbeddings(model, saveToRedis = false)

    // saveUserRecs 内部要为全量用户做 recommendForAllUsers（全量打分），通常是这里的主要耗时点，单独计时观察
    val userRecsStart = System.currentTimeMillis()
    saveUserRecs(model, topN = 10, saveToRedis = false)
    println(s"[计时] saveUserRecs(topN=10): ${System.currentTimeMillis() - userRecsStart}ms")

    spark.stop()
  }
}
