package com.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, sql}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object FeatureEngineering {
  /**
   * One-hot encoding example function
   * @param samples movie samples dataframe
   */
/**
   * 独热编码（One-Hot Encoding）示例
   * 
   * 独热编码是将离散型特征转换为稀疏向量的常用方法，
   * 适用于处理类别型特征（如电影ID、类型等），使其能够被机器学习模型处理。
   * 
   * 处理流程：
   * 1. 将字符串类型的 movieId 转换为整数类型
   * 2. 使用 OneHotEncoderEstimator 将整数ID转换为独热向量
   * 3. 输出转换后的数据集结构和前10条数据
   * 
   * @param samples 输入的 DataFrame，包含 movieId 列（字符串类型）
   */
  def oneHotEncoderExample(samples:DataFrame): Unit ={
    // 1. 将字符串类型的 movieId 转换为整数类型
    // 因为 OneHotEncoderEstimator 要求输入必须是数值类型
    // 将samples样本集中的movieId列转换为Integer类型，并命名为movieIdNumber
    // samplesWithIdNumber是一个在原有DataFrame基础上新增了一列的DataFrame
    val samplesWithIdNumber = samples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))

    // 2. 创建独热编码器实例
    // - setInputCols: 指定输入列（必须是数值类型），"输入列"的意思是：你准备把哪一列数据拿来做One-hot编码
    // - setOutputCols: 指定输出列（独热向量），告诉编码器：转换结果放到哪一列？
    // - setDropLast(false): 保留最后一个类别（默认true会丢弃，防止共线性）假设有100部不同的电影，One-hot向量就有100位。
    //如果 setDropLast(true)（默认值），编码器会丢掉最后一位，只保留99位，比如：电影100 → [0, 0, ..., 0]（全0就能表示，不需要专门的第100位）
    //这样做有时候可以避免数学上的多重共线性问题，但会丢失信息。
    //这里设置 false，就是保留完整的100位，每个电影都有自己专属的那一位是1。
    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("movieIdNumber"))
      .setOutputCols(Array("movieIdVector"))
      .setDropLast(false)

    // 3. 训练编码器并转换数据
    // fit(): 根据数据统计信息训练编码器
    // transform(): 使用训练好的编码器转换数据
    val oneHotEncoderSamples = oneHotEncoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    
    // 4. 输出结果：打印数据结构和前10条记录
    oneHotEncoderSamples.printSchema()  // 打印Schema结构
    oneHotEncoderSamples.show(10)       // 显示前10条数据
  }

/**
   * 自定义 UDF：将整数序列转换为稀疏向量（用于多热编码）
   * 
   * 输入：
   *   - a: Seq[Int] - 电影所属类型的索引序列（如 [0, 2, 4]）
   *   - length: Int - 向量长度（类型总数）
   * 
   * 输出：
   *   - SparseVector - 稀疏向量，对应索引位置为1，其余为0
   * 
   * 示例：
   *   输入: [0, 2], length=5
   *   输出: (5, [0, 2], [1.0, 1.0])  →  表示为 [1, 0, 1, 0, 0]
   */
  val array2vec: UserDefinedFunction = udf { (a: Seq[Int], length: Int) => 
    org.apache.spark.ml.linalg.Vectors.sparse(length, a.sortWith(_ < _).toArray, Array.fill[Double](a.length)(1.0)) 
  }

  /**
   * 多热编码（Multi-hot Encoding）示例
   * 
   * 多热编码用于处理多标签特征（如电影可能属于多个类型），
   * 与独热编码不同，多热编码允许向量中有多个1。
   * 
   * 处理流程：
   * 1. 将 genres 字符串按 "|" 分割并展开（explode）
   * 2. 使用 StringIndexer 将类型名称转换为数字索引
   * 3. 按 movieId 分组，收集每部电影的所有类型索引
   * 4. 使用 array2vec UDF 将索引序列转换为稀疏向量
   * 
   * @param samples 输入的 DataFrame，包含 movieId, title, genres 列
   */
  def multiHotEncoderExample(samples:DataFrame): Unit ={
    // 1. 将 genres 字段（如 "Action|Adventure|Comedy"）分割并展开
    // 每部电影会生成多行，每行对应一个类型
    val samplesWithGenre = samples.select(
      col("movieId"), 
      col("title"),
      explode(split(col("genres"), "\\|").cast("array<string>")).as("genre")
    )

    // 2. 创建 StringIndexer，将类型名称映射为数字索引
    // 例如：Action → 0, Comedy → 1, Drama → 2, ...
    val genreIndexer = new StringIndexer()
      .setInputCol("genre")
      .setOutputCol("genreIndex")

    // 3. 训练索引器并转换数据
    val stringIndexerModel: StringIndexerModel = genreIndexer.fit(samplesWithGenre)
    val genreIndexSamples = stringIndexerModel.transform(samplesWithGenre)
      .withColumn("genreIndexInt", col("genreIndex").cast(sql.types.IntegerType))

    // 4. 计算类型总数（最大索引 + 1）
    val indexSize = genreIndexSamples.agg(max(col("genreIndexInt"))).head().getAs[Int](0) + 1

    // 5. 按 movieId 分组，收集每部电影的所有类型索引
    val processedSamples = genreIndexSamples
      .groupBy(col("movieId"))
      .agg(collect_list("genreIndexInt").as("genreIndexes"))
      .withColumn("indexSize", typedLit(indexSize))

    // 6. 使用 UDF 将索引列表转换为稀疏向量
    val finalSample = processedSamples.withColumn("vector", array2vec(col("genreIndexes"), col("indexSize")))
    
    // 输出结果
    finalSample.printSchema()
    finalSample.show(10)
  }

  /**
   * 自定义 UDF：将 Double 转换为 DenseVector
   *
   * 输入：
   *   - value: Double - 输入值
   *
   * 输出：
   *   - DenseVector - 稠密向量，包含输入值
   *
   * 示例：
   *   输入: 3.5
   *   输出: [3.5]
   */
  val double2vec: UserDefinedFunction = udf { (value: Double) => org.apache.spark.ml.linalg.Vectors.dense(value) }

  /**
   * Process rating samples
   * @param samples rating samples
   */
  def ratingFeatures(samples:DataFrame): Unit ={
    // 打印输入样本集samples的Schema结构，查看包含哪些列及其数据类型
    samples.printSchema()
    // 打印输入样本集的前10行数据，预览原始打分数据的内容
    samples.show(10)

    //calculate average movie rating score and rating count
    // 按movieId分组，对打分表ratings进行聚合计算，得到每部电影的三个数值型特征
    // 利用打分表ratings计算电影的平均分、被打分次数等数值型特征
    val movieFeatures = samples.groupBy(col("movieId"))
      .agg(count(lit(1)).as("ratingCount"),
        avg(col("rating")).as("avgRating"),
        variance(col("rating")).as("ratingVar"))    // 计算每部电影的评分方差
        .withColumn("avgRatingVec", double2vec(col("avgRating")))   // 将平均评分转换为向量格式，便于后续特征处理

    // 打印聚合计算后的电影特征数据，查看前10条结果
    movieFeatures.show(10)

    //bucketing
    // 创建QuantileDiscretizer分桶器，根据分位数将打分次数离散化到100个桶中
    // 分桶处理，创建QuantileDiscretizer进行分桶，将打分次数这一特征分到100个桶中
    val ratingCountDiscretizer = new QuantileDiscretizer()
      .setInputCol("ratingCount")
      .setOutputCol("ratingCountBucket")
      .setNumBuckets(100)

    //Normalization
    // 创建MinMaxScaler归一化器，将平均评分向量缩放到[0,1]区间内
    // 归一化处理，创建MinMaxScaler进行归一化，将平均得分进行归一化
    val ratingScaler = new MinMaxScaler()
      .setInputCol("avgRatingVec")
      .setOutputCol("scaleAvgRating")

    // 将分桶器和归一化器组成一个处理阶段数组，用于构建Pipeline
    // 创建一个pipeline，依次执行两个特征处理过程
    val pipelineStage: Array[PipelineStage] = Array(ratingCountDiscretizer, ratingScaler)
    // 创建Pipeline实例，将两个特征处理阶段串联起来
    val featurePipeline = new Pipeline().setStages(pipelineStage)

    val movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    movieProcessedFeatures.show(10)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    // 为了去掉讨厌的报错
    sys.props("hadoop.home.dir") = "D:\\dev_software\\hadoop"

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("featureEngineering")
      .set("spark.submit.deployMode", "client")

    // 创建SparkSession实例
    val spark = SparkSession.builder.config(conf).getOrCreate()
    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
    println("Raw Movie Samples:")
    movieSamples.printSchema()
    movieSamples.show(10)

    println("OneHotEncoder Example:")
    oneHotEncoderExample(movieSamples)

    println("MultiHotEncoder Example:")
    multiHotEncoderExample(movieSamples)

    println("Numerical features Example:")
    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    ratingFeatures(ratingSamples)

  }
}
