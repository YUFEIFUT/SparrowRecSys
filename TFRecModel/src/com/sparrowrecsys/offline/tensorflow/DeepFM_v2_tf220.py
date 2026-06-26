"""
DeepFM_v2 for TensorFlow 2.20

从 DeepFM_v2.py（TF 2.3）迁移而来。

与 DeepFM_tf220.py（基于 DeepFM.py 的迁移版本）的主要区别：
  1. 一阶特征：将类别特征和数值特征分开处理，分别映射为标量后相加。
  2. 二阶特征：使用 FM 的向量化公式 (Σv)² - Σ(v²) 计算所有字段的整体交叉，
     而非逐对 dot product。
  3. Deep 部分：将所有字段的 Embedding 拼接为 3D 张量 (batch, num_fields, emb_dim)
     后展平送入 MLP，而非仅用 movie/user Embedding + 数值特征。

TF 2.20 适配要点：
  - 移除 tf.feature_column / DenseFeatures（Keras 3 已不再内置）。
  - 用 Keras 原生层（Embedding、StringLookup、Dense）替代。
  - 自定义 ReduceLayer 继承 tf.keras.layers.Layer，Keras 3 兼容。

数据路径：Google Colab 中 SparrowRecSys 项目的 ByTimeStamp 数据集。
"""

import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# 数据路径：Google Colab 中 SparrowRecSys 项目的样本文件
# ============================================================
train_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/trainingSamplesByTimeStamp.csv"
validation_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/validationSamplesByTimeStamp.csv"
test_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/testSamplesByTimeStamp.csv"


# ============================================================
# 超参数
# ============================================================
BATCH_SIZE = 12       # 每个 batch 的样本数，保持与原始代码一致
EPOCHS = 50           # 最大训练轮次（配合早停法，实际轮次可能更少）
EMBEDDING_DIM = 10    # Embedding 维度，保持与原始代码一致
FM_FIELD_DIM = 64     # 二阶 FM 特征的统一映射维度（原始代码中每个字段映射到 64 维）
MOVIE_BUCKETS = 1001  # movieId 的桶数量（ID 范围 0~1000）
USER_BUCKETS = 30001  # userId 的桶数量（ID 范围 0~30000）


# ============================================================
# 特征定义
# ============================================================
# 数值特征：原始 DeepFM_v2.py 中 Deep 部分使用的 7 个数值特征
NUMERIC_FEATURES = [
    "releaseYear",
    "movieRatingCount",
    "movieAvgRating",
    "movieRatingStddev",
    "userRatingCount",
    "userAvgRating",
    "userRatingStddev",
]

# ID 特征：直接参与 Embedding 的两个整数 ID
ID_FEATURES = ["movieId", "userId"]

# 类型特征：原始 DeepFM_v2.py 中 FM 部分使用的 genre 字符串特征
GENRE_FEATURES = ["userGenre1", "movieGenre1"]

# make_csv_dataset 只读取模型真正使用的列，避免 Keras 3 中多余列导致输入结构匹配混乱
SELECTED_COLUMNS = NUMERIC_FEATURES + ID_FEATURES + GENRE_FEATURES + ["label"]

# 电影类型词表，保持原始 DeepFM_v2.py 的枚举值
GENRE_VOCAB = [
    "Film-Noir", "Action", "Adventure", "Horror", "Romance", "War",
    "Comedy", "Western", "Documentary", "Sci-Fi", "Drama", "Thriller",
    "Crime", "Fantasy", "Animation", "IMAX", "Mystery", "Children", "Musical",
]


# ============================================================
# 数据集加载与 dtype/shape 规整
# ============================================================
def _prepare_features(features, label):
    """
    将 make_csv_dataset 解析出的特征规整成 Keras Functional API 需要的输入格式。

    make_csv_dataset 返回的每一列通常是形状为 (batch,) 的 Tensor。
    这里统一扩展为 (batch, 1)，与下面 Input(shape=(1,)) 对齐。
    同时显式转换 dtype，避免 CSV 自动类型推断在不同 TF/Keras 版本中产生差异。
    """
    model_inputs = {}

    # 数值特征：转为 float32，扩展维度
    for name in NUMERIC_FEATURES:
        model_inputs[name] = tf.expand_dims(tf.cast(features[name], tf.float32), axis=-1)

    # ID 特征：转为 int32，扩展维度
    for name in ID_FEATURES:
        model_inputs[name] = tf.expand_dims(tf.cast(features[name], tf.int32), axis=-1)

    # 类型特征：确保为 string 类型，扩展维度
    for name in GENRE_FEATURES:
        value = features[name]
        if value.dtype != tf.string:
            value = tf.strings.as_string(value)
        model_inputs[name] = tf.expand_dims(value, axis=-1)

    # 标签：转为 float32，扩展维度
    label = tf.expand_dims(tf.cast(label, tf.float32), axis=-1)
    return model_inputs, label


def get_dataset(file_path, shuffle=True):
    """
    从 CSV 构建 tf.data.Dataset。

    select_columns 只保留当前模型需要的列和 label。
    ignore_errors=True 保留原始代码的容错行为：遇到坏样本时跳过。
    """
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=BATCH_SIZE,
        label_name="label",
        select_columns=SELECTED_COLUMNS,
        na_value="0",
        num_epochs=1,
        shuffle=shuffle,
        ignore_errors=True,
    )
    return dataset.map(_prepare_features, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# ============================================================
# Keras 输入层
# ============================================================
def build_inputs():
    """为每个模型特征创建一个命名 Input，名称必须与 Dataset 字典中的 key 一致。"""
    inputs = {}

    for name in NUMERIC_FEATURES:
        inputs[name] = layers.Input(name=name, shape=(1,), dtype=tf.float32)

    for name in ID_FEATURES:
        inputs[name] = layers.Input(name=name, shape=(1,), dtype=tf.int32)

    for name in GENRE_FEATURES:
        inputs[name] = layers.Input(name=name, shape=(1,), dtype=tf.string)

    return inputs


# ============================================================
# 特征编码辅助函数
# ============================================================
def build_id_embedding(inputs, feature_name, num_buckets, output_dim, layer_prefix):
    """
    将整数 ID 转换为稠密 Embedding。

    对应原始代码中的：
    categorical_column_with_identity + embedding_column
    """
    embedding = layers.Embedding(
        input_dim=num_buckets,
        output_dim=output_dim,
        name=f"{layer_prefix}_{feature_name}_embedding",
    )(inputs[feature_name])
    return layers.Flatten(name=f"{layer_prefix}_{feature_name}_flatten")(embedding)


def build_genre_lookup_and_embedding(inputs, feature_name, output_dim, layer_prefix):
    """
    将 genre 字符串转换为词表索引，再转换为 Embedding。

    对应原始代码中的：
    categorical_column_with_vocabulary_list + embedding_column
    """
    genre_index = layers.StringLookup(
        vocabulary=GENRE_VOCAB,
        mask_token=None,
        num_oov_indices=1,
        name=f"{layer_prefix}_{feature_name}_lookup",
    )(inputs[feature_name])
    embedding = layers.Embedding(
        input_dim=len(GENRE_VOCAB) + 1,  # +1 for OOV token
        output_dim=output_dim,
        name=f"{layer_prefix}_{feature_name}_embedding",
    )(genre_index)
    return layers.Flatten(name=f"{layer_prefix}_{feature_name}_flatten")(embedding)


# ============================================================
# 自定义层：ReduceLayer —— 沿指定维度求和或求均值
# ============================================================
class ReduceLayer(layers.Layer):
    """
    沿指定维度对输入张量进行规约（求和或求均值）。

    用于 FM 二阶交叉计算中的两步规约：
      1. 对所有字段的 Embedding 沿字段维度求和 → (batch, emb_dim)
      2. 对平方后的 Embedding 沿字段维度求和 → (batch, emb_dim)
    """

    def __init__(self, axis, op='sum', **kwargs):
        super().__init__(**kwargs)
        # 沿哪个维度进行规约
        self.axis = axis
        # 规约操作类型：'sum'（求和）或 'mean'（求均值）
        self.op = op
        assert self.op in ['sum', 'mean'], f"op must be 'sum' or 'mean', got '{self.op}'"

    def call(self, inputs):
        if self.op == 'sum':
            return tf.reduce_sum(inputs, axis=self.axis)
        elif self.op == 'mean':
            return tf.reduce_mean(inputs, axis=self.axis)
        # 理论上不会走到这里，保留兜底逻辑
        return tf.reduce_sum(inputs, axis=self.axis)

    def get_config(self):
        """支持模型序列化（Keras 3 要求自定义层实现此方法）。"""
        config = super().get_config()
        config.update({"axis": self.axis, "op": self.op})
        return config


# ============================================================
# DeepFM_v2 模型定义
# ============================================================
def build_deepfm_v2_model():
    """
    构建 DeepFM_v2 模型，结构与原始 DeepFM_v2.py 保持一致：

    ┌─────────────────────────────────────────────────────────────┐
    │  First Order (一阶特征)                                      │
    │  ┌──────────────┐   ┌──────────────┐                        │
    │  │ Categorical   │   │ Dense        │                        │
    │  │ (Embedding→1) │   │ (Dense→1)    │                        │
    │  └──────┬───────┘   └──────┬───────┘                        │
    │         └──── Add ─────────┘                                │
    │                                                             │
    │  Second Order (二阶特征) — FM 向量化公式                      │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │ 每个字段 Embedding → Dense(64) → Reshape(batch,1,64) │   │
    │  │ 拼接为 (batch, num_fields, 64)                        │   │
    │  │ FM: (Σv)² - Σ(v²) → (batch, 64)                     │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                                                             │
    │  Deep (DNN 高阶特征)                                         │
    │  Flatten(二阶特征) → Dense(32,relu) → Dense(16,relu)        │
    │                                                             │
    │  Output                                                     │
    │  Concat(first_order, fm_second_order, deep) → sigmoid       │
    └─────────────────────────────────────────────────────────────┘
    """
    inputs = build_inputs()

    # ============================================================
    # 二阶特征：将每个字段的 Embedding 映射到统一维度并 reshape 为 3D
    # ============================================================
    # 对应原始代码中：
    #   second_order_cat_columns_emb = [DenseFeatures([item_genre_emb_col])(inputs), ...]
    #   然后每个经过 Dense(64) + Reshape((-1, 64))

    # 类型特征的 Embedding（二阶交叉用）
    second_order_genre_embs = []
    for genre_name in GENRE_FEATURES:
        emb = build_genre_lookup_and_embedding(inputs, genre_name, EMBEDDING_DIM, "second_order")
        # 映射到统一的 FM_FIELD_DIM 维度空间，再 reshape 为 (batch, 1, FM_FIELD_DIM)
        feature = layers.Dense(FM_FIELD_DIM, activation=None, name=f"second_order_{genre_name}_dense")(emb)
        feature = layers.Reshape((1, FM_FIELD_DIM), name=f"second_order_{genre_name}_reshape")(feature)
        second_order_genre_embs.append(feature)

    # ID 特征的 Embedding（二阶交叉用）
    second_order_id_embs = []
    for id_name in ID_FEATURES:
        num_buckets = MOVIE_BUCKETS if id_name == "movieId" else USER_BUCKETS
        emb = build_id_embedding(inputs, id_name, num_buckets, EMBEDDING_DIM, "second_order")
        # 映射到统一的 FM_FIELD_DIM 维度空间，再 reshape 为 (batch, 1, FM_FIELD_DIM)
        feature = layers.Dense(FM_FIELD_DIM, activation=None, name=f"second_order_{id_name}_dense")(emb)
        feature = layers.Reshape((1, FM_FIELD_DIM), name=f"second_order_{id_name}_reshape")(feature)
        second_order_id_embs.append(feature)

    # 数值特征：拼接后映射到 FM_FIELD_DIM 维度，再 reshape 为 (batch, 1, FM_FIELD_DIM)
    # 对应原始代码中：
    #   second_order_deep_columns = DenseFeatures(deep_columns)(inputs)
    #   second_order_deep_columns = Dense(64)(second_order_deep_columns)
    #   second_order_deep_columns = Reshape((-1, 64))(second_order_deep_columns)
    numeric_concat = layers.Concatenate(name="second_order_numeric_concat")(
        [inputs[name] for name in NUMERIC_FEATURES]
    )
    numeric_64d = layers.Dense(FM_FIELD_DIM, activation=None, name="second_order_numeric_dense")(numeric_concat)
    numeric_3d = layers.Reshape((1, FM_FIELD_DIM), name="second_order_numeric_reshape")(numeric_64d)

    # 沿字段维度（axis=1）拼接所有二阶特征，形成 (batch, num_fields, FM_FIELD_DIM) 的 3D 张量
    # 原始代码：4个类别字段 + 1个数值字段 = 5个字段
    second_order_fm_feature = layers.Concatenate(axis=1, name="second_order_all_fields_concat")(
        second_order_genre_embs + second_order_id_embs + [numeric_3d]
    )

    # ============================================================
    # Deep 部分（DNN）
    # ============================================================
    # 对应原始代码中：
    #   deep_feature = Flatten()(second_order_fm_feature)
    #   deep_feature = Dense(32, relu)(deep_feature)
    #   deep_feature = Dense(16, relu)(deep_feature)
    # 将二阶 FM 的 3D 特征展平后送入 DNN 进行高阶特征交叉学习
    deep_feature = layers.Flatten(name="deep_flatten")(second_order_fm_feature)
    deep_feature = layers.Dense(32, activation='relu', name="deep_dense_1")(deep_feature)
    deep_feature = layers.Dense(16, activation='relu', name="deep_dense_2")(deep_feature)

    # ============================================================
    # FM 二阶交叉特征计算（使用向量化公式优化）
    # ============================================================
    # FM 公式：0.5 * [(Σvi)² - Σ(vi²)]
    # 这里省略 0.5 系数（等价于乘以常数，不影响模型表达能力）
    #
    # 对应原始代码中的：
    #   second_order_sum_feature = ReduceLayer(1)(second_order_fm_feature)
    #   second_order_sum_square_feature = multiply([sum, sum])
    #   second_order_square_feature = multiply([fm, fm])
    #   second_order_square_sum_feature = ReduceLayer(1)(square)
    #   second_order_fm_feature = subtract([sum_square, square_sum])

    # 第一步：对所有字段的 Embedding 沿字段维度（axis=1）求和，得到 (batch, FM_FIELD_DIM)
    second_order_sum_feature = ReduceLayer(axis=1, op='sum', name="fm_reduce_sum")(second_order_fm_feature)

    # 第二步：对求和结果进行平方（逐元素自乘），得到 (batch, FM_FIELD_DIM)
    # (Σvi)² —— 先求和再平方
    second_order_sum_square = layers.Multiply(name="fm_sum_square")(
        [second_order_sum_feature, second_order_sum_feature]
    )

    # 第三步：先对每个字段的 Embedding 进行逐元素平方，得到 (batch, num_fields, FM_FIELD_DIM)
    # vi² —— 先平方
    second_order_square = layers.Multiply(name="fm_element_square")(
        [second_order_fm_feature, second_order_fm_feature]
    )

    # 第四步：对平方后的结果沿字段维度求和，得到 (batch, FM_FIELD_DIM)
    # Σ(vi²) —— 平方后再求和
    second_order_square_sum = ReduceLayer(axis=1, op='sum', name="fm_reduce_square_sum")(second_order_square)

    # 第五步：应用 FM 公式 —— (Σvi)² - Σ(vi²)，得到最终的二阶交叉特征
    # 形状为 (batch, FM_FIELD_DIM)
    fm_second_order_feature = layers.Subtract(name="fm_second_order")(
        [second_order_sum_square, second_order_square_sum]
    )

    # ============================================================
    # 一阶特征处理（First Order Features）
    # ============================================================
    # 对应原始代码中：
    #   first_order_cat_feature = DenseFeatures(cat_columns)(inputs)
    #   first_order_cat_feature = Dense(1)(first_order_cat_feature)
    #   first_order_deep_feature = DenseFeatures(deep_columns)(inputs)
    #   first_order_deep_feature = Dense(1)(first_order_deep_feature)
    #   first_order_feature = Add()([cat, deep])

    # 类别特征一阶：每个类别特征通过 output_dim=1 的 Embedding 学习标量权重
    # 数学上等价于 indicator one-hot 后接线性权重，但更节省内存
    movie_first_order = build_id_embedding(inputs, "movieId", MOVIE_BUCKETS, 1, "fm_first")
    user_first_order = build_id_embedding(inputs, "userId", USER_BUCKETS, 1, "fm_first")
    user_genre_first_order = build_genre_lookup_and_embedding(inputs, "userGenre1", 1, "fm_first")
    movie_genre_first_order = build_genre_lookup_and_embedding(inputs, "movieGenre1", 1, "fm_first")

    # 类别特征一阶求和
    first_order_cat = layers.Add(name="first_order_cat_sum")(
        [movie_first_order, user_first_order, user_genre_first_order, movie_genre_first_order]
    )

    # 数值特征一阶：拼接后通过 Dense(1) 映射为标量
    first_order_numeric_input = layers.Concatenate(name="first_order_numeric_concat")(
        [inputs[name] for name in NUMERIC_FEATURES]
    )
    first_order_deep = layers.Dense(1, activation=None, name="first_order_numeric_dense")(first_order_numeric_input)

    # 合并一阶特征：类别 + 数值
    # 对应 FM 公式中的 w0 + Σ(wi * xi) 部分
    first_order_feature = layers.Add(name="first_order_sum")([first_order_cat, first_order_deep])

    # ============================================================
    # 输出层：拼接所有特征并输出预测结果
    # ============================================================
    # first_order_feature:  (batch, 1)
    # fm_second_order_feature: (batch, FM_FIELD_DIM)
    # deep_feature:         (batch, 16)
    concatenated_outputs = layers.Concatenate(axis=1, name="deepfm_concat")(
        [first_order_feature, fm_second_order_feature, deep_feature]
    )
    output = layers.Dense(1, activation='sigmoid', name="prediction")(concatenated_outputs)

    return tf.keras.Model(inputs=inputs, outputs=output, name="DeepFM_v2_tf220")


# ============================================================
# 训练、验证、测试流程
# ============================================================
def main():
    # ----------------------------------------------------------
    # 加载数据集
    # ----------------------------------------------------------
    train_dataset = get_dataset(train_path, shuffle=True)
    validation_dataset = get_dataset(validation_path, shuffle=False)
    test_dataset = get_dataset(test_path, shuffle=False)

    # ----------------------------------------------------------
    # 构建模型
    # ----------------------------------------------------------
    model = build_deepfm_v2_model()
    model.summary()  # 打印模型结构，便于检查

    # ----------------------------------------------------------
    # 编译模型
    # ----------------------------------------------------------
    # 指定 AUC 的 name 为 "roc_auc"，这样验证集指标名就是 "val_roc_auc"，与早停法监控名一致
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )

    # ----------------------------------------------------------
    # 早停法（EarlyStopping）
    # ----------------------------------------------------------
    # 监控验证集的 ROC AUC，如果连续 5 个 epoch 指标没提升就停止训练，
    # 停止后恢复验证集上表现最好的权重。
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_roc_auc",    # 监控验证集的 AUC
        patience=5,               # 如果连续 5 个 epoch 指标没提升就停止
        mode="max",               # 目标是最大化 AUC
        restore_best_weights=True # 停止后恢复最优权重
    )

    # ----------------------------------------------------------
    # 训练模型
    # ----------------------------------------------------------
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    # ----------------------------------------------------------
    # 在测试集上评估模型
    # ----------------------------------------------------------
    test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
    print(
        "\n\nTest Loss: {}, Test Accuracy: {}, Test ROC AUC: {}, Test PR AUC: {}".format(
            test_loss, test_accuracy, test_roc_auc, test_pr_auc
        )
    )

    # ----------------------------------------------------------
    # 打印部分预测结果
    # ----------------------------------------------------------
    # 只取一个 batch，避免将整个 test_dataset 转成 list 占用过多内存
    for batch_features, batch_labels in test_dataset.take(1):
        predictions = model.predict(batch_features)
        for prediction, label in zip(predictions[:12], batch_labels[:12]):
            label_value = bool(label.numpy().item())
            print(
                "Predicted good rating: {:.2%}".format(prediction[0]),
                " | Actual rating label: ",
                "Good Rating" if label_value else "Bad Rating",
            )

    return history, model


history, model = main()
