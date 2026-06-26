"""
DIN (Deep Interest Network) for TensorFlow 2.20

从 DIN.py（TF 2.3）迁移而来。

DIN 的核心思想：
  用户的历史行为序列（最近看过的电影）对预测候选电影的点击率有不同的重要性。
  DIN 使用 Attention 机制（Activation Unit）来计算候选电影与每部历史电影的相似度，
  以此对历史行为 Embedding 进行加权求和，突出与候选电影相关的历史行为。

模型结构：
  ┌──────────────────────────────────────────────────────────────────────┐
  │  候选电影：movieId → 共享 Movie Embedding → (batch, emb_dim)         │
  │  历史行为：userRatedMovie1-5 → 共享 Movie Embedding → (batch, 5, emb) │
  │                                                                      │
  │  Activation Unit（注意力机制）：                                       │
  │    [历史行为 - 候选, 历史行为, 候选, 历史行为 * 候选] → Dense → σ → 权重 │
  │    加权求和历史行为 → (batch, emb_dim)                                │
  │                                                                      │
  │  用户画像：userId Emb + userGenre1 Emb + 3 数值特征                   │
  │  上下文特征：movieGenre1 Emb + 4 数值特征                             │
  │                                                                      │
  │  全连接层：Concat → Dense(128,PReLU) → Dense(64,PReLU) → sigmoid     │
  └──────────────────────────────────────────────────────────────────────┘

TF 2.20 适配要点：
  - 移除 tf.feature_column / DenseFeatures（Keras 3 已不再内置）。
  - 用 Keras 原生层（Embedding、StringLookup、Dense）替代。
  - Lambda 层内部使用 tf.reduce_sum（替代 tf.keras.backend.sum）。

数据路径：Google Colab 中 SparrowRecSys 项目的 ByTimeStamp 数据集。
"""

import tensorflow as tf
from tensorflow.keras import layers
import keras.ops as ops


# ============================================================
# 数据路径：Google Colab 中 SparrowRecSys 项目的样本文件
# ============================================================
train_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/trainingSamplesByTimeStamp.csv"
validation_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/validationSamplesByTimeStamp.csv"
test_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/testSamplesByTimeStamp.csv"


# ============================================================
# 超参数
# ============================================================
BATCH_SIZE = 12         # 每个 batch 的样本数，保持与原始代码一致
EPOCHS = 50             # 最大训练轮次（配合早停法，实际轮次可能更少）
EMBEDDING_SIZE = 10     # Embedding 维度，保持与原始代码一致
RECENT_MOVIES = 5       # 用户历史行为序列长度（userRatedMovie{1-5}）
MOVIE_BUCKETS = 1001    # movieId 的桶数量（ID 范围 0~1000）
USER_BUCKETS = 30001    # userId 的桶数量（ID 范围 0~30000）


# ============================================================
# 特征定义
# ============================================================
# 数值特征：原始 DIN.py 中用于 user_profile 和 context_features 的 7 个数值特征
NUMERIC_FEATURES = [
    "releaseYear",          # 电影上映年份
    "movieRatingCount",     # 电影被评分次数
    "movieAvgRating",       # 电影平均评分
    "movieRatingStddev",    # 电影评分标准差
    "userRatingCount",      # 用户评分次数
    "userAvgRating",        # 用户平均评分
    "userRatingStddev",     # 用户评分标准差
]

# ID 特征
ID_FEATURES = ["movieId", "userId"]

# 用户行为序列：用户最近评分过的 5 部电影的 ID
# DIN 的核心输入——用于构建用户兴趣序列
BEHAVIOR_FEATURES = [
    "userRatedMovie1",
    "userRatedMovie2",
    "userRatedMovie3",
    "userRatedMovie4",
    "userRatedMovie5",
]

# 类型特征：原始 DIN.py 中实际参与模型计算的 genre 特征
# 注意：原始代码的 inputs dict 中定义了 userGenre1-5 和 movieGenre1-3，
# 但只有 userGenre1 和 movieGenre1 实际被 DenseFeatures 使用。
# 这里只保留实际使用的列，未使用的 genre 列（userGenre2-5, movieGenre2-3）
# 不参与模型构建，避免 Keras 3 中多余的 Input 导致输入结构匹配混乱。
GENRE_FEATURES = ["userGenre1", "movieGenre1"]

# make_csv_dataset 只读取模型真正使用的列
SELECTED_COLUMNS = NUMERIC_FEATURES + ID_FEATURES + BEHAVIOR_FEATURES + GENRE_FEATURES + ["label"]

# 电影类型词表，保持原始 DIN.py 的枚举值
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

    # 数值特征：转为 float32，扩展维度 → (batch, 1)
    for name in NUMERIC_FEATURES:
        model_inputs[name] = tf.expand_dims(tf.cast(features[name], tf.float32), axis=-1)

    # ID 特征：转为 int32，扩展维度 → (batch, 1)
    for name in ID_FEATURES:
        model_inputs[name] = tf.expand_dims(tf.cast(features[name], tf.int32), axis=-1)

    # 用户行为序列：转为 int32，扩展维度 → (batch, 1)
    # userRatedMovie1-5 存储的是电影 ID（整数），不是评分
    for name in BEHAVIOR_FEATURES:
        model_inputs[name] = tf.expand_dims(tf.cast(features[name], tf.int32), axis=-1)

    # 类型特征：确保为 string 类型，扩展维度 → (batch, 1)
    for name in GENRE_FEATURES:
        value = features[name]
        if value.dtype != tf.string:
            value = tf.strings.as_string(value)
        model_inputs[name] = tf.expand_dims(value, axis=-1)

    # 标签：转为 float32，扩展维度 → (batch, 1)
    label = tf.expand_dims(tf.cast(label, tf.float32), axis=-1)
    return model_inputs, label


def get_dataset(file_path, shuffle=True):
    """
    从 CSV 构建 tf.data.Dataset。

    select_columns 只保留当前 DIN 模型需要的列和 label。
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
    """
    为每个模型特征创建一个命名 Input，名称必须与 Dataset 字典中的 key 一致。

    与原始 DIN.py 的差异：
      - 原始代码定义了 userGenre1-5 和 movieGenre1-3 的 Input，但只有 userGenre1
        和 movieGenre1 实际被 DenseFeatures 使用，其余是"幽灵输入"。
      - 这里只定义模型实际使用的输入，更干净且与 SELECTED_COLUMNS 对齐。
    """
    inputs = {}

    # 数值特征
    for name in NUMERIC_FEATURES:
        inputs[name] = layers.Input(name=name, shape=(1,), dtype=tf.float32)

    # ID 特征
    for name in ID_FEATURES:
        inputs[name] = layers.Input(name=name, shape=(1,), dtype=tf.int32)

    # 用户行为序列（电影 ID）
    for name in BEHAVIOR_FEATURES:
        inputs[name] = layers.Input(name=name, shape=(1,), dtype=tf.int32)

    # 类型特征
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
# DIN 模型定义
# ============================================================
def build_din_model():
    """
    构建 DIN (Deep Interest Network) 模型。

    DIN 的关键创新是 Activation Unit（注意力机制）：
      候选电影 Embedding 与每部历史电影 Embedding 一起输入一个小网络，
      计算出每部历史电影的"注意力权重"，然后对历史行为 Embedding 做加权求和。

    与原始 DIN.py 的结构完全对应，只是特征处理层从 DenseFeatures 改为 Keras 原生层。
    """
    inputs = build_inputs()

    # ============================================================
    # 共享 Movie Embedding 层
    # ============================================================
    # 候选电影和用户历史行为中的电影使用同一个 Embedding 空间。
    # 这样候选电影和历史电影的 Embedding 可以直接做减法/乘法等运算。
    # 对应原始代码：
    #   movie_emb_layer = Embedding(input_dim=1001, output_dim=EMBEDDING_SIZE, mask_zero=True)
    #   user_behaviors_emb_layer = movie_emb_layer(user_behaviors_layer)
    #   candidate_emb_layer = movie_emb_layer(candidate_layer)
    shared_movie_embedding = layers.Embedding(
        input_dim=MOVIE_BUCKETS,
        output_dim=EMBEDDING_SIZE,
        mask_zero=True,  # 掩码 ID=0 的位置（表示缺失的用户历史电影）
        name="shared_movie_embedding",
    )

    # ============================================================
    # 候选电影 Embedding
    # ============================================================
    # 对应原始代码：
    #   candidate_layer = DenseFeatures(candidate_movie_col)(inputs)  # movieId as numeric
    #   candidate_emb_layer = movie_emb_layer(candidate_layer)
    #   candidate_emb_layer = tf.squeeze(candidate_emb_layer, axis=1)
    # 输入 movieId → (batch, 1) → Embedding → (batch, 1, emb) → squeeze → (batch, emb)
    candidate_emb = shared_movie_embedding(inputs["movieId"])
    candidate_emb = ops.squeeze(candidate_emb, axis=1)  # 移除序列维度 → (batch, EMBEDDING_SIZE)

    # ============================================================
    # 用户行为序列 Embedding
    # ============================================================
    # 对应原始代码：
    #   user_behaviors_layer = DenseFeatures(recent_rate_col)(inputs)  # userRatedMovie1-5
    #   user_behaviors_emb_layer = movie_emb_layer(user_behaviors_layer)
    # 将 userRatedMovie1-5 沿 axis=1 拼接为 (batch, 5) 的张量
    behavior_ids = layers.Concatenate(axis=1, name="behavior_ids_concat")(
        [inputs[name] for name in BEHAVIOR_FEATURES]
    )
    # Embedding 查找 → (batch, 5, EMBEDDING_SIZE)
    user_behaviors_emb = shared_movie_embedding(behavior_ids)

    # ============================================================
    # Activation Unit（注意力机制）—— DIN 的核心
    # ============================================================
    # 对应原始代码中的注意力计算部分：
    #   repeated_candidate = RepeatVector(5)(candidate_emb)  # → (batch, 5, emb)
    #   sub = Subtract()([behaviors, repeated_candidate])
    #   mul = Multiply()([behaviors, repeated_candidate])
    #   concat = concatenate([sub, behaviors, repeated_candidate, mul])
    #   attn = Dense(32)(concat) → PReLU → Dense(1, sigmoid) → Flatten
    #   attn = RepeatVector(emb)(attn) → Permute → (batch, 5, emb)
    #   weighted = Multiply()([behaviors, attn])
    #   pooled = Lambda(sum, axis=1)(weighted)

    # 将候选电影 Embedding 扩展到与行为序列相同的序列长度
    # (batch, emb) → (batch, RECENT_MOVIES, emb)
    repeated_candidate = layers.RepeatVector(RECENT_MOVIES, name="repeat_candidate")(candidate_emb)

    # 计算候选与历史行为的差异和交互
    # 对应原始代码中的 Subtract 和 Multiply
    activation_sub = layers.Subtract(name="activation_subtract")(
        [user_behaviors_emb, repeated_candidate]
    )  # 差异特征：历史行为与候选的逐元素差 → (batch, 5, emb)
    activation_product = layers.Multiply(name="activation_multiply")(
        [user_behaviors_emb, repeated_candidate]
    )  # 交互特征：历史行为与候选的逐元素积 → (batch, 5, emb)

    # 拼接注意力网络的输入：4 组特征
    # [差异, 历史行为, 候选, 交互] → (batch, 5, 4 * emb)
    activation_all = layers.Concatenate(axis=-1, name="activation_concat")(
        [activation_sub, user_behaviors_emb, repeated_candidate, activation_product]
    )

    # 注意力网络：两层全连接，输出每个历史行为的注意力权重
    # 对应原始代码：
    #   Dense(32) → PReLU → Dense(1, sigmoid) → Flatten
    activation_unit = layers.Dense(32, name="attention_dense")(activation_all)
    activation_unit = layers.PReLU(name="attention_prelu")(activation_unit)
    activation_unit = layers.Dense(1, activation='sigmoid', name="attention_weight")(activation_unit)
    activation_unit = layers.Flatten(name="attention_flatten")(activation_unit)
    # (batch, 5) → 每个历史行为的注意力权重

    # 将注意力权重扩展为与 Embedding 维度相同的形状，用于逐元素加权
    # 对应原始代码：
    #   RepeatVector(EMBEDDING_SIZE)(activation_unit)  # (batch, emb, 5)
    #   Permute((2, 1))                                 # (batch, 5, emb)
    activation_unit = layers.RepeatVector(EMBEDDING_SIZE, name="attention_repeat")(activation_unit)
    activation_unit = layers.Permute((2, 1), name="attention_permute")(activation_unit)
    # → (batch, 5, EMBEDDING_SIZE)

    # 对历史行为 Embedding 进行注意力加权
    # 对应原始代码：Multiply()([user_behaviors_emb_layer, activation_unit])
    weighted_behaviors = layers.Multiply(name="attention_weighted_behaviors")(
        [user_behaviors_emb, activation_unit]
    )  # (batch, 5, EMBEDDING_SIZE)

    # 对加权后的历史行为沿序列维度求和，得到用户兴趣的聚合表示
    # 对应原始代码：Lambda(lambda x: tf.keras.backend.sum(x, axis=1))
    # 使用 tf.reduce_sum 替代 tf.keras.backend.sum（后者在 Keras 3 中已弃用）
    user_behaviors_pooled = layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        name="behavior_sum_pooling",
    )(weighted_behaviors)
    # → (batch, EMBEDDING_SIZE)

    # ============================================================
    # 用户画像特征（User Profile）
    # ============================================================
    # 对应原始代码：
    #   user_profile = [user_emb_col, user_genre_emb_col,
    #                   numeric('userRatingCount'), numeric('userAvgRating'), numeric('userRatingStddev')]
    #   user_profile_layer = DenseFeatures(user_profile)(inputs)
    user_embedding = build_id_embedding(inputs, "userId", USER_BUCKETS, EMBEDDING_SIZE, "user")
    user_genre_embedding = build_genre_lookup_and_embedding(inputs, "userGenre1", EMBEDDING_SIZE, "user")
    user_numeric = layers.Concatenate(name="user_numeric_concat")(
        [inputs["userRatingCount"], inputs["userAvgRating"], inputs["userRatingStddev"]]
    )
    # 拼接用户画像：userId Emb + userGenre1 Emb + 3 数值特征
    user_profile_layer = layers.Concatenate(name="user_profile_concat")(
        [user_embedding, user_genre_embedding, user_numeric]
    )

    # ============================================================
    # 上下文特征（Context Features）
    # ============================================================
    # 对应原始代码：
    #   context_features = [item_genre_emb_col,
    #                       numeric('releaseYear'), numeric('movieRatingCount'),
    #                       numeric('movieAvgRating'), numeric('movieRatingStddev')]
    #   context_features_layer = DenseFeatures(context_features)(inputs)
    movie_genre_embedding = build_genre_lookup_and_embedding(inputs, "movieGenre1", EMBEDDING_SIZE, "context")
    context_numeric = layers.Concatenate(name="context_numeric_concat")(
        [inputs["releaseYear"], inputs["movieRatingCount"],
         inputs["movieAvgRating"], inputs["movieRatingStddev"]]
    )
    # 拼接上下文特征：movieGenre1 Emb + 4 数值特征
    context_features_layer = layers.Concatenate(name="context_features_concat")(
        [movie_genre_embedding, context_numeric]
    )

    # ============================================================
    # 全连接层（FC Layers）
    # ============================================================
    # 拼接所有特征：
    #   用户画像 + 行为聚合 + 候选电影 Embedding + 上下文特征
    # 对应原始代码：
    #   concat_layer = concatenate([user_profile_layer, user_behaviors_pooled_layers,
    #                               candidate_emb_layer, context_features_layer])
    concat_layer = layers.Concatenate(name="din_concat")(
        [user_profile_layer, user_behaviors_pooled, candidate_emb, context_features_layer]
    )

    # 两层全连接网络，使用 PReLU 激活函数（与原始代码一致）
    # 对应原始代码：
    #   Dense(128) → PReLU → Dense(64) → PReLU → Dense(1, sigmoid)
    output_layer = layers.Dense(128, name="fc_dense_1")(concat_layer)
    output_layer = layers.PReLU(name="fc_prelu_1")(output_layer)
    output_layer = layers.Dense(64, name="fc_dense_2")(output_layer)
    output_layer = layers.PReLU(name="fc_prelu_2")(output_layer)
    output_layer = layers.Dense(1, activation='sigmoid', name="prediction")(output_layer)

    return tf.keras.Model(inputs=inputs, outputs=output_layer, name="DIN_tf220")


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
    model = build_din_model()
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


if __name__ == "__main__":
    main()
