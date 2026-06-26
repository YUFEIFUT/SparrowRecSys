"""
DIEN (Deep Interest Evolution Network) for TensorFlow 2.20.

这个版本从原始 DIEN.py 迁移而来，主要适配点：
1. 移除 tf.feature_column 和 DenseFeatures，改用 Keras 原生层。
2. 使用 TensorFlow 2.20 / Keras 3 兼容的 Functional API 写法，避免直接对 KerasTensor 调用裸 TensorFlow 算子。
3. 保留 DIEN 的核心结构：行为序列 Embedding、GRU 兴趣抽取、候选物品 Attention、AUGRU 兴趣演化、辅助负采样损失。
4. 数据路径改为 Google Colab 中 SparrowRecSys 项目的 ByTimeStamp 数据集。
5. 增加验证集，并使用 EarlyStopping 监控 val_roc_auc。

注意：本文件用于 Colab 运行。本地迁移时只做静态 review，不在本地执行训练。
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# 数据路径：Google Colab 中 SparrowRecSys 项目的样本文件
# ============================================================
train_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/trainingSamplesByTimeStamp.csv"
validation_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/validationSamplesByTimeStamp.csv"
test_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/testSamplesByTimeStamp.csv"


# ============================================================
# 超参数：尽量保持与原始 DIEN.py 的结构和规模一致
# ============================================================
BATCH_SIZE = 12
EPOCHS = 50
EMBEDDING_SIZE = 10
RECENT_MOVIES = 5
MOVIE_BUCKETS = 1001
USER_BUCKETS = 30001
AUXILIARY_LOSS_WEIGHT = 0.5


# ============================================================
# 特征定义
# ============================================================
NUMERIC_FEATURES = [
    "movieAvgRating",
    "movieRatingStddev",
    "movieRatingCount",
    "userAvgRating",
    "userRatingStddev",
    "userRatingCount",
    "releaseYear",
]

ID_FEATURES = ["movieId", "userId"]

BEHAVIOR_FEATURES = [
    "userRatedMovie1",
    "userRatedMovie2",
    "userRatedMovie3",
    "userRatedMovie4",
    "userRatedMovie5",
]

# DIEN 辅助损失需要使用 userRatedMovie2-5 的负样本，表示“下一步没有点击/评分的电影”。
NEGATIVE_BEHAVIOR_FEATURES = [
    "negative_userRatedMovie2",
    "negative_userRatedMovie3",
    "negative_userRatedMovie4",
    "negative_userRatedMovie5",
]

# 原始 DIEN.py 的 inputs 定义了 userGenre1-5 和 movieGenre1-3，
# 但真正进入 DenseFeatures 的只有 userGenre1 和 movieGenre1。
GENRE_FEATURES = ["userGenre1", "movieGenre1"]

SELECTED_COLUMNS = (
    NUMERIC_FEATURES
    + ID_FEATURES
    + BEHAVIOR_FEATURES
    + GENRE_FEATURES
    + ["label"]
)

GENRE_VOCAB = [
    "Film-Noir",
    "Action",
    "Adventure",
    "Horror",
    "Romance",
    "War",
    "Comedy",
    "Western",
    "Documentary",
    "Sci-Fi",
    "Drama",
    "Thriller",
    "Crime",
    "Fantasy",
    "Animation",
    "IMAX",
    "Mystery",
    "Children",
    "Musical",
]


# ============================================================
# 数据集加载与负样本生成
# ============================================================
def _sample_negative_movie_ids(positive_movie_ids, rng):
    """
    为一列正样本电影 ID 生成负样本电影 ID。

    负样本范围保持与原始代码一致：0 到 1000。
    如果随机出的负样本刚好等于正样本，就重新采样，确保二者不同。
    """
    positives = positive_movie_ids.astype(np.int32)
    negatives = rng.integers(0, MOVIE_BUCKETS, size=len(positives), dtype=np.int32)

    same_mask = negatives == positives
    while np.any(same_mask):
        negatives[same_mask] = rng.integers(
            0,
            MOVIE_BUCKETS,
            size=int(np.sum(same_mask)),
            dtype=np.int32,
        )
        same_mask = negatives == positives

    return negatives


def _load_dataframe_with_negative_movies(file_path, seed_num):
    """
    读取 CSV，并为 DIEN 的辅助损失构造负样本列。

    原始 DIEN.py 使用 pandas 读取全量 CSV 后生成负样本。
    这里保留这种方式，便于在 Colab 上直接运行，也方便确保每个 epoch 的负样本可复现。
    """
    df = pd.read_csv(file_path, usecols=SELECTED_COLUMNS)
    df = df.fillna(0)

    for name in NUMERIC_FEATURES:
        df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0).astype(np.float32)

    for name in ID_FEATURES + BEHAVIOR_FEATURES:
        df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0).astype(np.int32)

    for name in GENRE_FEATURES:
        df[name] = df[name].astype(str)

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(np.float32)

    rng = np.random.default_rng(seed_num)
    for positive_name, negative_name in zip(BEHAVIOR_FEATURES[1:], NEGATIVE_BEHAVIOR_FEATURES):
        df[negative_name] = _sample_negative_movie_ids(df[positive_name].to_numpy(), rng)

    return df


def get_dataset_with_negative_movies(file_path, batch_size, seed_num, shuffle=True):
    """
    构建 tf.data.Dataset。

    输出格式是 (features, label)，可以直接用于 model.fit / evaluate。
    每个标量特征都整理为 (batch, 1)，与 build_inputs() 中的 Input(shape=(1,)) 对齐。
    """
    df = _load_dataframe_with_negative_movies(file_path, seed_num)

    features = {}
    for name in NUMERIC_FEATURES:
        features[name] = df[name].to_numpy(dtype=np.float32).reshape(-1, 1)

    for name in ID_FEATURES + BEHAVIOR_FEATURES + NEGATIVE_BEHAVIOR_FEATURES:
        features[name] = df[name].to_numpy(dtype=np.int32).reshape(-1, 1)

    for name in GENRE_FEATURES:
        features[name] = df[name].to_numpy(dtype=str).reshape(-1, 1)

    labels = df["label"].to_numpy(dtype=np.float32).reshape(-1, 1)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 10000),
            seed=seed_num,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================
# Keras 输入层与特征编码
# ============================================================
def build_inputs():
    """为每个模型特征创建命名 Input，名称必须与 Dataset 字典中的 key 一致。"""
    inputs = {}

    for name in NUMERIC_FEATURES:
        inputs[name] = layers.Input(name=name, shape=(1,), dtype=tf.float32)

    for name in ID_FEATURES + BEHAVIOR_FEATURES + NEGATIVE_BEHAVIOR_FEATURES:
        inputs[name] = layers.Input(name=name, shape=(1,), dtype=tf.int32)

    for name in GENRE_FEATURES:
        inputs[name] = layers.Input(name=name, shape=(1,), dtype=tf.string)

    return inputs


def build_id_embedding(inputs, feature_name, num_buckets, output_dim, layer_prefix):
    """
    将整数 ID 转换为稠密 Embedding。

    对应原始代码中的 categorical_column_with_identity + embedding_column。
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

    对应原始代码中的 categorical_column_with_vocabulary_list + embedding_column。
    """
    genre_index = layers.StringLookup(
        vocabulary=GENRE_VOCAB,
        mask_token=None,
        num_oov_indices=1,
        name=f"{layer_prefix}_{feature_name}_lookup",
    )(inputs[feature_name])
    embedding = layers.Embedding(
        input_dim=len(GENRE_VOCAB) + 1,
        output_dim=output_dim,
        name=f"{layer_prefix}_{feature_name}_embedding",
    )(genre_index)
    return layers.Flatten(name=f"{layer_prefix}_{feature_name}_flatten")(embedding)


# ============================================================
# DIEN 自定义层
# ============================================================
class AttentionScore(layers.Layer):
    """
    DIEN 的 Activation Unit。

    输入：
      - candidate_embedding: 候选电影 Embedding，形状 (batch, emb)
      - gru_hidden_states: GRU 输出的行为隐藏状态，形状 (batch, time, emb)

    输出：
      - attention_score: 每个时间步对应的兴趣相关性权重，形状 (batch, time, emb)
    """

    def __init__(self, embedding_size=EMBEDDING_SIZE, time_length=RECENT_MOVIES, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.time_length = time_length
        self.repeat_time = layers.RepeatVector(time_length)
        self.repeat_embedding = layers.RepeatVector(embedding_size)
        self.multiply = layers.Multiply()
        self.dense_32 = layers.Dense(32, activation="sigmoid")
        self.dense_1 = layers.Dense(1, activation="sigmoid")
        self.permute = layers.Permute((2, 1))

    def call(self, inputs):
        candidate_embedding, gru_hidden_states = inputs

        repeated_candidate = self.repeat_time(candidate_embedding)
        activation_product = self.multiply([gru_hidden_states, repeated_candidate])
        attention = self.dense_32(activation_product)
        attention = self.dense_1(attention)
        attention = tf.squeeze(attention, axis=2)
        attention = self.repeat_embedding(attention)
        attention = self.permute(attention)

        return attention


class GRUGateParameter(layers.Layer):
    """
    AUGRU 中的门控参数层。

    use_reset_gate=False 时，计算 sigmoid 门控值。
    use_reset_gate=True 时，计算候选隐藏状态 h_tilde，隐藏状态会先乘以 reset gate。
    """

    def __init__(self, embedding_size=EMBEDDING_SIZE, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.activation = activation
        self.input_dense = layers.Dense(embedding_size, activation=None, use_bias=True)
        self.hidden_dense = layers.Dense(embedding_size, activation=None, use_bias=False)
        self.multiply = layers.Multiply()

    def call(self, inputs, reset_gate=None):
        gru_input, previous_hidden_state = inputs
        hidden_state = previous_hidden_state

        if reset_gate is not None:
            hidden_state = self.multiply([hidden_state, reset_gate])

        gate_value = self.input_dense(gru_input) + self.hidden_dense(hidden_state)

        if self.activation == "tanh":
            return tf.tanh(gate_value)
        return tf.sigmoid(gate_value)


class AUGRU(layers.Layer):
    """
    Attention based GRU。

    DIEN 使用注意力分数调节 GRU 的 update gate：
        update_gate = attention_score * z_t
        h_t = (1 - update_gate) * h_{t-1} + update_gate * h_tilde
    """

    def __init__(self, embedding_size=EMBEDDING_SIZE, time_length=RECENT_MOVIES, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.time_length = time_length
        self.reset_gate = GRUGateParameter(embedding_size, activation="sigmoid")
        self.update_gate = GRUGateParameter(embedding_size, activation="sigmoid")
        self.candidate_state = GRUGateParameter(embedding_size, activation="tanh")
        self.multiply = layers.Multiply()
        self.add = layers.Add()

    def call(self, inputs):
        gru_hidden_states, attention_scores = inputs
        batch_size = tf.shape(gru_hidden_states)[0]
        augru_hidden_state = tf.zeros((batch_size, self.embedding_size), dtype=gru_hidden_states.dtype)

        for time_index in range(self.time_length):
            current_hidden_state = gru_hidden_states[:, time_index, :]
            current_attention = attention_scores[:, time_index, :]

            r_t = self.reset_gate([current_hidden_state, augru_hidden_state])
            z_t = self.update_gate([current_hidden_state, augru_hidden_state])
            h_tilde = self.candidate_state(
                [current_hidden_state, augru_hidden_state],
                reset_gate=r_t,
            )

            attentional_update_gate = self.multiply([current_attention, z_t])
            keep_previous = self.multiply([(1.0 - attentional_update_gate), augru_hidden_state])
            use_current = self.multiply([attentional_update_gate, h_tilde])
            augru_hidden_state = self.add([keep_previous, use_current])

        return augru_hidden_state


class AuxiliaryLossLayer(layers.Layer):
    """
    DIEN 的辅助损失层。

    GRU 在每个时间步的隐藏状态应该能预测“下一部真实行为电影”，同时区分随机负样本电影。
    当下一部电影 ID 为 0 时，表示缺失/填充位置，不参与辅助损失计算。
    这个损失通过 add_loss 加入 model 的总训练损失，不改变模型最终输出。
    """

    def __init__(self, alpha=AUXILIARY_LOSS_WEIGHT, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.positive_dense_32 = layers.Dense(32, activation="sigmoid")
        self.positive_dense_1 = layers.Dense(1, activation="sigmoid")
        self.negative_dense_32 = layers.Dense(32, activation="sigmoid")
        self.negative_dense_1 = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        (
            negative_movie_embedding,
            behavior_embedding,
            behavior_ids,
            gru_hidden_states,
            pass_through,
        ) = inputs

        # 使用前 4 个隐藏状态预测后 4 个真实行为：h_1->movie_2, ..., h_4->movie_5。
        hidden_states = gru_hidden_states[:, :-1, :]
        positive_next_movies = behavior_embedding[:, 1:, :]
        positive_next_movie_ids = behavior_ids[:, 1:]

        positive_features = tf.concat([hidden_states, positive_next_movies], axis=-1)
        positive_scores = self.positive_dense_32(positive_features)
        positive_scores = self.positive_dense_1(positive_scores)

        negative_features = tf.concat([hidden_states, negative_movie_embedding], axis=-1)
        negative_scores = self.negative_dense_32(negative_features)
        negative_scores = self.negative_dense_1(negative_scores)

        positive_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(positive_scores),
            positive_scores,
        )
        negative_loss = tf.keras.losses.binary_crossentropy(
            tf.zeros_like(negative_scores),
            negative_scores,
        )

        # 只在真实存在下一步行为的位置计算辅助损失，跳过 ID=0 的 padding。
        valid_mask = tf.cast(tf.not_equal(positive_next_movie_ids, 0), positive_loss.dtype)
        weighted_loss = (positive_loss + negative_loss) * valid_mask
        auxiliary_loss = tf.reduce_sum(weighted_loss) / (tf.reduce_sum(valid_mask) + 1e-7)
        self.add_loss(self.alpha * auxiliary_loss)

        return pass_through


# ============================================================
# DIEN 模型定义
# ============================================================
def build_dien_model():
    """构建 TensorFlow 2.20 / Keras 3 兼容的 DIEN 模型。"""
    inputs = build_inputs()

    # ----------------------------------------------------------
    # 共享电影 Embedding：候选电影、用户历史行为、负样本电影共用同一个 Embedding 空间。
    # ----------------------------------------------------------
    shared_movie_embedding = layers.Embedding(
        input_dim=MOVIE_BUCKETS,
        output_dim=EMBEDDING_SIZE,
        mask_zero=True,
        name="shared_movie_embedding",
    )

    candidate_embedding = shared_movie_embedding(inputs["movieId"])
    candidate_embedding = layers.Flatten(name="candidate_movie_flatten")(candidate_embedding)

    behavior_ids = layers.Concatenate(axis=1, name="behavior_ids_concat")(
        [inputs[name] for name in BEHAVIOR_FEATURES]
    )
    behavior_embedding = shared_movie_embedding(behavior_ids)

    negative_behavior_ids = layers.Concatenate(axis=1, name="negative_behavior_ids_concat")(
        [inputs[name] for name in NEGATIVE_BEHAVIOR_FEATURES]
    )
    negative_behavior_embedding = shared_movie_embedding(negative_behavior_ids)

    # ----------------------------------------------------------
    # 兴趣抽取层：用 GRU 从历史行为序列中抽取用户兴趣状态。
    # ----------------------------------------------------------
    behavior_hidden_states = layers.GRU(
        EMBEDDING_SIZE,
        return_sequences=True,
        # mask_zero=True 会把 movieId=0 传成 RNN mask。
        # 行为序列中的 0 不一定都是严格右侧 padding，cuDNN GRU 不支持这种 mask，
        # 因此显式关闭 cuDNN 路径，避免在 Colab GPU 上触发断言失败。
        use_cudnn=False,
        name="interest_extractor_gru",
    )(behavior_embedding)

    # ----------------------------------------------------------
    # 兴趣演化层：候选电影参与 attention，AUGRU 输出与候选电影相关的演化兴趣。
    # ----------------------------------------------------------
    attention_scores = AttentionScore(name="attention_score")(
        [candidate_embedding, behavior_hidden_states]
    )
    augru_embedding = AUGRU(name="augru_interest_evolution")(
        [behavior_hidden_states, attention_scores]
    )

    # 将辅助损失挂到 augru_embedding 的计算路径上，确保 model.fit 时会纳入总 loss。
    augru_embedding = AuxiliaryLossLayer(name="auxiliary_loss")(
        [
            negative_behavior_embedding,
            behavior_embedding,
            behavior_ids,
            behavior_hidden_states,
            augru_embedding,
        ]
    )

    # ----------------------------------------------------------
    # 用户画像特征：userId Embedding + userGenre1 Embedding + 用户统计特征。
    # ----------------------------------------------------------
    user_embedding = build_id_embedding(inputs, "userId", USER_BUCKETS, EMBEDDING_SIZE, "user")
    user_genre_embedding = build_genre_lookup_and_embedding(
        inputs,
        "userGenre1",
        EMBEDDING_SIZE,
        "user",
    )
    user_numeric = layers.Concatenate(name="user_numeric_concat")(
        [
            inputs["userRatingCount"],
            inputs["userAvgRating"],
            inputs["userRatingStddev"],
        ]
    )
    user_profile_layer = layers.Concatenate(name="user_profile_concat")(
        [user_embedding, user_genre_embedding, user_numeric]
    )

    # ----------------------------------------------------------
    # 上下文特征：movieGenre1 Embedding + 电影统计特征。
    # ----------------------------------------------------------
    movie_genre_embedding = build_genre_lookup_and_embedding(
        inputs,
        "movieGenre1",
        EMBEDDING_SIZE,
        "context",
    )
    context_numeric = layers.Concatenate(name="context_numeric_concat")(
        [
            inputs["releaseYear"],
            inputs["movieRatingCount"],
            inputs["movieAvgRating"],
            inputs["movieRatingStddev"],
        ]
    )
    context_features_layer = layers.Concatenate(name="context_features_concat")(
        [movie_genre_embedding, context_numeric]
    )

    # ----------------------------------------------------------
    # 全连接预测层：演化兴趣 + 候选电影 + 用户画像 + 上下文特征。
    # ----------------------------------------------------------
    concat_layer = layers.Concatenate(name="dien_concat")(
        [
            augru_embedding,
            candidate_embedding,
            user_profile_layer,
            context_features_layer,
        ]
    )

    output_layer = layers.Dense(128, name="fc_dense_1")(concat_layer)
    output_layer = layers.PReLU(name="fc_prelu_1")(output_layer)
    output_layer = layers.Dense(64, name="fc_dense_2")(output_layer)
    output_layer = layers.PReLU(name="fc_prelu_2")(output_layer)
    output_layer = layers.Dense(1, activation="sigmoid", name="prediction")(output_layer)

    return tf.keras.Model(inputs=inputs, outputs=output_layer, name="DIEN_tf220")


# ============================================================
# 训练、验证、测试流程
# ============================================================
def main():
    train_dataset = get_dataset_with_negative_movies(
        train_path,
        BATCH_SIZE,
        seed_num=2020,
        shuffle=True,
    )
    validation_dataset = get_dataset_with_negative_movies(
        validation_path,
        BATCH_SIZE,
        seed_num=2021,
        shuffle=False,
    )
    test_dataset = get_dataset_with_negative_movies(
        test_path,
        BATCH_SIZE,
        seed_num=2022,
        shuffle=False,
    )

    model = build_dien_model()
    model.summary()

    # AUC 指标显式命名为 roc_auc，因此验证集指标名就是 val_roc_auc。
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_roc_auc",     # 监控验证集的 AUC
        patience=5,                # 如果连续 5 个 epoch 指标没提升就停止
        mode="max",                # 目标是最大化 AUC
        restore_best_weights=True, # 停止后恢复最优权重
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
    print(
        "\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}".format(
            test_loss,
            test_accuracy,
            test_roc_auc,
            test_pr_auc,
        )
    )

    # 打印一个 batch 的预测结果，避免把整个 test_dataset 转成 list 占用过多内存。
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
