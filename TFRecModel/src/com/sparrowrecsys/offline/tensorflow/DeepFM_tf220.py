"""
DeepFM for TensorFlow 2.20.

这个版本是从原始 DeepFM.py 迁移而来，主要适配点：
1. 移除 TensorFlow 2.3 时代的 tf.feature_column 和 DenseFeatures。
2. 使用 Keras 原生层实现 ID Embedding、StringLookup、FM 一阶项、FM 二阶交叉项和 Deep MLP。
3. 数据路径改为 Colab 中 SparrowRecSys 项目的 ByTimeStamp 数据集。
4. 增加验证集，并使用 EarlyStopping 监控 val_roc_auc。

注意：本文件只定义并运行训练流程；当前迁移过程中没有在本地执行训练代码。
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
# 超参数：尽量保持原始 DeepFM.py 的结构和规模
# ============================================================
BATCH_SIZE = 12
EPOCHS = 50
EMBEDDING_DIM = 10
MOVIE_BUCKETS = 1001
USER_BUCKETS = 30001


# ============================================================
# 特征定义
# ============================================================
# 原始 DeepFM.py 中 Deep 部分使用的 7 个数值特征。
# TODO 其实原本进行FeatureEngForRecModel操作之后得到的不止这里用到的这些特征，感觉后续可以尝试引入。
NUMERIC_FEATURES = [
    "releaseYear",
    "movieRatingCount",
    "movieAvgRating",
    "movieRatingStddev",
    "userRatingCount",
    "userAvgRating",
    "userRatingStddev",
]

# 原始 DeepFM.py 中直接参与 Embedding 的两个 ID 特征。
ID_FEATURES = ["movieId", "userId"]

# 原始 DeepFM.py 的 FM 部分只使用 userGenre1 和 movieGenre1 做一阶项与二阶交叉。
GENRE_FEATURES = ["userGenre1", "movieGenre1"]

# make_csv_dataset 只读取模型真正使用的列，避免 Keras 3 中多余列导致输入结构匹配混乱。
SELECTED_COLUMNS = NUMERIC_FEATURES + ID_FEATURES + GENRE_FEATURES + ["label"]

# 电影类型词表，保持原始 DeepFM.py 的枚举值。
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

    for name in NUMERIC_FEATURES:
        model_inputs[name] = tf.expand_dims(tf.cast(features[name], tf.float32), axis=-1)

    for name in ID_FEATURES:
        model_inputs[name] = tf.expand_dims(tf.cast(features[name], tf.int32), axis=-1)

    for name in GENRE_FEATURES:
        value = features[name]
        if value.dtype != tf.string:
            value = tf.strings.as_string(value)
        model_inputs[name] = tf.expand_dims(value, axis=-1)

    label = tf.expand_dims(tf.cast(label, tf.float32), axis=-1)
    return model_inputs, label


def get_dataset(file_path, shuffle=True):
    """
    从 CSV 构建 tf.data.Dataset。

    select_columns 只保留当前 DeepFM 模型需要的列和 label。
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

    这对应原始代码中的：
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

    这对应原始代码中的：
    categorical_column_with_vocabulary_list + embedding_column
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


def dot_product(name, left, right):
    """FM 二阶交叉中的内积，输出形状为 (batch, 1)。"""
    return layers.Dot(axes=1, name=name)([left, right])


# ============================================================
# DeepFM 模型定义
# ============================================================
def build_deepfm_model():
    """
    构建 DeepFM：
    - FM 一阶项：每个类别特征学习一个标量权重，相当于 indicator one-hot 后接线性权重。
    - FM 二阶项：对类别特征的 Embedding 做 pair-wise dot product。
    - Deep 部分：数值特征 + movieId/userId Embedding 拼接后进入 MLP。
    """
    inputs = build_inputs()

    # ---------- ID Embedding：movieId 和 userId ----------
    # 同一份 Embedding 同时用于 FM 二阶交叉和 Deep 部分，保持特征语义一致。
    movie_embedding = build_id_embedding(inputs, "movieId", MOVIE_BUCKETS, EMBEDDING_DIM, "shared")
    user_embedding = build_id_embedding(inputs, "userId", USER_BUCKETS, EMBEDDING_DIM, "shared")

    # ---------- Genre Embedding：userGenre1 和 movieGenre1 ----------
    user_genre_embedding = build_genre_lookup_and_embedding(inputs, "userGenre1", EMBEDDING_DIM, "fm")
    movie_genre_embedding = build_genre_lookup_and_embedding(inputs, "movieGenre1", EMBEDDING_DIM, "fm")

    # ---------- FM 一阶项 ----------
    # 原始实现用 indicator_column 生成 one-hot 后放入最终 Dense。
    # 这里用 output_dim=1 的 Embedding 学习每个类别的线性权重，数学上等价但更节省内存。
    movie_first_order = build_id_embedding(inputs, "movieId", MOVIE_BUCKETS, 1, "fm_first")
    user_first_order = build_id_embedding(inputs, "userId", USER_BUCKETS, 1, "fm_first")
    user_genre_first_order = build_genre_lookup_and_embedding(inputs, "userGenre1", 1, "fm_first")
    movie_genre_first_order = build_genre_lookup_and_embedding(inputs, "movieGenre1", 1, "fm_first")
    fm_first_order = layers.Add(name="fm_first_order_sum")(
        [movie_first_order, user_first_order, user_genre_first_order, movie_genre_first_order]
    )

    # ---------- FM 二阶交叉项 ----------
    # 保留原始 DeepFM.py 中的四组 pair-wise dot product。
    product_item_user = dot_product("fm_second_movie_user", movie_embedding, user_embedding)
    product_item_genre_user_genre = dot_product(
        "fm_second_movieGenre_userGenre",
        movie_genre_embedding,
        user_genre_embedding,
    )
    product_item_genre_user = dot_product(
        "fm_second_movieGenre_user",
        movie_genre_embedding,
        user_embedding,
    )
    product_user_genre_item = dot_product(
        "fm_second_userGenre_movie",
        user_genre_embedding,
        movie_embedding,
    )

    # ---------- Deep 部分 ----------
    # 数值特征已经在 Dataset 中转换为 float32 且形状为 (batch, 1)，这里直接拼接。
    deep_features = [inputs[name] for name in NUMERIC_FEATURES] + [movie_embedding, user_embedding]
    deep = layers.Concatenate(name="deep_feature_concat")(deep_features)
    deep = layers.Dense(64, activation="relu", name="deep_dense_1")(deep)
    deep = layers.Dense(64, activation="relu", name="deep_dense_2")(deep)

    # ---------- 输出层 ----------
    # 拼接 FM 一阶项、FM 二阶项和 Deep 输出，再用 sigmoid 输出二分类概率。
    concat = layers.Concatenate(name="deepfm_concat")(
        [
            fm_first_order,
            product_item_user,
            product_item_genre_user_genre,
            product_item_genre_user,
            product_user_genre_item,
            deep,
        ]
    )
    output = layers.Dense(1, activation="sigmoid", name="prediction")(concat)

    return tf.keras.Model(inputs=inputs, outputs=output, name="DeepFM_tf220")


# ============================================================
# 训练、验证、测试流程
# ============================================================
def main():
    train_dataset = get_dataset(train_path, shuffle=True)
    validation_dataset = get_dataset(validation_path, shuffle=False)
    test_dataset = get_dataset(test_path, shuffle=False)

    model = build_deepfm_model()

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )

    # 早停法：监控验证集 ROC AUC。
    # 因为 compile 中 AUC 的名字是 roc_auc，所以验证集指标名就是 val_roc_auc。
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_roc_auc",
        patience=5,
        mode="max",
        restore_best_weights=True,
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
