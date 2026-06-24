"""
Wide & Deep 模型 —— 适配 TensorFlow 2.20.0 (Keras 3) 的版本。

为什么要重写？
----------------
原始 WideNDeep.py 是 TF 2.3 时代的代码，大量依赖 `tf.feature_column`
(categorical_column_with_vocabulary_list / embedding_column / crossed_column /
indicator_column 等) 以及 `tf.keras.layers.DenseFeatures`。

从 TF 2.16 起，Keras 默认升级为 Keras 3，`tf.feature_column` 和
`DenseFeatures` 已经被彻底移除。所以在 TF 2.20.0 下原代码无法运行。

本文件用 Keras 预处理层(Preprocessing Layers)重新实现等价逻辑：
    - categorical_column_with_vocabulary_list + embedding_column
        -> StringLookup + Embedding
    - categorical_column_with_identity + embedding_column
        -> 直接用 Embedding (ID 本身就是整数索引)
    - crossed_column + indicator_column
        -> HashedCrossing(output_mode='one_hot')
    - numeric_column + DenseFeatures
        -> 直接把数值 Input 拼接起来

模型结构(Wide & Deep)、特征、超参数都与原版保持一致。
"""

import os
import tensorflow as tf

# ============ 样本数据路径 ============
# 改成你本地的 trainingSamples.csv / testSamples.csv 路径。
# 这里默认指向本仓库内的 sampledata 目录。
# 从本文件(.../TFRecModel/src/com/sparrowrecsys/offline/tensorflow/) 往上 6 级即仓库根目录
training_samples_file_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/trainingSamplesByTimeStamp.csv"
validation_samples_file_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/validationSamplesByTimeStamp.csv"
test_samples_file_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/testSamplesByTimeStamp.csv"


# ============ 只选用模型真正用到的列 ============
# 关键修复(Keras 3 / TF 2.20)：
# CSV 一共有 26 列，但本模型只定义了 18 个 Input。Keras 3 的函数式模型在
# 传入的 dict 结构与 Input 结构不一致时，会从“按列名映射”退化成“按顺序映射”，
# 于是 float 列(如 rating)被错误地喂进了 string 类型的 genre Input，
# 报错 "Cast float to string is not supported"。
# 通过 select_columns 把 dataset 限制为模型用到的列(+label)，
# 使 dict 的 key 与模型 Input 名一一对应，按名字正确映射。
SELECTED_COLUMNS = [
    'movieId', 'userId', 'releaseYear',
    'movieGenre1', 'movieGenre2', 'movieGenre3',
    'movieRatingCount', 'movieAvgRating', 'movieRatingStddev',
    'userRatedMovie1',
    'userRatingCount', 'userAvgRating', 'userRatingStddev',
    'userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5',
    'label',
]


# 每个特征对应的 dtype（与下面模型 Input 的 dtype 严格一致）
INT_FEATURES = ['movieRatingCount', 'userRatingCount', 'releaseYear',
                'movieId', 'userId', 'userRatedMovie1']
FLOAT_FEATURES = ['movieAvgRating', 'movieRatingStddev',
                  'userAvgRating', 'userRatingStddev']
STRING_FEATURES = ['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5',
                   'movieGenre1', 'movieGenre2', 'movieGenre3']


def _reorder_and_cast(features, label):
    """关键修复(Keras 3 / TF 2.20)：
    make_csv_dataset 返回的是 OrderedDict，且列顺序跟随 CSV header，
    与模型 Input 的定义顺序不同。Keras 3 在结构(OrderedDict vs dict、
    顺序不一致)不匹配时，会从“按列名映射”退化成“按位置映射”，
    导致 string 的 genre 列被喂进 int 的 ID Input，报
    'Cast string to int32 / float to string is not supported'。

    这里显式重建一个普通 dict，key 与模型 Input 一一对应，
    并把每列 cast 到模型期望的 dtype，彻底消除顺序/类型歧义。
    """
    out = {}
    # 同时把每列从 (batch,) expand 成 (batch, 1)，与 Input(shape=(1,)) 对齐
    for name in INT_FEATURES:
        out[name] = tf.expand_dims(tf.cast(features[name], tf.int32), axis=-1)
    for name in FLOAT_FEATURES:
        out[name] = tf.expand_dims(tf.cast(features[name], tf.float32), axis=-1)
    for name in STRING_FEATURES:
        col = features[name]
        if col.dtype != tf.string:
            col = tf.strings.as_string(col)
        out[name] = tf.expand_dims(col, axis=-1)
    return out, label


# ============ 加载样本为 tf.data.Dataset ============
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        select_columns=SELECTED_COLUMNS,
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    # 重排列顺序 + 统一 dtype，保证与模型 Input 按名字正确映射
    dataset = dataset.map(_reorder_and_cast)
    return dataset


train_dataset = get_dataset(training_samples_file_path)
validation_dataset = get_dataset(validation_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

# ============ genre(电影类型)词表 ============
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller',
               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

GENRE_FEATURES = {
    'userGenre1': genre_vocab,
    'userGenre2': genre_vocab,
    'userGenre3': genre_vocab,
    'userGenre4': genre_vocab,
    'userGenre5': genre_vocab,
    'movieGenre1': genre_vocab,
    'movieGenre2': genre_vocab,
    'movieGenre3': genre_vocab
}

# ============ 定义 Keras 模型的输入层 ============
# 在 Keras 3 中，预处理层直接接收 Input 张量。
# 这里 shape=(1,) 表示每个样本一个标量(make_csv_dataset 给出的每列形状是 (batch,)，
# Keras 会自动对齐到 (batch, 1))。
inputs = {
    'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(1,), dtype='float32'),
    'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(1,), dtype='float32'),
    'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(1,), dtype='int32'),
    'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(1,), dtype='float32'),
    'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(1,), dtype='float32'),
    'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(1,), dtype='int32'),
    'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(1,), dtype='int32'),

    'movieId': tf.keras.layers.Input(name='movieId', shape=(1,), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(1,), dtype='int32'),
    'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(1,), dtype='int32'),

    'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(1,), dtype='string'),
    'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(1,), dtype='string'),
    'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(1,), dtype='string'),
    'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(1,), dtype='string'),
    'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(1,), dtype='string'),
    'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(1,), dtype='string'),
    'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(1,), dtype='string'),
    'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(1,), dtype='string'),
}

# ============ Deep 部分的特征构造 ============
# 用一个列表收集所有 deep 侧的特征张量，最后 Concatenate 在一起。
deep_feature_tensors = []

# ---------- 数值特征 ----------
# 原来用 numeric_column；现在直接把数值 Input 转成 float32 后拼接。
numerical_feature_names = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev',
                           'userRatingCount', 'userAvgRating', 'userRatingStddev']
for feature_name in numerical_feature_names:
    # int 类型的数值列(如 releaseYear)统一 cast 成 float32，保证拼接时类型一致
    num_tensor = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, tf.float32), name='cast_' + feature_name
    )(inputs[feature_name])
    deep_feature_tensors.append(num_tensor)

# ---------- genre 类别特征(字符串 -> 索引 -> 嵌入) ----------
# 等价于原来的 categorical_column_with_vocabulary_list + embedding_column(dim=10)
for feature_name, vocab in GENRE_FEATURES.items():
    # StringLookup 默认带 1 个 OOV(词表外) 索引，所以 input_dim = len(vocab) + 1
    lookup = tf.keras.layers.StringLookup(
        vocabulary=vocab, num_oov_indices=1, name='lookup_' + feature_name
    )(inputs[feature_name])
    emb = tf.keras.layers.Embedding(
        input_dim=len(vocab) + 1, output_dim=10, name='emb_' + feature_name
    )(lookup)
    # Embedding 输出 (batch, 1, 10)，展平成 (batch, 10)
    emb = tf.keras.layers.Flatten(name='flat_' + feature_name)(emb)
    deep_feature_tensors.append(emb)

# ---------- movieId 嵌入 ----------
# 等价于 categorical_column_with_identity(num_buckets=1001) + embedding_column(dim=10)
movie_emb = tf.keras.layers.Embedding(input_dim=1001, output_dim=10, name='emb_movieId')(inputs['movieId'])
movie_emb = tf.keras.layers.Flatten(name='flat_movieId')(movie_emb)
deep_feature_tensors.append(movie_emb)

# ---------- userId 嵌入 ----------
# 等价于 categorical_column_with_identity(num_buckets=30001) + embedding_column(dim=10)
user_emb = tf.keras.layers.Embedding(input_dim=30001, output_dim=10, name='emb_userId')(inputs['userId'])
user_emb = tf.keras.layers.Flatten(name='flat_userId')(user_emb)
deep_feature_tensors.append(user_emb)

# ============ Wide & Deep 模型架构 ============

# ---------- Deep 部分(深度学习部分) ----------
# 把所有数值特征 + 类别嵌入特征拼接成一个稠密向量，再过两层全连接。
deep = tf.keras.layers.concatenate(deep_feature_tensors, name='deep_concat')
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)

# ---------- Wide 部分(线性/记忆部分) ----------
# 原来用 crossed_column([movieId, userRatedMovie1], 10000) + indicator_column。
# 现在用 HashedCrossing：把两个 ID 特征哈希交叉到 10000 个桶，输出 one-hot 向量。
# 含义和课程文章里 Google Wide&Deep 的 (已安装应用 x 曝光应用) 一致：
# movieId 是当前用户正在看并评分的电影，userRatedMovie1 是用户最近看过且喜欢的电影，
# 交叉后构成 "如果 A 则 B" 这样的简单共现规则，让模型具备记忆能力。
wide = tf.keras.layers.HashedCrossing(
    num_bins=10000, output_mode='one_hot', name='crossed_movie_ratedmovie'
)((inputs['movieId'], inputs['userRatedMovie1']))
# HashedCrossing one_hot 输出可能是 (batch, 1, 10000)，展平成 (batch, 10000)
wide = tf.keras.layers.Flatten(name='flat_wide')(wide)

# ---------- 拼接 Wide 与 Deep ----------
# Deep 提供泛化能力，Wide 提供记忆能力。
both = tf.keras.layers.concatenate([deep, wide], name='wide_and_deep_concat')
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(both)
model = tf.keras.Model(inputs, output_layer)

# ============ 编译模型 ============
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    # 给两个 AUC 显式命名，否则 Keras 会自动命名为 auc / auc_1，
    # 早停回调要监控的 val_auc_roc 名字才稳定可控。
    metrics=['accuracy',
             tf.keras.metrics.AUC(curve='ROC', name='auc_roc'),
             tf.keras.metrics.AUC(curve='PR', name='auc_pr')])

# ============ 早停法(Early Stopping) ============
# 监控验证集的 ROC AUC：当它连续 patience 个 epoch 不再提升时停止训练，
# 并用 restore_best_weights 回滚到验证表现最好的那一轮权重，避免过拟合。
# mode='max' 因为 AUC 越大越好。
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc_roc',
    mode='max',
    patience=3,
    restore_best_weights=True)

# ============ 训练 ============
# epochs 放大到 50，实际训练轮数由早停根据验证集表现自动决定。
history = model.fit(train_dataset,
          validation_data=validation_dataset,
          epochs=50,
          callbacks=[early_stopping])

# ============ 评估 ============
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))

# ============ 打印部分预测结果 ============
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))
