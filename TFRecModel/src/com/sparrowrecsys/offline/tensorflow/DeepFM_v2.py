import tensorflow as tf

"""
Diff with DeepFM:
    1. separate categorical features from dense features when processing first order features and second order features
    2. modify original fm part with a fully crossed fm part
"""
# 与DeepFM的区别：
#     1. 在处理一阶特征和二阶特征时，将类别特征和数值特征分开处理
#     2. 将原始FM部分修改为全交叉FM部分


# Training samples path, change to your local path
training_samples_file_path = tf.keras.utils.get_file("trainingSamples.csv",
                                                     "file:///Users/zhewang/Workspace/SparrowRecSys/src/main"
                                                     "/resources/webroot/sampledata/trainingSamples.csv")
# Test samples path, change to your local path
test_samples_file_path = tf.keras.utils.get_file("testSamples.csv",
                                                 "file:///Users/zhewang/Workspace/SparrowRecSys/src/main"
                                                 "/resources/webroot/sampledata/testSamples.csv")


# load sample as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset


# split as test dataset and training dataset
train_dataset = get_dataset(training_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

# define input for keras model
inputs = {
    'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
    'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
    'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
    'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
    'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
    'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
    'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),

    'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
    'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),

    'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
    'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
    'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
    'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
    'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
    'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
    'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
    'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),
}

# ============================================================
# 定义特征列（Feature Columns）
# ============================================================

# 电影ID特征列：使用identity映射，桶数量为1001（ID范围0~1000）
# movie id embedding feature
movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
# 将电影ID转换为10维的Embedding向量（用于二阶特征交互）
movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
# 将电影ID转换为One-Hot指示向量（用于一阶特征）
movie_ind_col = tf.feature_column.indicator_column(movie_col)  # movie id indicator columns

# user id embedding feature
user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
user_ind_col = tf.feature_column.indicator_column(user_col)  # user id indicator columns

# genre features vocabulary
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller',
               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

# user genre embedding feature
# 用户偏好类型特征列：基于词汇表匹配字符串到类别索引
user_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userGenre1",
                                                                           vocabulary_list=genre_vocab)
user_genre_ind_col = tf.feature_column.indicator_column(user_genre_col)
user_genre_emb_col = tf.feature_column.embedding_column(user_genre_col, 10)

# item genre embedding feature
item_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="movieGenre1",
                                                                           vocabulary_list=genre_vocab)
item_genre_ind_col = tf.feature_column.indicator_column(item_genre_col)
item_genre_emb_col = tf.feature_column.embedding_column(item_genre_col, 10)

# fm first-order categorical items
cat_columns = [movie_ind_col, user_ind_col, user_genre_ind_col, item_genre_ind_col]

deep_columns = [tf.feature_column.numeric_column('releaseYear'),
                tf.feature_column.numeric_column('movieRatingCount'),
                tf.feature_column.numeric_column('movieAvgRating'),
                tf.feature_column.numeric_column('movieRatingStddev'),
                tf.feature_column.numeric_column('userRatingCount'),
                tf.feature_column.numeric_column('userAvgRating'),
                tf.feature_column.numeric_column('userRatingStddev')]

# 一阶类别特征：先通过DenseFeatures将特征列转换为张量，再通过全连接层映射为1维标量

# DenseFeatures层：将字典形式的输入按照特征列定义转换为稠密张量
first_order_cat_feature = tf.keras.layers.DenseFeatures(cat_columns)(inputs)
# 全连接层：将类别特征的一阶输出映射为1个标量值，无激活函数（线性变换）
first_order_cat_feature = tf.keras.layers.Dense(1, activation=None)(first_order_cat_feature)

# 一阶数值特征：先通过DenseFeatures将数值特征列转换为张量，再通过全连接层映射为1维标量
first_order_deep_feature = tf.keras.layers.DenseFeatures(deep_columns)(inputs)
first_order_deep_feature = tf.keras.layers.Dense(1, activation=None)(first_order_deep_feature)
## first order feature

# 合并一阶特征：将类别特征和数值特征的一阶输出相加
# 对应FM公式中的 w0 + Σ(wi * xi) 部分
first_order_feature = tf.keras.layers.Add()([first_order_cat_feature, first_order_deep_feature])

# ============================================================
# 二阶特征处理（Second Order Features）
# ============================================================

# 将每个类别特征的Embedding分别提取出来，形成独立的二阶特征向量
second_order_cat_columns_emb = [tf.keras.layers.DenseFeatures([item_genre_emb_col])(inputs),
                                tf.keras.layers.DenseFeatures([movie_emb_col])(inputs),
                                tf.keras.layers.DenseFeatures([user_genre_emb_col])(inputs),
                                tf.keras.layers.DenseFeatures([user_emb_col])(inputs)
                                ]

# 对每个类别Embedding特征进行线性变换，统一映射到64维空间，并reshape为(batch, 1, 64)的3D张量
second_order_cat_columns = []
for feature_emb in second_order_cat_columns_emb:
    # 全连接层：将每个Embedding映射到统一的64维空间（无激活函数，保持线性）
    feature = tf.keras.layers.Dense(64, activation=None)(feature_emb)
    # Reshape层：将(batch, 64)变为(batch, 1, 64)，增加一个维度便于后续拼接
    feature = tf.keras.layers.Reshape((-1, 64))(feature)
    second_order_cat_columns.append(feature)

# 数值特征也映射到64维空间，并reshape为(batch, 1, 64)
second_order_deep_columns = tf.keras.layers.DenseFeatures(deep_columns)(inputs)
second_order_deep_columns = tf.keras.layers.Dense(64, activation=None)(second_order_deep_columns)
second_order_deep_columns = tf.keras.layers.Reshape((-1, 64))(second_order_deep_columns)
# 沿第1维（特征维度）拼接所有二阶特征，形成(batch, num_fields, 64)的3D张量
# TODO 不懂
# 将4个类别特征 + 1个数值特征 = 5个字段拼接在一起
# 最终形状为 (batch, 5, 64)，表示5个字段各64维
second_order_fm_feature = tf.keras.layers.Concatenate(axis=1)(second_order_cat_columns + [second_order_deep_columns])

# ============================================================
# Deep部分（DNN）
# ============================================================

# 将二阶FM特征展平后送入DNN进行高阶特征交叉学习
## second_order_deep_feature
deep_feature = tf.keras.layers.Flatten()(second_order_fm_feature)
deep_feature = tf.keras.layers.Dense(32, activation='relu')(deep_feature)
deep_feature = tf.keras.layers.Dense(16, activation='relu')(deep_feature)

# ============================================================
# 自定义层：ReduceLayer —— 沿指定维度求和或求均值
# ============================================================
class ReduceLayer(tf.keras.layers.Layer):
    def __init__(self, axis, op='sum', **kwargs):
        super().__init__()
        # 沿哪个维度进行规约
        self.axis = axis
         # 规约操作类型：'sum'（求和）或 'mean'（求均值）
        self.op = op
        # 断言：操作类型只能是'sum'或'mean'
        assert self.op in ['sum', 'mean']

    def build(self, input_shape):
        # 该层无可训练参数，build方法为空
        pass

    def call(self, input, **kwargs):
        # 前向传播逻辑
        if self.op == 'sum':
            # 沿axis维度求和
            return tf.reduce_sum(input, axis=self.axis)
        elif self.op == 'mean':
            return tf.reduce_mean(input, axis=self.axis)
        return tf.reduce_sum(input, axis=self.axis)

# ============================================================
# FM二阶交叉特征计算（使用公式优化）
# FM公式：0.5 * [ (Σvi·xi)² - Σ(vi·xi)² ] = 0.5 * [ (sum)² - sum_of_squares ]
# ============================================================

# 第一步：对所有字段的Embedding沿第1维（字段维度）求和，得到(batch, 64)
second_order_sum_feature = ReduceLayer(1)(second_order_fm_feature)
# 沿字段维度求和：将5个字段的64维向量逐元素相加

# 第二步：对求和结果进行平方（逐元素相乘 = 自乘），得到(batch, 64)
second_order_sum_square_feature = tf.keras.layers.multiply([second_order_sum_feature, second_order_sum_feature])
# (Σvi)² —— 先求和再平方

# 第三步：先对每个字段的Embedding进行逐元素平方，得到(batch, 5, 64)
second_order_square_feature = tf.keras.layers.multiply([second_order_fm_feature, second_order_fm_feature])
# vi² —— 先平方

# 第四步：对平方后的结果沿字段维度求和，得到(batch, 64)
second_order_square_sum_feature = ReduceLayer(1)(second_order_square_feature)
# Σ(vi²) —— 平方后再求和

# 第五步：应用FM公式：(Σvi)² - Σ(vi²)，得到最终的二阶交叉特征，形状为(batch, 64)
## second_order_fm_feature
second_order_fm_feature = tf.keras.layers.subtract([second_order_sum_square_feature, second_order_square_sum_feature])
# 注意：实际FM公式前还有一个0.5的系数，这里省略了（等价于乘以常数，不影响模型表达能力）

# ============================================================
# 输出层：拼接所有特征并输出预测结果
# ============================================================

# 将一阶特征、FM二阶交叉特征、DNN高阶特征沿第1维拼接
# first_order_feature: (batch, 1)
# second_order_fm_feature: (batch, 64)
# deep_feature: (batch, 16)
concatenated_outputs = tf.keras.layers.Concatenate(axis=1)([first_order_feature, second_order_fm_feature, deep_feature])
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concatenated_outputs)

model = tf.keras.Model(inputs, output_layer)
# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# train the model
model.fit(train_dataset, epochs=5)

# evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))

# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))
