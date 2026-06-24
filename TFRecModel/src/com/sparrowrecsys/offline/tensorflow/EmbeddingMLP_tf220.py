import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# 1. 指定训练集和测试集的路径
train_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/trainingSamplesByTimeStamp.csv"
validation_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/validationSamplesByTimeStamp.csv"
test_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/testSamplesByTimeStamp.csv"

# 2. 定义特征清单与词表
# 电影风格（Genre）的全量枚举值，用于构建 StringLookup 的词表
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller', 'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

# 类别特征列表（包含用户偏好风格和电影自身风格）
genre_features = ['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5',
                  'movieGenre1', 'movieGenre2', 'movieGenre3']
# 数值型特征列表（发行年份、评分统计信息等）
numeric_features = ['releaseYear', 'movieRatingCount', 'movieAvgRating',
                    'movieRatingStddev', 'userRatingCount', 'userAvgRating', 'userRatingStddev']

# 3. 构造输入层并进行预处理
all_inputs = {}  # 保存所有输入层
encoded_features = []  # 保存经过处理后的特征

# --- A. 处理数值特征 ---
for header in numeric_features:
    # 为每个数值特征定义一个 Input 层，接收 float32 类型数据
    # 为什么 shape=(1,)？因为一个特征只有一个数字
    numeric_col = tf.keras.Input(shape=(1,), name=header, dtype='float32')
    all_inputs[header] = numeric_col
    # 数值特征直接加入特征列表，无需额外转换
    encoded_features.append(numeric_col)

# --- B. 处理 Genre 类别特征 (String -> Embedding) ---
for header in genre_features:
    # 定义接收字符串的输入层
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    all_inputs[header] = categorical_col
    # 使用 StringLookup 将字符串映射为整数索引：
    # - vocabulary: 使用预定义的风格词表
    # - num_oov_indices: 设为 1，用于处理词表外的未知风格
    lookup_layer = layers.StringLookup(vocabulary=genre_vocab, mask_token=None, num_oov_indices=1)
    index = lookup_layer(categorical_col)
    # 使用 Embedding 层将索引转换为 10 维稠密向量
    # input_dim 为词表大小 + OOV + 1【这里原本的代码底层逻辑是咋样的？】
    embedding = layers.Embedding(input_dim=len(genre_vocab) + 2, output_dim=10)(index)
    # 将 Embedding 输出展平（从 [batch, 1, 10] 变为 [batch, 10]）以便拼接
    encoded_features.append(layers.Flatten()(embedding))

# --- C. 处理 ID 特征 (Int -> Embedding) ---
# 电影 ID：假设范围 1001 以内
movie_id_input = tf.keras.Input(shape=(1,), name='movieId', dtype='int64')
all_inputs['movieId'] = movie_id_input
movie_embedding = layers.Embedding(input_dim=1001, output_dim=10)(movie_id_input)
encoded_features.append(layers.Flatten()(movie_embedding))

# 用户 ID：假设范围 30001 以内
user_id_input = tf.keras.Input(shape=(1,), name='userId', dtype='int64')
all_inputs['userId'] = user_id_input
user_embedding = layers.Embedding(input_dim=30001, output_dim=10)(user_id_input)
encoded_features.append(layers.Flatten()(user_embedding))

# 4. 构建 MLP 网络 (多层感知机)
# 将所有处理后的特征（数值 + Embedding）拼接成一个长向量
#【之前的代码应该是多种特征加起来吧，现在的layers.Concatenate()(encoded_features)是什么逻辑？】
all_features = layers.Concatenate()(encoded_features)
# 两个隐藏层，每层 128 个神经元，使用 ReLU 激活函数增强非线性表达能力
x = layers.Dense(128, activation="relu")(all_features)
x = layers.Dense(128, activation="relu")(x)
# 输出层使用 Sigmoid 激活函数，输出 0 到 1 之间的点击概率值
output = layers.Dense(1, activation="sigmoid")(x)

# 定义 Keras 函数式模型
new_model = tf.keras.Model(all_inputs, output)

# 5. 编译模型
new_model.compile(
    optimizer='adam',                # Adam 优化器，自动调节学习率
    loss='binary_crossentropy',      # 二分类交叉熵损失函数
    metrics=['accuracy', tf.keras.metrics.AUC(name='roc_auc', curve='ROC')])

# 6. 优化的数据加载函数
def df_to_dataset(file_path, batch_size=32):
    """将 CSV 文件加载并转换为 tf.data.Dataset 格式"""
    df = pd.read_csv(file_path)
    feature_cols = numeric_features + genre_features + ['movieId', 'userId']

    # 标签转换为 float32
    labels = df['label'].values.astype('float32')

    # 数据清洗与类型转换
    # 1. 数值特征：填充缺失值为 0，强制转为 float32
    df[numeric_features] = df[numeric_features].fillna(0).astype('float32')
    # 2. 类别特征：填充缺失值为 'None'，强制转为 string 防止 Pandas 误识别类型
    df[genre_features] = df[genre_features].fillna('None').astype(str)
    # 3. ID 特征：取模运算确保数值不会超出 Embedding 层的 input_dim 范围
    df['movieId'] = df['movieId'].fillna(0).astype('int64') % 1001
    df['userId'] = df['userId'].fillna(0).astype('int64') % 30001

    # 构建输入字典
    input_dict = {col: df[col].values for col in feature_cols}
    # 转换为 TF 数据集
    ds = tf.data.Dataset.from_tensor_slices((input_dict, labels))
    return ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

# 准备训练和测试数据
train_ds = df_to_dataset(train_path)
validation_ds = df_to_dataset(validation_path)
test_ds = df_to_dataset(test_path)

# 7. 设置早停法回调 (EarlyStopping)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_roc_auc',    # 监控验证集的 AUC
    patience=5,               # 如果连续 5 个 epoch 指标没提升就停止
    mode='max',               # 目标是最大化 AUC
    restore_best_weights=True # 停止后恢复最优权重
)

# 8. 执行训练 (赋值给 history 变量以供绘图)
print("开始带有早停法的模型训练...")
history = new_model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=50,                # 设置较大的上限
    callbacks=[early_stopping]
)

# 9. 最终评估
loss, accuracy, auc = new_model.evaluate(test_ds)
print(f"\n最终评估结果: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")