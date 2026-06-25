import tensorflow as tf

# Training samples path, change to your local path
training_samples_file_path = tf.keras.utils.get_file("trainingSamples.csv",
                                                     "file:///Users/zhewang/Workspace/SparrowRecSys/src/main"
                                                     "/resources/webroot/sampledata/trainingSamples.csv")
# Test samples path, change to your local path
test_samples_file_path = tf.keras.utils.get_file("testSamples.csv",
                                                 "file:///Users/zhewang/Workspace/SparrowRecSys/src/main"
                                                 "/resources/webroot/sampledata/testSamples.csv")


# load sample as tf dataset
# 加载CSV样本文件并转换为TensorFlow Dataset数据集的函数
def get_dataset(file_path):
    # 使用experimental.make_csv_dataset创建CSV数据集
    # batch_size=12: 每批次12个样本
    # label_name='label': 指定标签列的名称为'label'
    # na_value="0": 将空值替换为0
    # num_epochs=1: 只遍历数据集一次
    # ignore_errors=True: 忽略数据读取中的错误
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

# movie id embedding feature
movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)

# user id embedding feature
user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)

# define input for keras model
# 定义Keras模型的输入层
# 创建两个输入：movieId和userId，数据类型为int32，标量形状shape=()
inputs = {
    'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
}

'''
下边两种模型的比较：https://chat.deepseek.com/share/60cqfmbcetyggvmun1
'''

# neural cf model arch two. only embedding in each tower, then MLP as the interaction layers
# 神经协同过滤模型架构一：双塔各自只包含嵌入层，然后通过MLP进行交互
def neural_cf_model_1(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    # 物品塔：使用DenseFeatures将特征列转换为稠密向量
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    # 用户塔：使用DenseFeatures将特征列转换为稠密向量
    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    # 将物品塔和用户塔的输出拼接在一起
    interact_layer = tf.keras.layers.concatenate([item_tower, user_tower])
    # 通过多层全连接神经网络（MLP）学习特征交互
    for num_nodes in hidden_units:
        interact_layer = tf.keras.layers.Dense(num_nodes, activation='relu')(interact_layer)
    # 输出层：使用sigmoid激活函数输出0-1之间的概率值
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(interact_layer)
    # 构建Keras模型
    neural_cf_model = tf.keras.Model(feature_inputs, output_layer)
    return neural_cf_model


# neural cf model arch one. embedding+MLP in each tower, then dot product layer as the output
# 神经协同过滤模型架构二：每塔使用嵌入+MLP，然后通过点积层输出
def neural_cf_model_2(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    # 物品塔：先通过DenseFeatures，然后经过MLP
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        item_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(item_tower)

    # 用户塔：先通过DenseFeatures，然后经过MLP
    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        user_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(user_tower)

    # 计算用户塔和物品塔输出的点积，衡量用户和物品的匹配程度
    output = tf.keras.layers.Dot(axes=1)([item_tower, user_tower])
    # 通过sigmoid激活函数输出点击/购买概率
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    neural_cf_model = tf.keras.Model(feature_inputs, output)
    return neural_cf_model


# neural cf model architecture
# 选择使用第一种神经协同过滤模型架构
# 传入输入层、物品特征列、用户特征列和隐藏层单元数[10, 10]
model = neural_cf_model_1(inputs, [movie_emb_col], [user_emb_col], [10, 10])

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

tf.keras.models.save_model(
    model,
    "file:///Users/zhewang/Workspace/SparrowRecSys/src/main/resources/webroot/modeldata/neuralcf/002",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
