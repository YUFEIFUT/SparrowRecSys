"""
NeuralCF — TensorFlow 3.x 适配版（改进版）
==========================================
原代码基于 TF 2.3，使用 tf.feature_column + DenseFeatures（TF 3.x 已移除）。
本版本改用手动 Embedding 层 + Functional API，保留原始模型结构不变。

改进点（针对训练指标持续下降问题）：
1. batch_size 12 → 256（减少梯度噪声，稳定训练）
2. Embedding 加 L2 正则化（防止 Embedding 向量过拟合噪声）
3. MLP 层间加 Dropout（防止交互层过拟合）
4. 加入 ReduceLROnPlateau 学习率调度（收敛后自动降低学习率）
5. 保留早停法 + 验证集

适配要点：
1. tf.feature_column → tf.keras.layers.Embedding（手动构建嵌入层）
2. DenseFeatures 移除，直接使用 Embedding 层输出
3. 新增验证集 + 早停法（monitor='val_roc_auc', patience=5, mode='max', restore_best_weights=True）
4. 数据路径改为 Google Colab 路径
5. CSV 共 26 列，但模型只使用 movieId 和 userId，用 select_columns 过滤
"""

import tensorflow as tf

# ============================================================
# 数据路径（Google Colab）
# ============================================================
train_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/trainingSamplesByTimeStamp.csv"
validation_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/validationSamplesByTimeStamp.csv"
test_path = "/content/SparrowRecSys/src/main/resources/webroot/sampledata/testSamplesByTimeStamp.csv"

# ============================================================
# 超参数
# ============================================================
BATCH_SIZE = 256        # 增大 batch_size，减少梯度噪声（原代码 12）
EMBEDDING_DIM = 10      # Embedding 维度（与原代码一致）
MOVIE_BUCKETS = 1001    # movieId 的 bucket 数（与原代码一致）
USER_BUCKETS = 30001    # userId 的 bucket 数（与原代码一致）
EPOCHS = 100            # 最大训练轮数（早停法会提前终止）
L2_REG = 1e-5           # Embedding L2 正则化系数
DROPOUT_RATE = 0.2      # Dropout 比率


# ============================================================
# 数据加载
# ============================================================
def get_dataset(file_path, batch_size=BATCH_SIZE):
    """
    使用 make_csv_dataset 加载 CSV 文件为 tf.data.Dataset。
    make_csv_dataset 会自动将 CSV 列解析为字典格式 (feature_name -> tensor)，
    并将 label_name 指定的列作为标签返回。

    适配变更：
    - CSV 共 26 列，但模型只使用 movieId 和 userId 两列
    - make_csv_dataset 默认返回所有列，多余列会导致模型输入不匹配
    - 因此用 select_columns 参数只加载需要的列
    - 原代码使用 na_value="0"，这里保留以保持一致性
    """
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batch_size,
        label_name='label',
        na_value="0",
        num_epochs=1,
        select_columns=['movieId', 'userId', 'label'],
        ignore_errors=True)
    return dataset


# 加载三个数据集：训练集、验证集、测试集
train_dataset = get_dataset(train_path)
validation_dataset = get_dataset(validation_path)
test_dataset = get_dataset(test_path)


# ============================================================
# 模型定义
# ============================================================
# --- 输入层 ---
inputs = {
    'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
}


def neural_cf_model_1(feature_inputs, movie_vocab_size, user_vocab_size, embedding_dim, hidden_units):
    """
    NeuralCF 模型架构一（原 neural_cf_model_1）：
    双塔结构 — 每个塔只有 Embedding，然后拼接后过 MLP 作为交互层。

    改进：
    - Embedding 加 L2 正则化，防止 Embedding 向量过拟合训练集噪声
    - MLP 层间加 Dropout，防止交互层过拟合
    """
    # --- Movie 塔 ---
    movie_embedding_layer = tf.keras.layers.Embedding(
        input_dim=movie_vocab_size,
        output_dim=embedding_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(L2_REG),
        name='movie_embedding'
    )
    movie_tower = movie_embedding_layer(feature_inputs['movieId'])

    # --- User 塔 ---
    user_embedding_layer = tf.keras.layers.Embedding(
        input_dim=user_vocab_size,
        output_dim=embedding_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(L2_REG),
        name='user_embedding'
    )
    user_tower = user_embedding_layer(feature_inputs['userId'])

    # --- 交互层 ---
    interact_layer = tf.keras.layers.concatenate([movie_tower, user_tower])
    for i, num_nodes in enumerate(hidden_units):
        interact_layer = tf.keras.layers.Dense(
            num_nodes, activation='relu', name=f'interact_dense_{i}'
        )(interact_layer)
        # 在 MLP 层间加 Dropout，防止交互层过拟合
        interact_layer = tf.keras.layers.Dropout(DROPOUT_RATE, name=f'interact_dropout_{i}')(interact_layer)

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(interact_layer)

    model = tf.keras.Model(feature_inputs, output_layer, name='NeuralCF_model_1')
    return model


def neural_cf_model_2(feature_inputs, movie_vocab_size, user_vocab_size, embedding_dim, hidden_units):
    """
    NeuralCF 模型架构二（原 neural_cf_model_2）：
    双塔结构 — 每个塔是 Embedding + MLP，然后做 Dot Product 作为输出。

    改进与 neural_cf_model_1 相同：L2 正则化 + Dropout。
    """
    # --- Movie 塔：Embedding + MLP ---
    movie_embedding_layer = tf.keras.layers.Embedding(
        input_dim=movie_vocab_size,
        output_dim=embedding_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(L2_REG),
        name='movie_embedding'
    )
    item_tower = movie_embedding_layer(feature_inputs['movieId'])
    for i, num_nodes in enumerate(hidden_units):
        item_tower = tf.keras.layers.Dense(
            num_nodes, activation='relu', name=f'item_dense_{i}'
        )(item_tower)
        item_tower = tf.keras.layers.Dropout(DROPOUT_RATE, name=f'item_dropout_{i}')(item_tower)

    # --- User 塔：Embedding + MLP ---
    user_embedding_layer = tf.keras.layers.Embedding(
        input_dim=user_vocab_size,
        output_dim=embedding_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(L2_REG),
        name='user_embedding'
    )
    user_tower = user_embedding_layer(feature_inputs['userId'])
    for i, num_nodes in enumerate(hidden_units):
        user_tower = tf.keras.layers.Dense(
            num_nodes, activation='relu', name=f'user_dense_{i}'
        )(user_tower)
        user_tower = tf.keras.layers.Dropout(DROPOUT_RATE, name=f'user_dropout_{i}')(user_tower)

    # --- Dot Product 交互 ---
    output = tf.keras.layers.Dot(axes=1, name='dot_product')([item_tower, user_tower])
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(output)

    model = tf.keras.Model(feature_inputs, output, name='NeuralCF_model_2')
    return model


# ============================================================
# 构建模型（使用架构一，与原代码一致）
# ============================================================
model = neural_cf_model_1(
    feature_inputs=inputs,
    movie_vocab_size=MOVIE_BUCKETS,
    user_vocab_size=USER_BUCKETS,
    embedding_dim=EMBEDDING_DIM,
    hidden_units=[10, 10]
)

# ============================================================
# 编译模型
# ============================================================
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(curve='ROC', name='roc_auc'),
        tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
    ]
)

model.summary()

# ============================================================
# 回调：早停 + 学习率调度
# ============================================================
# 早停法：监控验证集 ROC AUC，连续 5 个 epoch 无提升则停止
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_roc_auc',
    patience=5,
    mode='max',
    restore_best_weights=True,
    verbose=1
)

# 学习率调度：验证集 ROC AUC 连续 3 个 epoch 无提升时，学习率减半
# 作用：收敛后自动降低学习率，避免大步更新破坏已学到的特征
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_roc_auc',
    factor=0.5,        # 学习率乘以 0.5
    patience=3,        # 3 个 epoch 无提升则触发
    mode='max',
    min_lr=1e-6,       # 学习率下限
    verbose=1
)

# ============================================================
# 训练模型
# ============================================================
print("\n=== 开始训练 ===")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ============================================================
# 评估模型
# ============================================================
print("\n=== 测试集评估 ===")
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print(f'\nTest Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test ROC AUC: {test_roc_auc:.4f}')
print(f'Test PR AUC: {test_pr_auc:.4f}')

# ============================================================
# 打印部分预测结果
# ============================================================
print("\n=== 预测示例 ===")
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print(f"Predicted good rating: {prediction[0]:.2%}"
          f" | Actual rating label: {'Good Rating' if bool(goodRating) else 'Bad Rating'}")

# ============================================================
# 保存模型
# ============================================================
# TF 3.x 中 model.save() 只支持 .keras / .h5 扩展名
# 原项目 Java 端需要 SavedModel 格式目录，因此用 model.export() 导出
model.export("/content/SparrowRecSys/src/main/resources/webroot/modeldata/neuralcf/002")
print("\n模型已导出到 /content/SparrowRecSys/src/main/resources/webroot/modeldata/neuralcf/002")
