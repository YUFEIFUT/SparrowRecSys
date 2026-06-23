# WideNDeep.py 学习笔记

> 本文记录了在学习 `WideNDeep.py` 文件时的一些思考和解答。

---

## 1. `num_epochs=1` 在数据集中的含义

```python
dataset = tf.data.experimental.make_csv_dataset(
    file_path,
    batch_size=12,
    label_name='label',
    na_value="0",
    num_epochs=1,  # 这里容易误解！
    ignore_errors=True)
```

**理解要点**：这确实不是训练 epoch。这里的 `num_epochs` 指的是**数据集对象本身在迭代时重复数据的次数**。

- `num_epochs=1`：表示迭代器遍历完所有数据后就停止（不重复）
- `num_epochs=2`：表示遍历完一遍后，会**重新从文件开头再读一遍**，总共读 2 遍
- `num_epochs=None`：表示无限循环（永不停止）

**为什么设置为 1？**

因为在 `model.fit()` 中我们会指定 `epochs=5`，Keras 会自动管理整个数据集的重复遍历。如果这里设置 `num_epochs=5`，而 `model.fit(epochs=5)` 又重复 5 次，就变成总共 25 次了（5×5=25）。

类比理解：

```python
# 这就好比：
dataset = [1, 2, 3, 4, 5]  # 原始数据
# num_epochs=1 表示：只给你看一遍 [1,2,3,4,5]
# num_epochs=2 表示：给你看两遍 [1,2,3,4,5,1,2,3,4,5]
```

---

## 2. `DenseFeatures` 做了什么？

`DenseFeatures` 是一个**特征转换层**，它的作用是将各种类型的特征（数值、类别、嵌入等）统一转换成**稠密的数值张量**。

### 具体过程

```python
# 输入：多个特征列
numerical_columns = [numeric_column('releaseYear'), ...]  # 数值特征
categorical_columns = [embedding_column(...), ...]        # 类别嵌入特征

# DenseFeatures 做两件事：
# 1. 从 inputs 字典中提取对应的特征值
# 2. 按照 feature_column 的规则转换并拼接
deep = DenseFeatures(numerical_columns + categorical_columns)(inputs)
```

**具体例子：**

假设一个样本的输入是：

```python
inputs = {
    'releaseYear': 2019,           # 标量
    'movieAvgRating': 4.5,         # 标量
    'movieId': 123,                # 需要 embedding
    'userGenre1': 'Action',        # 需要 embedding
    # ... 更多特征
}
```

`DenseFeatures` 的处理流程：

1. **数值特征**：直接取出 `[2019, 4.5]`
2. **类别特征**：
   - `movieId=123` → 查 embedding 表 → `[0.1, 0.5, ..., 0.3]` (10 维)
   - `userGenre1='Action'` → 查 embedding 表 → `[0.2, -0.1, ..., 0.4]` (10 维)
3. **拼接所有特征**：`[2019, 4.5, 0.1, 0.5, ..., 0.3, 0.2, -0.1, ..., 0.4]`

**最终输出**：一个形状为 `(batch_size, total_feature_dim)` 的密集张量

---

## 3. `concatenate` 的操作

```python
both = tf.keras.layers.concatenate([deep, wide])
```

这是**向量拼接（Concatenation）**，不是相加！

### 具体操作

假设：

- `deep` 的输出形状：`(batch_size, 256)` （128+128 的两个全连接层）
- `wide` 的输出形状：`(batch_size, 10000)` （交叉特征 one-hot 编码）

```python
# 拼接后：
both 的形状：`(batch_size, 256 + 10000) = (batch_size, 10256)`
```

**可视化：**

```python
# 假设 batch_size=2
deep = [[0.1, 0.2, 0.3],    # 样本1的 deep 特征
        [0.4, 0.5, 0.6]]    # 样本2的 deep 特征
        # 形状: (2, 3)

wide = [[1.0, 0.0],         # 样本1的 wide 特征
        [0.0, 1.0]]         # 样本2的 wide 特征
        # 形状: (2, 2)

# concatenate 后：
both = [[0.1, 0.2, 0.3, 1.0, 0.0],   # 拼接在一起
        [0.4, 0.5, 0.6, 0.0, 1.0]]
        # 形状: (2, 5)
```

**为什么要拼接而不是相加？**

- Wide 部分的交叉特征和 Deep 部分的深度特征有**不同的语义**
- 拼接后通过输出层的全连接层学习**如何组合**这两类特征
- 这样模型可以灵活地决定给 Wide 和 Deep 特征不同的权重

---

## 4. 为什么都要 `inputs`？

这个问题很关键！`inputs` 的作用是**数据入口**（Input Placeholder）。

```python
# 1. 定义输入层（数据的入口）
inputs = {
    'movieId': Input(name='movieId', shape=(), dtype='int32'),
    'userId': Input(name='userId', shape=(), dtype='int32'),
    # ...
}

# 2. 所有层都需要知道数据从哪里来
#    通过传入 inputs 告诉它们从这些输入层获取数据

# 第一种写法（显式传参）：
deep = DenseFeatures(numerical_columns + categorical_columns)(inputs)
wide = DenseFeatures(crossed_feature)(inputs)

# 第二种写法（等价，但更清晰）：
x1 = DenseFeatures(numerical_columns + categorical_columns)(inputs)
x2 = DenseFeatures(crossed_feature)(inputs)
both = concatenate([x1, x2])
```

### 数据流示意图

```
                    inputs (字典)
                    /        \
                   /          \
    DenseFeatures(数值+类别)  DenseFeatures(交叉特征)
                   |            |
                  deep         wide
                   \            /
                    \          /
                  concatenate
                       |
                   Dense(输出)
                       |
                  output_layer
```

**如果把 `inputs` 比作水管总阀门：**

- `inputs` 是水的源头（原始数据）
- `DenseFeatures` 是过滤器/处理器（转换数据）
- `concatenate` 是管道连接器（合并数据）
- 最终输出是处理后的水（预测结果）

所以每个层都需要知道它的**上游数据来源**，这就是为什么都要传入 `inputs`。

### 额外小知识：如果不用 `inputs`？

在旧版 Keras 中，你可能看到这种写法：

```python
# 方式1：通过 inputs 传递
model = tf.keras.Model(inputs=inputs, outputs=output_layer)

# 方式2：如果不用 inputs（但不推荐）
input_tensor = tf.keras.layers.Input(shape=(10,))
x = tf.keras.layers.Dense(64)(input_tensor)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=input_tensor, outputs=output)
```

在推荐系统中，因为有**多种类型的数据**（数值、字符串、ID），所以用字典形式的 `inputs` 更灵活！

---

## 总结

| 概念 | 作用 |
|------|------|
| `num_epochs` | 控制数据集对象迭代时重复数据的次数，与训练 epoch 分开管理 |
| `DenseFeatures` | 将多种特征（数值、类别、嵌入）统一转换并拼接成稠密张量 |
| `concatenate` | 向量拼接，将 deep 和 wide 的特征向量连接成更长的向量 |
| `inputs` | 数据入口，所有层都需要知道从哪里获取原始数据 |
