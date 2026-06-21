# ALS 召回链路说明（离线产出 → 在线多路召回）

> 本文梳理「ALS 协同过滤」从离线训练到在线推荐的完整链路，以及每个关键设计为什么这么做。
> 涉及文件：
> - 离线（训练）：`offline/spark/model/CollaborativeFiltering.scala` —— 读数据/训练/评估/**保存模型**
> - 离线（导出）：`offline/spark/model/AlsModelExporter.scala` —— **加载模型**后导出隐向量 / 每用户Top-N
> - 在线：`online/recprocess/RecForYouProcess.java`、`online/datamanager/{DataManager,User}.java`、`online/RecSysServer.java`

---

## 1. 一张图看懂全局

```
┌─────────────────────────── 离线（Spark, 跑一次/每日） ───────────────────────────┐
│  CollaborativeFiltering.scala                                                    │
│    ALS.fit(ratings)  →  model                                                    │
│       ├─ itemFactors / userFactors  ──→  alsItemEmbeddings.csv / alsUserEmbeddings.csv │
│       │                                  （+ 可选 Redis: alsI2vEmb: / alsUEmb:）  │
│       └─ recommendForAllUsers(10)   ──→  userRecs.csv  （每用户Top-N）            │
│                                          （+ 可选 Redis: rec:userId）             │
└──────────────────────────────────────────┬──────────────────────────────────────┘
                                            │  文件 / Redis（两种数据源二选一）
                                            ▼
┌─────────────────────────── 在线（Jetty 服务，启动时加载 + 请求时召回） ──────────┐
│  RecSysServer 启动                                                               │
│    └─ DataManager.loadData(...)                                                  │
│         └─ loadUserRecs(userRecs.csv)  →  把每用户Top-N 灌进 User.alsRecMovieIds  │
│                                                                                  │
│  /getrecforyou 请求                                                              │
│    └─ RecForYouProcess.getRecList(userId, size, model)                           │
│         ├─ multipleRetrievalRecall(user)        ← 多路召回·配额制（本文重点）     │
│         │     ① 高分召回（兜底）                                                  │
│         │     ② 离线ALS召回（读 User.alsRecMovieIds 或 Redis rec:userId）         │
│         │     ③ Embedding/ANN 召回（用户向量近邻）                                │
│         └─ ranker(user, candidates, model)      ← 排序（emb / NeuralCF）         │
└──────────────────────────────────────────────────────────────────────────────┘
```

**一句话**：离线把「每个用户的 Top-N 推荐」算好存起来，在线把它当作多路召回中的**一路**读出来，和其他召回路合并去重后再交给排序层。

---

## 2. 离线侧：训练与导出（两个文件，各司其职）

离线拆成两个 object，避免一个文件什么都干、过于臃肿：

- **`CollaborativeFiltering`（训练）**：读数据 → 训练 → 评估 → 把模型保存成自包含目录 `modeldata/alsModel/`。
- **`AlsModelExporter`（导出）**：`ALSModel.load(...)` 加载上面的模型，再导出线上要的产物。**因为模型训练慢、导出快，分开后改 topN、只重导隐向量等都无需重训。**

`AlsModelExporter` 能导出两类产物，对应两条不同的上线路线：

| 产出 | 内容 | 导出方法 | 文件 | 给谁用 |
|---|---|---|---|---|
| **隐向量** | `itemFactors` / `userFactors`（k 维向量） | `saveEmbeddings` | `alsItemEmbeddings.csv` / `alsUserEmbeddings.csv` | 给「ANN 召回」用（把物品向量灌进索引） |
| **预计算推荐** | `recommendForAllUsers(10)`（每用户 Top-10 电影） | `saveUserRecs` | `userRecs.csv` | 给「离线ALS召回」用（直接查表） |

### 2.1 为什么隐向量和预计算结果是「二选一」的关系

- 走**方案A（离线预计算查表）** → 你要的是 `userRecs.csv`，**隐向量只是中间产物**，线上用不到。
- 走**ANN** → 你要的是隐向量（灌进索引），`recommendForAllUsers` 反而不需要。

两条路的持久化产物几乎互斥。代码里两套都写了，是为了**对照学习**；生产中按选定的路线保留一套即可。

### 2.2 文件格式约定（重要：和在线读取必须对齐）

```
# alsItemEmbeddings.csv / alsUserEmbeddings.csv  （隐向量）
id:emb1 emb2 ... embN        # 冒号分隔id，向量用空格分隔（与 Item2vec 的 Embedding 输出格式一致）

# userRecs.csv               （每用户Top-N推荐）
userId:m1 m2 m3             # 冒号分隔userId，movieId 用空格分隔、按推荐分降序（与项目其它 key:v1 v2 v3 约定一致）
```

`userRecs.csv` 里**冒号后的那段**（`m1 m2 m3`），刻意设计成**正好等于 Redis 里的 value**，这样文件和 Redis 两种来源的解析逻辑能复用，不会写两套。

### 2.3 为什么用 classpath 资源定位路径，而不是写死绝对路径

```scala
val recFolderPath = this.getClass.getResource("/webroot/modeldata/").getPath
```

写死 `src/main/resources/...` 这种相对路径依赖运行时的工作目录，换台机器/换 IDE 配置就崩。用 classpath 资源定位则跟着编译产物走，更稳——这也是项目里 `Embedding.scala` 的既有写法，保持一致。

---

## 3. 在线侧：多路召回是怎么运作的

入口是 `RecForYouProcess.getRecList`，它做两件事：**召回**（选候选）→ **排序**（精排打分）。本次改造重点是把原来「单路、非个性化」的召回，换成「多路、配额制」的召回。

### 3.1 改造前 vs 改造后

```java
// 改造前：单路，非个性化——谁来都是评分最高的800部
List<Movie> candidates = DataManager.getInstance().getMovies(800, "rating");

// 改造后：多路召回（配额制）
List<Movie> candidates = multipleRetrievalRecall(user);
```

### 3.2 三路召回与配额

`multipleRetrievalRecall(User user)` —— 入参就是**用户对象**。各路用「配额(取多少个)」控制贡献，是业界主流做法。

| 路 | 方法 | 配额常量 | 默认值 | 说明 |
|---|---|---|---|---|
| ① 高分召回 | `getMovies(quota,"rating")` | `QUOTA_HIGH_RATING` | **300** | 非个性化兜底 |
| ② 离线ALS召回 | `retrievalByAlsOffline` | `QUOTA_ALS_OFFLINE` | **100** | 实际受离线 `topN` 限制，通常远小于此 |
| ③ Embedding/ANN | `retrievalByEmbedding` | `QUOTA_EMB_ANN` | **400** | 个性化主力 |

配额和开关都抽成了类顶部的常量，方便调参。

### 3.3 为什么是「配额 + 去重」而不是「融合打分」

> **关键认知**：召回层的职责是「圈出一批不错的候选」，**不是精细排序**。精细排序是下游 `ranker`（emb / NeuralCF）的活——召回融合出来的顺序，到了 ranker 那里几乎会被重新打分覆盖。所以在召回层费力算「加权融合分」收益很低。

因此业界主流做法是：

1. **每路按配额取数** —— 各路的影响力体现为「占多少坑位」，而不是给每个候选算分。
2. **合并去重** —— 用 `LinkedHashMap<movieId, Movie>` 按 movieId 去重，保留首次出现顺序（去重键用 movieId，不依赖 `Movie.equals`，更稳妥）。
3. **整体交给 ranker** —— 候选最终顺序由排序层决定。

对应代码 `mergeChannel`：

```java
for (Movie m : candidates){
    if (null != m){
        candidateMap.putIfAbsent(m.getMovieId(), m);   // 按movieId去重，保留首次出现顺序
    }
}
```

> 注：早期版本用过「排名归一化分 × 权重」的加权融合（`(n-i)/n × weight` 累加）。它属于 rank-based fusion 家族（有名的标准算法是 **RRF**），适合「没有独立排序层」的轻量场景；但本项目下游有 ranker，融合排序会被覆盖，故改用更主流的配额制。

---

## 4. 关键决策：离线ALS这一路，为什么要「内存 or Redis」两种来源

这是最容易看晕的地方，单独讲。

### 4.1 项目的既有范式

这个项目（学习用途）的数据加载范式是：**启动时按 `Config.EMB_DATA_SOURCE` 决定从文件还是 Redis 把数据灌进内存，在线请求再读内存对象**。例如 movie embedding：

- `EMB_DATA_SOURCE = file` → 启动时从 csv 读进 `Movie.emb`
- `EMB_DATA_SOURCE = redis` → 启动时/请求时从 Redis 读

为了和项目风格一致，离线ALS召回这一路也做成**可切换数据源**，而不是写死只读 Redis。

### 4.2 具体实现

```java
private static List<Movie> retrievalByAlsOffline(User user, int size){
    List<Integer> recMovieIds;
    if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_REDIS)){
        // 数据源=Redis：请求时读 rec:userId
        recMovieIds = 解析 RedisClient.get("rec:" + userId);
    } else {
        // 数据源=文件：读启动时已加载到内存的 User.alsRecMovieIds
        recMovieIds = user.getAlsRecMovieIds();
    }
    // 两条来源在这里汇合：movieId -> Movie
    return recMovieIds 解析成 Movie 列表;
}
```

设计上让两条来源**都先得到 `List<Integer>`（movieId 列表），再统一解析成 `Movie`**，避免写两套解析逻辑。

### 4.3 内存这条路是怎么"灌"进去的

```
RecSysServer.loadData(..., "modeldata/userRecs.csv", ...)
   └─ DataManager.loadUserRecs(path)
        └─ 逐行 split(":") → userId + movieId列表 → user.setAlsRecMovieIds(...)
```

`User` 上为此新增了字段：

```java
@JsonIgnore
List<Integer> alsRecMovieIds;   // 离线ALS预计算的Top-N，启动时从文件加载
```

### 4.4 为什么 loadUserRecs 要"文件不存在就跳过"

```java
if (!recFile.exists()){
    System.out.println("User recs file not found, skip loading: ...");
    return;
}
```

`userRecs.csv` 是**可选的离线产物**——如果你还没跑离线作业，这个文件就不存在。项目里其他加载（如 emb）文件缺失会直接抛异常、服务起不来。这里特意做成**容错**：文件没有就跳过，服务照常启动，离线ALS这一路读到 `null` 返回空，多路召回自动退化为另外两路。

> **决策原则**：新增的可选数据源不应该破坏原有启动流程。「优雅降级」比「硬依赖」更适合这种增量功能。

---

## 5. Embedding/ANN 这一路：当前是占位实现

```java
// 当前：暴力遍历全库算余弦相似度，O(物品数)
for (Movie candidate : allCandidates){
    double sim = user.getEmb().calculateSimilarity(candidate.getEmb());
    ...
}
```

这一路**架构位置先占好了，但实现还是暴力扫描**（和 `SimilarMovieProcess.retrievalCandidatesByEmbedding` 同思路）。物品量大时这就是瓶颈。

**下一步**：用 `jelmerk/hnswlib`（纯 Java 的 HNSW 索引，支持内积/MIPS）替换暴力扫描——离线把物品向量灌进索引，在线用 `user.getEmb()` 当查询向量查 Top-N，把 O(N) 降到亚线性。

> 注意：ANN 用的是项目加载的 `user.getEmb()`（Item2vec/userEmb 体系），不是 ALS 隐向量。若要让 ANN 走 ALS 向量，需另把 ALS 物品向量灌进索引。

---

## 6. 召回 vs 排序：别混淆

| | 召回（recall） | 排序（rank） |
|---|---|---|
| 干什么 | 从百万物品里**快速、近似**筛出几百候选 | 在小候选集上**精确**打分 |
| 本文涉及 | ①②③ 三路 + 配额合并去重 | `ranker`（emb 相似度 / NeuralCF） |
| ANN/查表 的位置 | **都在这一层** | 不在 |

离线ALS（查表）和 ANN，**都是召回层的实现方式**，不是排序。排序始终是下游那个富特征模型（`emb` / `nerualcf`）。

---

## 7. 怎么把整条链路跑起来

### 默认状态（开箱）
- `Config.EMB_DATA_SOURCE = file`，且 `userRecs.csv` 尚未生成。
- 启动会打印 `skip loading`，离线ALS这一路为空，多路召回靠 ①③ 两路工作。

### 离线两步走（训练 / 导出 分离）
1. **训练 + 保存模型**：运行 `CollaborativeFiltering` 一次 → 在 `modeldata/alsModel/` 生成模型目录（自包含）。
2. **导出产物**：运行 `AlsModelExporter`（它**加载**上一步的模型，不再重训），在 `main` 里按需调用：
   - `saveEmbeddings(model, saveToRedis)` → 生成 `alsItemEmbeddings.csv` / `alsUserEmbeddings.csv`
   - `saveUserRecs(model, topN, saveToRedis)` → 生成 `userRecs.csv`
   > 拆分的意义：模型训练很慢，导出很快。把它们分开后，想换个 topN、或只重导隐向量，都不必再训一遍模型。

### 启用「离线ALS召回」（方案A）
1. 跑上面两步，确保生成了 `modeldata/userRecs.csv`（`saveUserRecs` 的 `saveToRedis=true` 则同时写 Redis）。
2. （可选）想走 Redis：在线把 `Config.EMB_DATA_SOURCE` 设为 `redis`。
3. 重启在线服务 → `loadUserRecs` 会把推荐结果灌进内存，第 ② 路生效。

### 调参
- 改三个 `QUOTA_*` 常量调整各路配额（取多少个）。
- 用 `ENABLE_ALS_OFFLINE` / `ENABLE_EMB_ANN` 开关单独开关某一路。

---

## 8. 数据契约速查表

| 数据 | 文件 | 文件格式 | Redis key | Redis value |
|---|---|---|---|---|
| 物品隐向量 | `alsItemEmbeddings.csv` | `id:e1 e2 ... eN` | `alsI2vEmb:id` | `e1 e2 ... eN` |
| 用户隐向量 | `alsUserEmbeddings.csv` | `id:e1 e2 ... eN` | `alsUEmb:id` | `e1 e2 ... eN` |
| 每用户Top-N | `userRecs.csv` | `userId:m1 m2 m3` | `rec:userId` | `m1 m2 m3` |

> 改任何一侧格式，另一侧的解析必须同步改——这就是「离线/在线通过文件耦合」的代价，也是为什么要把格式约定写清楚。
