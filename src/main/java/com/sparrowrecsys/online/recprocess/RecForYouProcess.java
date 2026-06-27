package com.sparrowrecsys.online.recprocess;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.datamanager.User;
import com.sparrowrecsys.online.datamanager.Movie;

import com.sparrowrecsys.online.datamanager.RedisClient;
import com.sparrowrecsys.online.util.Config;
import com.sparrowrecsys.online.util.Utility;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.*;

import static com.sparrowrecsys.online.util.HttpClient.asyncSinglePostRequest;

/**
 * Recommendation process of similar movies
 */

public class RecForYouProcess {

    // ==================== 多路召回各路配额配置 ====================
    // 业界主流范式：召回层只负责"产出候选集"，每一路按各自的"配额(取多少个)"贡献候选，
    // 合并去重后整体交给下游排序层(ranker)精排。召回层不做融合打分——精细排序是排序层的职责。
    // 配额越大，该路在候选集中占的"坑位"越多，即对最终结果的潜在影响越大。
    private static final int QUOTA_HIGH_RATING = 300;   // 高分召回（非个性化兜底）
    private static final int QUOTA_ALS_OFFLINE = 100;   // 离线ALS召回（实际受离线预计算的topN限制，通常远小于此）
    private static final int QUOTA_EMB_ANN     = 400;   // Embedding/ANN召回（个性化主力）

    // ==================== 各路召回开关 ====================
    // 离线ALS这一路依赖Redis中离线写入的 rec:userId，无Redis时可关掉
    private static final boolean ENABLE_ALS_OFFLINE = true;
    private static final boolean ENABLE_EMB_ANN = true;

    /**
     * get recommendation movie list
     * @param userId input user id
     * @param size  size of similar items
     * @param model model used for calculating similarity
     * @return  list of similar movies
     */
    public static List<Movie> getRecList(int userId, int size, String model){
        User user = DataManager.getInstance().getUserById(userId);
        if (null == user){
            return new ArrayList<>();
        }

        //load user emb from redis if data source is redis
        // 注意：要在召回之前加载好用户Embedding，因为 Embedding/ANN 召回路依赖 user.getEmb()
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_REDIS)){
            String userEmbKey = "uEmb:" + userId;
            String userEmb = RedisClient.getInstance().get(userEmbKey);
            if (null != userEmb){
                user.setEmb(Utility.parseEmbStr(userEmb));
            }
        }

        if (Config.IS_LOAD_USER_FEATURE_FROM_REDIS){
            String userFeaturesKey = "uf:" + userId;
            Map<String, String> userFeatures = RedisClient.getInstance().hgetAll(userFeaturesKey);
            if (null != userFeatures){
                user.setUserFeatures(userFeatures);
            }
        }

        // 多路召回（配额制）：替换原来单路的 getMovies(800,"rating")
        List<Movie> candidates = multipleRetrievalRecall(user);

        List<Movie> rankedList = ranker(user, candidates, model);

        if (rankedList.size() > size){
            return rankedList.subList(0, size);
        }
        return rankedList;
    }

    /**
     * 多路召回（配额制）：每一路按各自配额取若干候选，合并去重后整体交给排序层。
     * <p>
     * 这是业界主流的召回范式：召回只负责"圈出一批不错的候选"，不负责精细排序；
     * 候选的最终顺序由下游 {@link #ranker} 决定。各路的影响力通过"配额(取多少个)"体现，
     * 而不是给每个候选算融合分。
     * <p>
     * 当前包含三路：
     *   1. 高分召回（非个性化兜底，配额 {@link #QUOTA_HIGH_RATING}）
     *   2. 离线ALS预计算召回（方案A，配额 {@link #QUOTA_ALS_OFFLINE}）
     *   3. Embedding/ANN 召回（用户向量近邻，配额 {@link #QUOTA_EMB_ANN}）
     *
     * @param user 目标用户
     * @return 合并去重后的候选集合（顺序为各路依次、首次出现的顺序，仅作排序层输入，不代表最终排序）
     */
    public static List<Movie> multipleRetrievalRecall(User user){
        if (null == user){
            return new ArrayList<>();
        }

        // 用 LinkedHashMap 按 movieId 去重，同时保留"首次被召回"的顺序
        LinkedHashMap<Integer, Movie> candidateMap = new LinkedHashMap<>();

        // 第一路：高分召回（非个性化兜底）
        mergeChannel(candidateMap, DataManager.getInstance().getMovies(QUOTA_HIGH_RATING, "rating"));

        // 第二路：离线ALS预计算召回（方案A）
        if (ENABLE_ALS_OFFLINE){
            mergeChannel(candidateMap, retrievalByAlsOffline(user, QUOTA_ALS_OFFLINE));
        }

        // 第三路：Embedding/ANN 召回（用户向量近邻）
        if (ENABLE_EMB_ANN){
            mergeChannel(candidateMap, retrievalByEmbedding(user, QUOTA_EMB_ANN));
        }

        return new ArrayList<>(candidateMap.values());
    }

    /**
     * 把某一路的召回结果并入候选集：按 movieId 去重，保留首次出现的顺序。
     * 用 movieId 做去重键，不依赖 Movie 的 equals/hashCode，更稳妥。
     *
     * @param candidateMap 累计候选集（movieId -> Movie）
     * @param candidates   某一路的召回结果
     */
    private static void mergeChannel(LinkedHashMap<Integer, Movie> candidateMap, List<Movie> candidates){
        if (null == candidates){
            return;
        }
        for (Movie m : candidates){
            if (null != m){
                candidateMap.putIfAbsent(m.getMovieId(), m);
            }
        }
    }

    /**
     * 离线ALS预计算召回（方案A）：读取离线 CollaborativeFiltering 算好的每用户 Top-N 推荐结果。
     * <p>
     * 数据来源按 {@link Config#EMB_DATA_SOURCE} 决定（与项目里 user embedding 的处理方式一致）：
     *   - 文件：启动时已由 DataManager 加载进内存（{@link User#getAlsRecMovieIds()}），这里直接读内存；
     *   - Redis：按请求读取 key=rec:&lt;userId&gt;，value 为按推荐分降序、空格分隔的 movieId 串。
     * <p>
     * 无对应数据（如新用户、离线作业未运行、Redis 不可用）时返回空列表，这一路自动失效，不影响其他路。
     *
     * @param user 目标用户
     * @param size 该路最多取多少候选
     * @return 候选电影列表（已按离线推荐分排序）
     */
    private static List<Movie> retrievalByAlsOffline(User user, int size){
        List<Movie> candidates = new ArrayList<>();

        // 第一步：拿到该用户的推荐 movieId 列表（内存 或 Redis）
        List<Integer> recMovieIds;
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_REDIS)){
            // 数据源为 Redis：按请求读取 rec:userId
            recMovieIds = new ArrayList<>();
            try {
                String recStr = RedisClient.getInstance().get("rec:" + user.getUserId());
                if (null != recStr && !recStr.isEmpty()){
                    for (String idStr : recStr.split(" ")){
                        if (!idStr.trim().isEmpty()){
                            recMovieIds.add(Integer.parseInt(idStr.trim()));
                        }
                    }
                }
            } catch (Exception e){
                // Redis 不可用 / 解析失败：这一路降级为空
                return candidates;
            }
        } else {
            // 数据源为文件：读取启动时已加载到内存的离线推荐
            recMovieIds = user.getAlsRecMovieIds();
        }

        if (null == recMovieIds){
            return candidates;
        }

        // 第二步：把 movieId 解析成 Movie 对象（两条来源在这里汇合，逻辑统一）
        for (int i = 0; i < recMovieIds.size() && candidates.size() < size; i++){
            Movie m = DataManager.getInstance().getMovieById(recMovieIds.get(i));
            if (null != m){
                candidates.add(m);
            }
        }
        return candidates;
    }

    /**
     * Embedding/ANN 召回：用用户向量在物品向量空间里检索最相似的 Top-N 电影。
     * <p>
     * ⚠️ 当前实现是"暴力遍历全库算余弦相似度"（与 SimilarMovieProcess.retrievalCandidatesByEmbedding 同思路），
     *    时间复杂度 O(物品数)。物品量大时这正是性能瓶颈所在。
     *    生产中应替换为 jelmerk/hnswlib 的 HNSW 索引近邻检索（支持内积/MIPS），把暴力扫描换成亚线性近似检索——
     *    届时离线把物品向量灌进索引，这里用 user.getEmb() 作为查询向量查 Top-N 即可。
     *
     * @param user 目标用户
     * @param size 该路最多取多少候选
     * @return 候选电影列表（已按相似度从高到低排序）
     */
    private static List<Movie> retrievalByEmbedding(User user, int size){
        List<Movie> candidates = new ArrayList<>();
        if (null == user || null == user.getEmb()){
            return candidates;
        }

        List<Movie> allCandidates = DataManager.getInstance().getMovies(10000, "rating");
        HashMap<Movie, Double> movieScoreMap = new HashMap<>();
        for (Movie candidate : allCandidates){
            if (null == candidate.getEmb()){
                continue;
            }
            double similarity = user.getEmb().calculateSimilarity(candidate.getEmb());
            movieScoreMap.put(candidate, similarity);
        }

        movieScoreMap.entrySet().stream()
                .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                .limit(size)
                .forEach(e -> candidates.add(e.getKey()));
        return candidates;
    }

    /**
     * rank candidates
     * @param user    input user
     * @param candidates    movie candidates
     * @param model     model name used for ranking
     * @return  ranked movie list
     */
    public static List<Movie> ranker(User user, List<Movie> candidates, String model){
        HashMap<Movie, Double> candidateScoreMap = new HashMap<>();

        switch (model){
            case "emb":
                for (Movie candidate : candidates){
                    double similarity = calculateEmbSimilarScore(user, candidate);
                    candidateScoreMap.put(candidate, similarity);
                }
                break;
            case "nerualcf":
                callNeuralCFTFServing(user, candidates, candidateScoreMap);
                break;
            case "embmlp":
                callEmbeddingMLPTFServing(user, candidates, candidateScoreMap);
                break;
            default:
                //default ranking in candidate set
                for (int i = 0 ; i < candidates.size(); i++){
                    candidateScoreMap.put(candidates.get(i), (double)(candidates.size() - i));
                }
        }

        List<Movie> rankedList = new ArrayList<>();
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m -> rankedList.add(m.getKey()));
        return rankedList;
    }

    /**
     * function to calculate similarity score based on embedding
     * @param user     input user
     * @param candidate candidate movie
     * @return  similarity score
     */
    public static double calculateEmbSimilarScore(User user, Movie candidate){
        if (null == user || null == candidate || null == user.getEmb()){
            return -1;
        }
        return user.getEmb().calculateSimilarity(candidate.getEmb());
    }

    /**
     * call TenserFlow serving to get the NeuralCF model inference result
     * @param user              input user
     * @param candidates        candidate movies
     * @param candidateScoreMap save prediction score into the score map
     */
    public static void callNeuralCFTFServing(User user, List<Movie> candidates, HashMap<Movie, Double> candidateScoreMap){
        if (null == user || null == candidates || candidates.size() == 0){
            return;
        }

        JSONArray instances = new JSONArray();
        for (Movie m : candidates){
            JSONObject instance = new JSONObject();
            instance.put("userId", user.getUserId());
            instance.put("movieId", m.getMovieId());
            instances.put(instance);
        }

        JSONObject instancesRoot = new JSONObject();
        instancesRoot.put("instances", instances);

        //need to confirm the tf serving end point
        // curl -X POST "http://localhost:8501/v1/models/sparrow_ncf:predict" -H "Content-Type: application/json" -d "{\"instances\":[{\"movieId\":223,\"userId\":9077}]}"
        String predictionScores = asyncSinglePostRequest(Config.TF_SERVING_NCF_URL, instancesRoot.toString());
        System.out.println("send user" + user.getUserId() + " request to tf serving.");

        JSONObject predictionsObject = new JSONObject(predictionScores);
        JSONArray scores = predictionsObject.getJSONArray("predictions");
        for (int i = 0 ; i < candidates.size(); i++){
            candidateScoreMap.put(candidates.get(i), scores.getJSONArray(i).getDouble(0));
        }
    }

    /**
     * call TensorFlow Serving to get the EmbeddingMLP model inference result.
     * <p>
     * The request payload uses the TF Serving "instances" format. Each instance contains
     * 17 features that the EmbeddingMLP model expects:
     * <pre>
     *   movieId, userId, releaseYear,
     *   movieRatingCount, movieAvgRating, movieRatingStddev,
     *   userRatingCount, userAvgRating, userRatingStddev,
     *   userGenre1..userGenre5,
     *   movieGenre1..movieGenre3
     * </pre>
     * <p>
     * All features are pre-computed offline by Spark (FeatureEngForRecModel) and stored in Redis:
     * - User features (uf:userId) loaded in {@link RecForYouProcess#getRecList}
     * - Movie features (mf:movieId) loaded in {@link DataManager#loadMovieFeatures}
     *
     * @param user              input user
     * @param candidates        candidate movies
     * @param candidateScoreMap save prediction score into the score map
     */
    public static void callEmbeddingMLPTFServing(User user, List<Movie> candidates, HashMap<Movie, Double> candidateScoreMap){
        if (null == user || null == candidates || candidates.size() == 0){
            return;
        }

        Map<String, String> uf = user.getUserFeatures();

        JSONArray instances = new JSONArray();
        List<Movie> validCandidates = new ArrayList<>();
        for (Movie m : candidates){
            Map<String, String> mf = m.getMovieFeatures();
            if (null == mf || null == uf){
                continue;
            }

            // All values wrapped in single-element arrays so TF Serving parses them as [batch, 1]
            // matching the model's expected input shape.
            JSONObject instance = new JSONObject();

            // --- Movie features (from Redis mf:movieId) ---
            instance.put("movieId", new JSONArray().put(m.getMovieId()));
            instance.put("releaseYear", new JSONArray().put(Double.parseDouble(mf.getOrDefault("releaseYear", "0"))));
            instance.put("movieRatingCount", new JSONArray().put(Double.parseDouble(mf.getOrDefault("movieRatingCount", "0"))));
            instance.put("movieAvgRating", new JSONArray().put(Double.parseDouble(mf.getOrDefault("movieAvgRating", "0"))));
            instance.put("movieRatingStddev", new JSONArray().put(Double.parseDouble(mf.getOrDefault("movieRatingStddev", "0"))));
            instance.put("movieGenre1", new JSONArray().put(mf.getOrDefault("movieGenre1", "None")));
            instance.put("movieGenre2", new JSONArray().put(mf.getOrDefault("movieGenre2", "None")));
            instance.put("movieGenre3", new JSONArray().put(mf.getOrDefault("movieGenre3", "None")));

            // --- User features (from Redis uf:userId) ---
            instance.put("userId", new JSONArray().put(user.getUserId()));
            instance.put("userRatingCount", new JSONArray().put(Double.parseDouble(uf.getOrDefault("userRatingCount", "0"))));
            instance.put("userAvgRating", new JSONArray().put(Double.parseDouble(uf.getOrDefault("userAvgRating", "0"))));
            instance.put("userRatingStddev", new JSONArray().put(Double.parseDouble(uf.getOrDefault("userRatingStddev", "0"))));
            instance.put("userGenre1", new JSONArray().put(uf.getOrDefault("userGenre1", "None")));
            instance.put("userGenre2", new JSONArray().put(uf.getOrDefault("userGenre2", "None")));
            instance.put("userGenre3", new JSONArray().put(uf.getOrDefault("userGenre3", "None")));
            instance.put("userGenre4", new JSONArray().put(uf.getOrDefault("userGenre4", "None")));
            instance.put("userGenre5", new JSONArray().put(uf.getOrDefault("userGenre5", "None")));

            instances.put(instance);
            validCandidates.add(m);
        }

        if (instances.isEmpty()){
            return;
        }

        JSONObject requestBody = new JSONObject();
        requestBody.put("instances", instances);

        String predictionScores = asyncSinglePostRequest(Config.TF_SERVING_MLP_URL, requestBody.toString());
        System.out.println("send user" + user.getUserId() + " request to EmbeddingMLP tf serving.");

        JSONObject predictionsObject = new JSONObject(predictionScores);
        JSONArray scores = predictionsObject.getJSONArray("predictions");
        for (int i = 0; i < validCandidates.size(); i++){
            candidateScoreMap.put(validCandidates.get(i), scores.getJSONArray(i).getDouble(0));
        }
    }
}
