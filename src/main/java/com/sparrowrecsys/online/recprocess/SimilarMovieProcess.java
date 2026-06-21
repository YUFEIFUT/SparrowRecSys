package com.sparrowrecsys.online.recprocess;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.datamanager.Movie;
import java.util.*;

/**
 * Recommendation process of similar movies
 */

public class SimilarMovieProcess {

    /**
     * get recommendation movie list
     * @param movieId input movie id
     * @param size  size of similar items
     * @param model model used for calculating similarity
     * @return  list of similar movies
     */
    public static List<Movie> getRecList(int movieId, int size, String model){
        Movie movie = DataManager.getInstance().getMovieById(movieId);
        if (null == movie){
            return new ArrayList<>();
        }
        List<Movie> candidates = candidateGenerator(movie);
        List<Movie> rankedList = ranker(movie, candidates, model);

        if (rankedList.size() > size){
            return rankedList.subList(0, size);
        }
        return rankedList;
    }

    /**
     * generate candidates for similar movies recommendation
     * 单策略召回
     * @param movie input movie object
     * @return  movie candidates
     */
    public static List<Movie> candidateGenerator(Movie movie){
        HashMap<Integer, Movie> candidateMap = new HashMap<>();
        for (String genre : movie.getGenres()){
            // 按电影类型召回，每类取100部高分电影
            List<Movie> oneCandidates = DataManager.getInstance().getMoviesByGenre(genre, 100, "rating");
            for (Movie candidate : oneCandidates){
                candidateMap.put(candidate.getMovieId(), candidate);
            }
        }
        candidateMap.remove(movie.getMovieId());
        return new ArrayList<>(candidateMap.values());
    }

    /**
     * multiple-retrieval candidate generation method：多路召回
     * @param movie input movie object
     * @return movie candidates
     */
    public static List<Movie> multipleRetrievalCandidates(Movie movie){
        if (null == movie){
            return null;
        }

        HashSet<String> genres = new HashSet<>(movie.getGenres());

        HashMap<Integer, Movie> candidateMap = new HashMap<>();
        // 第一路：按类型召回（每类20部）
        for (String genre : genres){
            List<Movie> oneCandidates = DataManager.getInstance().getMoviesByGenre(genre, 20, "rating");
            for (Movie candidate : oneCandidates){
                candidateMap.put(candidate.getMovieId(), candidate);
            }
        }

        // 第二路：高分电影召回（100部）
        List<Movie> highRatingCandidates = DataManager.getInstance().getMovies(100, "rating");
        for (Movie candidate : highRatingCandidates){
            candidateMap.put(candidate.getMovieId(), candidate);
        }

        // 第三路：最新电影召回（100部）
        List<Movie> latestCandidates = DataManager.getInstance().getMovies(100, "releaseYear");
        for (Movie candidate : latestCandidates){
            candidateMap.put(candidate.getMovieId(), candidate);
        }

        candidateMap.remove(movie.getMovieId());
        return new ArrayList<>(candidateMap.values());
    }

    /**
     * embedding based candidate generation method：Embedding 召回
     * @param movie input movie
     * @param size  size of candidate pool
     * @return  movie candidates
     */
    public static List<Movie> retrievalCandidatesByEmbedding(Movie movie, int size){
        if (null == movie || null == movie.getEmb()){
            return null;
        }

        // 获取所有候选电影（10000部）
        List<Movie> allCandidates = DataManager.getInstance().getMovies(10000, "rating");
        HashMap<Movie,Double> movieScoreMap = new HashMap<>();
        // 计算每部电影的 Embedding 相似度
        for (Movie candidate : allCandidates){
            double similarity = calculateEmbSimilarScore(movie, candidate);
            movieScoreMap.put(candidate, similarity);
        }

        // 按相似度排序，返回 Top-K
        List<Map.Entry<Movie,Double>> movieScoreList = new ArrayList<>(movieScoreMap.entrySet());
        movieScoreList.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));

        List<Movie> candidates = new ArrayList<>();
        for (Map.Entry<Movie,Double> movieScoreEntry : movieScoreList){
            candidates.add(movieScoreEntry.getKey());
        }

        return candidates.subList(0, Math.min(candidates.size(), size));
    }

    /**
     * embedding based candidate generation method
     * @param movie input movie
     * @return  movie candidates
     */
    public static List<Movie> retrievalCandidatesByEmbeddingLSH(Movie movie){
        return retrievalCandidatesByEmbeddingLSH(movie, Integer.MAX_VALUE);
    }

    /**
     * embedding based candidate generation method with LSH
     * @param movie input movie
     * @param size  size of candidate pool
     * @return  movie candidates
     */
    public static List<Movie> retrievalCandidatesByEmbeddingLSH(Movie movie, int size){
        if (null == movie || null == movie.getEmb()){
            return null;
        }

        //use the LSH bucket index to narrow down the candidate pool in (near) constant time,
        //then rank the small candidate set by exact embedding similarity.
        List<Movie> allCandidates = retrievalCandidatesByLSH(movie);
        HashMap<Movie,Double> movieScoreMap = new HashMap<>();
        for (Movie candidate : allCandidates){
            double similarity = calculateEmbSimilarScore(movie, candidate);
            movieScoreMap.put(candidate, similarity);
        }

        List<Map.Entry<Movie,Double>> movieScoreList = new ArrayList<>(movieScoreMap.entrySet());
        //sort by similarity descending so the most similar movies come first
        movieScoreList.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));

        List<Movie> candidates = new ArrayList<>();
        for (Map.Entry<Movie,Double> movieScoreEntry : movieScoreList){
            candidates.add(movieScoreEntry.getKey());
        }

        return candidates.subList(0, Math.min(candidates.size(), size));
    }

    /**
     * retrieve the candidate pool by the offline-generated LSH buckets, using the "OR" multi-bucket
     * strategy with multi-probe (neighbouring buckets). Fall back to a full scan only when the bucket
     * data is missing or even multi-probe still returns too few candidates.
     * @param movie input movie
     * @return movie candidates (pre-rank pool)
     */
    public static List<Movie> retrievalCandidatesByLSH(Movie movie){
        List<Movie> candidates = DataManager.getInstance().getLshCandidates(movie);

        //fallback: bucket data not loaded or candidate pool too small, scan all movies with embedding
        if (candidates.size() < 10){
            List<Movie> allMovies = DataManager.getInstance().getMovies(10000, "rating");
            candidates = new ArrayList<>();
            for (Movie candidate : allMovies){
                if (candidate.getMovieId() != movie.getMovieId() && candidate.getEmb() != null){
                    candidates.add(candidate);
                }
            }
        }
        return candidates;
    }

    /**
     * rank candidates
     * @param movie    input movie
     * @param candidates    movie candidates
     * @param model     model name used for ranking
     * @return  ranked movie list
     */
    public static List<Movie> ranker(Movie movie, List<Movie> candidates, String model){
        HashMap<Movie, Double> candidateScoreMap = new HashMap<>();
        for (Movie candidate : candidates){
            double similarity;
            switch (model){
                case "emb":
                    similarity = calculateEmbSimilarScore(movie, candidate);
                    break;
                default:
                    similarity = calculateSimilarScore(movie, candidate);
            }
            candidateScoreMap.put(candidate, similarity);
        }
        List<Movie> rankedList = new ArrayList<>();
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m -> rankedList.add(m.getKey()));
        return rankedList;
    }

    /**
     * 计算候选电影与输入电影之间的相似度评分。
     * <p>
     * 评分由两部分加权组成：
     * 1. 类型相似度（权重 0.7）：衡量两部电影共有类型的比例，采用 Jaccard 系数形式，
     * 2. 评分分数（权重 0.3）：将候选电影的平均评分归一化到 [0, 1] 区间（满分 5 分）。
     * <p>
     * 最终得分 = genreSimilarity × 0.7 + ratingScore × 0.3
     *
     * @param movie     输入电影，作为比较的基准
     * @param candidate 候选电影，待评估的推荐对象
     * @return 相似度评分
     */
    public static double calculateSimilarScore(Movie movie, Movie candidate){
        // 统计两部电影共同拥有的类型数量
        int sameGenreCount = 0;
        for (String genre : movie.getGenres()){
            if (candidate.getGenres().contains(genre)){
                sameGenreCount++;
            }
        }

        // 计算类型相似度：共有类型数 / 两部电影类型总数之和 / 2
        // 计算类型相似度：相同类型数 / (两部电影类型总数)
        // 注意：这里 "/ 2" 的运算优先级有问题，实际等价于 sameGenreCount / (A + B) / 2
        // 两部电影类型完全相同时，最大值也只能到 0.25，导致 genreSimilarity 权重被大幅压缩
        //
        // 如果本意是 Jaccard 相似度，应为：
        //   genreSimilarity = (double)sameGenreCount / (movie.getGenres().size() + candidate.getGenres().size() - sameGenreCount);
        // 如果本意是按平均类型数归一化，应为：
        //   genreSimilarity = (double)sameGenreCount / ((movie.getGenres().size() + candidate.getGenres().size()) / 2.0);
        double genreSimilarity = (double)sameGenreCount / (movie.getGenres().size() + candidate.getGenres().size()) * 2;
        // 将候选电影的平均评分（满分 5 分）归一化到 [0, 1] 区间
        double ratingScore = candidate.getAverageRating() / 5;

        // 定义两个指标的权重：类型相似度占 70%，评分质量占 30%
        double similarityWeight = 0.7;
        double ratingScoreWeight = 0.3;

        // 返回加权综合得分
        return genreSimilarity * similarityWeight + ratingScore * ratingScoreWeight;
    }

    /**
     * function to calculate similarity score based on embedding
     * @param movie     input movie
     * @param candidate candidate movie
     * @return  similarity score
     */
    public static double calculateEmbSimilarScore(Movie movie, Movie candidate){
        if (null == movie || null == candidate){
            return -1;
        }
        return movie.getEmb().calculateSimilarity(candidate.getEmb());
    }
}
