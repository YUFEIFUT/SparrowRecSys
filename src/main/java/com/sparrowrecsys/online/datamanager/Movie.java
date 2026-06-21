package com.sparrowrecsys.online.datamanager;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.sparrowrecsys.online.model.Embedding;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Movie Class, contains attributes loaded from movielens movies.csv and other advanced data like averageRating, emb, etc.
 */
public class Movie {
    int movieId;
    String title;
    int releaseYear;
    String imdbId;
    String tmdbId;
    List<String> genres;
    //how many user rate the movie
    int ratingNumber;
    //average rating score
    double averageRating;

    //embedding of the movie
    @JsonIgnore
    Embedding emb;

    //LSH bucket keys the movie falls into (one key per hash table), used for embedding-based retrieval
    @JsonIgnore
    List<String> embBuckets;

    //all rating scores list
    @JsonIgnore
    List<Rating> ratings;

    @JsonIgnore
    Map<String, String> movieFeatures;

    final int TOP_RATING_SIZE = 10;

    @JsonSerialize(using = RatingListSerializer.class)
    List<Rating> topRatings;

    public Movie() {
        ratingNumber = 0;
        averageRating = 0;
        this.genres = new ArrayList<>();
        this.ratings = new ArrayList<>();
        this.topRatings = new LinkedList<>();
        this.emb = null;
        this.movieFeatures = null;
    }

    public int getMovieId() {
        return movieId;
    }

    public void setMovieId(int movieId) {
        this.movieId = movieId;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public int getReleaseYear() {
        return releaseYear;
    }

    public void setReleaseYear(int releaseYear) {
        this.releaseYear = releaseYear;
    }

    public List<String> getGenres() {
        return genres;
    }

    public void addGenre(String genre){
        this.genres.add(genre);
    }

    public void setGenres(List<String> genres) {
        this.genres = genres;
    }

    public List<Rating> getRatings() {
        return ratings;
    }

    /**
     * 添加用户评分并动态更新平均评分
     * 这是计算"历史用户平均打分"的核心方法，采用增量计算方式
     * 
     * @param rating 用户对该电影的评分对象
     */
    public void addRating(Rating rating) {
        // 增量计算平均评分：(当前平均分 * 已有评分数 + 新评分) / (已有评分数 + 1)
        // 这种方式避免了每次都重新遍历所有评分，提高计算效率
        averageRating = (averageRating * ratingNumber + rating.getScore()) / (ratingNumber + 1);
        
        // 评分数量加1
        ratingNumber++;
        
        // 将评分加入评分列表
        this.ratings.add(rating);
        
        // 维护Top评分列表（用于展示喜欢该电影的用户）
        addTopRating(rating);
    }

    /**
     * 将新评分插入到 topRatings 列表中的正确位置，保持升序排列
     * topRatings 用于记录对该电影评分最高的前N个用户（默认TOP_RATING_SIZE=10）
     * 列表升序排列，超过容量时删除最小的（第一个元素），保留最大的10个
     * 此方法在电影详情页显示"Who likes the movie most"时使用
     * 
     * @param rating 用户对该电影的评分对象
     */
    public void addTopRating(Rating rating){
        // 如果列表为空，直接添加
        if (this.topRatings.isEmpty()){
            this.topRatings.add(rating);
        }else{
            // 找到合适的插入位置（按评分升序排列）
            int index = 0;
            for (Rating topRating : this.topRatings){
                // 找到第一个评分 >= 当前评分的位置，插入到其前面（保持升序）
                if (topRating.getScore() >= rating.getScore()) {
                    break;
                }
                index++;
            }
            // 插入到指定位置
            topRatings.add(index, rating);
            
            // 如果超过最大容量，移除评分最低的（列表第一个元素，因为是升序排列）
            if (topRatings.size() > TOP_RATING_SIZE) {
                topRatings.remove(0);
            }
        }
    }

    public String getImdbId() {
        return imdbId;
    }

    public void setImdbId(String imdbId) {
        this.imdbId = imdbId;
    }

    public String getTmdbId() {
        return tmdbId;
    }

    public void setTmdbId(String tmdbId) {
        this.tmdbId = tmdbId;
    }

    public int getRatingNumber() {
        return ratingNumber;
    }

    public double getAverageRating() {
        return averageRating;
    }

    public Embedding getEmb() {
        return emb;
    }

    public void setEmb(Embedding emb) {
        this.emb = emb;
    }

    public List<String> getEmbBuckets() {
        return embBuckets;
    }

    public void setEmbBuckets(List<String> embBuckets) {
        this.embBuckets = embBuckets;
    }

    public Map<String, String> getMovieFeatures() {
        return movieFeatures;
    }

    public void setMovieFeatures(Map<String, String> movieFeatures) {
        this.movieFeatures = movieFeatures;
    }
}
