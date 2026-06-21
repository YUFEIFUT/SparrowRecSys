package com.sparrowrecsys.online.datamanager;

import com.sparrowrecsys.online.util.Config;
import com.sparrowrecsys.online.util.Utility;

import java.io.File;
import java.util.*;

/**
 * DataManager is an utility class, takes charge of all data loading logic.
 */

public class DataManager {
    //singleton instance
    private static volatile DataManager instance;
    HashMap<Integer, Movie> movieMap;
    HashMap<Integer, User> userMap;
    //genre reverse index for quick querying all movies in a genre
    HashMap<String, List<Movie>> genreReverseIndexMap;

    private DataManager(){
        this.movieMap = new HashMap<>();
        this.userMap = new HashMap<>();
        this.genreReverseIndexMap = new HashMap<>();
        instance = this;
    }

    public static DataManager getInstance(){
        if (null == instance){
            synchronized (DataManager.class){
                if (null == instance){
                    instance = new DataManager();
                }
            }
        }
        return instance;
    }

    //load data from file system including movie, rating, link data and model data like embedding vectors.
    public void loadData(String movieDataPath, String linkDataPath, String ratingDataPath, String movieEmbPath, String userEmbPath, String userRecPath, String movieRedisKey, String userRedisKey) throws Exception{
        loadMovieData(movieDataPath);
        loadLinkData(linkDataPath);
        loadRatingData(ratingDataPath);
        loadMovieEmb(movieEmbPath, movieRedisKey);
        if (Config.IS_LOAD_ITEM_FEATURE_FROM_REDIS){
            loadMovieFeatures("mf:");
        }

        loadUserEmb(userEmbPath, userRedisKey);
        loadUserRecs(userRecPath);
    }

    //load movie data from movies.csv
    private void loadMovieData(String movieDataPath) throws Exception{
        System.out.println("Loading movie data from " + movieDataPath + " ...");
        boolean skipFirstLine = true;
        try (Scanner scanner = new Scanner(new File(movieDataPath))) {
            while (scanner.hasNextLine()) {
                String movieRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] movieData = movieRawData.split(",");
                if (movieData.length == 3){
                    Movie movie = new Movie();
                    movie.setMovieId(Integer.parseInt(movieData[0]));
                    int releaseYear = parseReleaseYear(movieData[1].trim());
                    if (releaseYear == -1){
                        movie.setTitle(movieData[1].trim());
                    }else{
                        movie.setReleaseYear(releaseYear);
                        movie.setTitle(movieData[1].trim().substring(0, movieData[1].trim().length()-6).trim());
                    }
                    String genres = movieData[2];
                    if (!genres.trim().isEmpty()){
                        String[] genreArray = genres.split("\\|");
                        for (String genre : genreArray){
                            movie.addGenre(genre);
                            addMovie2GenreIndex(genre, movie);
                        }
                    }
                    this.movieMap.put(movie.getMovieId(), movie);
                }
            }
        }
        System.out.println("Loading movie data completed. " + this.movieMap.size() + " movies in total.");
    }

    //load movie embedding
    private void loadMovieEmb(String movieEmbPath, String embKey) throws Exception{
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)) {
            System.out.println("Loading movie embedding from " + movieEmbPath + " ...");
            int validEmbCount = 0;
            try (Scanner scanner = new Scanner(new File(movieEmbPath))) {
                while (scanner.hasNextLine()) {
                    String movieRawEmbData = scanner.nextLine();
                    String[] movieEmbData = movieRawEmbData.split(":");
                    if (movieEmbData.length == 2) {
                        Movie m = getMovieById(Integer.parseInt(movieEmbData[0]));
                        if (null == m) {
                            continue;
                        }
                        m.setEmb(Utility.parseEmbStr(movieEmbData[1]));
                        validEmbCount++;
                    }
                }
            }
            System.out.println("Loading movie embedding completed. " + validEmbCount + " movie embeddings in total.");
        }else{
            System.out.println("Loading movie embedding from Redis ...");
            Set<String> movieEmbKeys = RedisClient.getInstance().keys(embKey + "*");
            int validEmbCount = 0;
            for (String movieEmbKey : movieEmbKeys){
                String movieId = movieEmbKey.split(":")[1];
                Movie m = getMovieById(Integer.parseInt(movieId));
                if (null == m) {
                    continue;
                }
                m.setEmb(Utility.parseEmbStr(RedisClient.getInstance().get(movieEmbKey)));
                validEmbCount++;
            }
            System.out.println("Loading movie embedding completed. " + validEmbCount + " movie embeddings in total.");
        }
    }

    //load movie features
    private void loadMovieFeatures(String movieFeaturesPrefix) throws Exception{
        System.out.println("Loading movie features from Redis ...");
        Set<String> movieFeaturesKeys = RedisClient.getInstance().keys(movieFeaturesPrefix + "*");
        int validFeaturesCount = 0;
        for (String movieFeaturesKey : movieFeaturesKeys){
            String movieId = movieFeaturesKey.split(":")[1];
            Movie m = getMovieById(Integer.parseInt(movieId));
            if (null == m) {
                continue;
            }
            m.setMovieFeatures(RedisClient.getInstance().hgetAll(movieFeaturesKey));
            validFeaturesCount++;
        }
        System.out.println("Loading movie features completed. " + validFeaturesCount + " movie features in total.");
    }

    //load user embedding
    private void loadUserEmb(String userEmbPath, String embKey) throws Exception{
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)) {
            System.out.println("Loading user embedding from " + userEmbPath + " ...");
            int validEmbCount = 0;
            try (Scanner scanner = new Scanner(new File(userEmbPath))) {
                while (scanner.hasNextLine()) {
                    String userRawEmbData = scanner.nextLine();
                    String[] userEmbData = userRawEmbData.split(":");
                    if (userEmbData.length == 2) {
                        User u = getUserById(Integer.parseInt(userEmbData[0]));
                        if (null == u) {
                            continue;
                        }
                        u.setEmb(Utility.parseEmbStr(userEmbData[1]));
                        validEmbCount++;
                    }
                }
            }
            System.out.println("Loading user embedding completed. " + validEmbCount + " user embeddings in total.");
        }
    }

    //load user precomputed ALS recommendations (方案A：离线ALS预计算召回)
    //文件格式：每行 userId:m1 m2 m3 （冒号后为按推荐分降序、空格分隔的movieId串，与离线AlsModelExporter写出格式一致）
    //仅在数据源为文件时启动加载到内存（User对象）；数据源为Redis时由 RecForYouProcess 按请求读取 rec:userId
    private void loadUserRecs(String userRecPath) throws Exception{
        // 如果数据源不是文件模式，直接跳过（说明线上走的是Redis读取路径）
        if (!Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)){
            return;
        }
        File recFile = new File(userRecPath);
        //离线推荐结果是可选产物，文件不存在时跳过（如离线作业尚未运行），不阻断服务启动
        if (!recFile.exists()){
            System.out.println("User recs file not found, skip loading: " + userRecPath);
            return;
        }
        System.out.println("Loading user recs from " + userRecPath + " ...");
        int validCount = 0;
        try (Scanner scanner = new Scanner(recFile)) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] data = line.split(":");
                if (data.length == 2) {
                    User u = getUserById(Integer.parseInt(data[0].trim()));
                    if (null == u) {
                        continue;
                    }
                    List<Integer> recMovieIds = new ArrayList<>();
                    for (String idStr : data[1].split(" ")) {
                        if (!idStr.trim().isEmpty()) {
                            recMovieIds.add(Integer.parseInt(idStr.trim()));
                        }
                    }
                    u.setAlsRecMovieIds(recMovieIds);
                    validCount++;
                }
            }
        }
        System.out.println("Loading user recs completed. " + validCount + " user recs in total.");
    }

    //parse release year
    private int parseReleaseYear(String rawTitle){
        if (null == rawTitle || rawTitle.trim().length() < 6){
            return -1;
        }else{
            String yearString = rawTitle.trim().substring(rawTitle.length()-5, rawTitle.length()-1);
            try{
                return Integer.parseInt(yearString);
            }catch (NumberFormatException exception){
                return -1;
            }
        }
    }

    //load links data from links.csv
    private void loadLinkData(String linkDataPath) throws Exception{
        System.out.println("Loading link data from " + linkDataPath + " ...");
        int count = 0;
        boolean skipFirstLine = true;
        try (Scanner scanner = new Scanner(new File(linkDataPath))) {
            while (scanner.hasNextLine()) {
                String linkRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] linkData = linkRawData.split(",");
                if (linkData.length == 3){
                    int movieId = Integer.parseInt(linkData[0]);
                    Movie movie = this.movieMap.get(movieId);
                    if (null != movie){
                        count++;
                        movie.setImdbId(linkData[1].trim());
                        movie.setTmdbId(linkData[2].trim());
                    }
                }
            }
        }
        System.out.println("Loading link data completed. " + count + " links in total.");
    }

    //load ratings data from ratings.csv
    private void loadRatingData(String ratingDataPath) throws Exception{
        System.out.println("Loading rating data from " + ratingDataPath + " ...");
        boolean skipFirstLine = true;
        int count = 0;
        try (Scanner scanner = new Scanner(new File(ratingDataPath))) {
            while (scanner.hasNextLine()) {
                String ratingRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] linkData = ratingRawData.split(",");
                if (linkData.length == 4){
                    count ++;
                    Rating rating = new Rating();
                    rating.setUserId(Integer.parseInt(linkData[0]));
                    rating.setMovieId(Integer.parseInt(linkData[1]));
                    rating.setScore(Float.parseFloat(linkData[2]));
                    rating.setTimestamp(Long.parseLong(linkData[3]));
                    Movie movie = this.movieMap.get(rating.getMovieId());
                    if (null != movie){
                        movie.addRating(rating);
                    }
                    if (!this.userMap.containsKey(rating.getUserId())){
                        User user = new User();
                        user.setUserId(rating.getUserId());
                        this.userMap.put(user.getUserId(), user);
                    }
                    this.userMap.get(rating.getUserId()).addRating(rating);
                }
            }
        }

        System.out.println("Loading rating data completed. " + count + " ratings in total.");
    }

    //add movie to genre reversed index
    private void addMovie2GenreIndex(String genre, Movie movie){
        if (!this.genreReverseIndexMap.containsKey(genre)){
            this.genreReverseIndexMap.put(genre, new ArrayList<>());
        }
        this.genreReverseIndexMap.get(genre).add(movie);
    }

    /**
     * 根据电影类型获取电影列表，并按指定方式排序
     * 这是首页电影列表排序的核心方法，实现了按历史用户平均评分排序的逻辑
     * 
     * @param genre   电影类型（如Action、Romance）
     * @param size    返回的电影数量
     * @param sortBy  排序方式：
     *                - "rating": 按平均评分降序（首页默认使用）
     *                - "releaseYear": 按上映年份降序
     *                - "popularity": 按热度降序（即被评价次数）
     * @return 排序后的电影列表，最多返回size个
     */
    public List<Movie> getMoviesByGenre(String genre, int size, String sortBy){
        if (null != genre){
            // 从类型倒排索引中获取该类型的所有电影
            List<Movie> movies = new ArrayList<>(this.genreReverseIndexMap.get(genre));

            // 调用公共排序方法
            sortMovies(movies, sortBy);

            // 如果电影数量超过size，截取前size个返回
            if (movies.size() > size){
                return movies.subList(0, size);
            }
            return movies;
        }
        return null;
    }

    /**
     * 根据指定的排序方式对电影列表进行排序
     * @param movies 待排序的电影列表（会被原地修改）
     * @param sortBy 排序方式：
     *               - "rating": 按平均评分降序
     *               - "releaseYear": 按上映年份降序
     *               - "popularity": 按热度（被评价次数）降序
     */
    private void sortMovies(List<Movie> movies, String sortBy){
        switch (sortBy){
            // 按平均评分降序排序
            case "rating":
                movies.sort((m1, m2) -> Double.compare(m2.getAverageRating(), m1.getAverageRating()));
                break;
            // 按上映年份降序排序
            case "releaseYear":
                movies.sort((m1, m2) -> Integer.compare(m2.getReleaseYear(), m1.getReleaseYear()));
                break;
            // 按热度降序排序（热度 = 被评价次数）
            case "popularity":
                movies.sort((m1, m2) -> Integer.compare(m2.getRatingNumber(), m1.getRatingNumber()));
                break;
            default:
        }
    }

    //get top N movies order by sortBy method
    public List<Movie> getMovies(int size, String sortBy){
            List<Movie> movies = new ArrayList<>(movieMap.values());
            // 调用公共排序方法
            sortMovies(movies, sortBy);

            if (movies.size() > size){
                return movies.subList(0, size);
            }
            return movies;
    }

    //get movie object by movie id
    public Movie getMovieById(int movieId){
        return this.movieMap.get(movieId);
    }

    //get user object by user id
    public User getUserById(int userId){
        return this.userMap.get(userId);
    }
}
