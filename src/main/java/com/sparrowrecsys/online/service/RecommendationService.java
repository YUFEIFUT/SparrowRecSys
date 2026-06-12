package com.sparrowrecsys.online.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.datamanager.Movie;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;

/**
 * RecommendationService - 推荐服务
 * 提供基于不同输入条件的电影推荐服务
 * 主要用于首页各类型电影列表的获取，支持按评分或年份排序
 */

public class RecommendationService extends HttpServlet {
    
    /**
     * 处理GET请求，返回指定类型的电影列表
     * @param request  HTTP请求对象，包含三个关键参数：
     *                 - genre: 电影类型（如Action、Romance）
     *                 - size: 返回电影数量
     *                 - sortby: 排序方式（rating按评分，releaseYear按年份）
     * @param response HTTP响应对象，返回JSON格式的电影列表
     */
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws ServletException,
            IOException {
        try {
            // 设置响应头信息
            response.setContentType("application/json");      // 返回JSON格式
            response.setStatus(HttpServletResponse.SC_OK);    // HTTP 200状态码
            response.setCharacterEncoding("UTF-8");           // UTF-8编码
            response.setHeader("Access-Control-Allow-Origin", "*");  // 允许跨域

            // 从请求参数中获取查询条件
            String genre = request.getParameter("genre");   // 电影类型（如Action）
            String size = request.getParameter("size");     // 返回数量
            String sortby = request.getParameter("sortby"); // 排序方式

            // 核心逻辑：调用DataManager获取指定类型的电影列表
            // 排序逻辑在DataManager.getMoviesByGenre()中实现
            List<Movie> movies = DataManager.getInstance().getMoviesByGenre(genre, Integer.parseInt(size), sortby);

            // 将电影列表转换为JSON格式并返回
            ObjectMapper mapper = new ObjectMapper();
            String jsonMovies = mapper.writeValueAsString(movies);
            response.getWriter().println(jsonMovies);

        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println("");
        }
    }
}
