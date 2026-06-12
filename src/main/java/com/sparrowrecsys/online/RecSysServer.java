package com.sparrowrecsys.online;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.service.*;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.DefaultServlet;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.resource.Resource;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;

/**
 * 推荐系统在线服务主类
 * 负责启动Jetty服务器，提供推荐系统的Web服务和API接口
 * 包括电影推荐、用户信息、相似电影等RESTful服务
 */
public class RecSysServer {

    /**
     * 程序入口，创建服务器实例并启动
     * @param args 命令行参数（未使用）
     * @throws Exception 服务器启动异常
     */
    public static void main(String[] args) throws Exception {
        new RecSysServer().run();
    }

    /** 默认服务端口 */
    private static final int DEFAULT_PORT = 6010;

    /**
     * 启动推荐系统服务器
     * 包括：端口配置、静态资源加载、数据初始化、服务绑定、服务器启动
     * @throws Exception 服务器启动过程中的异常
     */
    public void run() throws Exception{

        // 配置服务端口：优先使用环境变量PORT，否则使用默认端口6010
        int port = DEFAULT_PORT;
        try {
            port = Integer.parseInt(System.getenv("PORT"));
        } catch (NumberFormatException ignored) {}

        // 创建服务器绑定地址：监听所有网络接口（0.0.0.0）
        InetSocketAddress inetAddress = new InetSocketAddress("0.0.0.0", port);
        Server server = new Server(inetAddress);

        // 获取Web根目录路径（包含index.html的目录）
        URL webRootLocation = this.getClass().getResource("/webroot/index.html");
        if (webRootLocation == null)
        {
            throw new IllegalStateException("Unable to determine webroot URL location");
        }

        // 将index.html路径转换为根目录URI，用于静态资源服务
        URI webRootUri = URI.create(webRootLocation.toURI().toASCIIString().replaceFirst("/index.html$","/"));
        System.out.printf("Web Root URI: %s%n", webRootUri.getPath());

        // 初始化数据管理器：加载电影数据、用户数据、Embedding向量等
        DataManager.getInstance().loadData(webRootUri.getPath() + "sampledata/movies.csv",
                webRootUri.getPath() + "sampledata/links.csv",webRootUri.getPath() + "sampledata/ratings.csv",
                webRootUri.getPath() + "modeldata/item2vecEmb.csv",
                webRootUri.getPath() + "modeldata/userEmb.csv",
                "i2vEmb", "uEmb");

        // 配置Servlet上下文处理器
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath("/");  // 设置上下文路径为根路径
        context.setBaseResource(Resource.newResource(webRootUri));  // 设置静态资源根目录
        context.setWelcomeFiles(new String[] { "index.html" });  // 设置欢迎页面
        context.getMimeTypes().addMimeMapping("txt","text/plain;charset=utf-8");  // 设置文本文件MIME类型

        // 绑定API服务到不同的URL路径
        context.addServlet(DefaultServlet.class,"/");  // 默认Servlet处理静态资源
        context.addServlet(new ServletHolder(new MovieService()), "/getmovie");  // 获取电影信息
        context.addServlet(new ServletHolder(new UserService()), "/getuser");  // 获取用户信息
        context.addServlet(new ServletHolder(new SimilarMovieService()), "/getsimilarmovie");  // 获取相似电影
        context.addServlet(new ServletHolder(new RecommendationService()), "/getrecommendation");  // 获取推荐结果
        context.addServlet(new ServletHolder(new RecForYouService()), "/getrecforyou");  // 为你推荐

        // 将上下文处理器设置到服务器
        server.setHandler(context);
        System.out.println("RecSys Server has started.");

        // 启动服务器并等待服务器线程结束
        server.start();
        server.join();
    }
}
