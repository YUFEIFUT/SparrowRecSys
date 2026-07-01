package com.sparrowrecsys.online.util;

public class Config {
    public static final String DATA_SOURCE_REDIS = "redis";
    public static final String DATA_SOURCE_FILE = "file";

    public static String EMB_DATA_SOURCE = Config.DATA_SOURCE_FILE;
    public static boolean IS_LOAD_USER_FEATURE_FROM_REDIS = true;
    public static boolean IS_LOAD_ITEM_FEATURE_FROM_REDIS = true;

    public static boolean IS_ENABLE_AB_TEST = false;

    // TF Serving endpoints
    public static String TF_SERVING_NCF_URL = "https://footbath-naming-impaired.ngrok-free.dev/v1/models/sparrow_ncf:predict";
    public static String TF_SERVING_MLP_URL = "https://footbath-naming-impaired.ngrok-free.dev/v1/models/sparrow_mlp:predict";
    public static String TF_SERVING_WD_URL = "https://footbath-naming-impaired.ngrok-free.dev/v1/models/sparrow_wd:predict";
    public static String TF_SERVING_DFM_URL = "https://footbath-naming-impaired.ngrok-free.dev/v1/models/sparrow_dfm:predict";
    public static String TF_SERVING_DFM_V2_URL = "https://footbath-naming-impaired.ngrok-free.dev/v1/models/sparrow_dfm_v2:predict";
    public static String TF_SERVING_DIN_URL = "https://footbath-naming-impaired.ngrok-free.dev/v1/models/sparrow_din:predict";
    public static String TF_SERVING_DIEN_URL = "https://footbath-naming-impaired.ngrok-free.dev/v1/models/sparrow_dien:predict";

}
