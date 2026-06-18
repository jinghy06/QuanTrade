"""
A股量化信号系统 - 集中配置
所有环境变量和硬编码参数统一在此管理
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------- 项目路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for _dir in [CACHE_DIR, MODELS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------- 数据库 ----------
DB_PATH = DATA_DIR / "quant.db"

# ---------- API密钥 ----------
load_dotenv(PROJECT_ROOT / ".env")

FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK", "")

# Tushare兜底（可选）
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# ---------- 股票池配置 ----------
# 默认关注沪深300成分股，可通过环境变量覆盖
DEFAULT_WATCHLIST = [
    "000001.SZ",  # 平安银行
    "000002.SZ",  # 万科A
    "000333.SZ",  # 美的集团
    "000568.SZ",  # 泸州老窖
    "000651.SZ",  # 格力电器
    "000725.SZ",  # 京东方A
    "000858.SZ",  # 五粮液
    "002001.SZ",  # 新和成
    "002230.SZ",  # 科大讯飞
    "002415.SZ",  # 海康威视
    "300001.SZ",  # 特锐德
    "300033.SZ",  # 同花顺
    "300059.SZ",  # 东方财富
    "300122.SZ",  # 智飞生物
    "300274.SZ",  # 阳光电源
    "300750.SZ",  # 宁德时代
    "600000.SH",  # 浦发银行
    "600009.SH",  # 上海机场
    "600016.SH",  # 民生银行
    "600028.SH",  # 中国石化
    "600030.SH",  # 中信证券
    "600031.SH",  # 三一重工
    "600036.SH",  # 招商银行
    "600276.SH",  # 恒瑞医药
    "600309.SH",  # 万华化学
    "600519.SH",  # 贵州茅台
    "600585.SH",  # 海螺水泥
    "600690.SH",  # 海尔智家
    "600745.SH",  # 闻泰科技
    "600887.SH",  # 伊利股份
    "601012.SH",  # 隆基绿能
    "601066.SH",  # 中信建投
    "601088.SH",  # 中国神华
    "601166.SH",  # 兴业银行
    "601288.SH",  # 农业银行
    "601318.SH",  # 中国平安
    "601398.SH",  # 工商银行
    "601888.SH",  # 中国中免
    "603259.SH",  # 药明康德
    "603501.SH",  # 韦尔股份
]

# 从环境变量加载自定义股票池
WATCHLIST_ENV = os.getenv("WATCHLIST", "")
WATCHLIST = WATCHLIST_ENV.split(",") if WATCHLIST_ENV else DEFAULT_WATCHLIST

# ---------- ML模型参数 ----------
# 预测目标
PREDICT_HORIZON = 1  # 预测未来1天
SIGNAL_THRESHOLD = 0.55  # 信号输出阈值，>0.55才输出
MIN_AUC = 0.52  # 模型可用最低AUC

# LightGBM默认参数
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 200,
}

# 时序交叉验证
N_SPLITS = 5  # PurgedGroupTimeSeriesSplit折数

# ---------- 特征工程参数 ----------
# 基础技术指标参数
MOMENTUM_WINDOWS = [5, 10, 20]
VOLATILITY_WINDOWS = [5, 20]
VOLUME_MA_WINDOWS = [5, 20]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14

# 训练数据长度
MIN_TRAIN_DAYS = 252 * 2  # 最少2年数据
MAX_FEATURES_HISTORY = 252 * 3  # 最多使用3年数据计算特征

# ---------- 回测参数 ----------
INITIAL_CAPITAL = 100000.0  # 回测初始资金
COMMISSION = 0.0003  # 手续费万3
SLIPPAGE = 0.001  # 滑点千1

# ---------- 信号生成时间 ----------
# 每日开盘前和收盘后触发
SCHEDULE_TIMES = ["09:00", "17:30"]

# ---------- 日志配置 ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FILE = LOGS_DIR / "quant.log"

# ---------- 风险控制 ----------
MAX_POSITION_PCT = 0.20  # 单票最大仓位20%
MAX_DRAWDOWN_PCT = 0.15  # 最大回撤15%预警
STOP_LOSS_PCT = 0.07  # 默认止损线7%

# ---------- 飞书通知模板 ----------
FEISHU_MSG_TEMPLATE = {
    "msg_type": "interactive",
    "card": {
        "header": {
            "title": {"tag": "plain_text", "content": ""},
            "template": "blue",
        },
        "elements": [],
    },
}
