# 📈 QuantTrade — A股量化信号系统

基于 **ML + 三情景分析** 的 ETF 量化交易信号生成与可视化监控平台。

## 系统架构

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  数据采集    │───▶│  特征工程     │───▶│  ML模型预测    │───▶│  信号生成     │
│  (akshare/   │    │  60+技术指标  │    │  LightGBM +   │    │  三情景决策   │
│   baostock)  │    │  120日长窗口  │    │  RandomForest │    │  + Monte Carlo│
└─────────────┘    └──────────────┘    └───────────────┘    └──────┬───────┘
                                                                   │
                    ┌──────────────┐    ┌───────────────┐          │
                    │  Streamlit   │◀───│  SQLite存储    │◀─────────┘
                    │  可视化监控台 │    │  (quant.db)    │
                    └──────────────┘    └───────────────┘
```

## 核心功能

### 🤖 ML模型
- **LightGBM + RandomForest** 双模型融合，AUC 0.86+
- **Assessment/Prediction 分离架构**：确定性状态评估 + 概率性趋势预测
- **OOD 验证**：训练集（4只大盘股）→ 测试集（5只ETF），严格无泄漏

### 📊 三情景分析框架
- **Base / Adverse / Favorable** 三情景概率输出
- 加权仓位决策引擎，非简单二分类
- 趋势跟踪动态加仓 + 股灾多指标预警

### 🔮 Monte Carlo 未来走势预测
- 基于几何布朗运动 + 情景漂移，生成未来 20 日价格路径
- P10 / 中位数 / P90 置信区间可视化

### 📰 新闻情绪分析
- 自动抓取财联社 / 东方财富新闻
- SnowNLP 情绪打分，近 1/3/7 日情绪聚合
- 重大事件标记（涨停/跌停/政策等）

### 📺 Streamlit 监控台（9大页面）
| 页面 | 功能 |
|------|------|
| 🏠 数据概览 | K线走势、成交量对比、ETF统计 |
| 🤖 模型评估 | 准确率/AUC、特征重要性、混淆矩阵 |
| 📈 回测分析 | 多策略对比、热力图、雷达图 |
| 🔔 信号监控 | 实时买卖信号、Kelly仓位、历史信号 |
| ⚙️ 策略调参 | 阈值/风控参数滑块、模拟收益 |
| 🔮 情景分析 | Assessment状态、三情景概率、推荐仓位 |
| 📡 未来走势 | 历史K线 + Monte Carlo预测区间 |
| 📰 新闻情绪 | 新闻时间线、情绪曲线、重大事件 |
| 🧪 OOD回测 | 训练/测试分离验证、Round 1/2/3对比 |

## 快速开始

### 1. 安装依赖

```bash
# 推荐使用 uv（更快）
uv pip install -e ".[dev]"

# 或 pip
pip install -e ".[dev]"
```

### 2. 配置环境变量

```bash
cp QuanTrade/quant_system/.env.example QuanTrade/quant_system/.env
# 编辑 .env，填入你的 API Key
```

需要配置的密钥：
- `MINIMAX_API_KEY` — MiniMax LLM API（用于信号解读）
- `FEISHU_WEBHOOK` — 飞书机器人 Webhook（用于信号推送）
- `TUSHARE_TOKEN` — Tushare 数据源（可选兜底）

### 3. 启动监控台

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可。

## 项目结构

```
QuanTrade/
├── app.py                          # Streamlit 监控台主入口
├── pyproject.toml                  # 项目配置 & 依赖声明
├── QuanTrade/
│   └── quant_system/
│       ├── config/settings.py      # 集中配置（参数、密钥、路径）
│       ├── data/                   # 数据采集 & 存储
│       │   ├── data_fetcher.py     # akshare 数据采集
│       │   ├── baostock_fetcher.py # baostock 数据采集
│       │   └── data_store.py       # SQLite 数据存储
│       ├── features/
│       │   └── feature_engine.py   # 特征工程（60+指标）
│       ├── models/
│       │   ├── ml_trainer.py       # ML 模型训练
│       │   ├── v4/                 # v4 模型权重 & 配置
│       │   └── v5/                 # v5 模型权重 & 配置
│       ├── strategies/             # 策略框架
│       │   ├── base.py             # 策略基类 + SignalResult
│       │   ├── registry.py         # 策略注册中心
│       │   ├── loader.py           # 动态策略加载器
│       │   ├── built_in.py         # 内置策略（MA/RSI/MACD/突破/均值回归/ML混合）
│       │   └── custom/             # 自定义策略目录
│       ├── scripts/                # 运行脚本
│       │   ├── feature_engineering_v4.py   # 特征工程
│       │   ├── train_model_v4.py           # 模型训练
│       │   ├── scenario_engine_v4.py       # 三情景引擎
│       │   ├── backtest_v4.py              # 回测
│       │   └── news_crawler_v4.py          # 新闻抓取
│       ├── notify/feishu_bot.py    # 飞书通知
│       └── sync/smb_sync.py        # 树莓派同步
├── tests/                          # 单元测试
├── data_test/                      # 测试数据
└── plan_v4.md                      # 系统升级计划文档
```

## 策略开发

系统内置 6 种策略，支持自定义扩展：

| 策略 | 说明 |
|------|------|
| MA Cross | 均线金叉/死叉 |
| RSI | 超买超卖反转 |
| MACD | DIF/DEA 交叉 + 零轴判断 |
| Breakout | N日高低点突破 |
| Mean Reversion | 布林带均值回归 |
| ML Hybrid | LightGBM + 技术面双重确认 |

自定义策略只需继承 `BaseStrategy`，实现 `generate_signal()` 方法即可。详见 [策略开发指南](QuanTrade/quant_system/strategies/README.md)。

## 技术栈

- **ML**: LightGBM, RandomForest, scikit-learn, Optuna
- **数据**: akshare, baostock, Tushare
- **可视化**: Streamlit, Plotly
- **存储**: SQLite
- **通知**: 飞书机器人
- **部署**: 树莓派 + SMB 同步

## 风险提示

> ⚠️ 本项目仅供学习和研究用途，**不构成任何投资建议**。量化模型存在过拟合风险，历史回测表现不代表未来收益。请在充分理解风险的前提下使用。
