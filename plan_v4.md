# 量化系统升级计划 v4 — FinSTaR 借鉴实现

## 目标
1. **OOD 训练/测试划分**：用4只大盘股（000002.SZ/000333.SZ/000568.SZ/000651.SZ，各585条）做训练集；5只ETF（562500.SH/588200.SH/588790.SH/159382.SZ/159241.SZ）做完全未见过的测试集
2. **Assessment/Prediction 分离架构**：特征计算（确定性）与趋势判断（概率性）解耦
3. **三情景分析框架**：Base/Adverse/Favorable 情景生成 + 加权决策
4. **120日长窗口特征**：回撤、波动率regime、支撑阻力proximity、事件检测
5. **未来K线走势预测**：生成未来N日价格区间预测（含置信带）
6. **新闻抓取与情绪分析**：自动匹配标的关键词，抓取财联社/东方财富新闻，情绪打分
7. **前端同步更新**：新增"情景分析""未来走势""新闻情绪"页面

## 阶段划分

### Stage 1 — 数据准备与特征工程扩展
- **输入**: `quant.db` 中 daily_prices
- **输出**: `features_v4` 表（含120日长窗口特征）
- **任务**:
  1. 从4只训练股票提取数据，构建120日滑动窗口
  2. 新增特征（论文借鉴）:
     - `drawdown_120d`: 当前价格距120日高点回撤%
     - `vol_regime_ratio`: 近20日波动率 / 近120日波动率
     - `trend_120d_return`: 120日累计收益
     - `support_proximity`: 价格距60日低点%
     - `resistance_proximity`: 价格距60日高点%
     - `event_zscore`: 当日收益z-score（|z|>2.5标记事件）
     - `post_event_momentum`: 事件后5日收益方向
     - `drawdown_recovery_prob`: 回撤>5%后20日恢复概率（历史统计）
  3. 保留原有45+特征
  4. 对5只ETF同样计算特征（仅用于测试，不参与训练）

### Stage 2 — Assessment/Prediction 分离模型训练
- **输入**: `features_v4`
- **输出**: `models_v4/` 目录下两个模型 + `predictions_v4` 表
- **任务**:
  1. **Assessment模块**（确定性计算，硬规则）:
     - 计算当前市场状态标签：`trend_state`（强涨/温和涨/震荡/温和跌/强跌）
     - 计算 `vol_state`（低/正常/高波动）
     - 计算 `drawdown_state`（峰值附近/回撤/深度回撤）
     - 计算 `support_resistance_state`（接近支撑/接近阻力/中间）
  2. **Prediction模块**（概率性判断，ML模型）:
     - LightGBM + RandomForest 融合，输入 = 原始特征 + Assessment状态标签
     - 输出不是简单 up/down，而是 **三情景概率**: P(延续), P(反转), P(加速)
     - 用5日/10日/20日未来收益方向作为标签（多horizon）
  3. **联合训练验证**: 4只股票做5-fold时间序列交叉验证

### Stage 3 — 三情景决策引擎 + 未来K线生成
- **输入**: Prediction模块输出的三情景概率 + Assessment状态
- **输出**: `scenario_signals_v4` 表 + 未来K线预测数据
- **任务**:
  1. **三情景决策规则**:
     - Base Case: 趋势延续概率 > 0.5 → 维持当前仓位方向
     - Adverse Case: 反转概率 > 0.4 → 减仓/对冲
     - Favorable Case: 加速概率 > 0.3 → 加仓
     - 最终仓位 = Base×w1 + Adverse×w2 + Favorable×w3（加权）
  2. **未来K线生成**（Monte Carlo）:
     - 基于历史波动率、当前趋势、三情景概率，生成100条未来20日价格路径
     - 输出：中位数路径 + 10%/90%置信区间
     - 保存到 `future_klines_v4` 表
  3. **回测**: 在5只ETF上运行三情景策略 vs 基准 vs 原策略

### Stage 4 — 新闻抓取与情绪分析脚本
- **输出**: `news_sentiment_v4` 表
- **任务**:
  1. **标的-关键词映射**:
     - 562500.SH → "中证A500"
     - 588200.SH → "科创芯片"
     - 588790.SH → "科创AI"
     - 159382.SZ → "创业板人工智能"
     - 159241.SZ → "创业板新能源"
     - 训练股票同理映射
  2. **抓取源**:
     - 财联社电报 (cls.cn) — 7×24快讯
     - 东方财富新闻 — 个股/板块新闻
     - 新浪财经 — 行业新闻
  3. **情绪分析**:
     - 用 SnowNLP / 自定义词典 对标题+摘要打分 (-1 ~ +1)
     - 按标的聚合：近1日/3日/7日平均情绪
     - 标记"重大事件"（标题含涨停/跌停/大跌/大涨/政策等关键词）
  4. **入库**: `news_sentiment_v4(symbol, date, source, title, sentiment_score, is_major_event)`

### Stage 5 — Streamlit 前端更新
- **输出**: 更新 `app.py`
- **新增页面/功能**:
  1. **"情景分析"页面**: 
     - 展示当前 Assessment 状态（趋势/波动/回撤/支撑阻力）
     - 三情景概率雷达图/条形图
     - 推荐仓位（Base/Adverse/Favorable加权）
  2. **"未来走势"页面**:
     - 选择标的，展示历史K线 + 未来20日预测区间（置信带）
     - 100条Monte Carlo路径的密度热力图
  3. **"新闻情绪"页面**:
     - 标的筛选器
     - 新闻时间线（带情绪颜色标记）
     - 近7日情绪曲线
     - 重大事件高亮
  4. **"OOD回测"页面**:
     - 训练集 vs 测试集划分说明
     - 5只ETF上各策略对比（基准/原融合/三情景/新闻增强）
     - 性能表格 + 累计收益曲线

## 文件结构

```
QuanTrade/quant_system/
├── data/
│   └── quant.db  (新增表: features_v4, predictions_v4, scenario_signals_v4, future_klines_v4, news_sentiment_v4, backtest_results_v4)
├── models/
│   └── v4/
│       ├── assessment_engine.pkl      # Assessment硬规则
│       ├── prediction_lgbm.pkl        # LightGBM三情景模型
│       ├── prediction_rf.pkl          # RF三情景模型
│       ├── scenario_weights.json      # 三情景仓位权重
│       └── sentiment_dict.json        # 情绪词典
├── scripts/
│   ├── feature_engineering_v4.py     # Stage 1
│   ├── train_model_v4.py             # Stage 2
│   ├── scenario_engine_v4.py         # Stage 3
│   ├── news_crawler_v4.py            # Stage 4
│   └── backtest_v4.py                # Stage 3回测
└── app.py                            # Stage 5 (更新)
```

## 依赖
- 新增: `requests`, `beautifulsoup4`, `snownlp` (或自定义词典), `numpy`, `pandas`, `lightgbm`, `sklearn`, `plotly`, `streamlit`

## 关键设计决策
1. **Assessment用硬规则而非ML**: 论文明确指出评估任务是确定性的，用程序化计算保证100%正确性
2. **Prediction输出三概率而非二分类**: 直接输出 P(延续)/P(反转)/P(加速)，供决策引擎使用
3. **Monte Carlo用几何布朗运动+情景漂移**: 而非简单线性外推，更贴合金融随机过程
4. **新闻情绪作为独立特征输入**: 不直接修改模型，而是作为 `sentiment_1d/3d/7d` 特征加入Prediction模块
5. **OOD严格分离**: 训练集股票代码绝不出现在测试集，时间上也做walk-forward
