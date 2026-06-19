# QuanTrade 2.0 - ETF智能择时系统

> 版本：2.0  
> 日期：2026-06-19  
> 基于：QuanTrade 1.x 策略C（轮动+择时，+9.31%）  
> 更新：2026-06-19 - 扩展至24只ETF，策略C重新训练

---

## 一、版本历史

| 版本 | 说明 | 核心策略 | 回测收益 |
|------|------|----------|----------|
| 1.0 | 基础版，22只个股 | 三情景分类 | -18.02% |
| 1.1 | 改进版，阈值调整 | 方案A/B/C对比 | 全部跑输基准 |
| 1.2 | 重构版，ETF择时（8只） | 大盘择时+ETF轮动 | **+9.31%** |
| **2.0** | **全面升级版（24只ETF）** | **5层架构** | **目标+20-30%** |

---

## 二、1.x 版本回顾

### 1.0 版本的问题
- 尝试预测个股涨跌，准确率接近随机（50%）
- 三情景分类（favorable/base/adverse）效果差
- 所有方案都跑输基准

### 1.2 版本的突破
- **关键发现**：选股能力为负，择时能力为正
- **策略C**：大盘择时 + ETF轮动 = +9.31%
- **核心逻辑**：不选股，只做"买不买"的判断

### 1.2 版本保留的核心组件
- 大盘择时模型（LightGBM，基于沪深300特征）
- ETF轮动模型（LightGBM，基于技术指标）
- 回测框架（周频调仓，考虑交易成本）

### 2.0 版本的升级点
- ETF池从8只扩展到24只
- 策略C需要在24只ETF上重新训练
- 新增热点赛道选股模块
- 新增黄金对冲模块
- 新增情绪/政策/政治因子（3个独立工程）
- 新增数据驱动的建仓/减仓逻辑优化

---

## 二、ETF标的池（23只ETF + 1只基准）

> ⚠️ 数据库中只有ETF，没有个股！
> 
> 数据库表名：`etf_daily_prices`
> 
> 标的数：24只（23只ETF + 1只基准指数）

### 标的汇总

| 类别 | 数量 | 说明 |
|------|------|------|
| AI/计算/机器人 | 5只 | 科技成长 |
| 航空航天/军工 | 3只 | 国防安全 |
| 新能源/光伏 | 4只 | 碳中和 |
| 医药/消费 | 4只 | 内需消费 |
| 科技/半导体 | 4只 | 硬科技 |
| 金融/周期 | 3只 | 传统行业 |
| 避险资产 | 1只 | 黄金对冲 |
| **ETF合计** | **23只** | |
| 基准指数 | 1只 | 沪深300 |
| **总计** | **24只** | |

---

### AI/计算/机器人赛道（5只）
| 代码 | 名称 | 跟踪指数 |
|------|------|----------|
| 562500 | 机器人ETF | 中证机器人 |
| 515070 | 人工智能ETF | CS人工智能 |
| 159995 | 芯片ETF | 国证芯片 |
| 159550 | 算力ETF | 算力基础设施 |
| 516510 | 云计算ETF | 中证云计算 |

### 航空航天/军工赛道（3只）
| 代码 | 名称 | 跟踪指数 |
|------|------|----------|
| 512660 | 军工ETF | 中证军工 |
| 512670 | 国防ETF | 中证国防 |
| 515960 | 航天军工ETF | 航天军工 |

### 新能源/光伏赛道（4只）
| 代码 | 名称 | 跟踪指数 |
|------|------|----------|
| 515790 | 光伏ETF | 光伏产业 |
| 516160 | 新能源ETF | 新能源 |
| 561160 | 锂电池ETF | 锂电池 |
| 159790 | 碳中和ETF | 碳中和 |

### 医药/消费赛道（4只）
| 代码 | 名称 | 跟踪指数 |
|------|------|----------|
| 512010 | 医药ETF | 医药卫生 |
| 159928 | 消费ETF | 主要消费 |
| 512690 | 白酒ETF | 中证白酒 |
| 515170 | 食品饮料ETF | 食品饮料 |

### 科技/半导体赛道（4只）
| 代码 | 名称 | 跟踪指数 |
|------|------|----------|
| 512480 | 半导体ETF | 半导体 |
| 588000 | 科创50ETF | 科创50 |
| 159915 | 创业板ETF | 创业板指 |
| 513180 | 恒生科技ETF | 恒生科技 |

### 金融/周期赛道（3只）
| 代码 | 名称 | 跟踪指数 |
|------|------|----------|
| 512880 | 证券ETF | 中证全指证券 |
| 512800 | 银行ETF | 中证银行 |
| 512200 | 地产ETF | 中证800地产 |

### 避险资产（1只）
| 代码 | 名称 | 跟踪指数 |
|------|------|----------|
| 518880 | 黄金ETF | Au99.99 |

### 基准指数
| 代码 | 名称 | 用途 |
|------|------|------|
| 510300 | 沪深300ETF | 大盘基准 |

---

## 三、2.0 系统架构

### 3.1 总体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    QuanTrade 2.0 架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  第1层       │    │  第2层       │    │  第3层       │     │
│  │  策略C核心   │ →  │  热点赛道    │ →  │  黄金对冲    │     │
│  │  (已有)      │    │  选股       │    │  (新增)      │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ↓                   ↓                   ↓           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    第4层                              │   │
│  │              情绪/政策/政治因子                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │ 工程A    │  │ 工程B    │  │ 工程C    │          │   │
│  │  │ 市场情绪 │  │ 国际政治 │  │ 国内政策 │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    第5层                              │   │
│  │              建仓/减仓逻辑优化                         │   │
│  │           (数据驱动，非规则驱动)                        │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    输出层                              │   │
│  │              交易信号 + 仓位建议                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
输入数据：
  ├── ETF日线数据（8只AI/军工 + 黄金ETF）
  ├── 沪深300数据（大盘基准）
  ├── 新闻数据（东方财富API）
  ├── 政策数据（国务院/发改委/证监会）
  └── 国际政治数据（外交部/商务部/美联储）
      ↓
特征工程：
  ├── 技术指标（34个）
  ├── 资金流向代理（4个）
  ├── 情绪因子（6个）
  ├── 政策因子（3个）
  └── 地缘政治因子（3个）
      ↓
模型预测：
  ├── 大盘择时模型 → 上涨概率
  ├── 热点赛道模型 → ETF排名
  └── 综合决策模型 → 最终信号
      ↓
输出信号：
  ├── 仓位建议（0%/30%/50%/80%/100%）
  ├── ETF选择（1-2只）
  └── 黄金对冲比例（0-30%）
```

---

## 四、各层详细设计

### 4.1 第1层：策略C核心（重新训练）

**文件**：`run_etf_system_v2.py`（新建）

**1.2版本回顾**：
- 8只ETF + 沪深300基准
- 回测结果：+9.31%
- 核心框架有效，但ETF池太小

**2.0版本升级**：
- 扩展到24只ETF + 黄金ETF + 沪深300基准
- 策略C需要在新ETF池上重新训练
- 预期效果：选股范围更大，更容易选到强势赛道

**核心逻辑**（保持不变）：
```python
# 大盘择时（基于沪深300特征）
market_prob = timing_model.predict(benchmark_features)
if market_prob > 0.55:
    position = 1.0  # 满仓
elif market_prob < 0.45:
    position = 0.0  # 空仓
else:
    position = 0.5  # 半仓

# ETF轮动（基于24只ETF特征）
etf_scores = rotation_model.predict(etf_features)
selected_etfs = top_k(etf_scores, k=2)

# 最终收益
return = mean(selected_etfs_return) * position - transaction_cost
```

**训练数据**：
- 训练集：2022-2023年
- 验证集：2024年上半年
- 测试集：2024年下半年至今

**注意**：策略C是基础框架，后续4层都是在此基础上叠加

---

### 4.2 第2层：热点赛道选股（新增）

**目标**：提升选股能力，从"随机选2只"变成"选最强赛道"

**文件**：`sector_rotation.py`（新建）

**选股指标**：

| 指标 | 计算方式 | 权重 | 说明 |
|------|----------|------|------|
| 动量得分 | 5/10/20日涨幅排名 | 30% | 越涨越强 |
| 资金流向 | 成交量变化率排名 | 30% | 放量=资金流入 |
| 趋势强度 | MA5>MA10>MA20 得分 | 20% | 均线多头=趋势强 |
| 相对强弱 | vs沪深300超额收益 | 20% | 跑赢大盘=强势 |

**计算代码**：
```python
def calculate_sector_score(etf_df, benchmark_df):
    """计算每只ETF的热点赛道得分"""
    
    # 1. 动量得分（30%）
    etf_df['momentum_5d'] = etf_df['close'].pct_change(5)
    etf_df['momentum_10d'] = etf_df['close'].pct_change(10)
    etf_df['momentum_20d'] = etf_df['close'].pct_change(20)
    etf_df['momentum_score'] = (
        rank(etf_df['momentum_5d']) * 0.4 +
        rank(etf_df['momentum_10d']) * 0.3 +
        rank(etf_df['momentum_20d']) * 0.3
    )
    
    # 2. 资金流向得分（30%）
    etf_df['fund_flow_5d'] = etf_df['volume'].pct_change(5)
    etf_df['fund_flow_10d'] = etf_df['volume'].pct_change(10)
    etf_df['fund_flow_score'] = (
        rank(etf_df['fund_flow_5d']) * 0.5 +
        rank(etf_df['fund_flow_10d']) * 0.5
    )
    
    # 3. 趋势强度得分（20%）
    etf_df['ma5'] = etf_df['close'].rolling(5).mean()
    etf_df['ma10'] = etf_df['close'].rolling(10).mean()
    etf_df['ma20'] = etf_df['close'].rolling(20).mean()
    etf_df['trend_score'] = (
        (etf_df['ma5'] > etf_df['ma10']).astype(int) * 0.4 +
        (etf_df['ma10'] > etf_df['ma20']).astype(int) * 0.3 +
        (etf_df['close'] > etf_df['ma5']).astype(int) * 0.3
    )
    
    # 4. 相对强弱得分（20%）
    etf_df['relative_5d'] = etf_df['momentum_5d'] - benchmark_df['bm_return_5d']
    etf_df['relative_10d'] = etf_df['momentum_10d'] - benchmark_df['bm_return_10d']
    etf_df['relative_score'] = (
        rank(etf_df['relative_5d']) * 0.5 +
        rank(etf_df['relative_10d']) * 0.5
    )
    
    # 综合得分
    etf_df['sector_score'] = (
        etf_df['momentum_score'] * 0.30 +
        etf_df['fund_flow_score'] * 0.30 +
        etf_df['trend_score'] * 0.20 +
        etf_df['relative_score'] * 0.20
    )
    
    return etf_df
```

**与策略C的融合**：
```python
# 原来：ML模型预测概率选ETF
selected = decision_df.nlargest(2, 'ml_probability')

# 现在：ML预测概率 × 热点赛道得分
decision_df['final_score'] = (
    decision_df['ml_probability'] * 0.5 +
    decision_df['sector_score'] * 0.5
)
selected = decision_df.nlargest(2, 'final_score')
```

---

### 4.3 第3层：黄金对冲（新增）

**目标**：市场下跌时，用黄金ETF替代空仓，既能避险又能保值

**文件**：`gold_hedge.py`（新建）

**黄金ETF**：
- 黄金ETF (518880) - 华安黄金
- 黄金ETF (159934) - 易方达黄金

**对冲逻辑**：
```python
def calculate_gold_allocation(market_signal, market_prob):
    """计算黄金仓位"""
    
    if market_prob < 0.35:
        # 强烈看跌：30%黄金 + 70%现金
        return {'gold': 0.30, 'stock': 0.00, 'cash': 0.70}
    
    elif market_prob < 0.45:
        # 看跌：20%黄金 + 30%股票 + 50%现金
        return {'gold': 0.20, 'stock': 0.30, 'cash': 0.50}
    
    elif market_prob < 0.55:
        # 中性：10%黄金 + 40%股票 + 50%现金
        return {'gold': 0.10, 'stock': 0.40, 'cash': 0.50}
    
    elif market_prob < 0.65:
        # 看涨：0%黄金 + 70%股票 + 30%现金
        return {'gold': 0.00, 'stock': 0.70, 'cash': 0.30}
    
    else:
        # 强烈看涨：0%黄金 + 90%股票 + 10%现金
        return {'gold': 0.00, 'stock': 0.90, 'cash': 0.10}
```

**黄金的优势**：
- 与股市负相关（股市跌，黄金涨）
- 避险属性（地缘政治风险时上涨）
- 长期保值（抗通胀）

**预期效果**：
- 降低最大回撤（从-18%降到-10%左右）
- 在熊市中提供正收益
- 不影响牛市收益（牛市时黄金仓位为0）

---

### 4.4 第4层：情绪/政策/政治因子（3个独立工程）

#### 工程A：市场情绪分析

**文件**：`sentiment_engine/`（独立目录）

**数据源**：
- 东方财富新闻搜索API
- 新浪财经滚动新闻
- 同花顺财经

**关键词词库**（大幅扩充）：

```python
# 正面词库（扩充到100+）
POSITIVE_WORDS = [
    # 涨跌类
    '涨停', '大涨', '反弹', '突破', '创新高', '飙升', '暴涨', '强势', '领涨',
    '上涨', '走高', '拉升', '冲高', '新高', '翻倍', '暴涨',
    
    # 利好类
    '利好', '增持', '买入', '看多', '买入评级', '增持评级',
    '业绩增长', '盈利', '分红', '回购', '战略合作', '订单', '中标',
    
    # 政策类
    '政策支持', '补贴', '减税', '降准', '降息', '宽松', '刺激',
    '鼓励', '扶持', '改革', '开放', '创新', '发展',
    
    # 行业类（AI/芯片）
    'AI突破', '大模型', '算力', '芯片', '半导体', '集成电路',
    '人工智能', '机器人', '智能制造', '自动驾驶', '数字化',
    
    # 行业类（军工/航天）
    '国防', '军工', '航天', '卫星', '导弹', '战斗机', '航母',
    '北斗', '火箭', '航天器', '军事现代化',
    
    # 行业类（新能源）
    '光伏', '新能源', '储能', '锂电池', '风电', '碳中和',
    '绿色能源', '清洁能源', '电动', '氢能',
]

# 负面词库（扩充到100+）
NEGATIVE_WORDS = [
    # 涨跌类
    '跌停', '大跌', '暴跌', '跳水', '重挫', '弱势', '领跌',
    '下跌', '走低', '杀跌', '崩盘', '腰斩', '破位', '新低',
    
    # 利空类
    '利空', '减持', '卖出', '看空', '卖出评级', '减持评级',
    '业绩下滑', '亏损', '退市', 'ST', '违规', '处罚', '调查',
    
    # 政策类
    '加息', '收紧', '监管', '限制', '制裁', '整顿', '打压',
    '去杠杆', '紧缩', '调控', '限购', '限贷',
    
    # 风险类
    '贸易战', '制裁', '冲突', '战争', '脱钩', '断供',
    '地缘政治', '金融危机', '泡沫', '黑天鹅', '灰犀牛',
    
    # 行业类
    '产能过剩', '库存积压', '需求下滑', '价格战', '技术封锁',
]

# 重大事件词库（扩充到50+）
MAJOR_EVENT_WORDS = [
    # 公司事件
    '财报', '业绩', '分红', '回购', '增持', '减持',
    '重组', '并购', 'IPO', '增发', '配股', '可转债',
    
    # 政策事件
    '降准', '降息', '加息', 'MLF', 'LPR', '逆回购',
    '财政政策', '货币政策', '产业政策', '监管政策',
    
    # 国际事件
    '美联储', '欧央行', 'G7', 'G20', 'APEC', '联合国',
    '贸易战', '关税', '制裁', '脱钩', '断供',
    
    # 市场事件
    '停牌', '复牌', '退市', 'ST', '*ST',
    '暴跌', '熔断', '千股跌停', '千股涨停',
]
```

**情绪评分方法**：
```python
def analyze_sentiment(text, positive_words, negative_words, major_event_words):
    """分析文本情绪"""
    if not text:
        return 0, False
    
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    is_major = any(w in text for w in major_event_words)
    
    # 归一化到 [-1, 1]
    total = pos_count + neg_count
    if total == 0:
        score = 0
    else:
        score = (pos_count - neg_count) / total
    
    # 重大事件加权
    if is_major:
        score *= 1.5
    
    return np.clip(score, -1, 1), is_major
```

**情绪因子输出**：
```python
sentiment_factors = {
    'sentiment_1d': 1日情绪均值,
    'sentiment_3d': 3日情绪均值,
    'sentiment_7d': 7日情绪均值,
    'sentiment_momentum': sentiment_3d - sentiment_3d.shift(3),
    'news_count_1d': 1日新闻数量,
    'major_events_7d': 7日重大事件数,
}
```

---

#### 工程B：国际政治分析

**文件**：`geopolitical_engine/`（独立目录）

**数据源**：
- 外交部新闻
- 商务部公告
- 美联储声明
- 国际新闻（Reuters、Bloomberg中文）

**关键词词库**：

```python
# 地缘政治风险词库
GEOPOLITICAL_RISK_WORDS = [
    # 贸易摩擦
    '贸易战', '关税', '加征关税', '贸易摩擦', '贸易争端',
    '反倾销', '反补贴', '贸易壁垒', '出口管制',
    
    # 技术封锁
    '技术封锁', '芯片禁令', '实体清单', '出口管制',
    '技术脱钩', '供应链断裂', '断供', '卡脖子',
    
    # 地缘冲突
    '军事冲突', '战争', '军事行动', '导弹试射',
    '台海', '南海', '中东', '乌克兰', '俄罗斯',
    
    # 制裁
    '制裁', '金融制裁', '贸易制裁', '技术制裁',
    '冻结资产', '禁止交易', '黑名单',
    
    # 金融风险
    '金融危机', '银行倒闭', '债务危机', '货币危机',
    '资本外流', '汇率波动', '美元走强',
]

# 地缘政治缓和词库
GEOPOLITICAL_POSITIVE_WORDS = [
    '贸易谈判', '达成协议', '取消关税', '缓和',
    '对话', '合作', '互利', '共赢', '稳定',
    '和平', '停火', '谈判', '协商', '解决',
]
```

**地缘政治风险评分**：
```python
def calculate_geopolitical_risk(news_list):
    """计算地缘政治风险评分"""
    risk_score = 0
    for news in news_list:
        risk_count = sum(1 for w in GEOPOLITICAL_RISK_WORDS if w in news)
        positive_count = sum(1 for w in GEOPOLITICAL_POSITIVE_WORDS if w in news)
        risk_score += risk_count - positive_count
    
    # 归一化到 [0, 1]
    return min(max(risk_score / 10, 0), 1)
```

---

#### 工程C：国内政策分析

**文件**：`policy_engine/`（独立目录）

**数据源**：
- 国务院政策文件
- 发改委公告
- 央行公告
- 证监会公告
- 工信部公告

**关键词词库**：

```python
# 利好政策词库
POSITIVE_POLICY_WORDS = [
    # 货币政策
    '降准', '降息', 'MLF下调', 'LPR下调', '逆回购',
    '宽松', '流动性充裕', '货币供应增加',
    
    # 财政政策
    '减税', '降费', '财政补贴', '专项债', '国债',
    '财政支出', '转移支付', '税收优惠',
    
    # 产业政策
    '产业扶持', '技术创新', '研发补贴', '产业基金',
    '专精特新', '智能制造', '数字化转型',
    
    # 行业政策（AI/芯片）
    '人工智能', '芯片', '半导体', '集成电路', '算力',
    '大模型', '数字经济', '新基建',
    
    # 行业政策（新能源）
    '新能源', '光伏', '储能', '碳中和', '碳达峰',
    '绿色金融', '清洁能源', '电动化',
    
    # 行业政策（军工）
    '国防现代化', '军事装备', '航天', '北斗',
    '军民融合', '国防科工',
]

# 利空政策词库
NEGATIVE_POLICY_WORDS = [
    # 货币政策
    '加息', '收紧', '去杠杆', '紧缩', '流动性收紧',
    
    # 监管政策
    '监管', '整顿', '规范', '限制', '禁止',
    '处罚', '罚款', '吊销', '关停',
    
    # 行业限制
    '产能过剩', '淘汰落后', '环保限产', '能耗限制',
    '房地产调控', '限购', '限贷', '限售',
    
    # 金融风险
    '金融风险', '债务风险', '影子银行', '非法集资',
    'P2P暴雷', '信托违约', '债券违约',
]
```

**政策影响评分**：
```python
def calculate_policy_score(news_list):
    """计算政策影响评分"""
    positive_count = sum(
        1 for news in news_list 
        for w in POSITIVE_POLICY_WORDS if w in news
    )
    negative_count = sum(
        1 for news in news_list 
        for w in NEGATIVE_POLICY_WORDS if w in news
    )
    
    total = positive_count + negative_count
    if total == 0:
        return 0
    
    return (positive_count - negative_count) / total
```

---

### 4.5 第5层：建仓/减仓逻辑优化（数据驱动）

**目标**：用历史数据验证最优参数，而非凭感觉设定规则

**文件**：`optimize_trading_rules.py`（新建）

**方法**：网格搜索 + 历史回测

**需要优化的参数**：

| 参数 | 搜索范围 | 说明 |
|------|----------|------|
| 建仓阈值 | 跌1%/2%/3%/4%/5% | 沪指跌多少时建仓 |
| ETF跌幅阈值 | 跌1%/2%/3%/4%/5% | ETF跌多少时建仓 |
| RSI超卖阈值 | 20/25/30/35 | RSI低于多少算超卖 |
| 回撤阈值 | 10%/15%/20%/25% | 从高点回撤多少建仓 |
| 目标收益 | 10%/15%/20%/25%/30% | 达到多少收益减仓 |
| 减仓比例 | 30%/40%/50%/60% | 每次减仓多少 |
| RSI超买阈值 | 65/70/75/80 | RSI高于多少减仓 |

**优化代码框架**：
```python
def optimize_trading_rules(etf_df, benchmark_df):
    """用历史数据优化交易规则参数"""
    
    best_params = None
    best_return = -np.inf
    
    # 网格搜索
    for entry_threshold in [0.01, 0.02, 0.03, 0.04, 0.05]:
        for rsi_oversold in [20, 25, 30, 35]:
            for target_profit in [0.10, 0.15, 0.20, 0.25, 0.30]:
                for reduce_ratio in [0.3, 0.4, 0.5, 0.6]:
                    
                    params = {
                        'entry_threshold': entry_threshold,
                        'rsi_oversold': rsi_oversold,
                        'target_profit': target_profit,
                        'reduce_ratio': reduce_ratio,
                    }
                    
                    # 回测这组参数
                    result = backtest_with_params(etf_df, benchmark_df, params)
                    
                    if result['total_return'] > best_return:
                        best_return = result['total_return']
                        best_params = params
    
    return best_params, best_return
```

**建仓规则（数据驱动版）**：
```python
def should_open_position(etf, market, params):
    """是否应该建仓（参数由数据优化得出）"""
    
    # 条件1: 沪指大跌
    market_drop = market['pct_change'] < -params['entry_threshold']
    
    # 条件2: ETF大跌
    etf_drop = etf['pct_change'] < -params['entry_threshold']
    
    # 条件3: RSI超卖
    oversold = etf['rsi_14'] < params['rsi_oversold']
    
    # 条件4: 从高点回撤
    drawdown = (etf['close'] / etf['high_20d'] - 1) < -params['drawdown_threshold']
    
    # 满足条件1+2，以及条件3或4
    return market_drop and etf_drop and (oversold or drawdown)
```

**减仓规则（数据驱动版）**：
```python
def should_reduce_position(etf, profit_pct, params):
    """是否应该减仓（参数由数据优化得出）"""
    
    # 条件1: 达到目标收益
    if profit_pct > params['target_profit']:
        return params['reduce_ratio']
    
    # 条件2: RSI超买
    if etf['rsi_14'] > params['rsi_overbought']:
        return params['reduce_ratio'] * 0.5
    
    # 条件3: 情绪过热
    if etf['sentiment_7d'] > 0.8:
        return params['reduce_ratio'] * 0.3
    
    return 0
```

---

## 五、文件结构

```
QuanTrade/
├── Quantrade2.0.md                    # 本文档
├── run_etf_system.py                  # 1.2版本主脚本（保留）
├── sector_rotation.py                 # 2.0 新增：热点赛道选股
├── gold_hedge.py                      # 2.0 新增：黄金对冲
├── optimize_trading_rules.py          # 2.0 新增：交易规则优化
├── run_etf_system_v2.py               # 2.0 主脚本（整合所有层）
│
├── sentiment_engine/                  # 2.0 新增：市场情绪工程
│   ├── __init__.py
│   ├── crawler.py                     # 新闻爬虫
│   ├── analyzer.py                    # 情绪分析
│   ├── keywords.py                    # 关键词词库
│   └── aggregator.py                  # 情绪聚合
│
├── geopolitical_engine/               # 2.0 新增：国际政治工程
│   ├── __init__.py
│   ├── crawler.py                     # 政治新闻爬虫
│   ├── analyzer.py                    # 风险分析
│   └── keywords.py                    # 关键词词库
│
├── policy_engine/                     # 2.0 新增：国内政策工程
│   ├── __init__.py
│   ├── crawler.py                     # 政策公告爬虫
│   ├── analyzer.py                    # 政策影响分析
│   └── keywords.py                    # 关键词词库
│
├── quant_system/
│   ├── data/
│   │   ├── quant.db                   # 数据库
│   │   ├── data_fetcher.py            # 数据获取
│   │   └── baostock_fetcher.py        # Baostock获取
│   │
│   ├── models/
│   │   ├── etf_system_results.json    # 1.2版本结果
│   │   └── etf_system_v2_results.json # 2.0版本结果
│   │
│   └── scripts/
│       ├── expand_full_data_v3.py     # 数据扩展脚本
│       └── news_crawler_v4.py         # 新闻爬虫（已有）
```

---

## 六、执行计划

### Phase 1: 数据准备（预计2小时）

| 任务 | 文件 | 预计时间 |
|------|------|----------|
| 下载黄金ETF数据 | `expand_gold_data.py` | 10分钟 |
| 构建新闻爬虫 | `sentiment_engine/crawler.py` | 30分钟 |
| 构建政治爬虫 | `geopolitical_engine/crawler.py` | 20分钟 |
| 构建政策爬虫 | `policy_engine/crawler.py` | 20分钟 |
| 扩充关键词词库 | `*/keywords.py` | 20分钟 |
| 数据测试验证 | - | 20分钟 |

**数据更新频率**：
- ETF日线数据：每个交易日收盘后更新
- 新闻/政策/政治数据：每个交易日爬取一次
- 情绪因子：每个交易日计算一次

### Phase 2: 特征工程（预计1小时）

| 任务 | 文件 | 预计时间 |
|------|------|----------|
| 热点赛道得分 | `sector_rotation.py` | 20分钟 |
| 黄金对冲特征 | `gold_hedge.py` | 10分钟 |
| 情绪因子计算 | `sentiment_engine/analyzer.py` | 15分钟 |
| 政策因子计算 | `policy_engine/analyzer.py` | 10分钟 |
| 地缘政治因子 | `geopolitical_engine/analyzer.py` | 10分钟 |

### Phase 3: 策略实现（预计2小时）

| 任务 | 文件 | 预计时间 |
|------|------|----------|
| 整合策略C + 热点赛道 | `run_etf_system_v2.py` | 30分钟 |
| 实现黄金对冲逻辑 | `gold_hedge.py` | 20分钟 |
| 实现情绪/政策因子集成 | `run_etf_system_v2.py` | 30分钟 |
| 交易规则参数优化 | `optimize_trading_rules.py` | 40分钟 |

### Phase 4: 回测验证（预计1小时）

| 任务 | 文件 | 预计时间 |
|------|------|----------|
| 运行2.0系统 | `run_etf_system_v2.py` | 20分钟 |
| 对比1.2和2.0结果 | - | 15分钟 |
| 参数敏感性分析 | - | 15分钟 |
| 输出最终报告 | - | 10分钟 |

### 总计：约6小时

---

## 七、预期效果

| 指标 | 1.2版本 | 2.0预期 | 改进来源 |
|------|---------|---------|----------|
| 总收益 | +9.31% | +20-30% | 热点赛道选股 |
| 最大回撤 | -18.06% | -10-15% | 黄金对冲 |
| 夏普比率 | 0.35 | 0.5-0.8 | 情绪/政策因子 |
| 胜率 | 15.7% | 30-40% | 交易规则优化 |

**各层贡献预估**：

| 层 | 预期贡献 | 说明 |
|----|----------|------|
| 第1层：策略C | +9% | 基础择时+轮动 |
| 第2层：热点赛道 | +5-10% | 选到强势ETF |
| 第3层：黄金对冲 | +3-5% | 降低回撤 |
| 第4层：情绪/政策 | +2-3% | 提前识别转折 |
| 第5层：规则优化 | +1-2% | 参数最优 |

---

## 八、风险提示

1. **过拟合风险**：参数优化可能导致过拟合历史数据
2. **数据质量**：新闻爬虫可能不稳定，情绪因子可能有噪声
3. **交易成本**：频繁调仓会增加交易成本
4. **市场变化**：历史规律可能在未来失效

**应对措施**：
- 使用交叉验证避免过拟合
- 多数据源容灾
- 控制调仓频率（周频）
- 定期重训模型

---

## 九、安全与配置

### 9.1 敏感信息存储

所有Token和密钥存储在 `.secrets/` 文件夹中，已配置 `.gitignore` 排除：

```
.secrets/
├── tokens.md          # Token存储（GitHub、Tushare等）
└── README.md          # 说明文档
```

### 9.2 数据更新频率

| 数据类型 | 更新频率 | 说明 |
|----------|----------|------|
| ETF日线数据 | 每个交易日 | 收盘后自动更新 |
| 新闻情绪数据 | 每个交易日 | 交易日爬取 |
| 政策/政治数据 | 每个交易日 | 交易日爬取 |
| 情绪因子 | 每个交易日 | 基于新闻计算 |
| 模型重训 | 每月/每季度 | 视效果衰减情况 |

---

## 十、后续迭代方向

### 2.1 版本（短期）
- 加入更多ETF（新能源、医药、消费等）
- 优化情绪词库（加入更多行业术语）
- 实现实盘信号推送

### 2.2 版本（中期）
- 加入分钟级数据
- 实现日内交易策略
- 加入期权对冲

### 3.0 版本（长期）
- 深度学习模型（LSTM/Transformer）
- 强化学习策略
- 多市场联动（A股+港股+美股）
