# A股量化信号系统项目计划书

> **版本**：v1.2  
> **日期**：2026-05-19  
> **硬件**：RTX 3060 笔记本（训练）+ 树莓派 CM4/5（信号）  
> **角色**：LLM 提供决策建议，ML 提供预测，人工执行交易  
> **通知**：飞书自定义机器人 Webhook  

---

## 一、项目定位

不做自动交易。A股个人实盘 API 门槛极高（中泰 XTP 需 300 万金融资产认证），本系统定位为**信号顾问系统**：

- **输入**：A股日K数据 + 财务数据 + 宏观代理变量
- **处理**：LightGBM 预测明日涨跌概率 + MiniMax LLM 生成结构化建议
- **输出**：飞书推送（操作建议、目标价、止损、理由）
- **执行**：人工在券商 APP 下单或设置条件单

---

## 二、整合的 GitHub 项目（全部基于实际调研）

| 项目 | Stars | 职责 | 协议 | 备注 |
|------|-------|------|------|------|
| **akshare-team/akshare** | ~7.5K | A股数据获取 | MIT | 日K/分钟K/财务/宏观 |
| **microsoft/LightGBM** | ~25K | ML模型训练 | MIT | **官方声明跨平台文本格式** |
| **scikit-learn/sklearn** | ~61K | 数据预处理 | BSD | StandardScaler/Pipeline |
| **pmorissette/bt** | ~2.3K | 回测引擎 | MIT | FinRL-X 实际使用的回测库 |
| **pmorissette/ffn** | ~1.3K | 绩效评估 | MIT | 夏普/最大回撤/IC |
| **LinShuyue2003/qbt-lite** | ~200 | **轻量级回测 + Streamlit UI** | MIT | 事件驱动回测，内网Dashboard |
| **kernc/backtesting.py** | ~3.5K | **交互式回测引擎** | MIT | Bokeh图表，参数优化，策略模板 |
| **AI4FinanceFoundation/ai-trading-agent** | ~200 | **LLM Agent 架构参考** | 未明确 | function calling + 定时循环 + 审计日志 |
| **kgrajski/trading_etf** | ~150 | **多Agent分析架构参考** | 未明确 | Quant/News/Synthesis/Review 四Agent协作 |
| **microsoft/RD-Agent** | ~3K | **自动化因子挖掘** | MIT | LLM自动生成量化因子，配合Qlib使用 |

### 项目分层说明

**基础设施层**：
- `akshare`：A股数据唯一免费且覆盖全面的源
- `LightGBM` + `sklearn`：ML训练与预处理

**回测验证层（三选一或组合）**：
- `bt`：FinRL-X同款，适合portfolio级回测
- `qbt-lite`：轻量级，带Streamlit UI，适合快速原型和树莓派内网展示
- `backtesting.py`：交互式Bokeh图表，适合参数优化和可视化分析

**LLM决策层（架构借鉴，不直接整合）**：
- `ai-trading-agent`：借鉴其**function calling约束**、**定时循环**、**审计日志**设计
- `trading_etf`：借鉴其**多Agent协作**架构（Quant/News/Synthesis/Review）
- `RD-Agent`：用于笔记本上的**自动化因子挖掘**，生成新特征供LightGBM训练

**通知层**：
- 飞书自定义机器人（Webhook）：替代Telegram，与你现有OpenClaw+Feishu生态一致

---

## 三、LLM Agent 架构参考详解

### 3.1 ai-trading-agent（AI4Finance Foundation）

**实际状态**：TypeScript + Bun，2025-2026年实验项目，~200 stars。  
**可借鉴的设计**：

| 设计点 | 本系统应用方式 |
|--------|-------------|
| **定时循环**（每5分钟`setInterval`） | 改为每日开盘前（9:00）和收盘后（17:30）触发 |
| **Function Calling约束** | LLM只能调用3个预定义工具：`get_technical` / `get_ml_pred` / `get_macro` |
| **审计日志**（PostgreSQL Invocations+ToolCalls） | SQLite记录每次LLM输入输出，用于事后复盘和绩效归因 |
| **Prompt与数据分离**（`prompt.ts` / `markets.ts`） | `system_prompt.md`（策略文本）与`feature_engine.py`（特征代码）分离 |

**不直接整合的原因**：
- TypeScript/Bun栈与你的Python生态不匹配
- 存在**position side inversion bug**（README明确警告需审计）
- 仅支持Lighter.xyz交易所，A股无法使用
- 实验性质，免责声明明确"教育用途"

### 3.2 trading_etf（Multi-Agent AI Analyst）

**实际状态**：Python，ETF周度交易系统，~150 stars，R&D Workbench定位。  
**核心架构（直接借鉴）**：

```
用户查询（如"分析平安银行"）
    │
    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ QuantAgent  │    │ NewsAgent   │    │ SynthesisAgent│   │ ReviewAgent   │
│ 技术面分析   │───→│ 新闻情绪    │───→│ 综合研判     │───→│ 风险审查     │
│ 指标计算    │    │ 舆情摘要    │    │ 生成建议     │    │ 合规检查     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**本系统简化版（两Agent）**：
- **QuantAgent** → 本地Python执行：技术面摘要 + ML预测 + 宏观环境
- **SynthesisAgent** → MiniMax LLM：接收QuantAgent输出，生成结构化建议

### 3.3 RD-Agent（Microsoft）

**实际状态**：~3K stars，2025年发布，微软亚洲研究院出品。  
**在本系统中的角色**：
- 部署在**笔记本**上，用于**自动化因子挖掘**
- 输入：原始A股数据
- 输出：新因子表达式（如`ts_mean(volume, 20) / ts_std(close, 60)`）
- 人工审核后，将有效因子加入`feature_engine.py`
- 每月运行一次，为LightGBM提供新弹药

---

## 四、硬件分工架构

```
┌──────────────────────────────┐         ┌──────────────────────────────┐
│   RTX 3060 笔记本              │         │   树莓派 CM4/5 (ARM64)       │
│   Windows / WSL2               │         │   Raspberry Pi OS 64-bit     │
│   ── "训练工厂+研究站" ──       │         │   ── "信号哨兵" ──            │
│                                │         │                              │
│  1. 数据实验室                  │  模型   │  1. 数据同步                  │
│     • AkShare拉取历史数据       │  文件   │     • AkShare增量更新        │
│     • SQLite本地仓库            │  同步   │     • 特征计算                │
│     • 特征工程                  │         │                              │
│                                │         │  2. 信号工厂                  │
│  2. ML训练车间                  │         │     • 加载lgb.txt             │
│     • LightGBM分类器            │         │     • 重建预处理管道          │
│     • 时序交叉验证              │         │     • 生成预测分数            │
│     • 超参数调优(Optuna)        │         │     • 过滤(>0.55才输出)      │
│     • 保存: lgb.txt + json      │         │                              │
│                                │         │  3. LLM分析师                 │
│  3. 回测验证(多引擎)            │         │     • MiniMax API(云端)       │
│     • bt引擎(portfolio级)       │         │     • function calling       │
│     • qbt-lite(快速原型+UI)     │         │     • 结构化建议输出          │
│     • backtesting.py(交互图表)  │         │                              │
│                                │         │  4. 信号推送                  │
│  4. LLM Prompt工坊              │         │     • 飞书Webhook机器人       │
│     • MiniMax prompt调优       │         │     • SQLite历史记录          │
│     • function calling测试      │         │     • WebUI(可选)             │
│                                │         │                              │
│  5. RD-Agent因子挖掘(每月)      │         │                              │
│     • 自动生成新因子            │         │                              │
│     • 人工审核后入库            │         │                              │
│                                │         │                              │
└──────────────────────────────┘         └──────────────────────────────┘
           │                                              │
           └────────────── git push / rsync ──────────────┘
```

---

## 五、模型跨平台迁移方案（核心）

**问题**：sklearn的`joblib` pickle在版本不一致时会崩溃（GitHub issue #32090已证实）。

**三层保险**：

### 保险1：LightGBM原生文本格式（100%跨平台）
LightGBM官方文档明确声明文本格式与平台/架构无关。

```python
# 笔记本（训练后保存）
model.booster_.save_model('models/lgb_classifier.txt')

# 树莓派（加载）
import lightgbm as lgb
booster = lgb.Booster(model_file='models/lgb_classifier.txt')
```

### 保险2：版本锁定（避免pickle崩溃）
```bash
# 笔记本
pip freeze > requirements.txt

# 树莓派（安装完全相同版本）
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 保险3：JSON参数导出（彻底不用pickle）
```python
# 笔记本导出
import json
params = {
    'feature_names': feature_names,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'model_path': 'lgb_classifier.txt',
    'threshold': 0.55
}
json.dump(params, open('model_config.json', 'w'))

# 树莓派重建
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.mean_ = np.array(config['scaler_mean'])
scaler.scale_ = np.array(config['scaler_scale'])
scaler.n_features_in_ = len(config['feature_names'])
```

---

## 六、数据流与信号流

### 6.1 数据源（双源容灾）

| 数据源 | 用途 | 成本 | 备注 |
|--------|------|------|------|
| **AkShare** | 日K、分钟K、财务数据 | 免费 | 依赖东方财富，偶有反爬升级 |
| **Tushare** | 补充规范数据、指数成分 | 免费额度 | 稳定性高，AkShare失效时兜底 |
| **iTick** | 实时tick（可选） | 免费层 | WebSocket，限流 |

### 6.2 特征工程（Alpha158简化版 + RD-Agent新因子）

**基础特征（手工）**：
- **动量**：5/10/20日收益率、RSI、MACD
- **波动率**：5/20日标准差、ATR
- **量价**：成交额均线、OBV
- **宏观代理**：10年期国债收益率（AkShare可获取）、汇率

**RD-Agent生成因子（每月更新）**：
- 示例：`ts_mean(volume, 20) / ts_std(close, 60)`（量稳比）
- 审核标准：IC > 0.03，与现有因子相关性 < 0.7

### 6.3 预测目标

1. **明日涨跌概率**（二分类）：LightGBM，AUC目标>0.52（A股噪音大，0.55已算可用）
2. **5日波动率区间**（回归）：用于LLM仓位建议
3. **相对强弱评分**（截面）：同一板块内比较

### 6.4 LLM决策输入（借鉴ai-trading-agent + trading_etf架构）

MiniMax-M2.7通过function calling按需调用本地工具：

| 工具名 | 功能 | 对应参考设计 |
|--------|------|-------------|
| `get_technical_summary(symbol)` | 技术面摘要 | ai-trading-agent的`stockData.ts` + trading_etf的QuantAgent |
| `get_ml_prediction(symbol)` | ML预测结果 | ai-trading-agent的enrich prompt信号层 |
| `get_macro_context()` | 宏观环境 | trading_etf的宏观段落 |
| `get_news_sentiment(symbol)` | 新闻情绪（可选） | trading_etf的NewsAgent |

### 6.5 输出格式（结构化JSON）

```json
{
  "timestamp": "2026-05-19T09:35:00",
  "symbol": "000001.SZ",
  "name": "平安银行",
  "signal": "轻仓试探",
  "confidence": 0.62,
  "ml_prediction": {"up_prob": 0.58, "volatility": "中"},
  "technical": {"trend": "短期反弹", "rsi": 45, "macd": "金叉初期"},
  "macro": {"rate_env": "降息周期", "market_sentiment": "谨慎"},
  "suggestion": {
    "action": "轻仓试探",
    "target_price": 12.50,
    "stop_loss": 11.80,
    "position_pct": 0.05,
    "rationale": "ML模型显示58%上涨概率，MACD刚形成金叉，处于降息周期有利于银行板块。但大盘情绪谨慎，建议不超过5%仓位。",
    "risk_factors": ["大盘情绪谨慎", "银行股波动性低但弹性弱", "ML置信度仅58%未达强信号阈值"]
  },
  "holding_advice": null
}
```

---

## 七、回测引擎选型对比

| 引擎 | 类型 | 适合场景 | 优势 | 劣势 |
|------|------|---------|------|------|
| **bt** | 向量+事件驱动 | Portfolio级策略，多资产 | FinRL-X同款，成熟稳定 | 学习曲线陡，文档分散 |
| **qbt-lite** | 轻量级事件驱动 | 快速原型，单资产信号验证 | Streamlit UI，CLI友好，MIT | 新仓库，社区小 |
| **backtesting.py** | 向量+交互式 | 参数优化，可视化分析 | Bokeh交互图表，参数热力图 | 对A股数据需适配 |

**建议**：
- **主力回测**：bt（ FinRL-X验证过，适合portfolio回测）
- **快速验证**：qbt-lite（单股信号快速测试，Streamlit给非技术用户展示）
- **参数优化**：backtesting.py（网格搜索+交互图表，找最优参数组合）

---

## 八、飞书通知设计

### 8.1 推送方式
飞书自定义机器人通过**Webhook**接收POST请求，支持Markdown格式和卡片消息。

### 8.2 消息格式示例

```python
import requests

def send_feishu(signal):
    webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxx"

    content = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"📊 量化信号：{signal['name']} ({signal['symbol']})"},
                "template": "blue" if signal['signal'] == '轻仓试探' else "red"
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**信号强度**：{signal['confidence']}"}},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**操作建议**：{signal['suggestion']['action']}"}},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**目标价**：{signal['suggestion']['target_price']} | **止损**：{signal['suggestion']['stop_loss']}"}},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**ML预测**：上涨概率 {signal['ml_prediction']['up_prob']}"}},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**理由**：{signal['suggestion']['rationale']}"}},
                {"tag": "hr"},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"⚠️ **风险**：{', '.join(signal['suggestion']['risk_factors'])}"}}
            ]
        }
    }

    requests.post(webhook, json=content)
```

### 8.3 交互命令（飞书机器人回调，可选）
如部署公网可访问服务，配置飞书机器人回调：
- `@量化助手 分析 000001` — 触发单股分析
- `@量化助手 今日信号` — 推送今日全部关注列表
- `@量化助手 绩效` — 返回本周信号准确率统计

---

## 九、实施路线图

### Phase 0：笔记本基建（Week 1-2）
- [ ] 安装环境：`pip install akshare lightgbm scikit-learn bt ffn optuna requests`
- [ ] AkShare下载沪深300成分股3年历史日K数据
- [ ] SQLite建表：`daily_prices`、`features`、`signals`
- [ ] 计算第一批技术指标（SMA、RSI、MACD），可视化验证
- [ ] 训练第一个LightGBM，AUC目标>0.52

### Phase 1：回测验证（Week 3-4）
- [ ] 将ML信号包装为`bt.Algo`
- [ ] 回测2022-2024年，评估夏普比率、最大回撤、胜率
- [ ] **淘汰标准**：夏普<0.5或最大回撤>25%，则调整特征或放弃模型
- [ ] 蒙特卡洛敏感性分析
- [ ] （可选）用qbt-lite做单股快速验证，用backtesting.py做参数优化

### Phase 2：LLM接入（Week 5-6）
- [ ] 注册MiniMax开放平台，获取API Key
- [ ] 测试function calling（3-4个工具定义，借鉴ai-trading-agent约束思想）
- [ ] 调优system prompt，约束输出JSON schema
- [ ] 验证LLM能正确调用`get_ml_prediction`并返回结构化建议

### Phase 3：飞书推送测试（Week 6）
- [ ] 飞书群创建自定义机器人，获取Webhook URL
- [ ] 用Python`requests`发送测试卡片消息
- [ ] 验证Markdown渲染和移动端显示效果

### Phase 4：RD-Agent因子挖掘（Week 7，每月重复）
- [ ] 笔记本部署RD-Agent（WSL2/Docker）
- [ ] 输入A股数据，自动生成新因子
- [ ] 人工审核：IC>0.03且与现有因子低相关则入库
- [ ] 重训练LightGBM，观察AUC提升

### Phase 5：树莓派部署（Week 8）
- [ ] 树莓派安装与笔记本完全相同的Python包版本
- [ ] 传输`lgb.txt`+`config.json`
- [ ] **一致性测试**：同一股票同一天，两边预测值误差<1e-6
- [ ] systemd服务化+飞书推送测试
- [ ] （可选）部署qbt-lite Streamlit UI，内网`192.168.1.x:8501`访问

### Phase 6：模拟验证（Month 3-6）
- [ ] 树莓派每日9:00生成信号，飞书推送
- [ ] 在券商模拟盘或极小资金（<<1000元）跟随信号
- [ ] 记录：信号方向准确率、盈亏比、最大回撤
- [ ] 每月在笔记本上重训练模型，更新树莓派

### Phase 7：人工实盘（Month 6+）
- [ ] 仅投入可完全损失的资金
- [ ] 严格止损，每日审查交易日志
- [ ] 存活6个月，最大回撤<15%

---

## 十、风险控制（诚实版）

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| A股T+1+无做空 | 只能做多，熊市失效 | 设置大盘趋势过滤器，熊市降低仓位或空仓 |
| AkShare反爬升级 | 数据中断 | Tushare兜底；本地SQLite缓存至少60天历史 |
| sklearn版本不兼容 | 树莓派加载失败 | 版本锁定+JSON参数导出双重保险 |
| ML预测上限低 | AUC 0.55已算可用 | 不追求高准确率，追求盈亏比>1.2 |
| MiniMax API成本 | 月费用50-150元 | 每日仅分析关注列表（≤20只），控制token |
| 信号≠盈利 | 方向对但盈亏比差 | LLM强制要求输出止损价，人工严格执行 |
| ai-trading-agent参考风险 | 实验性架构借鉴可能不适用 | 仅借鉴其function calling约束和审计日志思想，不复制交易逻辑 |
| RD-Agent幻觉因子 | 生成无意义因子 | 人工审核IC和相关性，不自动入库 |

---

## 十一、附录

### A. MiniMax API配置
- **Base URL**：`https://api.minimaxi.com/v1`
- **模型**：`MiniMax-M2.7`（支持function calling + reasoning_split）
- **关键参数**：`extra_body={"reasoning_split": True}`（分离推理链）
- **工具定义**：3-4个function（technical / ml_pred / macro / news_sentiment）

### B. 飞书机器人配置
1. 飞书群设置→添加机器人→自定义机器人
2. 复制Webhook URL（格式：`https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxx`）
3. 安全设置：可配置IP白名单（你的树莓派公网IP）或关键词验证
4. 树莓派上环境变量：`FEISHU_WEBHOOK=https://...`

### C. 树莓派systemd服务
```ini
# /etc/systemd/system/quantbot.service
[Unit]
Description=A股量化信号系统
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/quant_system
Environment="MINIMAX_API_KEY=your_key"
Environment="FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxx"
ExecStart=/home/pi/quant_env/bin/python signal_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable quantbot
sudo systemctl start quantbot
sudo systemctl status quantbot
```

### D. 项目文件结构建议
```
quant_system/
├── config/
│   └── settings.py              # 集中配置（MiniMax key、飞书Webhook、股票池）
├── data/
│   ├── data_fetcher.py          # AkShare + Tushare双源获取
│   ├── data_store.py            # SQLite操作封装
│   └── cache/                   # 本地数据缓存
├── features/
│   ├── feature_engine.py        # Alpha158简化 + 自定义因子
│   └── rd_agent_new/            # RD-Agent生成的新因子（每月更新）
├── models/
│   ├── ml_trainer.py            # LightGBM训练 + 时序CV
│   ├── lgb_classifier.txt       # 跨平台模型文件（git-lfs）
│   └── model_config.json        # 预处理参数（无pickle）
├── backtest/
│   ├── bt_engine.py             # bt回测封装
│   ├── qbt_wrapper.py           # qbt-lite快速验证（可选）
│   └── backtestingpy_wrapper.py # backtesting.py参数优化（可选）
├── llm/
│   ├── advisor.py               # MiniMax function calling封装
│   ├── tools.py                 # 4个工具实现
│   └── system_prompt.md         # 策略文本（与代码分离）
├── notify/
│   └── feishu_bot.py            # 飞书卡片消息推送
├── signal_bot.py                # 树莓派主入口（定时任务）
├── requirements.txt             # 严格版本锁定
└── deploy.md                    # 树莓派部署手册
```

---

> **下一步**：确认此计划后，可开始写具体代码模块。建议先从`data_fetcher.py`+`feature_engine.py`开始，这是整个系统的地基。
