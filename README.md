# QuanTrade 3.0

ETF 中低频交易系统：统一数据、模型训练、预测 K 线、情绪／政策／地缘决策、组合执行与份额账本。常规月度或季度轮动，风险事件可生成受约束减仓建议，人工执行。

## 安装与入口

在 Git 根目录，使用 Python 3.11+：

```powershell
.venv/Scripts/python.exe -m pip install -e ".[dev,data,ui]"
.venv/Scripts/python.exe -m quantrade --help
```

安装后可直接用 `quantrade`。默认路径相对仓库 `configs/default.toml`，不依赖启动目录。自定义配置：`quantrade --config C:/path/default.toml ...`。核心依赖为 pandas、numpy、LightGBM、sklearn；采集与页面为可选依赖，legacy 依赖组仅用于旧实现恢复。

## 数据

```powershell
quantrade data migrate
quantrade data validate
quantrade data snapshot
quantrade data import-instruments --file configs/instruments.csv
quantrade data update --symbols 510300 510500 --start 2025-01-01 --end 2026-10-12
```

旧库只读，新库为 `data/quantrade.sqlite`。原始导出与异常保存在 `data/migration/`。相同重复价格合并；冲突隔离；极端行情保留待核查。

标的信息不自动认证名称、行业、上市日期和交易规则。编辑 instruments.csv 后导入；核验来源后才能设 verified=true，不能将行情开始日期视为上市日期。实际账户应确认 T+0／T+1、交易单位、费用、现金预留。默认配置是研究假设。

- `data import-prices --file prices.csv --basis raw`：symbol,date,open,high,low,close,volume,amount；可带 source,fetched_at,price_basis,verified。拒绝覆盖冲突价格。
- `data import-events --file events.csv`：event_id,category,published_at,available_at,valid_until,score,confidence,source,content。时间必须带时区，category 为 sentiment/policy/geopolitical，score 为 [-1,1]。修订用新 ID。
- `data import-actions --file actions.csv`：event_id,date,symbol,split,dividend。dividend 为当日实际支付的每份现金额；split 为份额倍数。复杂除息应收和跨日付款须转换为可核对事件，系统不猜测公司行动。
- `data resolve-issue --id ISSUE_ID --explanation 证据说明`：记录解释后关闭异常。标的与价格口径另行核验。

## 回测与审计

```powershell
quantrade backtest --start 2025-01-02 --end 2026-06-18 --allow-unverified
quantrade backtest --start 2025-01-02 --end 2026-06-18 --walk-forward --allow-unverified --cost-levels 0.0003 0.001
```

比较等权满仓买入持有、固定风险敞口、动量、动量＋模型、完整多因子。满仓基准保留；其他策略共用集中度约束。收盘决策，下一交易日开盘成交，收盘估值；持仓缺价标为陈旧，不归零、不假定成交。

滚动验证在调仓前执行历史训练、标签清洗、验证和后续预测；窗口保留训练边界，持仓跨窗口连续。无足够模型数据时降级动量。成本测试重新执行成交和账本。

`runs/v3-实验编号/` 保存配置、数据快照、代码摘要、净值、持仓、目标、订单、成交、模型窗口、预测、评价与 `audit.json`；独立审计从流水重建现金、份额，核对每一天 NAV。阅读 `REPORT.md` 查看结果。

默认拒绝校验未通过的数据；`--allow-unverified` 显式生成 exploratory 结果。旧行情口径、标的规则和跳变尚未核验时不能形成正式实盘结论。模拟没有盘口排队与冲击模型；涨跌停和停牌信息不足须作为执行限制披露。

## 模型、预测 K 线与多因子

```powershell
quantrade model train --cutoff 2026-05-01
quantrade model validate --version model-训练结果编号
quantrade model publish --version model-训练结果编号
quantrade forecast --date 2026-10-12 --symbol 510300
```

统一 ETF 特征与多期限回归，按日期分区、按标签期限清洗；未知标签保留缺失。缺失中位数仅从训练区间学习，训练与推理输入一致。模型不能预测其训练／验证／测试信息边界以前的日期，发布记录可获取时间，过期模型降级。

validate 是技术验收，publish 是本地研究发布，均不代表实盘准入。滚动交易回放与因子消融评价增量价值。

预测图保留模型锚点、插值路径与误差区间；固定种子的模拟影线不是训练出的 OHLC。锚点区间使用验证集绝对误差的 90% 分位数，测试覆盖率留档，未来覆盖率仍需观察。图表日期使用工作日近似，不能用于认定成交或止损。

回看历史预测可使用 `quantrade forecast --date YYYY-MM-DD --symbol ETF_CODE --model-version MODEL_ID --research`；该模型的信息边界仍必须早于预测日期，输出标为历史研究，不修改生产发布记录。

情绪／政策／地缘读取当时可获得、未过期的冻结事件；没有真实来源时明确标为行情代理或 unavailable。因子影响统一风险预算，不各自买卖；事件锁定、冷却与计划调仓恢复防止每日重复减仓。LLM 解析可按冻结事件格式导入，首轮不绑定外部模型服务。

## 账户与页面

```powershell
quantrade account init --id actual --mode actual --cash 50000 --complete
quantrade account import-fills --id actual --file fills.csv
quantrade report --account actual --date 2026-06-18
quantrade dashboard
```

初始化现金应是完整回放起点，不是忽略已有持仓后的当前现金。未声明 complete 的账户不生成订单建议；已有账户拒绝重置。成交字段见 configs/fills.example.csv，side 为 buy/sell。同一成交幂等，冲突 ID、历史倒序记录拒绝入账；整批成功才保存。

报告冻结当天决策，不成交、不推送、不修改持仓。建议数量按已知收盘价估算；下一交易日须确认交易所日历。分红记录、账户流水或未来日历不完整时，不能将建议当成可立即执行的订单。

只读监控页面整合账户、调仓、回测、数据、预测与决策五个视图。

## 代码与测试

正式包 src/quantrade，配置 configs，页面 ui。历史功能和重复入口按迁移清单归档；模型、预测与因子能力迁入正式包。见 [重构计划](ETF_REFACTOR_PLAN.md) 和 [升级记录](V3_UPGRADE_REPORT.md)。

```powershell
.venv/Scripts/python.exe -m pytest -q
```

数据库、模型、实验与 `backups/v3-pre-refactor/` 不提交 Git，迁移清单纳入版本管理。先核对恢复材料再处理备份。工程正确与策略有效分开验收。
