# 文件清理报告 v2

> 执行时间：2026-07-16  
> 策略：归档到 `archive/` 目录，保留核心资产，不直接删除

## 一、测试结果

清理前后均运行测试，结果一致：

```
28 passed in ~1.5s
```

## 二、根目录最终保留结构

```
QuanTrade/
├── README.md
├── TECHNICAL_REPORT_v2.0.2c.md
├── TECHNICAL_REPORT_v2.2.0.md
├── pyproject.toml
├── app.py                          # Streamlit 监控台
├── evaluator_agent.py              # 独立评价 Agent
├── sentiment_engine.py             # 情绪引擎
├── cross_market_data.py            # 跨市场数据
├── geopolitical_engine.py          # 地缘风险代理
├── policy_engine.py                # 政策周期代理
├── sector_rotation.py              # 板块轮动
├── gold_hedge.py                   # 黄金对冲
├── calc_buy_hold.py                # 买入持有基准计算
├── check_buy_hold.py               # 买入持有验证
├── run_etf_system_v2.0.2c.py       # README 推荐稳定版
├── run_etf_system_v2.2.0_final.py  # 当前最新版
├── tests/                          # 28 个测试用例
├── data_test/                      # 测试数据样本
├── QuanTrade/quant_system/         # 主包（个股信号机器人）
└── archive/                        # 本次归档目录
```

## 三、归档文件统计

| 归档目录 | 文件数 | 说明 |
|---|---|---|
| `archive/old_versions/` | 13 | 旧版本回测主脚本 |
| `archive/debug/` | 6 | 调试/诊断脚本 |
| `archive/optimize/` | 13 | 优化/分析脚本 |
| `archive/data_scripts/` | 9 | 数据扩展/补充脚本 |
| `archive/experiments/` | 14 | 实验/对比/监督脚本 |
| `archive/outputs/` | 35+ | 历史输出/图片 |
| `archive/docs/` | 10 | 旧版计划/报告 |
| `archive/misc/` | 4 | 用户数据/无关文件 |
| **合计** | **约 121 个文件** | |

## 四、特殊处理

| 文件 | 处理方式 | 原因 |
|---|---|---|
| `GITHUB_TOKEN.md` | 移至 `.secrets/GITHUB_TOKEN.md` | 敏感信息，.gitignore 已忽略 |
| `INTERVIEW_PREP.md` | 移至 `archive/misc/` | 与项目无关 |
| `interview_cheatsheet.png` | 移至 `archive/outputs/` | 与项目无关 |
| `streamlit.log` | 删除 | 日志文件，可重新生成 |
| `cleanup_report.md` | 移至 `archive/docs/` | 旧版清理报告已过时 |
| `train_stocks_v5.txt` | 移至 `archive/misc/` | 空文件 |

## 五、后续建议

1. **观察期**：保留 `archive/` 一段时间（建议 1-2 周），确认核心工作流不受影响。
2. **彻底清理**：确认无误后，可删除 `archive/` 目录释放空间。
3. **Git 提交**：当前 git 状态显示大量删除 + `archive/` 未跟踪。如需提交，建议分两步：
   - 先提交核心文件保留
   - 再决定是否将 `archive/` 加入版本控制（通常不建议，占用仓库空间）
4. **继续精简**：
   - 合并 `calc_buy_hold.py` / `check_buy_hold.py`
   - 将核心模块逐步移入 `QuanTrade/quant_system/` 主包
   - 统一数据库 schema，清理 `quant.db` 中碎片化实验表
