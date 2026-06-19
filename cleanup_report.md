# 文件清理报告

## 根目录文件分析

### 📌 保留（核心文件）

| 文件 | 说明 | 建议 |
|------|------|------|
| `run_etf_system.py` | 当前运行的1.2版本主脚本 | ✅ 保留 |
| `run_etf_system_v1.2.py` | 1.2版本备份 | ✅ 保留 |
| `Quantrade2.0.md` | 2.0计划文档 | ✅ 保留 |
| `README.md` | 项目说明 | ✅ 保留 |

### 🔧 临时/调试文件（建议删除）

| 文件 | 说明 | 建议 |
|------|------|------|
| `debug_returns.py` | 调试收益计算 | ❌ 删除 |
| `debug_returns2.py` | 调试收益计算v2 | ❌ 删除 |
| `debug_returns3.py` | 调试收益计算v3 | ❌ 删除 |
| `diagnose_root_cause.py` | 诊断策略问题 | ❌ 删除 |
| `analyze_contribution.py` | 分析收益贡献 | ❌ 删除 |
| `analyze_strategy.py` | 分析策略问题 | ❌ 删除 |
| `analyze_thresholds.py` | 分析阈值 | ❌ 删除 |
| `calc_buy_hold.py` | 计算买入不动收益 | ❌ 删除 |
| `check_etf_prices.py` | 检查ETF价格 | ❌ 删除 |

### 🧪 实验文件（可选删除）

| 文件 | 说明 | 建议 |
|------|------|------|
| `run_experiments.py` | 1.0版本实验 | ⚠️ 可选删除 |
| `run_experiments.py.bak` | 实验备份 | ❌ 删除 |
| `run_improvement_experiments.py` | 改进方案实验 | ⚠️ 可选删除 |
| `run_comprehensive_experiments.py` | 全面实验 | ⚠️ 可选删除 |
| `run_small_capital_experiments.py` | 小资金实验 | ⚠️ 可选删除 |
| `run_etf_quant_model.py` | ETF量化模型 | ⚠️ 可选删除 |
| `run_quant_model.py` | 量化模型 | ⚠️ 可选删除 |

### 📊 数据扩展文件（可选删除）

| 文件 | 说明 | 建议 |
|------|------|------|
| `expand_etf_data.py` | 扩展ETF数据 | ⚠️ 可选删除 |
| `expand_data_baostock.py` | Baostock数据扩展 | ⚠️ 可选删除 |
| `expand_full_data.py` | 全面数据扩展 | ⚠️ 可选删除 |
| `expand_full_data_v2.py` | 全面数据扩展v2 | ⚠️ 可选删除 |
| `expand_full_data_v3.py` | 全面数据扩展v3 | ⚠️ 可选删除 |

### 📝 旧版本文件（可选删除）

| 文件 | 说明 | 建议 |
|------|------|------|
| `fix_backtest.py` | 修复回测 | ⚠️ 可选删除 |
| `phase1_enhanced_model.py` | Phase1增强模型 | ⚠️ 可选删除 |
| `phase23_compact.py` | Phase23紧凑版 | ⚠️ 可选删除 |
| `phase23_enhanced_strategy.py` | Phase23增强策略 | ⚠️ 可选删除 |
| `app.py` | 应用程序 | ⚠️ 可选删除 |
| `plan_v4.md` | v4计划文档 | ⚠️ 可选删除 |
| `quant_research_report.md` | 研究报告 | ⚠️ 可选删除 |

### 🗑️ 数据库备份（可选删除）

| 文件 | 大小 | 说明 | 建议 |
|------|------|------|------|
| `quant.db.bak_20260618_153054` | 107MB | 数据库备份 | ⚠️ 可选删除 |

### 📊 实验结果文件（可选删除）

| 文件 | 说明 | 建议 |
|------|------|------|
| `comprehensive_experiment_results.json` | 全面实验结果 | ⚠️ 可选删除 |
| `experiment_results.json` | 实验结果 | ⚠️ 可选删除 |
| `improvement_experiment_results.json` | 改进实验结果 | ⚠️ 可选删除 |
| `small_capital_results.json` | 小资金实验结果 | ⚠️ 可选删除 |
| `etf_system_results.json` | ETF系统结果 | ✅ 保留 |

---

## 统计

| 类别 | 文件数 | 建议删除 |
|------|--------|----------|
| 调试文件 | 9 | 9 |
| 实验文件 | 7 | 1-7 |
| 数据扩展文件 | 5 | 5 |
| 旧版本文件 | 6 | 0-6 |
| 数据库备份 | 1 | 0-1 |
| 实验结果 | 4 | 3 |
| **总计** | 32 | 18-31 |

## 删除后预计节省空间

- 调试文件：约50KB
- 实验文件：约100KB
- 数据扩展文件：约50KB
- 数据库备份：约107MB
- **总计：约107MB**
