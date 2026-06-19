# GitHub 上传 Token 使用说明

## 生成 Token

1. 访问 https://github.com/settings/tokens
2. 点击 **Generate new token (classic)**
3. 勾选权限：
   - `repo` (完整仓库权限)
   - `workflow` (可选，用于 GitHub Actions)
4. 点击生成，**复制并保存 Token**

## 存放 Token

将 Token 粘贴到下方（保留 `ghp_` 前缀）：

```
ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ **安全提醒**：
> - 此文件已加入 `.gitignore`，不会被提交到仓库
> - 不要将此文件内容分享给他人
> - Token 具有仓库写入权限，请妥善保管

## 使用 Token 推送

### 方式一：命令行直接推送

```bash
# 配置 credential helper（Windows 一次性）
git config --global credential.helper wincred

# 推送时会弹出用户名/密码窗口
# 用户名：你的 GitHub 用户名
# 密码：粘贴上面生成的 Token
git push origin main
```

### 方式二：使用 Token 直接推送

```bash
# 在 Git Bash 中
cd /c/Users/HY/PycharmProjects/QuanTrade

# 使用 token 作为密码推送
git push https://<TOKEN>@github.com/jinghy06/QuanTrade.git main

# 或者先设置远程 URL（替换 <TOKEN>）
git remote set-url origin https://<TOKEN>@github.com/jinghy06/QuanTrade.git
git push origin main
```

### 方式三：使用 SSH（推荐长期配置）

```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 添加到 GitHub Settings -> SSH and GPG keys -> New SSH key
# 然后修改远程 URL
git remote set-url origin git@github.com:jinghy06/QuanTrade.git

# 测试连接
ssh -T git@github.com
```

## 当前仓库状态

- 远程仓库：`https://github.com/jinghy06/QuanTrade.git`
- 当前分支：`main`
- 本地领先远程：`1 commit`
- 大量未跟踪文件待上传

## 需要上传的关键文件清单

### v2.0.1 核心文件（含机器学习）
- `run_etf_system_v2.0.1.py` — 五层架构 ML 策略（LightGBM/RandomForest/GradientBoosting）
- `evaluator_agent.py` — 独立评价 Agent
- `supervision_agent.py` — 独立监督 Agent
- `sentiment_engine.py` — 情绪/政策/地缘引擎
- `sector_rotation.py` — 热点板块轮动
- `gold_hedge.py` — 黄金对冲模块

### v2.0.2 迭代文件（纯动量）
- `run_etf_system_v2.0.2c.py` — 最终版本（纯动量 + 三因子 + 用户策略）
- `run_etf_system_v2.0.2b.py` — 季度调仓版
- `run_etf_system_v2.0.2.py` — 月度调仓版
- `geopolitical_engine.py` — 国际政治因子引擎
- `policy_engine.py` — 国内政策因子引擎
- `optimize_trading_rules.py` — 交易规则参数优化

### 数据与报告
- `user_trades.xlsx` / `user_trades_raw.json` — 用户交易记录
- `QUANTRADE_2.0.1_REPORT.md` — 2.0.1 版本报告
- `OPTIMIZATION_PLAN_v2.0.2.md` — 2.0.2 优化计划
- `QUANTRADE_2.1_PLAN.md` — 2.1 版本规划

## 上传命令参考

```bash
cd /c/Users/HY/PycharmProjects/QuanTrade

# 添加所有 v2 相关文件
git add run_etf_system_v2*.py evaluator_agent.py supervision_agent.py \
    sentiment_engine.py sector_rotation.py gold_hedge.py \
    geopolitical_engine.py policy_engine.py optimize_trading_rules.py \
    *.md

# 提交
git commit -m "feat: v2.0.1 ML strategy + v2.0.2c momentum strategy

- v2.0.1: Five-layer ML strategy (LightGBM/RF/GB)
- v2.0.2c: Pure momentum + three-factor + user strategy integration
- Independent evaluation agent (anti-overfitting)
- User trade analysis and strategy extraction
- Three-factor engine: sentiment/geopolitical/policy"

# 推送（使用上方生成的 Token）
git push origin main
```
