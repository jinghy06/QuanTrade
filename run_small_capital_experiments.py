"""
小资金量重构方案 - 每周选1-2只最看好的股票
核心思路: 不是"仓位管理"，而是"选股"

方案A: ML排名选股 - 每周用模型选出最看好的1-2只
方案B: 动量轮动 - 每周选近期涨得最好的1-2只
方案C: ML+动量混合 - 综合模型信号和动量排名

基准: 等权持有全部22只 / 随机选1-2只
"""
import sqlite3
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'
BASE_MODEL_DIR = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\models'

# ============================================================
# 交易成本设定（小资金）
# ============================================================
COMMISSION_RATE = 0.0003  # 佣金万三
STAMP_TAX_RATE = 0.001    # 印花税千一（卖出时收）
SLIPPAGE = 0.002          # 滑点0.2%
TOTAL_COST = COMMISSION_RATE * 2 + STAMP_TAX_RATE + SLIPPAGE  # 一次完整交易约0.56%


# ============================================================
# 1. 特征工程
# ============================================================

def add_features(df):
    """增加动量和统计特征"""
    result = []
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        s = df.loc[mask].copy()

        # 动量特征
        for w in [3, 5, 10, 20]:
            s[f'return_{w}d'] = s['close'].pct_change(w)

        # 波动率
        for w in [5, 10, 20]:
            s[f'vol_{w}d'] = s['close'].pct_change().rolling(w).std()

        # 相对强弱
        s['rsi_14'] = compute_rsi(s['close'], 14)

        # 价格位置
        for w in [10, 20]:
            s[f'high_{w}d'] = s['high'].rolling(w).max()
            s[f'low_{w}d'] = s['low'].rolling(w).min()
            s[f'pos_{w}d'] = (s['close'] - s[f'low_{w}d']) / (s[f'high_{w}d'] - s[f'low_{w}d'] + 1e-10)

        # 成交量变化
        for w in [5, 10]:
            s[f'vol_ma_{w}'] = s['volume'].rolling(w).mean()
            s[f'vol_ratio_{w}'] = s['volume'] / (s[f'vol_ma_{w}'] + 1e-10)

        # MACD
        exp1 = s['close'].ewm(span=12, adjust=False).mean()
        exp2 = s['close'].ewm(span=26, adjust=False).mean()
        s['macd'] = exp1 - exp2
        s['macd_signal'] = s['macd'].ewm(span=9, adjust=False).mean()
        s['macd_hist'] = s['macd'] - s['macd_signal']

        # 趋势强度
        s['trend_strength'] = s['close'].pct_change(5) + s['close'].pct_change(10) + s['close'].pct_change(20)

        result.append(s)
    return pd.concat(result, ignore_index=True)


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ============================================================
# 2. 选股策略
# ============================================================

def weekly_stock_selection(df, feature_cols, strategy='ml_rank', n_stocks=2, top_n_train=5):
    """
    每周选股策略

    strategy:
      - 'ml_rank': 用ML模型预测收益率，选最高的
      - 'momentum': 选近期涨得最好的
      - 'ml_momentum': 综合ML和动量
    """

    # 训练集: 2023年之前
    train_mask = df['trade_date'] < '2023-01-01'
    val_mask = (df['trade_date'] >= '2023-01-01') & (df['trade_date'] < '2024-01-01')

    train_df = df[train_mask].copy()

    # 训练模型（如果是ML策略）
    model = None
    if strategy in ['ml_rank', 'ml_momentum']:
        X_train = train_df[feature_cols].fillna(0)
        y_train = (train_df['target_return_5d'] > 0).astype(int)

        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1, n_jobs=-1
        )
        model.fit(X_train, y_train)

    # 测试集: 2024-2025，按周遍历
    test_df = df[df['trade_date'] >= '2024-01-01'].copy()
    weeks = test_df['trade_date'].dt.isocalendar().week
    years = test_df['trade_date'].dt.year
    test_df['year_week'] = years * 100 + weeks

    weekly_results = []
    all_selections = []

    for yw in sorted(test_df['year_week'].unique()):
        week_df = test_df[test_df['year_week'] == yw].copy()
        if len(week_df) == 0:
            continue

        # 每周第一个交易日做决策
        decision_day = week_df['trade_date'].min()
        decision_df = week_df[week_df['trade_date'] == decision_day].copy()

        if len(decision_df) < 2:
            continue

        # 选股
        if strategy == 'ml_rank':
            # ML模型预测，选概率最高的
            X = decision_df[feature_cols].fillna(0)
            proba = model.predict_proba(X)[:, 1]  # 上涨概率
            decision_df['score'] = proba

        elif strategy == 'momentum':
            # 动量排名：选5日涨幅最大的
            decision_df['score'] = decision_df['return_5d'].fillna(0)

        elif strategy == 'ml_momentum':
            # 综合：ML概率 * 0.5 + 动量排名 * 0.5
            X = decision_df[feature_cols].fillna(0)
            proba = model.predict_proba(X)[:, 1]
            momentum = decision_df['return_5d'].fillna(0)
            # 标准化后混合
            proba_norm = (proba - proba.mean()) / (proba.std() + 1e-10)
            mom_norm = (momentum - momentum.mean()) / (momentum.std() + 1e-10)
            decision_df['score'] = proba_norm * 0.5 + mom_norm * 0.5

        elif strategy == 'random':
            # 随机选股（对照组）
            decision_df['score'] = np.random.random(len(decision_df))

        elif strategy == 'equal_weight':
            # 等权持有全部（基准）
            decision_df['score'] = 1.0

        # 排序选股
        decision_df = decision_df.sort_values('score', ascending=False)

        if strategy == 'equal_weight':
            selected = decision_df  # 全选
        else:
            selected = decision_df.head(n_stocks)

        # 计算本周收益
        week_returns = []
        for _, stock in selected.iterrows():
            symbol = stock['symbol']
            stock_week = week_df[week_df['symbol'] == symbol].sort_values('trade_date')
            if len(stock_week) >= 2:
                week_ret = stock_week['close'].iloc[-1] / stock_week['close'].iloc[0] - 1
                week_returns.append(week_ret)

        if week_returns:
            avg_return = np.mean(week_returns)
        else:
            avg_return = 0

        # 扣除交易成本（每周调仓一次，双向交易）
        if strategy != 'equal_weight':
            avg_return -= TOTAL_COST

        weekly_results.append({
            'year_week': yw,
            'decision_date': decision_day,
            'n_stocks': len(selected),
            'week_return': avg_return,
            'selected_stocks': list(selected['symbol'].values),
        })

        all_selections.append({
            'year_week': yw,
            'stocks': list(selected['symbol'].values),
            'scores': list(selected['score'].values),
        })

    return pd.DataFrame(weekly_results), all_selections, model


# ============================================================
# 3. 回测计算
# ============================================================

def compute_strategy_metrics(weekly_results, strategy_name):
    """计算策略指标"""
    returns = weekly_results['week_return'].values

    # 累计收益
    cumulative = (1 + returns).cumprod()
    total_return = cumulative[-1] - 1

    # 年化收益
    n_weeks = len(returns)
    years = n_weeks / 52
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 夏普比率（年化）
    weekly_std = np.std(returns)
    sharpe = np.mean(returns) / (weekly_std + 1e-10) * np.sqrt(52)

    # 最大回撤
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_drawdown = np.min(drawdown)

    # 胜率
    win_rate = np.sum(returns > 0) / len(returns)

    return {
        'strategy': strategy_name,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_weeks': n_weeks,
    }


# ============================================================
# 4. 主流程
# ============================================================

def main():
    print("=" * 100)
    print("小资金量重构方案 - 每周选1-2只最看好的股票")
    print("=" * 100)

    # 加载数据
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM features_v5", conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # 生成5日收益率标签
    df['target_return_5d'] = df.groupby('symbol')['close'].pct_change(5).shift(-5)

    # 增加特征
    df = add_features(df)
    df = df.dropna(subset=['target_return_5d'])

    # 特征列
    EXCLUDE = {'trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount',
               'target_return_10d', 'target_direction_10d', 'scenario_label_10d',
               'target_return_5d'}
    feature_cols = [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    print(f"\n特征数: {len(feature_cols)}")
    print(f"数据: {len(df)}条, {df['symbol'].nunique()}只股票")
    print(f"交易成本: {TOTAL_COST*100:.2f}%/次")

    all_metrics = []

    # ============================================================
    # 基准1: 等权持有全部
    # ============================================================
    print(f"\n{'=' * 80}")
    print("基准1: 等权持有全部22只")
    print(f"{'=' * 80}")
    weekly_results, _, _ = weekly_stock_selection(df, feature_cols, 'equal_weight', n_stocks=22)
    metrics = compute_strategy_metrics(weekly_results, '基准:等权全持')
    all_metrics.append(metrics)
    print(f"  总收益: {metrics['total_return']*100:.2f}%  年化: {metrics['annual_return']*100:.2f}%  夏普: {metrics['sharpe']:.2f}  最大回撤: {metrics['max_drawdown']*100:.2f}%  胜率: {metrics['win_rate']*100:.1f}%")

    # ============================================================
    # 基准2: 随机选2只
    # ============================================================
    print(f"\n{'=' * 80}")
    print("基准2: 每周随机选2只")
    print(f"{'=' * 80}")
    np.random.seed(42)
    weekly_results, _, _ = weekly_stock_selection(df, feature_cols, 'random', n_stocks=2)
    metrics = compute_strategy_metrics(weekly_results, '基准:随机2只')
    all_metrics.append(metrics)
    print(f"  总收益: {metrics['total_return']*100:.2f}%  年化: {metrics['annual_return']*100:.2f}%  夏普: {metrics['sharpe']:.2f}  最大回撤: {metrics['max_drawdown']*100:.2f}%  胜率: {metrics['win_rate']*100:.1f}%")

    # ============================================================
    # 方案A: ML排名选股（选1只）
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案A1: ML排名选股 - 每周选1只最看好的")
    print(f"{'=' * 80}")
    weekly_results, selections, model = weekly_stock_selection(df, feature_cols, 'ml_rank', n_stocks=1)
    metrics = compute_strategy_metrics(weekly_results, 'A1:ML选1只')
    all_metrics.append(metrics)
    print(f"  总收益: {metrics['total_return']*100:.2f}%  年化: {metrics['annual_return']*100:.2f}%  夏普: {metrics['sharpe']:.2f}  最大回撤: {metrics['max_drawdown']*100:.2f}%  胜率: {metrics['win_rate']*100:.1f}%")

    # ============================================================
    # 方案A2: ML排名选股（选2只）
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案A2: ML排名选股 - 每周选2只最看好的")
    print(f"{'=' * 80}")
    weekly_results, _, _ = weekly_stock_selection(df, feature_cols, 'ml_rank', n_stocks=2)
    metrics = compute_strategy_metrics(weekly_results, 'A2:ML选2只')
    all_metrics.append(metrics)
    print(f"  总收益: {metrics['total_return']*100:.2f}%  年化: {metrics['annual_return']*100:.2f}%  夏普: {metrics['sharpe']:.2f}  最大回撤: {metrics['max_drawdown']*100:.2f}%  胜率: {metrics['win_rate']*100:.1f}%")

    # ============================================================
    # 方案B1: 动量轮动（选1只）
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案B1: 动量轮动 - 每周选近期涨得最多的1只")
    print(f"{'=' * 80}")
    weekly_results, _, _ = weekly_stock_selection(df, feature_cols, 'momentum', n_stocks=1)
    metrics = compute_strategy_metrics(weekly_results, 'B1:动量选1只')
    all_metrics.append(metrics)
    print(f"  总收益: {metrics['total_return']*100:.2f}%  年化: {metrics['annual_return']*100:.2f}%  夏普: {metrics['sharpe']:.2f}  最大回撤: {metrics['max_drawdown']*100:.2f}%  胜率: {metrics['win_rate']*100:.1f}%")

    # ============================================================
    # 方案B2: 动量轮动（选2只）
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案B2: 动量轮动 - 每周选近期涨得最多的2只")
    print(f"{'=' * 80}")
    weekly_results, _, _ = weekly_stock_selection(df, feature_cols, 'momentum', n_stocks=2)
    metrics = compute_strategy_metrics(weekly_results, 'B2:动量选2只')
    all_metrics.append(metrics)
    print(f"  总收益: {metrics['total_return']*100:.2f}%  年化: {metrics['annual_return']*100:.2f}%  夏普: {metrics['sharpe']:.2f}  最大回撤: {metrics['max_drawdown']*100:.2f}%  胜率: {metrics['win_rate']*100:.1f}%")

    # ============================================================
    # 方案C1: ML+动量混合（选1只）
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案C1: ML+动量混合 - 每周选综合得分最高的1只")
    print(f"{'=' * 80}")
    weekly_results, _, _ = weekly_stock_selection(df, feature_cols, 'ml_momentum', n_stocks=1)
    metrics = compute_strategy_metrics(weekly_results, 'C1:混合选1只')
    all_metrics.append(metrics)
    print(f"  总收益: {metrics['total_return']*100:.2f}%  年化: {metrics['annual_return']*100:.2f}%  夏普: {metrics['sharpe']:.2f}  最大回撤: {metrics['max_drawdown']*100:.2f}%  胜率: {metrics['win_rate']*100:.1f}%")

    # ============================================================
    # 方案C2: ML+动量混合（选2只）
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案C2: ML+动量混合 - 每周选综合得分最高的2只")
    print(f"{'=' * 80}")
    weekly_results, _, _ = weekly_stock_selection(df, feature_cols, 'ml_momentum', n_stocks=2)
    metrics = compute_strategy_metrics(weekly_results, 'C2:混合选2只')
    all_metrics.append(metrics)
    print(f"  总收益: {metrics['total_return']*100:.2f}%  年化: {metrics['annual_return']*100:.2f}%  夏普: {metrics['sharpe']:.2f}  最大回撤: {metrics['max_drawdown']*100:.2f}%  胜率: {metrics['win_rate']*100:.1f}%")

    # ============================================================
    # 汇总对比
    # ============================================================
    print(f"\n\n{'=' * 100}")
    print("汇总对比")
    print(f"{'=' * 100}")

    print(f"\n{'方案':<20s} {'总收益':>8s} {'年化':>8s} {'夏普':>6s} {'最大回撤':>8s} {'胜率':>6s}")
    print("-" * 60)
    for m in all_metrics:
        print(f"{m['strategy']:<20s} {m['total_return']*100:>7.2f}% {m['annual_return']*100:>7.2f}% "
              f"{m['sharpe']:>6.2f} {m['max_drawdown']*100:>7.2f}% {m['win_rate']*100:>5.1f}%")

    # 排名
    print(f"\n{'=' * 100}")
    print("排名")
    print(f"{'=' * 100}")

    sorted_by_return = sorted(all_metrics, key=lambda x: x['total_return'], reverse=True)
    print(f"\n按总收益排名:")
    for i, m in enumerate(sorted_by_return):
        print(f"  {i+1}. {m['strategy']}: {m['total_return']*100:.2f}%")

    sorted_by_sharpe = sorted(all_metrics, key=lambda x: x['sharpe'], reverse=True)
    print(f"\n按夏普比率排名:")
    for i, m in enumerate(sorted_by_sharpe):
        print(f"  {i+1}. {m['strategy']}: {m['sharpe']:.2f}")

    # 保存结果
    import json
    results = {
        'experiments': all_metrics,
        'best_return': sorted_by_return[0]['strategy'],
        'best_sharpe': sorted_by_sharpe[0]['strategy'],
    }
    with open(os.path.join(BASE_MODEL_DIR, 'small_capital_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到 {os.path.join(BASE_MODEL_DIR, 'small_capital_results.json')}")


if __name__ == '__main__':
    import os
    main()
