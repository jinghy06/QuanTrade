"""
Backtest v4 — 多策略回测引擎
- 基准(持有) / 原融合策略 / 三情景策略 / 三情景+Kelly
- 输出 backtest_results_v4 和 backtest_daily_v4
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DB_PATH = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
TEST_ETFS = ['562500.SH', '588200.SH', '588790.SH', '159382.SZ', '159241.SZ']

FEE_RATE = 0.0001  # 0.01%
SLIPPAGE = 0.0001  # 0.01%
INIT_CAPITAL = 1_000_000.0

print("=" * 70)
print("【Backtest v4 — 多策略回测引擎】")
print("=" * 70)

# ==================== 1. 读取数据 ====================
conn = sqlite3.connect(DB_PATH)

# 读取 daily_prices 计算实际日收益
placeholders = ','.join([f"'{s}'" for s in TEST_ETFS])

df_prices = pd.read_sql_query(
    f"""
    SELECT trade_date, symbol, open, high, low, close
    FROM daily_prices
    WHERE symbol IN ({placeholders})
    ORDER BY symbol, trade_date
    """,
    conn
)

# 读取 predictions_v4 (原融合策略信号)
df_pred = pd.read_sql_query(
    f"""
    SELECT trade_date, symbol, rf_binary_pred, target_return_10d
    FROM predictions_v4
    WHERE symbol IN ({placeholders})
    ORDER BY symbol, trade_date
    """,
    conn
)

# 读取 scenario_signals_v4 (三情景策略信号)
df_scenario = pd.read_sql_query(
    f"""
    SELECT trade_date, symbol, position_size, signal_direction, scenario_decision
    FROM scenario_signals_v4
    WHERE symbol IN ({placeholders})
    ORDER BY symbol, trade_date
    """,
    conn
)

conn.close()

if df_prices.empty:
    print("[错误] daily_prices 为空")
    sys.exit(1)

print(f"[数据] daily_prices: {len(df_prices)} 条")
print(f"[数据] predictions_v4: {len(df_pred)} 条")
print(f"[数据] scenario_signals_v4: {len(df_scenario)} 条")

# 转换日期
df_prices['trade_date'] = pd.to_datetime(df_prices['trade_date'], format='mixed')
df_pred['trade_date'] = pd.to_datetime(df_pred['trade_date'], format='mixed')
df_scenario['trade_date'] = pd.to_datetime(df_scenario['trade_date'], format='mixed')

# 去重
df_prices = df_prices.drop_duplicates(subset=['symbol', 'trade_date'], keep='last')
df_pred = df_pred.drop_duplicates(subset=['symbol', 'trade_date'], keep='last')
df_scenario = df_scenario.drop_duplicates(subset=['symbol', 'trade_date'], keep='last')

# 计算日收益率
df_prices = df_prices.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
df_prices['daily_return'] = df_prices.groupby('symbol')['close'].pct_change().shift(-1)
df_prices = df_prices.dropna(subset=['daily_return']).copy()

print(f"[数据] 价格数据已准备: {len(df_prices)} 条")

# ==================== 2. 回测引擎 ====================
def run_backtest(df_sym, strategy_name, signal_col, position_col=None, kelly=False):
    """
    运行单策略回测
    - signal_col: 信号列名 (1=买入/持有, 0=卖出/观望)
    - position_col: 仓位列名 (可选，默认signal_col即为仓位)
    - kelly: 是否启用Kelly仓位优化
    """
    df_sym = df_sym.sort_values('trade_date').reset_index(drop=True)
    n = len(df_sym)
    if n == 0:
        return None, None

    capital = INIT_CAPITAL
    nav = [capital]
    positions = [0.0]
    signals = [0]
    benchmark_nav = [capital]

    # 基准始终满仓
    benchmark_cum = INIT_CAPITAL

    for i in range(n):
        row = df_sym.iloc[i]
        daily_ret = row['daily_return']

        # 基准
        benchmark_cum *= (1 + daily_ret)
        benchmark_nav.append(benchmark_cum)

        # 信号
        if kelly:
            # Kelly仓位优化: 基于最近10日预测准确率
            if i < 10:
                kelly_pos = 0.5
            else:
                recent = df_sym.iloc[max(0, i-10):i]
                # 用 rf_binary_pred 作为预测方向，daily_return>0 作为实际方向
                pred_dir = recent['rf_binary_pred'].values
                actual_dir = (recent['daily_return'].values > 0).astype(int)
                wins = recent[pred_dir == actual_dir]
                losses = recent[pred_dir != actual_dir]
                p = len(wins) / len(recent) if len(recent) > 0 else 0.5
                win_rets = wins['daily_return'].values
                loss_rets = losses['daily_return'].values
                avg_win = np.mean(win_rets[win_rets > 0]) if len(win_rets[win_rets > 0]) > 0 else 0.001
                avg_loss = abs(np.mean(loss_rets[loss_rets < 0])) if len(loss_rets[loss_rets < 0]) > 0 else 0.001
                b = avg_win / avg_loss if avg_loss > 0 else 1.0
                q = 1 - p
                kelly_frac = (p * b - q) / b if b > 0 else 0.0
                kelly_pos = max(0.0, min(1.0, kelly_frac * 0.5))

            # 三情景信号方向决定是否在市场中
            sig_dir = row['signal_direction']
            if sig_dir == 1:  # long
                pos = kelly_pos
            else:
                pos = 0.0
        else:
            if position_col and position_col in df_sym.columns:
                pos = row[position_col]
            else:
                pos = float(row[signal_col])

        # 交易费用 (仓位变化时产生)
        prev_pos = positions[-1]
        if abs(pos - prev_pos) > 1e-6:
            cost = abs(pos - prev_pos) * (FEE_RATE + SLIPPAGE)
            capital *= (1 - cost)

        # 当日收益
        ret = daily_ret * pos
        capital *= (1 + ret)

        nav.append(capital)
        positions.append(pos)
        signals.append(1 if pos > 0 else 0)

    # 计算指标
    nav_arr = np.array(nav)
    total_ret = nav_arr[-1] / INIT_CAPITAL - 1
    n_days = len(nav_arr) - 1
    annual_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 and total_ret > -1 else -1.0

    peak = np.maximum.accumulate(nav_arr)
    dd = (nav_arr - peak) / peak
    max_dd = np.min(dd)

    daily_rets = np.diff(nav_arr) / nav_arr[:-1]
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0.0
    win_rate = np.sum(daily_rets > 0) / len(daily_rets) if len(daily_rets) > 0 else 0.0

    benchmark_total = benchmark_nav[-1] / INIT_CAPITAL - 1

    result = {
        'symbol': df_sym['symbol'].iloc[0],
        'strategy': strategy_name,
        'total_return': total_ret,
        'annual_return': annual_ret,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'benchmark_return': benchmark_total,
    }

    daily_df = pd.DataFrame({
        'symbol': df_sym['symbol'].iloc[0],
        'strategy': strategy_name,
        'trade_date': df_sym['trade_date'].tolist() + [df_sym['trade_date'].iloc[-1]],
        'nav': nav_arr,
        'position': positions,
        'signal': signals,
        'benchmark_nav': benchmark_nav,
    })

    return result, daily_df

# ==================== 3. 运行各策略回测 ====================
all_results = []
all_daily = []

for sym in TEST_ETFS:
    sym_prices = df_prices[df_prices['symbol'] == sym].copy()
    if len(sym_prices) < 20:
        print(f"[跳过] {sym}: 价格数据不足 ({len(sym_prices)})")
        continue

    # 策略1: 基准(持有) — 使用全部价格数据
    sym_prices['benchmark_signal'] = 1.0
    r, d = run_backtest(sym_prices, '基准(持有)', 'benchmark_signal')
    if r:
        all_results.append(r)
        all_daily.append(d)
        print(f"\n>>> {sym}: 基准(持有) {len(sym_prices)} 个交易日")
        print(f"  基准(持有): 总收益={r['total_return']*100:+.2f}% 最大回撤={r['max_drawdown']*100:.2f}%")

    # 策略2: 原融合策略 — 仅使用 predictions_v4 有信号的日期
    sym_pred = df_pred[df_pred['symbol'] == sym][['trade_date', 'symbol', 'rf_binary_pred']].copy()
    if len(sym_pred) > 0:
        sym_merged = sym_prices.merge(sym_pred, on=['trade_date', 'symbol'], how='inner')
        if len(sym_merged) >= 20:
            r, d = run_backtest(sym_merged, '原融合策略', 'rf_binary_pred')
            if r:
                all_results.append(r)
                all_daily.append(d)
                print(f"  原融合策略: {len(sym_merged)} 个交易日 | 总收益={r['total_return']*100:+.2f}% 最大回撤={r['max_drawdown']*100:.2f}%")
        else:
            print(f"  原融合策略: 数据不足 ({len(sym_merged)} 天)，跳过")
    else:
        print(f"  原融合策略: 无预测数据，跳过")

    # 策略3: 三情景策略 — 仅使用 scenario_signals_v4 有信号的日期
    sym_scen = df_scenario[df_scenario['symbol'] == sym][['trade_date', 'symbol', 'position_size', 'signal_direction', 'scenario_decision']].copy()
    if len(sym_scen) > 0:
        sym_merged = sym_prices.merge(sym_scen, on=['trade_date', 'symbol'], how='inner')
        if len(sym_merged) >= 20:
            # 当 signal_direction == -1 (short/hedge) 时，视为空仓
            sym_merged['scenario_pos'] = sym_merged.apply(
                lambda row: row['position_size'] if row['signal_direction'] == 1 else 0.0, axis=1
            )
            r, d = run_backtest(sym_merged, '三情景策略', 'scenario_pos', position_col='scenario_pos')
            if r:
                all_results.append(r)
                all_daily.append(d)
                print(f"  三情景策略: {len(sym_merged)} 个交易日 | 总收益={r['total_return']*100:+.2f}% 最大回撤={r['max_drawdown']*100:.2f}%")
        else:
            print(f"  三情景策略: 数据不足 ({len(sym_merged)} 天)，跳过")
    else:
        print(f"  三情景策略: 无信号数据，跳过")

    # 策略4: 三情景+Kelly — 在三情景有效日期上叠加Kelly
    if len(sym_scen) > 0:
        sym_merged = sym_prices.merge(sym_scen, on=['trade_date', 'symbol'], how='inner')
        # 需要 rf_binary_pred 用于Kelly计算，从 predictions_v4 合并（若缺失则默认0.5）
        sym_merged = sym_merged.merge(df_pred[df_pred['symbol'] == sym][['trade_date', 'rf_binary_pred']], on='trade_date', how='left')
        sym_merged['rf_binary_pred'] = sym_merged['rf_binary_pred'].fillna(0).astype(int)
        if len(sym_merged) >= 20:
            r, d = run_backtest(sym_merged, '三情景+Kelly', 'signal_direction', kelly=True)
            if r:
                all_results.append(r)
                all_daily.append(d)
                print(f"  三情景+Kelly: {len(sym_merged)} 个交易日 | 总收益={r['total_return']*100:+.2f}% 最大回撤={r['max_drawdown']*100:.2f}%")
        else:
            print(f"  三情景+Kelly: 数据不足 ({len(sym_merged)} 天)，跳过")
    else:
        print(f"  三情景+Kelly: 无信号数据，跳过")

# ==================== 4. 汇总输出 ====================
if not all_results:
    print("[错误] 没有生成回测结果")
    sys.exit(1)

results_df = pd.DataFrame(all_results)
daily_df = pd.concat(all_daily, ignore_index=True)

print("\n" + "=" * 70)
print("【策略对比汇总】")
print("=" * 70)
print(f"{'ETF':<12} {'策略':<14} {'总收益':<10} {'年化收益':<10} {'最大回撤':<10} {'夏普':<8} {'胜率':<8} {'基准收益':<10}")
print("-" * 90)
for _, r in results_df.iterrows():
    print(f"{r['symbol']:<12} {r['strategy']:<14} {r['total_return']*100:>8.2f}% {r['annual_return']*100:>8.2f}% {r['max_drawdown']*100:>8.2f}% {r['sharpe']:>6.2f} {r['win_rate']*100:>6.1f}% {r['benchmark_return']*100:>8.2f}%")

# 按策略平均
print("\n" + "=" * 70)
print("【策略平均表现】")
print("=" * 70)
avg_df = results_df.groupby('strategy')[['total_return', 'annual_return', 'max_drawdown', 'sharpe', 'win_rate', 'benchmark_return']].mean()
for strategy, row in avg_df.iterrows():
    print(f"{strategy:<14} 总收益={row['total_return']*100:+.2f}% 年化={row['annual_return']*100:+.2f}% 最大回撤={row['max_drawdown']*100:.2f}% 夏普={row['sharpe']:.2f} 胜率={row['win_rate']*100:.1f}%")

# ==================== 5. 保存到数据库 ====================
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# backtest_results_v4
cursor.execute("DROP TABLE IF EXISTS backtest_results_v4")
cursor.execute("""
CREATE TABLE backtest_results_v4 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    strategy TEXT,
    total_return REAL,
    annual_return REAL,
    max_drawdown REAL,
    sharpe REAL,
    win_rate REAL,
    benchmark_return REAL
)
""")
conn.commit()

for _, r in results_df.iterrows():
    cursor.execute("""
        INSERT INTO backtest_results_v4 (symbol, strategy, total_return, annual_return, max_drawdown, sharpe, win_rate, benchmark_return)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (r['symbol'], r['strategy'], r['total_return'], r['annual_return'], r['max_drawdown'], r['sharpe'], r['win_rate'], r['benchmark_return']))

# backtest_daily_v4
cursor.execute("DROP TABLE IF EXISTS backtest_daily_v4")
cursor.execute("""
CREATE TABLE backtest_daily_v4 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    strategy TEXT,
    trade_date TEXT,
    nav REAL,
    position REAL,
    signal INTEGER,
    benchmark_nav REAL
)
""")
conn.commit()

# 批量插入每日数据
for _, r in daily_df.iterrows():
    cursor.execute("""
        INSERT INTO backtest_daily_v4 (symbol, strategy, trade_date, nav, position, signal, benchmark_nav)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (r['symbol'], r['strategy'], r['trade_date'].strftime('%Y-%m-%d'), r['nav'], r['position'], r['signal'], r['benchmark_nav']))

conn.commit()
conn.close()

print(f"\n[保存] backtest_results_v4: {len(results_df)} 条")
print(f"[保存] backtest_daily_v4: {len(daily_df)} 条")

print("\n" + "=" * 70)
print("【Backtest v4 完成】")
print("=" * 70)
