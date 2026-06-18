"""
Backtest v4 — Tuned Scenario Strategy
使用调优后的 scenario_signals_v4 重新回测
对比: 基准(持有) / 原融合策略 / 三情景调优策略 / 三情景调优+Kelly
"""
import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
TEST_ETFS = ['562500.SH', '588200.SH', '588790.SH', '159382.SZ', '159241.SZ']

print("=" * 70)
print("【Backtest v4 TUNED — 调优版三情景策略回测】")
print("=" * 70)

# ==================== 1. 加载数据 ====================
conn = sqlite3.connect(DB_PATH)

# 日K数据
placeholders = ','.join(["'" + s + "'" for s in TEST_ETFS])
df_price = pd.read_sql_query(
    f"SELECT * FROM daily_prices WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date",
    conn
)
df_price['trade_date'] = pd.to_datetime(df_price['trade_date'], format='mixed')
df_price = df_price.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

# 原融合策略信号 (从 predictions_v4 或 strategy_signals_v3)
try:
    df_old_signals = pd.read_sql_query(
        f"SELECT trade_date, symbol, rf_binary_pred FROM predictions_v4 WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date",
        conn
    )
    df_old_signals['trade_date'] = pd.to_datetime(df_old_signals['trade_date'], format='mixed')
    has_old_signals = True
    print(f"[数据] 原融合策略信号: {len(df_old_signals)} 条")
except:
    has_old_signals = False
    print("[警告] 无原融合策略信号，跳过对比")

# 调优版三情景信号
df_scenario = pd.read_sql_query(
    f"SELECT trade_date, symbol, close, position_size, signal_direction, scenario_decision, fusion_adverse_proba, fusion_base_proba, fusion_favorable_proba FROM scenario_signals_v4 WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date",
    conn
)
df_scenario['trade_date'] = pd.to_datetime(df_scenario['trade_date'], format='mixed')
print(f"[数据] 调优版三情景信号: {len(df_scenario)} 条")

conn.close()

# ==================== 2. 回测引擎 ====================
def backtest_strategy(df_price, df_signals, strategy_name, position_col='position_size',
                      fee_rate=0.0001, slippage=0.0001, initial_capital=1_000_000):
    """
    通用回测引擎
    df_signals 必须包含: trade_date, symbol, position_size (0~1)
    """
    results = []
    daily_records = []
    
    for symbol in TEST_ETFS:
        price_df = df_price[df_price['symbol'] == symbol].sort_values('trade_date').reset_index(drop=True)
        sig_df = df_signals[df_signals['symbol'] == symbol].sort_values('trade_date').reset_index(drop=True)
        
        if len(price_df) < 10 or len(sig_df) < 5:
            continue
        
        # 合并信号到价格数据
        merged = pd.merge(price_df, sig_df[['trade_date', position_col, 'signal_direction']], 
                         on='trade_date', how='left')
        merged[position_col] = merged[position_col].fillna(0)
        merged['signal_direction'] = merged['signal_direction'].fillna(0)
        
        # 计算收益
        merged['return'] = merged['close'].pct_change()
        
        capital = initial_capital
        position = 0.0
        shares = 0.0
        nav_history = []
        
        for i in range(1, len(merged)):
            row = merged.iloc[i]
            prev_row = merged.iloc[i-1]
            
            target_pos = row[position_col]  # 0~1
            
            # 简化: 直接使用position_size作为做多仓位
            # 不再强制根据signal_direction空仓 (避免binary_pred错误导致踏空)
            effective_pos = float(target_pos) if not pd.isna(target_pos) else 0.0
            
            # 调仓
            if abs(effective_pos - position) > 0.01:
                trade_value = abs(effective_pos - position) * capital
                fee = trade_value * (fee_rate + slippage)
                capital -= fee
                position = effective_pos
            
            # 当日收益
            daily_return = row['return'] if not pd.isna(row['return']) else 0
            capital *= (1 + daily_return * position)
            
            nav_history.append({
                'trade_date': row['trade_date'],
                'symbol': symbol,
                'strategy': strategy_name,
                'nav': capital,
                'position': position,
                'signal': 1 if position > 0 else 0,
                'daily_return': daily_return * position,
                'close': row['close']
            })
        
        # 基准(持有)
        benchmark_nav = initial_capital
        benchmark_history = []
        for i in range(1, len(merged)):
            row = merged.iloc[i]
            daily_return = row['return'] if not pd.isna(row['return']) else 0
            benchmark_nav *= (1 + daily_return)
            benchmark_history.append({
                'trade_date': row['trade_date'],
                'symbol': symbol,
                'strategy': '基准(持有)',
                'nav': benchmark_nav,
                'position': 1.0,
                'signal': 1,
                'daily_return': daily_return,
                'close': row['close']
            })
        
        # 计算指标
        nav_series = pd.Series([n['nav'] for n in nav_history])
        if len(nav_series) > 0:
            total_return = nav_series.iloc[-1] / initial_capital - 1
            days = len(nav_series)
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            
            # 最大回撤
            cummax = nav_series.cummax()
            drawdown = (nav_series - cummax) / cummax
            max_drawdown = drawdown.min()
            
            # 夏普
            daily_rets = pd.Series([n['daily_return'] for n in nav_history])
            sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
            
            # 胜率
            win_rate = (daily_rets > 0).mean()
            
            # 基准收益
            benchmark_total = benchmark_history[-1]['nav'] / initial_capital - 1 if benchmark_history else 0
            
            results.append({
                'symbol': symbol,
                'strategy': strategy_name,
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'benchmark_return': benchmark_total,
                'trading_days': days
            })
        
        daily_records.extend(nav_history)
        daily_records.extend(benchmark_history)
    
    return pd.DataFrame(results), pd.DataFrame(daily_records)


# ==================== 3. 运行回测 ====================
all_results = []
all_daily = []

# 3.1 调优版三情景策略
print("\n【回测1: 三情景调优策略】")
res_tuned, daily_tuned = backtest_strategy(df_price, df_scenario, '三情景调优', 'position_size')
all_results.append(res_tuned)
all_daily.append(daily_tuned)
print(res_tuned.to_string(index=False))

# 3.2 三情景调优 + Kelly (半Kelly)
print("\n【回测2: 三情景调优 + Kelly】")
df_kelly = df_scenario.copy()
# Kelly仓位 = 2*p - 1 (简化版), 再与三情景仓位取较小值
df_kelly['kelly_pos'] = (2 * df_kelly['fusion_favorable_proba'] - 1).clip(0, 1)
df_kelly['position_size'] = (df_kelly['position_size'] * 0.5 + df_kelly['kelly_pos'] * 0.5).clip(0.2, 1.0)
res_kelly, daily_kelly = backtest_strategy(df_price, df_kelly, '三情景调优+Kelly', 'position_size')
all_results.append(res_kelly)
all_daily.append(daily_kelly)
print(res_kelly.to_string(index=False))

# 3.3 原融合策略 (如果有)
if has_old_signals:
    print("\n【回测3: 原融合策略】")
    df_old = df_old_signals.copy()
    df_old['position_size'] = df_old['rf_binary_pred'].astype(float) * 0.8  # 80%仓位
    df_old['signal_direction'] = 1
    res_old, daily_old = backtest_strategy(df_price, df_old, '原融合策略', 'position_size')
    all_results.append(res_old)
    all_daily.append(daily_old)
    print(res_old.to_string(index=False))

# ==================== 4. 汇总对比 ====================
print("\n" + "=" * 70)
print("【策略对比汇总】")
print("=" * 70)

summary = pd.concat(all_results, ignore_index=True)
summary_display = summary.copy()
summary_display['总收益率(%)'] = summary_display['total_return'] * 100
summary_display['年化收益率(%)'] = summary_display['annual_return'] * 100
summary_display['最大回撤(%)'] = summary_display['max_drawdown'] * 100
summary_display['基准收益(%)'] = summary_display['benchmark_return'] * 100
summary_display['胜率(%)'] = summary_display['win_rate'] * 100

print("\n按ETF分组:")
for sym in TEST_ETFS:
    sym_df = summary_display[summary_display['symbol'] == sym]
    if len(sym_df) == 0:
        continue
    print(f"\n>>> {sym}")
    print(sym_df[['strategy', 'total_return', 'benchmark_return', 'max_drawdown', 'sharpe', 'win_rate']].to_string(index=False))

print("\n\n按策略平均:")
avg_df = summary_display.groupby('strategy').agg({
    'total_return': 'mean',
    'benchmark_return': 'mean',
    'max_drawdown': 'mean',
    'sharpe': 'mean',
    'win_rate': 'mean'
}).round(4).reset_index()
print(avg_df.to_string(index=False))

# ==================== 5. 保存到数据库 ====================
print("\n【保存结果到数据库...】")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 保存回测结果
cursor.execute("DROP TABLE IF EXISTS backtest_results_v4_tuned")
cursor.execute("""
CREATE TABLE backtest_results_v4_tuned (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    strategy TEXT,
    total_return REAL,
    annual_return REAL,
    max_drawdown REAL,
    sharpe REAL,
    win_rate REAL,
    benchmark_return REAL,
    trading_days INTEGER
)
""")
conn.commit()

summary.to_sql('backtest_results_v4_tuned', conn, if_exists='append', index=False)

# 保存每日净值
cursor.execute("DROP TABLE IF EXISTS backtest_daily_v4_tuned")
cursor.execute("""
CREATE TABLE backtest_daily_v4_tuned (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TIMESTAMP,
    symbol TEXT,
    strategy TEXT,
    nav REAL,
    position REAL,
    signal INTEGER,
    daily_return REAL,
    close REAL
)
""")
conn.commit()

daily_all = pd.concat(all_daily, ignore_index=True)
daily_all.to_sql('backtest_daily_v4_tuned', conn, if_exists='append', index=False)

conn.commit()
conn.close()

print(f"[保存] backtest_results_v4_tuned: {len(summary)} 条")
print(f"[保存] backtest_daily_v4_tuned: {len(daily_all)} 条")

print("\n" + "=" * 70)
print("【Backtest v4 TUNED 完成】")
print("=" * 70)
