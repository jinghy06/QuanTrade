"""调试：对比买入不动和等权持有的计算"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

conn = sqlite3.connect(DB_PATH)
etf_df = pd.read_sql_query("SELECT * FROM etf_daily_prices", conn)
etf_df['trade_date'] = pd.to_datetime(etf_df['trade_date'])
conn.close()

test_start = pd.Timestamp('2024-01-01')

# 只看有完整数据的7只ETF
full_etfs = ['159995', '512660', '512670', '515070', '515960', '516510', '562500']
etf_full = etf_df[etf_df['symbol'].isin(full_etfs)]

print("=" * 80)
print("对比两种计算方法")
print("=" * 80)

# 方法1：买入不动
print("\n方法1：买入不动（2024年初买入，持有至今）")
first_prices = {}
last_prices = {}
for symbol in full_etfs:
    sym_data = etf_full[etf_full['symbol'] == symbol].sort_values('trade_date')
    test_data = sym_data[sym_data['trade_date'] >= test_start]
    if len(test_data) > 0:
        first_prices[symbol] = test_data['close'].iloc[0]
        last_prices[symbol] = test_data['close'].iloc[-1]
        ret = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100
        print(f"  {symbol}: {first_prices[symbol]:.3f} -> {last_prices[symbol]:.3f} ({ret:+.2f}%)")

# 等权：每只投1万
initial = 10000 * len(first_prices)
final = sum(10000 / first_prices[s] * last_prices[s] for s in first_prices)
buy_hold_ret = (final / initial - 1) * 100
print(f"\n  总投入: {initial:.0f}")
print(f"  总市值: {final:.0f}")
print(f"  收益率: {buy_hold_ret:.2f}%")

# 方法2：等权持有（每周调仓）
print("\n" + "=" * 80)
print("方法2：等权持有（每周调仓）")
print("=" * 80)

etf_full_copy = etf_full.copy()
etf_full_copy['year_week'] = etf_full_copy['trade_date'].dt.isocalendar().year * 100 + etf_full_copy['trade_date'].dt.isocalendar().week

# 只看测试集
test_data = etf_full_copy[etf_full_copy['trade_date'] >= test_start]

weekly_returns = []
for yw in sorted(test_data['year_week'].unique()):
    week_df = test_data[test_data['year_week'] == yw]
    rets = []
    for sym in week_df['symbol'].unique():
        sd = week_df[week_df['symbol'] == sym].sort_values('trade_date')
        if len(sd) >= 2:
            rets.append(sd['close'].iloc[-1] / sd['close'].iloc[0] - 1)
    if rets:
        avg_ret = np.mean(rets)
        weekly_returns.append(avg_ret)

# 累积收益
cumulative = 1.0
for r in weekly_returns:
    cumulative *= (1 + r)
rebal_ret = (cumulative - 1) * 100

print(f"  周数: {len(weekly_returns)}")
print(f"  累积收益: {rebal_ret:.2f}%")

# 对比
print("\n" + "=" * 80)
print("对比结果")
print("=" * 80)
print(f"  买入不动: {buy_hold_ret:.2f}%")
print(f"  等权持有: {rebal_ret:.2f}%")
print(f"  差异: {buy_hold_ret - rebal_ret:.2f}%")

# 检查是否有异常周
print("\n" + "=" * 80)
print("检查异常周（收益 > 50% 或 < -50%）")
print("=" * 80)
for i, r in enumerate(weekly_returns):
    if abs(r) > 0.5:
        print(f"  第{i+1}周: {r*100:+.2f}%")
