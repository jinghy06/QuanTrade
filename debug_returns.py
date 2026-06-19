"""调试：为什么买入不动和等权持有的收益差这么多？"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

conn = sqlite3.connect(DB_PATH)
etf_df = pd.read_sql_query("SELECT * FROM etf_daily_prices", conn)
etf_df['trade_date'] = pd.to_datetime(etf_df['trade_date'])
conn.close()

BENCHMARK = '510300'
test_start = pd.Timestamp('2024-01-01')

# 只看有完整数据的ETF（2022年就开始的）
etf_counts = etf_df.groupby('symbol')['trade_date'].count()
full_etfs = etf_counts[etf_counts > 500].index.tolist()
print(f"有完整数据的ETF（>500条）: {len(full_etfs)}只")
print(f"  {full_etfs}")

# 方法1：买入不动（只用有完整数据的ETF）
print("\n" + "=" * 70)
print("方法1：买入不动（2024年初买入，持有至今）")
print("=" * 70)

etf_full = etf_df[etf_df['symbol'].isin(full_etfs) & (etf_df['symbol'] != BENCHMARK)]

returns = []
for symbol in sorted(etf_full['symbol'].unique()):
    sym_data = etf_full[etf_full['symbol'] == symbol].sort_values('trade_date')
    test_data = sym_data[sym_data['trade_date'] >= test_start]
    if len(test_data) > 0:
        ret = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100
        returns.append(ret)
        print(f"  {symbol}: {ret:.2f}%")

print(f"\n  平均收益: {np.mean(returns):.2f}%")

# 方法2：等权持有（每周调仓）
print("\n" + "=" * 70)
print("方法2：等权持有（每周调仓）")
print("=" * 70)

etf_full_copy = etf_full.copy()
etf_full_copy['year_week'] = etf_full_copy['trade_date'].dt.isocalendar().year * 100 + etf_full_copy['trade_date'].dt.isocalendar().week

weekly_returns = []
for yw in sorted(etf_full_copy['year_week'].unique()):
    week_df = etf_full_copy[etf_full_copy['year_week'] == yw]
    rets = []
    for sym in week_df['symbol'].unique():
        sd = week_df[week_df['symbol'] == sym].sort_values('trade_date')
        if len(sd) >= 2:
            rets.append(sd['close'].iloc[-1] / sd['close'].iloc[0] - 1)
    if rets:
        weekly_returns.append(np.mean(rets))

# 累积收益
cumulative = np.cumprod([1 + r for r in weekly_returns])
total_return = (cumulative[-1] - 1) * 100
print(f"  周数: {len(weekly_returns)}")
print(f"  累积收益: {total_return:.2f}%")

# 方法3：直接用首尾价格计算
print("\n" + "=" * 70)
print("方法3：直接用首尾价格（验证）")
print("=" * 70)

first_prices = {}
last_prices = {}
for symbol in sorted(etf_full['symbol'].unique()):
    sym_data = etf_full[etf_full['symbol'] == symbol].sort_values('trade_date')
    test_data = sym_data[sym_data['trade_date'] >= test_start]
    if len(test_data) > 0:
        first_prices[symbol] = test_data['close'].iloc[0]
        last_prices[symbol] = test_data['close'].iloc[-1]

# 等权：每只ETF投同样的钱
initial_investment = 10000  # 每只投1万
total_initial = initial_investment * len(first_prices)
total_final = 0

for symbol in first_prices:
    shares = initial_investment / first_prices[symbol]
    value = shares * last_prices[symbol]
    total_final += value
    ret = (last_prices[symbol] / first_prices[symbol] - 1) * 100

total_ret = (total_final / total_initial - 1) * 100
print(f"  总投入: {total_initial:.0f}")
print(f"  总市值: {total_final:.0f}")
print(f"  总收益: {total_ret:.2f}%")
