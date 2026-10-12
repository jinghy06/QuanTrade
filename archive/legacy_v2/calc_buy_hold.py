"""计算买入不动策略的收益率"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

conn = sqlite3.connect(DB_PATH)
etf_df = pd.read_sql_query("SELECT * FROM etf_daily_prices", conn)
etf_df['trade_date'] = pd.to_datetime(etf_df['trade_date'])
conn.close()

# 只看ETF（不含基准）
etf_only = etf_df[etf_df['symbol'] != '510300']
test_start = pd.Timestamp('2024-01-01')

print("=" * 80)
print("买入不动策略：2024年初买入，持有至今")
print("=" * 80)

print(f"\n{'ETF':<12s} {'买入价':>8s} {'当前价':>8s} {'收益率':>8s}")
print("-" * 50)

returns = []
for symbol in sorted(etf_only['symbol'].unique()):
    sym_data = etf_only[etf_only['symbol'] == symbol].sort_values('trade_date')
    test_data = sym_data[sym_data['trade_date'] >= test_start]

    if len(test_data) > 0:
        buy_price = test_data['close'].iloc[0]
        current_price = test_data['close'].iloc[-1]
        ret = (current_price / buy_price - 1) * 100
        returns.append(ret)
        print(f"{symbol:<12s} {buy_price:>8.3f} {current_price:>8.3f} {ret:>7.2f}%")

print("-" * 50)
avg_ret = np.mean(returns)
print(f"{'平均（等权）':<12s} {'':>8s} {'':>8s} {avg_ret:>7.2f}%")

print(f"\n结论:")
print(f"  如果在2024年初买入所有ETF不动，到现在平均亏损 {abs(avg_ret):.2f}%")
print()
print("对比策略C（轮动+择时）: +0.93%")
print(f"  策略C比买入不动多赚了 {avg_ret * (-1) + 0.93:.2f}%")
