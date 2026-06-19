"""调试：找出等权持有计算的问题"""
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

# 只看有完整数据的7只ETF
full_etfs = ['159995', '512660', '512670', '515070', '515960', '516510', '562500']
etf_full = etf_df[etf_df['symbol'].isin(full_etfs)]

# 检查每周的数据
etf_full_copy = etf_full.copy()
etf_full_copy['year_week'] = etf_full_copy['trade_date'].dt.isocalendar().year * 100 + etf_full_copy['trade_date'].dt.isocalendar().week

print("检查前几周的数据:")
print("=" * 70)

count = 0
for yw in sorted(etf_full_copy['year_week'].unique()):
    if count >= 5:
        break

    week_df = etf_full_copy[etf_full_copy['year_week'] == yw]

    # 只看测试集
    if week_df['trade_date'].min() < test_start:
        continue

    print(f"\n第{yw}周:")
    print(f"  日期范围: {week_df['trade_date'].min().date()} ~ {week_df['trade_date'].max().date()}")
    print(f"  ETF数量: {week_df['symbol'].nunique()}")

    for sym in sorted(week_df['symbol'].unique()):
        sd = week_df[week_df['symbol'] == sym].sort_values('trade_date')
        if len(sd) >= 2:
            ret = sd['close'].iloc[-1] / sd['close'].iloc[0] - 1
            print(f"    {sym}: {sd['close'].iloc[0]:.3f} -> {sd['close'].iloc[-1]:.3f} ({ret*100:+.2f}%)")

    count += 1
