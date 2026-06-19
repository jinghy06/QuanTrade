"""检查买入持有策略收益"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("QuanTrade/quant_system/data/quant.db")

etfs = ['510300', '515070', '159995', '512660', '512670', '562500', '516510', '515960']

print("=== ETF买入持有对比 ===\n")
print(f"{'ETF':8s} | {'总收益':>8s} | {'年化':>6s}")
print("-" * 30)

best_return = -999
best_etf = ""

for symbol in etfs:
    df = pd.read_sql(f"SELECT close FROM etf_daily_prices WHERE symbol='{symbol}' ORDER BY trade_date", conn)
    if len(df) < 100:
        continue
    
    total = df['close'].iloc[-1] / df['close'].iloc[0] - 1
    n_years = len(df) / 252
    annual = (1 + total) ** (1/n_years) - 1 if n_years > 0 else 0
    
    print(f"{symbol:8s} | {total:>8.2%} | {annual:>6.2%}")
    
    if total > best_return:
        best_return = total
        best_etf = symbol

print(f"\n最佳: {best_etf} ({best_return:.2%})")
bench_df = pd.read_sql("SELECT close FROM etf_daily_prices WHERE symbol='510300' ORDER BY trade_date", conn)
bench_return = bench_df.iloc[-1][0] / bench_df.iloc[0][0] - 1
print(f"沪深300基准: {bench_return:.2%}")

conn.close()
