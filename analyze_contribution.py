"""分析买入不动收益贡献"""
import sqlite3
import pandas as pd

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

conn = sqlite3.connect(DB_PATH)
etf = pd.read_sql_query("SELECT * FROM etf_daily_prices", conn)
etf['trade_date'] = pd.to_datetime(etf['trade_date'])
conn.close()

test_start = pd.Timestamp('2024-01-01')
full_etfs = ['159995', '512660', '512670', '515070', '515960', '516510', '562500']
etf = etf[etf['symbol'].isin(full_etfs)]

names = {
    '159995': '芯片ETF',
    '512660': '军工ETF',
    '512670': '国防ETF',
    '515070': '人工智能ETF',
    '515960': '航天军工ETF',
    '516510': '云计算ETF',
    '562500': '机器人ETF',
}

print("=" * 70)
print("买入不动收益贡献分析（2024年初至今）")
print("=" * 70)
print()
print(f"{'ETF':<15s} {'收益率':>10s}")
print("-" * 30)

results = []
for sym in full_etfs:
    s = etf[etf['symbol'] == sym].sort_values('trade_date')
    t = s[s['trade_date'] >= test_start]
    if len(t) > 0:
        ret = (t['close'].iloc[-1] / t['close'].iloc[0] - 1) * 100
        results.append((sym, ret))

results.sort(key=lambda x: x[1], reverse=True)

for sym, ret in results:
    name = names.get(sym, sym)
    print(f"{name:<15s} {ret:>+9.2f}%")

print("-" * 30)
avg_ret = sum(r[1] for r in results) / len(results)
print(f"{'平均':<15s} {avg_ret:>+9.2f}%")

print()
print("=" * 70)
print("贡献最大的3只ETF:")
print("=" * 70)
for i, (sym, ret) in enumerate(results[:3]):
    name = names.get(sym, sym)
    print(f"  {i+1}. {name}: {ret:+.2f}%")

print()
print("贡献最小的3只ETF:")
for i, (sym, ret) in enumerate(results[-3:]):
    name = names.get(sym, sym)
    print(f"  {i+1}. {name}: {ret:+.2f}%")
