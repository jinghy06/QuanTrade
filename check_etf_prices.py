"""检查ETF价格走势，确认收益是否真实"""
import sqlite3
import pandas as pd

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

conn = sqlite3.connect(DB_PATH)
etf = pd.read_sql_query("SELECT * FROM etf_daily_prices WHERE symbol IN ('159995', '515070', '516510')", conn)
etf['trade_date'] = pd.to_datetime(etf['trade_date'])
conn.close()

names = {
    '159995': '芯片ETF',
    '515070': '人工智能ETF',
    '516510': '云计算ETF',
}

print("检查ETF价格走势，确认收益是否真实")
print("=" * 70)

for sym in ['159995', '515070', '516510']:
    s = etf[etf['symbol'] == sym].sort_values('trade_date')
    test_start = pd.Timestamp('2024-01-01')
    test_data = s[s['trade_date'] >= test_start]

    name = names[sym]
    print(f"\n{name} ({sym}):")
    print(f"  测试期第一天: {test_data['trade_date'].iloc[0].date()} 收盘价: {test_data['close'].iloc[0]:.3f}")
    print(f"  测试期最后一天: {test_data['trade_date'].iloc[-1].date()} 收盘价: {test_data['close'].iloc[-1]:.3f}")
    total_ret = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100
    print(f"  总收益率: {total_ret:+.2f}%")

    print(f"  分年收益:")
    for year in [2024, 2025, 2026]:
        year_data = test_data[test_data['trade_date'].dt.year == year]
        if len(year_data) > 0:
            year_ret = (year_data['close'].iloc[-1] / year_data['close'].iloc[0] - 1) * 100
            print(f"    {year}: {year_data['close'].iloc[0]:.3f} -> {year_data['close'].iloc[-1]:.3f} ({year_ret:+.2f}%)")
