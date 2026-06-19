"""保存黄金ETF完整数据"""

import akshare as ak
import sqlite3
import pandas as pd

print("=== 获取黄金ETF完整数据 ===\n")

# 获取基金净值数据
print("正在下载数据...")
df = ak.fund_etf_fund_info_em(fund="518880", start_date="20140101", end_date="20260618")
print(f"记录数: {len(df)}")
print(f"列名: {df.columns.tolist()}")
print(f"时间范围: {df['净值日期'].min()} ~ {df['净值日期'].max()}")

# 转换格式
df_save = pd.DataFrame({
    "date": pd.to_datetime(df["净值日期"]),
    "code": "518880.SH",
    "open": df["单位净值"],
    "high": df["单位净值"],
    "low": df["单位净值"],
    "close": df["单位净值"],
    "volume": 0,
    "amount": 0
})

print(f"\n转换后数据:")
print(df_save.head())

# 保存到数据库
print("\n保存到数据库...")
conn = sqlite3.connect("QuanTrade/quant_system/data/quant.db")
df_save.to_sql("gold_daily_prices", conn, if_exists="replace", index=False)

# 验证
count = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM gold_daily_prices").fetchone()
print(f"数据库验证: {count[0]}条, {count[1]} ~ {count[2]}")

conn.close()
print("\n✅ 黄金ETF数据保存成功!")
