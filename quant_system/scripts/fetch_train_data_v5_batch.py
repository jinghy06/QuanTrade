#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用baostock分批获取2005年至今的A股历史数据
每批5只，避免超时
"""

import baostock as bs
import pandas as pd
import sqlite3
import time

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

TRAIN_STOCKS_V5 = [
    ('600036.SH', '招商银行'), ('600030.SH', '中信证券'), ('600050.SH', '中国联通'),
    ('600028.SH', '中国石化'), ('600019.SH', '宝钢股份'), ('600188.SH', '兖矿能源'),
    ('600011.SH', '华能国际'), ('600900.SH', '长江电力'),
    ('600519.SH', '贵州茅台'), ('000858.SZ', '五粮液'), ('000651.SZ', '格力电器'),
    ('000333.SZ', '美的集团'), ('000568.SZ', '泸州老窖'), ('600600.SH', '青岛啤酒'),
    ('000002.SZ', '万科A'), ('600031.SH', '三一重工'),
    ('600276.SH', '恒瑞医药'),
    ('000725.SZ', '京东方A'), ('000063.SZ', '中兴通讯'), ('000625.SZ', '长安汽车'),
    ('600104.SH', '上汽集团'), ('600029.SH', '南方航空'),
]

START_DATE = "2005-01-01"
END_DATE = "2025-12-31"
BATCH_SIZE = 5

def fetch_one(code, name):
    bs_code = code.replace('.SH', '.sh').replace('.SZ', '.sz').lower()
    print(f"  {code} {name}...", end=" ", flush=True)
    
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,code,open,high,low,close,volume,amount,turn,pctChg",
        start_date=START_DATE, end_date=END_DATE,
        frequency="d", adjustflag="2"
    )
    
    data = []
    while (rs.error_code == '0') & rs.next():
        data.append(rs.get_row_data())
    
    if not data:
        print(f"无数据({rs.error_msg})")
        return None
    
    df = pd.DataFrame(data, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    for col in ['open','high','low','close','volume','amount','turn','pctChg']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.rename(columns={
        'date': 'trade_date', 'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'volume': 'volume', 'amount': 'amount',
        'turn': 'turnover', 'pctChg': 'pct_change'
    })
    df['symbol'] = code
    
    cols = ['trade_date','symbol','open','high','low','close','volume','amount','pct_change','turnover']
    df = df[[c for c in cols if c in df.columns]]
    
    print(f"{len(df)}条")
    return df

def save_batch(df_list, conn, first_batch):
    if not df_list:
        return
    combined = pd.concat(df_list, ignore_index=True)
    
    if first_batch:
        conn.execute("DROP TABLE IF EXISTS daily_prices_v5")
        conn.execute("""
            CREATE TABLE daily_prices_v5 (
                trade_date TIMESTAMP, symbol TEXT,
                open REAL, high REAL, low REAL, close REAL,
                volume REAL, amount REAL, pct_change REAL, turnover REAL
            )
        """)
    
    combined.to_sql('daily_prices_v5', conn, if_exists='append', index=False)
    print(f"  -> 已保存 {len(combined)}条")

def main():
    print("=" * 70)
    print(f"baostock分批获取 {START_DATE}~{END_DATE} | {len(TRAIN_STOCKS_V5)}只股票")
    print("=" * 70)
    
    lg = bs.login()
    print(f"登录: {lg.error_msg}\n")
    
    conn = sqlite3.connect(DB_PATH)
    all_success = []
    first_batch = True
    
    for batch_start in range(0, len(TRAIN_STOCKS_V5), BATCH_SIZE):
        batch = TRAIN_STOCKS_V5[batch_start:batch_start+BATCH_SIZE]
        print(f"\n--- 批次 {batch_start//BATCH_SIZE + 1}/{(len(TRAIN_STOCKS_V5)-1)//BATCH_SIZE + 1} ---")
        
        batch_data = []
        for code, name in batch:
            df = fetch_one(code, name)
            if df is not None and len(df) > 100:
                batch_data.append(df)
                all_success.append((code, len(df)))
            time.sleep(0.3)
        
        save_batch(batch_data, conn, first_batch)
        first_batch = False
    
    # 创建索引
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dp5_date ON daily_prices_v5(trade_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dp5_symbol ON daily_prices_v5(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dp5_sym_date ON daily_prices_v5(symbol, trade_date)")
    conn.commit()
    conn.close()
    
    bs.logout()
    
    print(f"\n{'='*70}")
    print(f"完成: {len(all_success)}/{len(TRAIN_STOCKS_V5)} 只成功")
    for code, cnt in all_success:
        print(f"  {code}: {cnt}条")

if __name__ == '__main__':
    main()
