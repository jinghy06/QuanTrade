#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用baostock获取2005年至今的A股历史数据
"""

import baostock as bs
import pandas as pd
import sqlite3
import time
from datetime import datetime

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

# 2005年前已上市的大盘股
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

def fetch_with_baostock(code, name):
    """使用baostock获取历史数据"""
    # baostock格式: sh.600036 或 sz.000002
    bs_code = code.replace('.SH', '.sh').replace('.SZ', '.sz').lower()
    
    print(f"  获取 {code} {name}...", end=" ")
    
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
            start_date=START_DATE,
            end_date=END_DATE,
            frequency="d",
            adjustflag="2"  # 前复权
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            print(f"无数据 (error: {rs.error_msg})")
            return None
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 转换数据类型
        df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 标准化列名
        df = df.rename(columns={
            'date': 'trade_date',
            'code': 'symbol_raw',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'preclose': 'pre_close',
            'volume': 'volume',
            'amount': 'amount',
            'turn': 'turnover',
            'pctChg': 'pct_change'
        })
        
        df['symbol'] = code
        
        # 只保留需要的列
        cols = ['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change', 'turnover']
        df = df[[c for c in cols if c in df.columns]]
        
        print(f"{len(df)}条 ({df['trade_date'].min().date()}~{df['trade_date'].max().date()})")
        return df
        
    except Exception as e:
        print(f"失败: {e}")
        return None

def main():
    print("=" * 80)
    print(f"使用baostock获取历史数据 - {START_DATE} 至 {END_DATE}")
    print(f"股票池: {len(TRAIN_STOCKS_V5)}只")
    print("=" * 80)
    
    # 登录baostock
    print("\n登录baostock...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return
    print(f"登录成功: {lg.error_msg}")
    
    all_data = []
    success_count = 0
    
    for i, (code, name) in enumerate(TRAIN_STOCKS_V5, 1):
        print(f"\n[{i}/{len(TRAIN_STOCKS_V5)}]", end="")
        df = fetch_with_baostock(code, name)
        if df is not None and len(df) > 100:
            all_data.append(df)
            success_count += 1
        time.sleep(0.3)
    
    # 登出
    bs.logout()
    
    print(f"\n{'='*80}")
    print(f"数据获取完成: {success_count}/{len(TRAIN_STOCKS_V5)} 只成功")
    
    if not all_data:
        print("无数据可保存")
        return
    
    # 合并保存
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n合并数据: {len(combined)}条")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 创建新表
    conn.execute("DROP TABLE IF EXISTS daily_prices_v5")
    conn.execute("""
        CREATE TABLE daily_prices_v5 (
            trade_date TIMESTAMP,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            pct_change REAL,
            turnover REAL
        )
    """)
    
    combined.to_sql('daily_prices_v5', conn, if_exists='append', index=False)
    
    # 创建索引
    conn.execute("CREATE INDEX idx_dp5_date ON daily_prices_v5(trade_date)")
    conn.execute("CREATE INDEX idx_dp5_symbol ON daily_prices_v5(symbol)")
    conn.execute("CREATE INDEX idx_dp5_sym_date ON daily_prices_v5(symbol, trade_date)")
    
    conn.commit()
    conn.close()
    
    print(f"已保存到 daily_prices_v5 表")
    
    # 统计
    print(f"\n数据统计:")
    print(f"  总记录数: {len(combined)}")
    print(f"  时间范围: {combined['trade_date'].min().date()} ~ {combined['trade_date'].max().date()}")
    print(f"  股票数量: {combined['symbol'].nunique()}")
    print(f"  平均每只: {len(combined) // combined['symbol'].nunique()}条")
    
    for sym in sorted(combined['symbol'].unique()):
        cnt = len(combined[combined['symbol'] == sym])
        print(f"    {sym}: {cnt}条")

if __name__ == '__main__':
    main()
