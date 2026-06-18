#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展训练集数据获取 - 2005年至今
覆盖22只2005年前已上市的A股大盘股
"""

import akshare as ak
import pandas as pd
import sqlite3
import time
from datetime import datetime

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

# 2005年前已上市的大盘股（覆盖主要行业）
TRAIN_STOCKS_V5 = [
    # 金融
    ('600036.SH', '招商银行', 2002),
    ('600030.SH', '中信证券', 2003),
    ('600050.SH', '中国联通', 2002),
    # 能源/原材料
    ('600028.SH', '中国石化', 2001),
    ('600019.SH', '宝钢股份', 2000),
    ('600188.SH', '兖矿能源', 1998),
    ('600011.SH', '华能国际', 2001),
    ('600900.SH', '长江电力', 2003),
    # 消费
    ('600519.SH', '贵州茅台', 2001),
    ('000858.SZ', '五粮液', 1998),
    ('000651.SZ', '格力电器', 1996),
    ('000333.SZ', '美的集团', 1993),
    ('000568.SZ', '泸州老窖', 1994),
    ('600600.SH', '青岛啤酒', 1993),
    # 地产/基建
    ('000002.SZ', '万科A', 1991),
    ('600031.SH', '三一重工', 2003),
    # 医药
    ('600276.SH', '恒瑞医药', 2000),
    # 科技/制造
    ('000725.SZ', '京东方A', 2001),
    ('000063.SZ', '中兴通讯', 1997),
    ('000625.SZ', '长安汽车', 1997),
    ('600104.SH', '上汽集团', 1997),
    ('600029.SH', '南方航空', 2003),
]

START_DATE = "20050101"
END_DATE = "20251231"

def fetch_stock_data(code, name, list_year):
    """获取单只股票历史数据"""
    pure_code = code.replace('.SH', '').replace('.SZ', '')
    
    try:
        print(f"  获取 {code} {name} (上市{list_year}年)...", end=" ")
        
        # 使用akshare获取历史数据
        df = ak.stock_zh_a_hist(
            symbol=pure_code,
            period="daily",
            start_date=START_DATE,
            end_date=END_DATE,
            adjust="qfq"  # 前复权
        )
        
        if df is None or df.empty:
            print(f"无数据")
            return None
        
        # 标准化列名
        df = df.rename(columns={
            '日期': 'trade_date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        })
        
        df['symbol'] = code
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 只保留需要的列
        cols = ['trade_date', 'symbol', 'open', 'close', 'high', 'low', 'volume', 'amount', 'pct_change', 'turnover']
        df = df[[c for c in cols if c in df.columns]]
        
        print(f"{len(df)}条 (从{df['trade_date'].min().date()}到{df['trade_date'].max().date()})")
        return df
        
    except Exception as e:
        print(f"失败: {e}")
        return None

def save_to_db(all_data):
    """保存到数据库"""
    if not all_data:
        print("无数据可保存")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n合并数据: {len(combined)}条")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 创建新表或清空旧表
    conn.execute("DROP TABLE IF EXISTS daily_prices_v5")
    conn.execute("""
        CREATE TABLE daily_prices_v5 (
            trade_date TIMESTAMP,
            symbol TEXT,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
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

def main():
    print("=" * 80)
    print(f"扩展训练集数据获取 - {START_DATE} 至 {END_DATE}")
    print(f"股票池: {len(TRAIN_STOCKS_V5)}只2005年前上市大盘股")
    print("=" * 80)
    
    all_data = []
    success_count = 0
    
    for i, (code, name, list_year) in enumerate(TRAIN_STOCKS_V5, 1):
        print(f"\n[{i}/{len(TRAIN_STOCKS_V5)}]", end="")
        df = fetch_stock_data(code, name, list_year)
        if df is not None and len(df) > 100:
            all_data.append(df)
            success_count += 1
        time.sleep(0.5)  # 避免请求过快
    
    print(f"\n{'='*80}")
    print(f"数据获取完成: {success_count}/{len(TRAIN_STOCKS_V5)} 只成功")
    
    if all_data:
        save_to_db(all_data)
        
        # 统计
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n数据统计:")
        print(f"  总记录数: {len(combined)}")
        print(f"  时间范围: {combined['trade_date'].min().date()} ~ {combined['trade_date'].max().date()}")
        print(f"  股票数量: {combined['symbol'].nunique()}")
        
        for sym in sorted(combined['symbol'].unique()):
            cnt = len(combined[combined['symbol'] == sym])
            print(f"    {sym}: {cnt}条")

if __name__ == '__main__':
    main()
