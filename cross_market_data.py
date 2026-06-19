"""
QuanTrade 跨市场数据接口
支持：VIX恐慌指数、黄金ETF、港股ETF、US10Y国债收益率
用于三因子中的地缘政治和风险情绪因子

网络环境限制时，脚本可能无法直接获取数据，但可作为接口框架运行。
"""
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
import warnings
warnings.filterwarnings('ignore')

DB_PATH = 'QuanTrade/quant_system/data/quant.db'

# ========== VIX 恐慌指数 ==========
VIX_SYMBOL = '^VIX'
VIX_TABLE = 'vix_daily'

# ========== 黄金ETF ==========
GOLD_ETF_CN = '518880'  # 华安黄金ETF
GOLD_TABLE = 'gold_daily_prices'

# ========== 港股ETF ==========
HK_ETFS = {
    '513180': '恒生科技ETF',  # 跨市场：港股科技
    '513130': '港股通互联网',  # 跨市场：港股互联网
}


def ensure_table(conn, table_name, schema_sql):
    """确保表存在"""
    cursor = conn.cursor()
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema_sql})")
    conn.commit()


def fetch_vix_from_yahoo(start='2022-01-01', end='2026-06-30'):
    """从Yahoo Finance获取VIX数据"""
    try:
        # Yahoo Finance API (public)
        period1 = int(pd.Timestamp(start).timestamp())
        period2 = int(pd.Timestamp(end).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?period1={period1}&period2={period2}&interval=1d"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"[VIX] Yahoo API status: {resp.status_code}")
            return None
        data = resp.json()
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        closes = result['indicators']['quote'][0]['close']
        
        df = pd.DataFrame({
            'trade_date': pd.to_datetime(timestamps, unit='s'),
            'close': closes
        }).dropna()
        df['symbol'] = 'VIX'
        return df[['trade_date', 'symbol', 'close']]
    except Exception as e:
        print(f"[VIX] Yahoo fetch failed: {e}")
        return None


def fetch_gold_from_akshare(start='20220101', end='20260630'):
    """从akshare获取黄金ETF数据"""
    try:
        import akshare as ak
        df = ak.fund_etf_hist_em(symbol=GOLD_ETF_CN, period="daily", start_date=start, end_date=end, adjust="qfq")
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                '日期': 'trade_date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '涨跌幅': 'pct_change'
            })
            df['symbol'] = GOLD_ETF_CN
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        print(f"[Gold] akshare fetch failed: {e}")
    return None


def fetch_hk_etf_from_akshare(symbol, start='20220101', end='20260630'):
    """从akshare获取港股ETF数据"""
    try:
        import akshare as ak
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                '日期': 'trade_date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '涨跌幅': 'pct_change'
            })
            df['symbol'] = symbol
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        print(f"[HK ETF {symbol}] akshare fetch failed: {e}")
    return None


def save_vix_to_db(df, conn):
    """保存VIX数据到数据库"""
    if df is None or len(df) == 0:
        return 0
    ensure_table(conn, VIX_TABLE, 
        "trade_date TIMESTAMP, symbol TEXT, close REAL")
    df.to_sql(VIX_TABLE, conn, if_exists='append', index=False)
    return len(df)


def save_gold_to_db(df, conn):
    """保存黄金数据到数据库"""
    if df is None or len(df) == 0:
        return 0
    ensure_table(conn, GOLD_TABLE,
        "trade_date TIMESTAMP, symbol TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, amount REAL, pct_change REAL")
    df.to_sql(GOLD_TABLE, conn, if_exists='append', index=False)
    return len(df)


def save_etf_to_db(df, conn, table='etf_daily_prices'):
    """保存ETF数据到数据库"""
    if df is None or len(df) == 0:
        return 0
    df.to_sql(table, conn, if_exists='append', index=False)
    return len(df)


def check_data_coverage(conn):
    """检查所有数据覆盖情况"""
    print("\n" + "=" * 60)
    print("Data Coverage Report")
    print("=" * 60)
    
    # ETF data
    etf = pd.read_sql('''
        SELECT symbol, COUNT(*) as cnt, MIN(trade_date) as start, MAX(trade_date) as end 
        FROM etf_daily_prices GROUP BY symbol
    ''', conn)
    print(f"\n[ETF Daily Prices] {len(etf)} symbols")
    for _, row in etf.iterrows():
        status = "FULL" if row['cnt'] > 500 else "PARTIAL"
        start_str = str(row['start'])[:10]
        end_str = str(row['end'])[:10]
        print(f"  {row['symbol']}: {row['cnt']} records ({start_str} ~ {end_str}) [{status}]")
    
    # Gold data
    try:
        gold = pd.read_sql(f'SELECT COUNT(*) as cnt, MIN(trade_date) as start, MAX(trade_date) as end FROM {GOLD_TABLE}', conn)
        g_start = str(gold['start'].iloc[0])[:10]
        g_end = str(gold['end'].iloc[0])[:10]
        print(f"\n[Gold] {gold['cnt'].iloc[0]} records ({g_start} ~ {g_end})")
    except:
        print(f"\n[Gold] Table {GOLD_TABLE} not found")
    
    # VIX data
    try:
        vix = pd.read_sql(f'SELECT COUNT(*) as cnt, MIN(trade_date) as start, MAX(trade_date) as end FROM {VIX_TABLE}', conn)
        v_start = str(vix['start'].iloc[0])[:10]
        v_end = str(vix['end'].iloc[0])[:10]
        print(f"[VIX] {vix['cnt'].iloc[0]} records ({v_start} ~ {v_end})")
    except:
        print(f"[VIX] Table {VIX_TABLE} not found")


def main():
    print("=" * 60)
    print("QuanTrade Cross-Market Data Manager")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. 检查当前数据覆盖
    check_data_coverage(conn)
    
    # 2. 尝试获取VIX数据
    print("\n[VIX] Fetching from Yahoo Finance...")
    vix_df = fetch_vix_from_yahoo()
    if vix_df is not None:
        n = save_vix_to_db(vix_df, conn)
        print(f"  Saved {n} VIX records")
    else:
        print("  Failed (network restriction or API limit)")
    
    # 3. 尝试获取黄金数据
    print("\n[Gold] Fetching from akshare...")
    gold_df = fetch_gold_from_akshare()
    if gold_df is not None:
        n = save_gold_to_db(gold_df, conn)
        print(f"  Saved {n} gold records")
    else:
        print("  Failed (network restriction)")
    
    # 4. 尝试获取港股ETF数据
    for symbol, name in HK_ETFS.items():
        print(f"\n[HK ETF {symbol} - {name}] Fetching from akshare...")
        hk_df = fetch_hk_etf_from_akshare(symbol)
        if hk_df is not None:
            n = save_etf_to_db(hk_df, conn)
            print(f"  Saved {n} records")
        else:
            print("  Failed (network restriction)")
        time.sleep(1)
    
    # 5. 再次检查
    check_data_coverage(conn)
    
    conn.close()
    print("\n" + "=" * 60)
    print("Done. If some data failed due to network restrictions,")
    print("run this script again when network is available,")
    print("or manually download data from:")
    print("  - VIX: https://www.cboe.com/tradable_products/vix/")
    print("  - Gold ETF: akshare or Tushare")
    print("  - HK ETF: akshare or Tushare")
    print("=" * 60)


if __name__ == '__main__':
    main()
