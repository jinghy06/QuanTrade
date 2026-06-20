import akshare as ak
import sqlite3
import pandas as pd
from datetime import datetime
import time
import sys

DB_PATH = 'QuanTrade/quant_system/data/quant.db'

# 仅有2026年数据的ETF（需要补充历史）
MISSING_ETFS = [
    '159790', '159915', '159928', '512010', '512200', '512480',
    '512690', '512800', '512880', '513180', '515170', '515790',
    '516160', '561160', '588000'
]

def sina_symbol(symbol):
    """转换为新浪财经格式: shxxx 或 szxxx"""
    if symbol.startswith('15') or symbol.startswith('16') or symbol.startswith('588'):
        return f'sz{symbol}'
    return f'sh{symbol}'

def get_etf_hist_sina(symbol):
    """用新浪财经接口获取ETF历史数据"""
    sina_sym = sina_symbol(symbol)
    try:
        df = ak.fund_etf_hist_sina(symbol=sina_sym)
        if df is not None and len(df) > 0:
            # 新浪接口返回: date, open, high, low, close, volume
            df = df.rename(columns={
                'date': 'trade_date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            df['symbol'] = symbol
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['amount'] = 0
            df['pct_change'] = df['close'].pct_change() * 100
            return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        print(f"  sina failed for {symbol}: {e}")
    return None

def save_to_db(df, conn):
    """保存到etf_daily_prices表，如果获取数据更早则替换"""
    if df is None or len(df) == 0:
        return 0
    symbol = df['symbol'].iloc[0]
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    cursor = conn.cursor()
    cursor.execute('SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM etf_daily_prices WHERE symbol = ?', (symbol,))
    row = cursor.fetchone()
    db_min, db_max, db_cnt = row[0], row[1], row[2]
    
    data_min = df['trade_date'].min()
    data_max = df['trade_date'].max()
    
    if db_cnt > 0:
        # 如果获取数据比数据库更早或更晚，删除旧数据重新插入
        if data_min < pd.Timestamp(db_min) or data_max > pd.Timestamp(db_max):
            print(f"    Replacing {db_cnt} old records (DB: {db_min}~{db_max}) with {len(df)} new records ({data_min.date()}~{data_max.date()})")
            cursor.execute('DELETE FROM etf_daily_prices WHERE symbol = ?', (symbol,))
            conn.commit()
        else:
            print(f"    DB already has full range {db_min}~{db_max}, skipping")
            return 0
    
    df.to_sql('etf_daily_prices', conn, if_exists='append', index=False)
    return len(df)

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 60)
    print("ETF History Data Supplement - Sina API")
    print("=" * 60)
    conn = sqlite3.connect(DB_PATH)
    total_new = 0
    for symbol in MISSING_ETFS:
        print(f"\n[{symbol}] Fetching...")
        df = get_etf_hist_sina(symbol)
        if df is not None and len(df) > 0:
            n = save_to_db(df, conn)
            total_new += n
            print(f"  Added {n} new, total available {len(df)}")
        else:
            print(f"  [FAIL]")
        time.sleep(0.5)
    conn.close()
    print(f"\n{'=' * 60}")
    print(f"Total new records: {total_new}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
