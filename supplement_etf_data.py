import akshare as ak
import sqlite3
import pandas as pd
from datetime import datetime
import time

DB_PATH = 'QuanTrade/quant_system/data/quant.db'

# 仅有2026年数据的ETF（需要补充2022-2025年历史）
MISSING_ETFS = [
    '159790', '159915', '159928', '512010', '512200', '512480',
    '512690', '512800', '512880', '513180', '515170', '515790',
    '516160', '561160', '588000'
]

# 已有完整数据的ETF（用于对比验证）
COMPLETE_ETFS = ['159995', '510300', '512660', '512670', '515070', '515960', '516510', '562500']

def get_etf_hist(symbol, start='20220101', end='20251231'):
    """用akshare获取ETF历史数据"""
    try:
        # 尝试获取基金行情
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
        print(f"  akshare fund_etf_hist_em failed for {symbol}: {e}")
    
    # 备选：尝试stock_zh_index_daily_em
    try:
        # ETF代码格式
        if symbol.startswith('15') or symbol.startswith('16') or symbol.startswith('56'):
            code = f"{symbol}.SZ"
        else:
            code = f"{symbol}.SH"
        df = ak.stock_zh_index_daily_em(symbol=code)
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                'date': 'trade_date',
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'volume': 'volume'
            })
            df['symbol'] = symbol
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['amount'] = 0
            df['pct_change'] = df['close'].pct_change() * 100
            return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        print(f"  akshare stock_zh_index_daily_em failed for {symbol}: {e}")
    
    return None

def save_to_db(df, conn):
    """保存到etf_daily_prices表"""
    if df is None or len(df) == 0:
        return 0
    
    # 检查已有数据
    symbol = df['symbol'].iloc[0]
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM etf_daily_prices WHERE symbol = ?', (symbol,))
    existing = cursor.fetchone()[0]
    
    # 只插入新数据
    if existing > 0:
        cursor.execute('SELECT MAX(trade_date) FROM etf_daily_prices WHERE symbol = ?', (symbol,))
        max_date = cursor.fetchone()[0]
        df = df[df['trade_date'] > max_date]
    
    if len(df) == 0:
        return 0
    
    df.to_sql('etf_daily_prices', conn, if_exists='append', index=False)
    return len(df)

def main():
    print("=" * 60)
    print("ETF History Data Supplement Script")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    
    total_new = 0
    for symbol in MISSING_ETFS:
        print(f"\n[{symbol}] Fetching history...")
        df = get_etf_hist(symbol)
        if df is not None and len(df) > 0:
            n = save_to_db(df, conn)
            total_new += n
            print(f"  Added {n} records, total {len(df)} records")
        else:
            print(f"  [FAIL] Fetch failed")
        time.sleep(1)  # 避免请求过快
    
    conn.close()
    
    print(f"\n{'=' * 60}")
    print(f"Total new records: {total_new}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
