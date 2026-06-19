#!/usr/bin/env python3
"""
补充ETF数据 + 跨市场数据脚本
==============================
目标：
1. 为现有24只ETF补充更长的历史数据（特别是2022-2025年）
2. 补充用户实际交易的ETF：159382, 588790, 159605
3. 获取跨市场数据：黄金(518880)、VIX代理、港股ETF、美股ETF
"""

import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

DB_PATH = 'QuanTrade/quant_system/data/quant.db'

# ============================================================
# ETF池
# ============================================================
ETF_POOL = [
    # 完整数据8只
    '562500', '516510', '515960', '515070', '512670', '512660', '510300', '159995',
    # 约半年数据1只
    '159550',
    # 仅2026年数据15只
    '588000', '561160', '516160', '515790', '515170', '513180',
    '512880', '512800', '512690', '512480', '512200', '512010',
    '159928', '159915', '159790',
    # 用户实际交易（新增）
    '159382', '588790', '159605',
]

# 跨市场数据
CROSS_MARKET = {
    '518880': '黄金ETF',           # 已有
    '513050': '中概互联网ETF',    # 港股/美股
    '513100': '纳斯达克ETF',      # 美股
    '159941': '纳指100ETF',       # 美股
    '159920': '恒生ETF',          # 港股
}

# ============================================================
# 数据获取
# ============================================================
def get_etf_data_akshare(symbol, start_date='20220101', end_date='20260630'):
    """使用akshare获取ETF历史数据"""
    try:
        import akshare as ak
        # 判断交易所
        if symbol.startswith('5'):
            # 上海ETF
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        elif symbol.startswith('1'):
            # 深圳ETF
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        else:
            # 其他
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        
        if df is None or len(df) == 0:
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
            '换手率': 'turnover',
        })
        df['symbol'] = symbol
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        
        # 确保数值列正确
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'pct_change', 'change', 'turnover']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        print(f"  [错误] {symbol}: {e}")
        return None


def get_etf_data_tushare(symbol, start_date='20220101', end_date='20260630'):
    """使用tushare获取ETF历史数据（备用）"""
    try:
        import tushare as ts
        # 需要用户设置token
        pro = ts.pro_api()
        
        # tushare ETF代码格式
        if symbol.startswith('5'):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        if df is None or len(df) == 0:
            return None
        
        df = df.rename(columns={
            'trade_date': 'trade_date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
        })
        df['symbol'] = symbol
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        df['pct_change'] = df['close'].pct_change() * 100
        
        return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        print(f"  [Tushare错误] {symbol}: {e}")
        return None


# ============================================================
# 数据写入
# ============================================================
def save_to_db(df, db_path=DB_PATH):
    """将DataFrame写入数据库，自动建表/更新"""
    if df is None or len(df) == 0:
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 建表（如果不存在）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS etf_daily_prices (
            trade_date TEXT,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            pct_change REAL,
            PRIMARY KEY (trade_date, symbol)
        )
    ''')
    
    # 使用INSERT OR REPLACE避免重复
    inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO etf_daily_prices 
                (trade_date, symbol, open, high, low, close, volume, amount, pct_change)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['trade_date'], row['symbol'],
                row.get('open'), row.get('high'), row.get('low'), row.get('close'),
                row.get('volume'), row.get('amount'), row.get('pct_change')
            ))
            inserted += 1
        except Exception as e:
            print(f"  [写入错误] {row['symbol']} {row['trade_date']}: {e}")
    
    conn.commit()
    conn.close()
    return inserted


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("QuanTrade 数据补充脚本")
    print("=" * 60)
    
    # 检查akshare可用性
    try:
        import akshare as ak
        print(f"[OK] akshare 版本: {ak.__version__}")
    except ImportError:
        print("[错误] akshare 未安装，请先安装: pip install akshare")
        return
    
    # 1. 补充现有24只ETF数据
    print("\n[1/3] 补充现有24只ETF数据...")
    for symbol in ETF_POOL[:24]:
        print(f"  获取 {symbol}...", end=' ')
        df = get_etf_data_akshare(symbol)
        if df is not None and len(df) > 0:
            n = save_to_db(df)
            print(f"OK ({n} 条, {df['trade_date'].min()} ~ {df['trade_date'].max()})")
        else:
            print(f"失败，尝试Tushare...", end=' ')
            df = get_etf_data_tushare(symbol)
            if df is not None and len(df) > 0:
                n = save_to_db(df)
                print(f"OK ({n} 条)")
            else:
                print("也失败")
    
    # 2. 补充用户实际交易ETF
    print("\n[2/3] 补充用户实际交易ETF...")
    for symbol in ['159382', '588790', '159605']:
        print(f"  获取 {symbol}...", end=' ')
        df = get_etf_data_akshare(symbol)
        if df is not None and len(df) > 0:
            n = save_to_db(df)
            print(f"OK ({n} 条, {df['trade_date'].min()} ~ {df['trade_date'].max()})")
        else:
            print("失败")
    
    # 3. 补充跨市场数据
    print("\n[3/3] 补充跨市场数据...")
    for symbol, name in CROSS_MARKET.items():
        print(f"  获取 {symbol} ({name})...", end=' ')
        df = get_etf_data_akshare(symbol)
        if df is not None and len(df) > 0:
            # 写入gold_daily_prices或新表
            n = save_to_db(df)
            print(f"OK ({n} 条)")
        else:
            print("失败")
    
    # 4. 统计最终数据
    print("\n[统计] 最终数据覆盖：")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT symbol, COUNT(*) as cnt, MIN(trade_date) as min_d, MAX(trade_date) as max_d
        FROM etf_daily_prices
        GROUP BY symbol
        ORDER BY cnt DESC
    ''')
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} 条, {row[2]} ~ {row[3]}")
    conn.close()
    
    print("\n数据补充完成！")


if __name__ == '__main__':
    main()
