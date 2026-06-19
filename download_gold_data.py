"""
下载黄金ETF(518880)数据
用于QuanTrade 2.0的黄金对冲层
"""

import baostock as bs
import pandas as pd
import sqlite3
from datetime import datetime

DB_PATH = "QuanTrade/quant_system/data/quant.db"

def download_gold_etf():
    """下载黄金ETF数据并存入数据库"""
    
    print("[1/3] 登录Baostock...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return
    print("  登录成功")
    
    print("\n[2/3] 下载黄金ETF(518880)数据...")
    rs = bs.query_history_k_data_plus(
        "sh.518880",
        "date,code,open,high,low,close,volume,amount",
        start_date="2014-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        frequency="d",
        adjustflag="2"  # 前复权
    )
    
    if rs.error_code != '0':
        print(f"下载失败: {rs.error_msg}")
        bs.logout()
        return
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 转换数据类型
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['date'] = pd.to_datetime(df['date'])
    
    # 转换code格式: sh.518880 -> 518880.SH
    df['code'] = df['code'].apply(lambda x: x.split('.')[1] + '.' + x.split('.')[0].upper())
    
    print(f"  下载完成: {len(df)} 条记录")
    print(f"  时间范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  最新价格: {df['close'].iloc[-1]:.3f}")
    
    print("\n[3/3] 保存到数据库...")
    conn = sqlite3.connect(DB_PATH)
    
    # 创建表（如果不存在）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold_daily_prices (
            date TEXT NOT NULL,
            code TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            PRIMARY KEY (date, code)
        )
    """)
    
    # 保存数据
    df.to_sql('gold_daily_prices', conn, if_exists='replace', index=False)
    
    # 验证
    count = conn.execute("SELECT COUNT(*) FROM gold_daily_prices").fetchone()[0]
    print(f"  数据库记录数: {count}")
    
    conn.close()
    bs.logout()
    
    print("\n✅ 黄金ETF数据下载完成!")
    return df

if __name__ == "__main__":
    download_gold_etf()
