"""
ETF数据扩展 - 用Baostock下载2018年至今的数据
"""
import sqlite3
import pandas as pd
import baostock as bs
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

# 新增15只ETF
NEW_ETFS = {
    'sh.515790': '光伏ETF',
    'sh.516160': '新能源ETF',
    'sh.561160': '锂电池ETF',
    'sz.159790': '碳中和ETF',
    'sh.512010': '医药ETF',
    'sz.159928': '消费ETF',
    'sh.512690': '白酒ETF',
    'sh.515170': '食品饮料ETF',
    'sh.512480': '半导体ETF',
    'sh.588000': '科创50ETF',
    'sz.159915': '创业板ETF',
    'sh.513180': '恒生科技ETF',
    'sh.512880': '证券ETF',
    'sh.512800': '银行ETF',
    'sh.512200': '地产ETF',
}


def download_one(bs_code, start_date='2018-01-01'):
    """下载单只ETF"""
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount,pctChg",
            start_date=start_date,
            end_date=datetime.now().strftime('%Y-%m-%d'),
            frequency="d",
            adjustflag="2"
        )
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={'date': 'trade_date', 'pctChg': 'pct_change'})

        # 转换symbol格式
        code = bs_code.split('.')[1]
        df['symbol'] = code

        df['trade_date'] = pd.to_datetime(df['trade_date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[df['close'] > 0].copy()
        return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        return None


def main():
    print("=" * 80)
    print("ETF数据扩展 - 下载新ETF")
    print("=" * 80)

    lg = bs.login()
    print(f"Baostock登录: {lg.error_msg}")

    # 1. 下载新ETF
    print(f"\n下载 {len(NEW_ETFS)} 只新ETF...")
    all_new = []
    success = 0

    for bs_code, name in NEW_ETFS.items():
        print(f"  {name} ({bs_code})...", end='')
        df = download_one(bs_code)
        if df is not None and len(df) > 0:
            all_new.append(df)
            success += 1
            print(f" OK ({len(df)}条)")
        else:
            print(" FAIL")
        time.sleep(0.3)

    bs.logout()

    # 2. 加载已有数据
    conn = sqlite3.connect(DB_PATH)
    existing = pd.read_sql_query("SELECT * FROM etf_daily_prices", conn)
    existing['trade_date'] = pd.to_datetime(existing['trade_date'])
    print(f"\n已有数据: {len(existing)}条, {existing['symbol'].nunique()}只")

    # 3. 合并
    if all_new:
        new_data = pd.concat(all_new, ignore_index=True)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

        combined.to_sql('etf_daily_prices', conn, if_exists='replace', index=False)
        print(f"新数据: {len(new_data)}条")
        print(f"合并后: {len(combined)}条, {combined['symbol'].nunique()}只")

        # 打印各ETF统计
        print(f"\n各ETF数据量:")
        for sym in sorted(combined['symbol'].unique()):
            s = combined[combined['symbol'] == sym]
            print(f"  {sym:<10s} {len(s):>5d}条 ({s['trade_date'].min().date()} ~ {s['trade_date'].max().date()})")

    conn.close()
    print("\n完成!")


if __name__ == '__main__':
    main()
