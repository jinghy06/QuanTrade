"""
用Baostock下载龙头股数据（备用数据源）
"""
import sqlite3
import pandas as pd
import numpy as np
import baostock as bs
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

SECTOR_STOCKS = {
    # AI/机器人赛道龙头
    'sz.002230': {'name': '科大讯飞', 'sector': 'AI'},
    'sz.002415': {'name': '海康威视', 'sector': 'AI'},
    'sz.300124': {'name': '汇川技术', 'sector': 'AI'},
    'sz.002747': {'name': '埃斯顿', 'sector': 'AI'},
    'sh.688017': {'name': '绿的谐波', 'sector': 'AI'},
    'sz.300308': {'name': '中际旭创', 'sector': 'AI'},
    'sh.688256': {'name': '寒武纪', 'sector': 'AI'},
    'sh.688041': {'name': '海光信息', 'sector': 'AI'},
    'sz.002236': {'name': '大华股份', 'sector': 'AI'},
    'sz.300496': {'name': '中科创达', 'sector': 'AI'},
    'sz.002405': {'name': '四维图新', 'sector': 'AI'},
    'sh.688111': {'name': '金山办公', 'sector': 'AI'},

    # 航空航天/军工赛道龙头
    'sh.600760': {'name': '中航沈飞', 'sector': 'military'},
    'sz.002179': {'name': '中航光电', 'sector': 'military'},
    'sh.600893': {'name': '航发动力', 'sector': 'military'},
    'sz.000768': {'name': '中航西飞', 'sector': 'military'},
    'sh.600118': {'name': '中国卫星', 'sector': 'military'},
    'sz.002025': {'name': '航天电器', 'sector': 'military'},
    'sh.600038': {'name': '中直股份', 'sector': 'military'},
    'sz.300699': {'name': '光威复材', 'sector': 'military'},
    'sh.600862': {'name': '中航高科', 'sector': 'military'},
    'sh.600372': {'name': '中航机载', 'sector': 'military'},
}


def download_stock(bs_code, start_date='2022-01-01'):
    """用Baostock下载单只股票"""
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount,pctChg",
            start_date=start_date,
            end_date=datetime.now().strftime('%Y-%m-%d'),
            frequency="d",
            adjustflag="2"  # 前复权
        )

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={
            'date': 'trade_date',
            'pctChg': 'pct_change',
        })

        # 转换为标准symbol格式
        code = bs_code.split('.')[1]
        market = 'SH' if bs_code.startswith('sh') else 'SZ'
        df['symbol'] = f"{code}.{market}"

        df['trade_date'] = pd.to_datetime(df['trade_date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        print(f"  下载 {bs_code} 失败: {e}")
        return None


def main():
    print("=" * 80)
    print("用Baostock下载龙头股数据")
    print("=" * 80)

    # 登录
    lg = bs.login()
    print(f"Baostock登录: {lg.error_msg}")

    all_data = []
    success = 0

    for bs_code, info in SECTOR_STOCKS.items():
        print(f"  {info['name']} ({bs_code})...", end='')
        df = download_stock(bs_code)
        if df is not None and len(df) > 0:
            # 过滤无效数据
            df = df[df['close'] > 0].copy()
            if len(df) > 0:
                all_data.append(df)
                success += 1
                print(f" OK ({len(df)}条)")
            else:
                print(" 无有效数据")
        else:
            print(" FAIL")

    print(f"\n成功: {success}/{len(SECTOR_STOCKS)}")

    # 加载已有ETF数据
    conn = sqlite3.connect(DB_PATH)
    etf_data = pd.read_sql_query("SELECT * FROM etf_daily_prices", conn)
    etf_data['trade_date'] = pd.to_datetime(etf_data['trade_date'])

    # 合并
    if all_data:
        stock_data = pd.concat(all_data, ignore_index=True)
        combined = pd.concat([etf_data, stock_data], ignore_index=True)
        combined = combined.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

        combined.to_sql('expanded_daily_prices', conn, if_exists='replace', index=False)
        print(f"\n合并数据已保存: {len(combined)}条, {combined['symbol'].nunique()}只标的")

        # 打印统计
        print(f"\n标的列表:")
        for sym in sorted(combined['symbol'].unique()):
            s = combined[combined['symbol'] == sym]
            print(f"  {sym:<12s} {len(s):>5d}条")

    bs.logout()
    conn.close()
    print("\n完成!")


if __name__ == '__main__':
    main()
