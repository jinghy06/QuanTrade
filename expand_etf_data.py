"""
扩展数据集：加入AI/机器人和航空航天/军工赛道龙头股
"""
import sqlite3
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

# ============================================================
# 龙头股标的池
# ============================================================
SECTOR_STOCKS = {
    # AI/机器人赛道龙头
    '002230': {'name': '科大讯飞', 'sector': 'AI', 'desc': 'AI语音龙头'},
    '002415': {'name': '海康威视', 'sector': 'AI', 'desc': 'AI+安防龙头'},
    '300124': {'name': '汇川技术', 'sector': 'AI', 'desc': '机器人+自动化'},
    '002747': {'name': '埃斯顿', 'sector': 'AI', 'desc': '工业机器人'},
    '688017': {'name': '绿的谐波', 'sector': 'AI', 'desc': '机器人核心零部件'},
    '300308': {'name': '中际旭创', 'sector': 'AI', 'desc': '光模块/算力龙头'},
    '688256': {'name': '寒武纪', 'sector': 'AI', 'desc': 'AI芯片'},
    '688041': {'name': '海光信息', 'sector': 'AI', 'desc': '国产GPU'},
    '002236': {'name': '大华股份', 'sector': 'AI', 'desc': 'AI+视频监控'},
    '300496': {'name': '中科创达', 'sector': 'AI', 'desc': '智能操作系统'},
    '002405': {'name': '四维图新', 'sector': 'AI', 'desc': '自动驾驶地图'},
    '688111': {'name': '金山办公', 'sector': 'AI', 'desc': 'AI办公'},

    # 航空航天/军工赛道龙头
    '600760': {'name': '中航沈飞', 'sector': 'military', 'desc': '战斗机龙头'},
    '002179': {'name': '中航光电', 'sector': 'military', 'desc': '军工连接器'},
    '600893': {'name': '航发动力', 'sector': 'military', 'desc': '航空发动机'},
    '000768': {'name': '中航西飞', 'sector': 'military', 'desc': '运输机'},
    '600118': {'name': '中国卫星', 'sector': 'military', 'desc': '卫星龙头'},
    '002025': {'name': '航天电器', 'sector': 'military', 'desc': '航天连接器'},
    '600038': {'name': '中直股份', 'sector': 'military', 'desc': '直升机'},
    '300699': {'name': '光威复材', 'sector': 'military', 'desc': '碳纤维材料'},
    '600862': {'name': '中航高科', 'sector': 'military', 'desc': '航空复合材料'},
    '600372': {'name': '中航机载', 'sector': 'military', 'desc': '航空电子'},
}


def download_stock_data(symbol, start_date='20220101'):
    """下载单只股票数据（带市场前缀）"""
    try:
        # 判断市场
        if symbol.startswith('6'):
            market = 'sh'
        elif symbol.startswith(('0', '3')):
            market = 'sz'
        else:
            market = 'bj'

        full_symbol = f"{market}{symbol}"

        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                 start_date=start_date, end_date=datetime.now().strftime('%Y%m%d'),
                                 adjust="qfq")
        df = df.rename(columns={
            '日期': 'trade_date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume',
            '成交额': 'amount', '涨跌幅': 'pct_change',
        })
        df['symbol'] = f"{symbol}.{'SH' if market == 'sh' else 'SZ'}"
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        print(f"  下载 {symbol} 失败: {e}")
        return None


def main():
    print("=" * 80)
    print("扩展数据集：加入AI+军工龙头股")
    print("=" * 80)

    # 1. 下载龙头股数据
    print(f"\n下载 {len(SECTOR_STOCKS)} 只龙头股数据...")
    all_data = []
    success = 0

    for symbol, info in SECTOR_STOCKS.items():
        print(f"  {info['name']} ({symbol})...", end='')
        df = download_stock_data(symbol)
        if df is not None and len(df) > 0:
            all_data.append(df)
            success += 1
            print(f" OK ({len(df)}条)")
        else:
            print(" FAIL")
        time.sleep(0.5)  # 避免请求过快

    print(f"\n成功下载: {success}/{len(SECTOR_STOCKS)}")

    # 2. 加载已有ETF数据
    conn = sqlite3.connect(DB_PATH)
    etf_data = pd.read_sql_query("SELECT * FROM etf_daily_prices", conn)
    etf_data['trade_date'] = pd.to_datetime(etf_data['trade_date'])
    print(f"已有ETF数据: {len(etf_data)}条")

    # 3. 合并数据
    if all_data:
        stock_data = pd.concat(all_data, ignore_index=True)
        combined = pd.concat([etf_data, stock_data], ignore_index=True)
        combined = combined.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

        # 保存
        combined.to_sql('expanded_daily_prices', conn, if_exists='replace', index=False)
        print(f"\n合并数据已保存到 expanded_daily_prices: {len(combined)}条")
        print(f"总标的数: {combined['symbol'].nunique()}")

        # 统计
        print(f"\n标的列表:")
        names = {f"{k}.{'SH' if k.startswith('6') else 'SZ'}": v['name'] for k, v in SECTOR_STOCKS.items()}
        etf_names = {
            '562500': '机器人ETF', '515070': '人工智能ETF', '159995': '芯片ETF',
            '159550': '算力ETF', '516510': '云计算ETF', '512660': '军工ETF',
            '512670': '国防ETF', '515960': '航天军工ETF', '510300': '沪深300ETF',
        }
        all_names = {**names, **etf_names}

        for sym in sorted(combined['symbol'].unique()):
            s = combined[combined['symbol'] == sym]
            name = all_names.get(sym, sym)
            print(f"  {sym:<12s} {name:<15s} {len(s):>5d}条 ({s['trade_date'].min().date()} ~ {s['trade_date'].max().date()})")

    conn.close()
    print("\n完成!")


if __name__ == '__main__':
    main()
