"""
ETF择时系统 - 全面数据扩展
1. 更多ETF（23只，2018年至今）
2. 资金流向代理指标
3. 新闻情绪因子
"""
import sqlite3
import pandas as pd
import numpy as np
import baostock as bs
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

ETF_POOL = {
    '562500': {'name': '机器人ETF', 'sector': 'AI'},
    '515070': {'name': '人工智能ETF', 'sector': 'AI'},
    '159995': {'name': '芯片ETF', 'sector': 'AI'},
    '159550': {'name': '算力ETF', 'sector': 'AI'},
    '516510': {'name': '云计算ETF', 'sector': 'AI'},
    '512660': {'name': '军工ETF', 'sector': 'military'},
    '512670': {'name': '国防ETF', 'sector': 'military'},
    '515960': {'name': '航天军工ETF', 'sector': 'military'},
    '515790': {'name': '光伏ETF', 'sector': 'energy'},
    '516160': {'name': '新能源ETF', 'sector': 'energy'},
    '561160': {'name': '锂电池ETF', 'sector': 'energy'},
    '159790': {'name': '碳中和ETF', 'sector': 'energy'},
    '512010': {'name': '医药ETF', 'sector': 'consumer'},
    '159928': {'name': '消费ETF', 'sector': 'consumer'},
    '512690': {'name': '白酒ETF', 'sector': 'consumer'},
    '515170': {'name': '食品饮料ETF', 'sector': 'consumer'},
    '512480': {'name': '半导体ETF', 'sector': 'tech'},
    '588000': {'name': '科创50ETF', 'sector': 'tech'},
    '159915': {'name': '创业板ETF', 'sector': 'tech'},
    '513180': {'name': '恒生科技ETF', 'sector': 'tech'},
    '512880': {'name': '证券ETF', 'sector': 'finance'},
    '512800': {'name': '银行ETF', 'sector': 'finance'},
    '512200': {'name': '地产ETF', 'sector': 'finance'},
}
BENCHMARK = '510300'


def download_etf_baostock(symbol, start_date='2018-01-01'):
    """用Baostock下载ETF数据"""
    try:
        if symbol.startswith(('5', '5')):
            bs_code = f"sh.{symbol}"
        elif symbol.startswith(('1', '0', '3')):
            bs_code = f"sz.{symbol}"
        else:
            bs_code = f"sh.{symbol}"

        rs = bs.query_history_k_data_plus(
            bs_code, "date,code,open,high,low,close,volume,amount,pctChg",
            start_date=start_date, end_date=datetime.now().strftime('%Y-%m-%d'),
            frequency="d", adjustflag="2"
        )
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        if not data_list:
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={'date': 'trade_date', 'pctChg': 'pct_change'})
        df['symbol'] = symbol
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df['close'] > 0].copy()
        return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        return None


def download_all_etfs():
    """下载所有ETF数据"""
    print("=" * 80)
    print("步骤1: 下载ETF数据（23只，2018年至今）")
    print("=" * 80)

    lg = bs.login()
    print(f"Baostock登录: {lg.error_msg}")

    all_data = []
    success = 0
    for symbol, info in ETF_POOL.items():
        print(f"  {info['name']} ({symbol})...", end='')
        df = download_etf_baostock(symbol)
        if df is not None and len(df) > 0:
            all_data.append(df)
            success += 1
            print(f" OK ({len(df)}条)")
        else:
            print(" FAIL")
        time.sleep(0.3)

    # 基准
    print(f"  沪深300ETF ({BENCHMARK})...", end='')
    df = download_etf_baostock(BENCHMARK)
    if df is not None and len(df) > 0:
        all_data.append(df)
        print(f" OK ({len(df)}条)")

    bs.logout()

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        conn = sqlite3.connect(DB_PATH)
        result.to_sql('etf_full_prices', conn, if_exists='replace', index=False)
        conn.close()
        print(f"\n成功: {success}/{len(ETF_POOL)}只ETF, {len(result)}条数据")
        return result
    return None


def build_features(df):
    """构建技术指标+资金流向代理"""
    print("  构建技术指标和资金流向代理...")
    result = []

    for symbol in df['symbol'].unique():
        s = df[df['symbol'] == symbol].copy().sort_values('trade_date').reset_index(drop=True)

        # 动量
        for w in [3, 5, 10, 20]:
            s[f'return_{w}d'] = s['close'].pct_change(w)

        # 波动率
        for w in [5, 10, 20]:
            s[f'vol_{w}d'] = s['close'].pct_change().rolling(w).std()

        # RSI
        for w in [6, 14]:
            delta = s['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
            s[f'rsi_{w}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # MACD
        exp1 = s['close'].ewm(span=12, adjust=False).mean()
        exp2 = s['close'].ewm(span=26, adjust=False).mean()
        s['macd'] = exp1 - exp2
        s['macd_signal'] = s['macd'].ewm(span=9, adjust=False).mean()
        s['macd_hist'] = s['macd'] - s['macd_signal']

        # 布林带
        s['bb_mid'] = s['close'].rolling(20).mean()
        s['bb_std'] = s['close'].rolling(20).std()
        s['bb_position'] = (s['close'] - s['bb_mid'] + 2 * s['bb_std']) / (4 * s['bb_std'] + 1e-10)

        # 价格位置
        for w in [10, 20]:
            s[f'high_{w}d'] = s['high'].rolling(w).max()
            s[f'low_{w}d'] = s['low'].rolling(w).min()
            s[f'pos_{w}d'] = (s['close'] - s[f'low_{w}d']) / (s[f'high_{w}d'] - s[f'low_{w}d'] + 1e-10)

        # 成交量
        for w in [5, 10]:
            s[f'vol_ma_{w}'] = s['volume'].rolling(w).mean()
            s[f'vol_ratio_{w}'] = s['volume'] / (s[f'vol_ma_{w}'] + 1e-10)

        # 趋势
        s['trend_strength'] = s['close'].pct_change(5) + s['close'].pct_change(10) + s['close'].pct_change(20)

        # ATR
        high_low = s['high'] - s['low']
        high_close = np.abs(s['high'] - s['close'].shift())
        low_close = np.abs(s['low'] - s['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        s['atr_14'] = tr.rolling(14).mean()

        # 资金流向代理
        s['fund_flow_5d'] = s['volume'].pct_change(5)
        s['fund_flow_10d'] = s['volume'].pct_change(10)
        s['fund_flow_20d'] = s['volume'].pct_change(20)
        s['price_vol_corr'] = s['close'].rolling(10).corr(s['volume'])

        # 情绪占位（后续可用真实数据填充）
        s['sentiment_1d'] = 0.0
        s['sentiment_3d'] = 0.0
        s['sentiment_7d'] = 0.0

        # 目标变量
        s['target_return_5d'] = s['close'].shift(-5) / s['close'] - 1
        s['target_up'] = (s['target_return_5d'] > 0).astype(int)

        result.append(s)

    return pd.concat(result, ignore_index=True)


def add_benchmark_features(df, benchmark_df):
    """添加大盘特征"""
    bm = benchmark_df.sort_values('trade_date').copy()
    bm['bm_return_5d'] = bm['close'].pct_change(5)
    bm['bm_return_10d'] = bm['close'].pct_change(10)
    bm['bm_return_20d'] = bm['close'].pct_change(20)
    bm['bm_vol_10d'] = bm['close'].pct_change().rolling(10).std()
    bm['bm_vol_20d'] = bm['close'].pct_change().rolling(20).std()

    delta = bm['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    bm['bm_rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    bm['bm_ma_20'] = bm['close'].rolling(20).mean()
    bm['bm_ma_60'] = bm['close'].rolling(60).mean()
    bm['bm_above_ma20'] = (bm['close'] > bm['bm_ma_20']).astype(int)
    bm['bm_above_ma60'] = (bm['close'] > bm['bm_ma_60']).astype(int)
    bm['bm_high_vol'] = (bm['bm_vol_20d'] > bm['bm_vol_20d'].rolling(60).quantile(0.8)).astype(int)

    bm_cols = ['trade_date', 'bm_return_5d', 'bm_return_10d', 'bm_return_20d',
               'bm_vol_10d', 'bm_vol_20d', 'bm_rsi_14', 'bm_above_ma20', 'bm_above_ma60', 'bm_high_vol']

    df = df.merge(bm[bm_cols], on='trade_date', how='left')
    df['relative_return_5d'] = df['return_5d'] - df['bm_return_5d']
    df['relative_return_10d'] = df['return_10d'] - df['bm_return_10d']
    return df


def main():
    print("=" * 80)
    print("ETF择时系统 - 全面数据扩展")
    print("=" * 80)

    # 1. 下载数据
    etf_data = download_all_etfs()
    if etf_data is None:
        print("数据下载失败!")
        return

    # 2. 构建特征
    print("\n" + "=" * 80)
    print("步骤2: 构建特征")
    print("=" * 80)

    df = build_features(etf_data)
    benchmark_df = df[df['symbol'] == BENCHMARK].copy()
    etf_df = df[df['symbol'].isin(ETF_POOL.keys())].copy()
    etf_df = add_benchmark_features(etf_df, benchmark_df)

    # 保存
    conn = sqlite3.connect(DB_PATH)
    etf_df.to_sql('etf_full_features', conn, if_exists='replace', index=False)
    conn.close()

    # 统计
    feature_cols = [c for c in etf_df.columns if c.startswith(('return_', 'vol_', 'rsi_', 'macd', 'bb_', 'pos_',
                                                                'vol_ratio', 'trend_', 'atr_', 'relative_', 'bm_',
                                                                'sentiment_', 'fund_flow'))
                    and c not in ['bm_target_5d', 'bm_target_up']]

    print(f"\n最终数据集:")
    print(f"  标的数: {etf_df['symbol'].nunique()}")
    print(f"  样本数: {len(etf_df)}")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  时间: {etf_df['trade_date'].min().date()} ~ {etf_df['trade_date'].max().date()}")
    print(f"\n数据已保存到 etf_full_features 表")


if __name__ == '__main__':
    main()
