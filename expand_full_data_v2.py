"""
ETF择时系统 - 全面数据扩展（AkShare版）
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


def download_etf_akshare(symbol, start_date='20180101'):
    """用AkShare下载ETF数据"""
    try:
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                  start_date=start_date, end_date=datetime.now().strftime('%Y%m%d'),
                                  adjust="qfq")
        df = df.rename(columns={
            '日期': 'trade_date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume',
            '成交额': 'amount', '涨跌幅': 'pct_change',
        })
        df['symbol'] = symbol
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]
    except Exception as e:
        print(f"  下载 {symbol} 失败: {e}")
        return None


def download_all_etfs():
    """下载所有ETF数据"""
    print("=" * 80)
    print("下载ETF数据（23只，2018年至今）")
    print("=" * 80)

    all_data = []
    success = 0

    for symbol, info in ETF_POOL.items():
        print(f"  {info['name']} ({symbol})...", end='')
        df = download_etf_akshare(symbol, start_date='20180101')
        if df is not None and len(df) > 0:
            all_data.append(df)
            success += 1
            print(f" OK ({len(df)}条)")
        else:
            print(" FAIL")
        time.sleep(1)  # 避免请求过快

    # 基准
    print(f"  沪深300ETF ({BENCHMARK})...", end='')
    df = download_etf_akshare(BENCHMARK, start_date='20180101')
    if df is not None and len(df) > 0:
        all_data.append(df)
        print(f" OK ({len(df)}条)")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        conn = sqlite3.connect(DB_PATH)
        result.to_sql('etf_full_prices', conn, if_exists='replace', index=False)
        conn.close()
        print(f"\n成功: {success}/{len(ETF_POOL)}只ETF, {len(result)}条数据")
        print(f"时间: {result['trade_date'].min().date()} ~ {result['trade_date'].max().date()}")
        return result
    return None


def build_features(df):
    """构建技术指标+资金流向代理"""
    print("  构建特征...")
    result = []

    for symbol in df['symbol'].unique():
        s = df[df['symbol'] == symbol].copy().sort_values('trade_date').reset_index(drop=True)

        for w in [3, 5, 10, 20]:
            s[f'return_{w}d'] = s['close'].pct_change(w)
        for w in [5, 10, 20]:
            s[f'vol_{w}d'] = s['close'].pct_change().rolling(w).std()
        for w in [6, 14]:
            delta = s['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
            s[f'rsi_{w}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        exp1 = s['close'].ewm(span=12, adjust=False).mean()
        exp2 = s['close'].ewm(span=26, adjust=False).mean()
        s['macd'] = exp1 - exp2
        s['macd_signal'] = s['macd'].ewm(span=9, adjust=False).mean()
        s['macd_hist'] = s['macd'] - s['macd_signal']

        s['bb_mid'] = s['close'].rolling(20).mean()
        s['bb_std'] = s['close'].rolling(20).std()
        s['bb_position'] = (s['close'] - s['bb_mid'] + 2 * s['bb_std']) / (4 * s['bb_std'] + 1e-10)

        for w in [10, 20]:
            s[f'high_{w}d'] = s['high'].rolling(w).max()
            s[f'low_{w}d'] = s['low'].rolling(w).min()
            s[f'pos_{w}d'] = (s['close'] - s[f'low_{w}d']) / (s[f'high_{w}d'] - s[f'low_{w}d'] + 1e-10)

        for w in [5, 10]:
            s[f'vol_ma_{w}'] = s['volume'].rolling(w).mean()
            s[f'vol_ratio_{w}'] = s['volume'] / (s[f'vol_ma_{w}'] + 1e-10)

        s['trend_strength'] = s['close'].pct_change(5) + s['close'].pct_change(10) + s['close'].pct_change(20)

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

        # 情绪占位
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
    print("ETF择时系统 - 全面数据扩展（AkShare版）")
    print("=" * 80)

    # 1. 下载数据
    etf_data = download_all_etfs()
    if etf_data is None:
        print("数据下载失败!")
        return

    # 2. 构建特征
    print("\n" + "=" * 80)
    print("构建特征")
    print("=" * 80)

    df = build_features(etf_data)
    benchmark_df = df[df['symbol'] == BENCHMARK].copy()
    etf_df = df[df['symbol'].isin(ETF_POOL.keys())].copy()
    etf_df = add_benchmark_features(etf_df, benchmark_df)

    # 保存
    conn = sqlite3.connect(DB_PATH)
    etf_df.to_sql('etf_full_features', conn, if_exists='replace', index=False)
    conn.close()

    feature_cols = [c for c in etf_df.columns if c.startswith(('return_', 'vol_', 'rsi_', 'macd', 'bb_', 'pos_',
                                                                'vol_ratio', 'trend_', 'atr_', 'relative_', 'bm_',
                                                                'sentiment_', 'fund_flow'))
                    and c not in ['bm_target_5d', 'bm_target_up']]

    print(f"\n最终数据集:")
    print(f"  标的数: {etf_df['symbol'].nunique()}")
    print(f"  样本数: {len(etf_df)}")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  时间: {etf_df['trade_date'].min().date()} ~ {etf_df['trade_date'].max().date()}")


if __name__ == '__main__':
    main()
