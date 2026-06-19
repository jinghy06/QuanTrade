"""
小资金ETF择时系统 - AI+军工赛道
策略A: ETF轮动选股（每周选最好的1-2只）
策略B: 大盘择时（判断买不买）
策略C: 轮动+择时组合
"""
import sqlite3
import pandas as pd
import numpy as np
import akshare as ak
import warnings
from datetime import datetime
import lightgbm as lgb
import json
import os

warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'
BASE_MODEL_DIR = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\models'

ETF_POOL = {
    # ETF
    '562500': {'name': '机器人ETF', 'sector': 'AI'},
    '515070': {'name': '人工智能ETF', 'sector': 'AI'},
    '159995': {'name': '芯片ETF', 'sector': 'AI'},
    '159550': {'name': '算力ETF', 'sector': 'AI'},
    '516510': {'name': '云计算ETF', 'sector': 'AI'},
    '512660': {'name': '军工ETF', 'sector': 'military'},
    '512670': {'name': '国防ETF', 'sector': 'military'},
    '515960': {'name': '航天军工ETF', 'sector': 'military'},
    # AI龙头股
    '002230.SZ': {'name': '科大讯飞', 'sector': 'AI'},
    '002415.SZ': {'name': '海康威视', 'sector': 'AI'},
    '300124.SZ': {'name': '汇川技术', 'sector': 'AI'},
    '002747.SZ': {'name': '埃斯顿', 'sector': 'AI'},
    '688017.SH': {'name': '绿的谐波', 'sector': 'AI'},
    '300308.SZ': {'name': '中际旭创', 'sector': 'AI'},
    '688256.SH': {'name': '寒武纪', 'sector': 'AI'},
    '688041.SH': {'name': '海光信息', 'sector': 'AI'},
    '002236.SZ': {'name': '大华股份', 'sector': 'AI'},
    '300496.SZ': {'name': '中科创达', 'sector': 'AI'},
    '002405.SZ': {'name': '四维图新', 'sector': 'AI'},
    '688111.SH': {'name': '金山办公', 'sector': 'AI'},
    # 军工龙头股
    '600760.SH': {'name': '中航沈飞', 'sector': 'military'},
    '002179.SZ': {'name': '中航光电', 'sector': 'military'},
    '600893.SH': {'name': '航发动力', 'sector': 'military'},
    '000768.SZ': {'name': '中航西飞', 'sector': 'military'},
    '600118.SH': {'name': '中国卫星', 'sector': 'military'},
    '002025.SZ': {'name': '航天电器', 'sector': 'military'},
    '600038.SH': {'name': '中直股份', 'sector': 'military'},
    '300699.SZ': {'name': '光威复材', 'sector': 'military'},
    '600862.SH': {'name': '中航高科', 'sector': 'military'},
    '600372.SH': {'name': '中航机载', 'sector': 'military'},
}

BENCHMARK = '510300'  # 沪深300ETF
COMMISSION = 0.0003
STAMP_TAX = 0.001
SLIPPAGE = 0.002
TOTAL_COST = COMMISSION * 2 + STAMP_TAX + SLIPPAGE


def download_etf_data(symbol, start_date='20220101'):
    """下载单只ETF数据"""
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


def download_all_etf_data():
    """下载所有ETF数据"""
    print("下载ETF数据...")
    all_data = []
    for symbol in list(ETF_POOL.keys()) + [BENCHMARK]:
        name = ETF_POOL.get(symbol, {}).get('name', '沪深300ETF')
        print(f"  {name} ({symbol})...", end='')
        df = download_etf_data(symbol)
        if df is not None and len(df) > 0:
            all_data.append(df)
            print(f" OK ({len(df)}条)")
        else:
            print(" FAIL")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        conn = sqlite3.connect(DB_PATH)
        result.to_sql('etf_daily_prices', conn, if_exists='replace', index=False)
        conn.close()
        print(f"已保存: {len(result)}条")
        return result
    return None


def build_features(df):
    """构建特征"""
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


def strategy_a_rotation(etf_df, feature_cols, n_stocks=2):
    """策略A：每周选最好的1-2只ETF"""
    train = etf_df[etf_df['trade_date'] < '2024-01-01'].dropna(subset=feature_cols + ['target_up'])
    test = etf_df[etf_df['trade_date'] >= '2024-01-01'].copy()

    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                 num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, verbosity=-1, n_jobs=-1)
    model.fit(train[feature_cols], train['target_up'])

    test['year_week'] = test['trade_date'].dt.isocalendar().year * 100 + test['trade_date'].dt.isocalendar().week
    weekly_returns = []

    for yw in sorted(test['year_week'].unique()):
        week_df = test[test['year_week'] == yw]
        if len(week_df) == 0:
            continue
        decision_day = week_df['trade_date'].min()
        decision_df = week_df[week_df['trade_date'] == decision_day].copy()
        if len(decision_df) < 2:
            continue

        proba = model.predict_proba(decision_df[feature_cols].fillna(0))[:, 1]
        decision_df['score'] = proba
        selected = decision_df.nlargest(n_stocks, 'score')

        rets = []
        for _, stock in selected.iterrows():
            sym_data = week_df[week_df['symbol'] == stock['symbol']].sort_values('trade_date')
            if len(sym_data) >= 2:
                rets.append(sym_data['close'].iloc[-1] / sym_data['close'].iloc[0] - 1)

        week_ret = np.mean(rets) - TOTAL_COST if rets else 0
        weekly_returns.append({'year_week': yw, 'return': week_ret})

    return pd.DataFrame(weekly_returns), model


def strategy_b_timing(etf_df, benchmark_df, bm_feature_cols):
    """策略B：判断大盘风险，决定买不买"""
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
    bm['bm_target_up'] = (bm['close'].shift(-5) / bm['close'] - 1 > 0).astype(int)

    bm = bm.dropna(subset=bm_feature_cols + ['bm_target_up'])
    train = bm[bm['trade_date'] < '2024-01-01']
    test = bm[bm['trade_date'] >= '2024-01-01']

    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                 num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, verbosity=-1, n_jobs=-1)
    model.fit(train[bm_feature_cols], train['bm_target_up'])

    test = test.copy()
    test['year_week'] = test['trade_date'].dt.isocalendar().year * 100 + test['trade_date'].dt.isocalendar().week
    etf_df = etf_df.copy()
    etf_df['year_week'] = etf_df['trade_date'].dt.isocalendar().year * 100 + etf_df['trade_date'].dt.isocalendar().week

    weekly_returns = []
    for yw in sorted(test['year_week'].unique()):
        week_bm = test[test['year_week'] == yw]
        if len(week_bm) == 0:
            continue
        decision_row = week_bm.iloc[0:1]
        proba = model.predict_proba(decision_row[bm_feature_cols].fillna(0))[:, 1][0]

        week_etf = etf_df[etf_df['year_week'] == yw]
        etf_rets = []
        for sym in week_etf['symbol'].unique():
            sym_data = week_etf[week_etf['symbol'] == sym].sort_values('trade_date')
            if len(sym_data) >= 2:
                etf_rets.append(sym_data['close'].iloc[-1] / sym_data['close'].iloc[0] - 1)
        avg_ret = np.mean(etf_rets) if etf_rets else 0

        if proba > 0.55:
            position, week_ret = 1.0, avg_ret - TOTAL_COST
        elif proba < 0.45:
            position, week_ret = 0.0, 0.0
        else:
            position, week_ret = 0.5, avg_ret * 0.5 - TOTAL_COST * 0.5

        weekly_returns.append({'year_week': yw, 'return': week_ret, 'position': position, 'market_prob': proba})

    return pd.DataFrame(weekly_returns), model


def compute_metrics(weekly_returns, name):
    """计算策略指标"""
    returns = weekly_returns['return'].values
    cumulative = (1 + returns).cumprod()
    total_return = cumulative[-1] - 1
    years = len(returns) / 52
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(52)
    peak = np.maximum.accumulate(cumulative)
    max_drawdown = np.min((cumulative - peak) / peak)
    win_rate = np.sum(returns > 0) / len(returns)

    return {
        'strategy': name,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_weeks': len(returns),
    }


def main():
    print("=" * 80)
    print("小资金ETF择时系统 - AI+军工赛道")
    print("=" * 80)
    n_ai = sum(1 for v in ETF_POOL.values() if v['sector'] == 'AI')
    n_mil = sum(1 for v in ETF_POOL.values() if v['sector'] == 'military')
    print(f"\n标的池: {len(ETF_POOL)}只ETF ({n_ai}只AI + {n_mil}只军工)")
    print(f"调仓频率: 每周一")
    print(f"交易成本: {TOTAL_COST*100:.2f}%/次")

    # 1. 数据
    print(f"\n{'=' * 80}")
    print("步骤1: 加载/下载ETF数据")
    print(f"{'=' * 80}")

    conn = sqlite3.connect(DB_PATH)
    try:
        etf_data = pd.read_sql_query("SELECT * FROM etf_daily_prices", conn)
        etf_data['trade_date'] = pd.to_datetime(etf_data['trade_date'])

        # 只用有完整数据的ETF（至少500条记录，覆盖2022年至今）
        etf_counts = etf_data.groupby('symbol')['trade_date'].count()
        full_etfs = etf_counts[etf_counts > 500].index.tolist()
        etf_data = etf_data[etf_data['symbol'].isin(full_etfs)]

        print(f"  从数据库加载: {len(etf_data)}条, {etf_data['symbol'].nunique()}只ETF（有完整数据）")
    except:
        conn.close()
        etf_data = download_all_etf_data()
        if etf_data is None:
            print("数据下载失败!")
            return
        conn = sqlite3.connect(DB_PATH)

    # 2. 特征
    print(f"\n{'=' * 80}")
    print("步骤2: 构建特征")
    print(f"{'=' * 80}")

    df = build_features(etf_data)
    benchmark_df = df[df['symbol'] == BENCHMARK].copy()
    etf_df = df[df['symbol'].isin(ETF_POOL.keys())].copy()
    etf_df = add_benchmark_features(etf_df, benchmark_df)

    FEATURE_COLS = [c for c in etf_df.columns
                    if c.startswith(('return_', 'vol_', 'rsi_', 'macd', 'bb_', 'pos_', 'vol_ratio', 'trend_', 'atr_', 'relative_', 'bm_'))
                    and c not in ['bm_target_5d', 'bm_target_up']]
    BM_FEATURE_COLS = ['bm_return_5d', 'bm_return_10d', 'bm_return_20d',
                       'bm_vol_10d', 'bm_vol_20d', 'bm_rsi_14',
                       'bm_above_ma20', 'bm_above_ma60', 'bm_high_vol']

    print(f"  ETF特征: {len(FEATURE_COLS)}个")
    print(f"  大盘特征: {len(BM_FEATURE_COLS)}个")

    # 3. 策略
    print(f"\n{'=' * 80}")
    print("步骤3: 运行策略")
    print(f"{'=' * 80}")

    # 基准
    print("\n基准: 等权持有所有ETF")
    etf_df_copy = etf_df.copy()
    etf_df_copy['year_week'] = etf_df_copy['trade_date'].dt.isocalendar().year * 100 + etf_df_copy['trade_date'].dt.isocalendar().week
    bm_weekly = []
    for yw in sorted(etf_df_copy['year_week'].unique()):
        week_df = etf_df_copy[etf_df_copy['year_week'] == yw]
        rets = []
        for sym in week_df['symbol'].unique():
            sd = week_df[week_df['symbol'] == sym].sort_values('trade_date')
            if len(sd) >= 2:
                rets.append(sd['close'].iloc[-1] / sd['close'].iloc[0] - 1)
        bm_weekly.append({'year_week': yw, 'return': np.mean(rets) if rets else 0})
    bm_weekly_df = pd.DataFrame(bm_weekly)

    print("\n策略A: ETF轮动选股")
    sa_df, _ = strategy_a_rotation(etf_df, FEATURE_COLS, n_stocks=2)

    print("策略B: 大盘择时")
    sb_df, _ = strategy_b_timing(etf_df, benchmark_df, BM_FEATURE_COLS)

    print("策略C: 轮动+择时组合")
    sa2, _ = strategy_a_rotation(etf_df, FEATURE_COLS, n_stocks=2)
    sb2, _ = strategy_b_timing(etf_df, benchmark_df, BM_FEATURE_COLS)
    sc_df = sa2.merge(sb2[['year_week', 'position']], on='year_week', how='inner')
    sc_df['return'] = sc_df['return'] * sc_df['position']

    # 4. 结果
    print(f"\n{'=' * 80}")
    print("步骤4: 结果对比")
    print(f"{'=' * 80}")

    all_metrics = [
        compute_metrics(bm_weekly_df, '基准:等权持有'),
        compute_metrics(sa_df, 'A:轮动选股'),
        compute_metrics(sb_df, 'B:大盘择时'),
        compute_metrics(sc_df, 'C:轮动+择时'),
    ]

    print(f"\n{'策略':<20s} {'总收益':>8s} {'年化':>8s} {'夏普':>6s} {'最大回撤':>8s} {'胜率':>6s}")
    print("-" * 60)
    for m in all_metrics:
        print(f"{m['strategy']:<20s} {m['total_return']*100:>7.2f}% {m['annual_return']*100:>7.2f}% "
              f"{m['sharpe']:>6.2f} {m['max_drawdown']*100:>7.2f}% {m['win_rate']*100:>5.1f}%")

    print(f"\n按总收益排名:")
    for i, m in enumerate(sorted(all_metrics, key=lambda x: x['total_return'], reverse=True)):
        print(f"  {i+1}. {m['strategy']}: {m['total_return']*100:.2f}%")

    print(f"\n按夏普比率排名:")
    for i, m in enumerate(sorted(all_metrics, key=lambda x: x['sharpe'], reverse=True)):
        print(f"  {i+1}. {m['strategy']}: {m['sharpe']:.2f}")

    # 保存
    results = {
        'experiments': all_metrics,
        'best_return': sorted(all_metrics, key=lambda x: x['total_return'], reverse=True)[0]['strategy'],
        'best_sharpe': sorted(all_metrics, key=lambda x: x['sharpe'], reverse=True)[0]['strategy'],
        'etf_pool': {k: v['name'] for k, v in ETF_POOL.items()},
    }
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    with open(os.path.join(BASE_MODEL_DIR, 'etf_system_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到 {os.path.join(BASE_MODEL_DIR, 'etf_system_results.json')}")

    conn.close()


if __name__ == '__main__':
    main()
