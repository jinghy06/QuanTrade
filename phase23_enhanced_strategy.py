"""
Phase 2+3: 策略体系全面升级
- 趋势过滤器优化
- Kelly公式仓位管理
- 动态阈值调整
- 多策略融合（分类+回归+趋势+波动率）
- 扩展因子（交叉ETF、时间特征、价格行为）
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, mean_squared_error, r2_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("【Phase 2+3: 策略体系全面升级】")
print("=" * 70)

# ==================== 1. 数据加载 ====================
db_path = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
conn = sqlite3.connect(db_path)

etf_list = ['562500.SH', '159382.SZ', '588790.SH', '159241.SZ', '588200.SH']

all_data = []
for etf in etf_list:
    df = pd.read_sql_query(f"SELECT * FROM daily_prices WHERE symbol='{etf}' ORDER BY trade_date", conn)
    all_data.append(df)
    print(f"{etf}: {len(df)} 条")

# 也加载个股数据作为市场代理
market_proxy = pd.read_sql_query(
    "SELECT * FROM daily_prices WHERE symbol='000001.SZ' ORDER BY trade_date", conn)
print(f"000001.SZ(平安银行): {len(market_proxy)} 条 (市场代理)")

conn.close()

df_all = pd.concat(all_data, ignore_index=True)
df_all['trade_date'] = pd.to_datetime(df_all['trade_date'], format='mixed')
df_all = df_all.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

market_proxy['trade_date'] = pd.to_datetime(market_proxy['trade_date'], format='mixed')
market_proxy = market_proxy.sort_values('trade_date').reset_index(drop=True)
market_proxy['market_return_1d'] = market_proxy['close'].pct_change()
market_proxy['market_return_5d'] = market_proxy['close'].pct_change(5)
market_proxy['market_return_20d'] = market_proxy['close'].pct_change(20)
market_proxy['market_ma20'] = market_proxy['close'].rolling(20).mean()
market_proxy['market_trend'] = (market_proxy['close'] > market_proxy['market_ma20']).astype(int)

print(f"\n总数据量: {len(df_all)} 条, ETF数量: {df_all['symbol'].nunique()}")

# ==================== 2. 扩展特征工程 (80+ 特征) ====================
print("\n" + "=" * 70)
print("【2. 扩展特征工程 (80+ Alpha因子)】")
print("=" * 70)

def calc_features_v3(group, market_df=None):
    """完整版特征工程 - 80+ Alpha因子"""
    g = group.sort_values('trade_date').copy()
    n = len(g)
    if n < 30:
        return g

    # --- 基础价格特征 ---
    g['return_1d'] = g['close'].pct_change()
    for w in [2, 3, 5, 10, 20, 60]:
        g[f'return_{w}d'] = g['close'].pct_change(w)

    # --- 移动平均线体系 ---
    for w in [3, 5, 10, 20, 60]:
        g[f'ma_{w}'] = g['close'].rolling(w).mean()
        g[f'ma_dist_{w}'] = (g['close'] - g[f'ma_{w}']) / g[f'ma_{w}']
        g[f'ema_{w}'] = g['close'].ewm(span=w, adjust=False).mean()

    # MA排列信号
    g['ma_alignment'] = np.where(
        (g['ma_5'] > g['ma_10']) & (g['ma_10'] > g['ma_20']), 1,
        np.where((g['ma_5'] < g['ma_10']) & (g['ma_10'] < g['ma_20']), -1, 0)
    )
    g['ma_golden_cross'] = ((g['ma_5'] > g['ma_10']) & (g['ma_5'].shift(1) <= g['ma_10'].shift(1))).astype(int)
    g['ma_death_cross'] = ((g['ma_5'] < g['ma_10']) & (g['ma_5'].shift(1) >= g['ma_10'].shift(1))).astype(int)
    g['ma_bullish'] = (g['close'] > g['ma_5']) & (g['ma_5'] > g['ma_10']) & (g['ma_10'] > g['ma_20'])
    g['ma_bearish'] = (g['close'] < g['ma_5']) & (g['ma_5'] < g['ma_10']) & (g['ma_10'] < g['ma_20'])

    # --- 布林带 (BOLL) ---
    for w in [20]:
        ma = g['close'].rolling(w).mean()
        std = g['close'].rolling(w).std()
        g[f'boll_upper_{w}'] = ma + 2 * std
        g[f'boll_lower_{w}'] = ma - 2 * std
        g[f'boll_width_{w}'] = (g[f'boll_upper_{w}'] - g[f'boll_lower_{w}']) / ma
        g[f'boll_position_{w}'] = (g['close'] - g[f'boll_lower_{w}']) / (g[f'boll_upper_{w}'] - g[f'boll_lower_{w}'])
        g[f'boll_squeeze'] = (g[f'boll_width_{w}'] < g[f'boll_width_{w}'].rolling(20).mean() * 0.8).astype(int)

    # --- RSI体系 ---
    delta = g['close'].diff()
    for w in [6, 14, 28]:
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / loss
        g[f'rsi_{w}'] = 100 - (100 / (1 + rs))
    g['rsi_diff'] = g['rsi_6'] - g['rsi_14']
    g['rsi_oversold'] = (g['rsi_14'] < 30).astype(int)
    g['rsi_overbought'] = (g['rsi_14'] > 70).astype(int)

    # --- MACD体系 ---
    ema_12 = g['close'].ewm(span=12, adjust=False).mean()
    ema_26 = g['close'].ewm(span=26, adjust=False).mean()
    g['macd_dif'] = ema_12 - ema_26
    g['macd_dea'] = g['macd_dif'].ewm(span=9, adjust=False).mean()
    g['macd_hist'] = g['macd_dif'] - g['macd_dea']
    g['macd_golden'] = ((g['macd_dif'] > g['macd_dea']) & (g['macd_dif'].shift(1) <= g['macd_dea'].shift(1))).astype(int)
    g['macd_death'] = ((g['macd_dif'] < g['macd_dea']) & (g['macd_dif'].shift(1) >= g['macd_dea'].shift(1))).astype(int)
    g['macd_divergence'] = np.where(
        (g['close'] > g['close'].shift(10)) & (g['macd_dif'] < g['macd_dif'].shift(10)), -1,
        np.where((g['close'] < g['close'].shift(10)) & (g['macd_dif'] > g['macd_dif'].shift(10)), 1, 0)
    )

    # --- KDJ ---
    low_min = g['low'].rolling(9).min()
    high_max = g['high'].rolling(9).max()
    rsv = (g['close'] - low_min) / (high_max - low_min) * 100
    g['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    g['kdj_d'] = g['kdj_k'].ewm(com=2, adjust=False).mean()
    g['kdj_j'] = 3 * g['kdj_k'] - 2 * g['kdj_d']
    g['kdj_golden'] = ((g['kdj_k'] > g['kdj_d']) & (g['kdj_k'].shift(1) <= g['kdj_d'].shift(1))).astype(int)

    # --- 波动率体系 ---
    for w in [5, 10, 20]:
        g[f'std_{w}d'] = g['return_1d'].rolling(w).std()
    g['atr_14'] = (g['high'] - g['low']).rolling(14).mean() / g['close']
    g['volatility_regime'] = (g['std_20d'] > g['std_20d'].rolling(60).mean()).astype(int)
    g['volatility_percentile'] = g['std_20d'].rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 60 else np.nan, raw=False)

    # --- 成交量体系 ---
    for w in [5, 20]:
        g[f'volume_ma_{w}'] = g['volume'].rolling(w).mean()
    g['volume_ratio'] = g['volume'] / g['volume_ma_5']
    g['volume_zscore'] = (g['volume'] - g['volume'].rolling(20).mean()) / g['volume'].rolling(20).std()
    g['amount_ma5'] = g['amount'].rolling(5).mean()
    g['amount_ratio'] = g['amount'] / g['amount_ma5']
    g['volume_price_corr_20'] = g['volume'].rolling(20).corr(g['close'])

    # --- 价格位置与支撑阻力 ---
    for w in [20, 60]:
        high = g['high'].rolling(w).max()
        low = g['low'].rolling(w).min()
        g[f'price_position_{w}'] = (g['close'] - low) / (high - low)
        g[f'dist_to_support_{w}'] = (g['close'] - low) / g['close']
        g[f'dist_to_resistance_{w}'] = (high - g['close']) / g['close']

    # --- 趋势斜率 ---
    def linear_slope(x):
        if len(x) < 5 or np.all(np.isnan(x)):
            return np.nan
        xi = np.arange(len(x))
        mask = ~np.isnan(x)
        if mask.sum() < 5:
            return np.nan
        return np.polyfit(xi[mask], x[mask], 1)[0] / np.mean(x[mask])

    for w in [10, 20]:
        g[f'trend_slope_{w}'] = g['close'].rolling(w).apply(linear_slope, raw=True)

    # --- OBV ---
    obv = [0]
    for i in range(1, len(g)):
        if g['close'].iloc[i] > g['close'].iloc[i-1]:
            obv.append(obv[-1] + g['volume'].iloc[i])
        elif g['close'].iloc[i] < g['close'].iloc[i-1]:
            obv.append(obv[-1] - g['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    g['obv'] = obv
    g['obv_ma20'] = g['obv'].rolling(20).mean()
    g['obv_ratio'] = g['obv'] / g['obv_ma20']

    # --- 振幅与价格行为 ---
    g['amplitude_pct'] = g['amplitude'] / 100
    g['body_pct'] = (g['close'] - g['open']) / g['open']
    g['upper_shadow'] = (g['high'] - g[['close', 'open']].max(axis=1)) / g['close']
    g['lower_shadow'] = (g[['close', 'open']].min(axis=1) - g['low']) / g['close']
    g['doji'] = (abs(g['close'] - g['open']) / (g['high'] - g['low']) < 0.1).astype(int)
    g['hammer'] = ((g['lower_shadow'] > 2 * abs(g['body_pct'])) & (g['upper_shadow'] < abs(g['body_pct']))).astype(int)

    # --- 动量与反转 ---
    g['momentum_10'] = g['close'] / g['close'].shift(10) - 1
    g['momentum_20'] = g['close'] / g['close'].shift(20) - 1
    g['roc_12'] = (g['close'] - g['close'].shift(12)) / g['close'].shift(12)

    # --- 威廉指标 (WR) ---
    for w in [10, 20]:
        high_w = g['high'].rolling(w).max()
        low_w = g['low'].rolling(w).min()
        g[f'wr_{w}'] = (high_w - g['close']) / (high_w - low_w) * 100

    # --- CCI ---
    tp = (g['high'] + g['low'] + g['close']) / 3
    for w in [14, 20]:
        ma_tp = tp.rolling(w).mean()
        md = tp.rolling(w).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        g[f'cci_{w}'] = (tp - ma_tp) / (0.015 * md)

    # --- 量价背离 ---
    g['price_volume_corr_20'] = g['close'].rolling(20).corr(g['volume'])

    # --- 时间特征 ---
    g['dayofweek'] = g['trade_date'].dt.dayofweek
    g['month'] = g['trade_date'].dt.month
    g['quarter'] = g['trade_date'].dt.quarter
    g['is_month_end'] = (g['trade_date'].dt.day >= 25).astype(int)
    g['is_week_start'] = (g['dayofweek'] == 0).astype(int)
    g['is_week_end'] = (g['dayofweek'] == 4).astype(int)

    # --- 统计特征 ---
    g['skew_20'] = g['return_1d'].rolling(20).skew()
    g['kurt_20'] = g['return_1d'].rolling(20).kurt()
    g['zscore_20'] = (g['close'] - g['close'].rolling(20).mean()) / g['close'].rolling(20).std()

    # --- 缺口特征 ---
    g['gap_up'] = (g['low'] > g['high'].shift(1)).astype(int)
    g['gap_down'] = (g['high'] < g['low'].shift(1)).astype(int)
    g['gap_size'] = (g['open'] - g['close'].shift(1)) / g['close'].shift(1)

    # --- 目标变量 ---
    g['target_next_day_return'] = g['close'].shift(-1) / g['close'] - 1
    g['target_direction'] = (g['target_next_day_return'] > 0).astype(int)
    g['target_3d_return'] = g['close'].shift(-3) / g['close'] - 1
    g['target_5d_return'] = g['close'].shift(-5) / g['close'] - 1

    return g

# 按股票分组计算特征
print("\n计算扩展特征 (80+ Alpha因子)...")
df_features = []
for symbol, group in df_all.groupby('symbol'):
    df_features.append(calc_features_v3(group))
df_features = pd.concat(df_features, ignore_index=True)

# 合并市场代理变量
if len(market_proxy) > 0:
    market_cols = ['trade_date', 'market_return_1d', 'market_return_5d', 'market_return_20d', 'market_trend']
    df_features = df_features.merge(market_proxy[market_cols], on='trade_date', how='left')

# 选择特征列
exclude_cols = ['trade_date', 'symbol', 'created_at',
                'target_next_day_return', 'target_direction',
                'target_3d_return', 'target_5d_return',
                'ma_5', 'ma_10', 'ma_20', 'ma_60',
                'ema_3', 'ema_5', 'ema_10', 'ema_20', 'ema_60',
                'boll_upper_20', 'boll_lower_20',
                'volume_ma_5', 'volume_ma_20', 'amount_ma5',
                'obv', 'obv_ma20',
                'high_w', 'low_w', 'ma_tp', 'md', 'tp',
                'low_min', 'high_max', 'rsv']

feature_cols = [c for c in df_features.columns if c not in exclude_cols]
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_features[c])]

print(f"特征列数量: {len(feature_cols)}")

# 去除缺失值
df_model = df_features.dropna(subset=feature_cols + ['target_direction']).copy()
print(f"\n去除缺失值后可用数据: {len(df_model)} 条")
print(f"各ETF数据量:")
print(df_model['symbol'].value_counts())

# ==================== 3. Walk-Forward 滚动训练 ====================
print("\n" + "=" * 70)
print("【3. Walk-Forward 滚动训练 + 多模型融合】")
print("=" * 70)

df_model = df_model.sort_values('trade_date').reset_index(drop=True)

TRAIN_WINDOW = 60    # 训练窗口: 60天 (~3个月)
TEST_WINDOW = 15     # 测试窗口: 15天
STEP = 15            # 滚动步长: 15天

min_date = df_model['trade_date'].min()
max_date = df_model['trade_date'].max()

windows = []
start = min_date + pd.Timedelta(days=TRAIN_WINDOW)
while start + pd.Timedelta(days=TEST_WINDOW) <= max_date:
    train_end = start - pd.Timedelta(days=1)
    test_end = start + pd.Timedelta(days=TEST_WINDOW) - pd.Timedelta(days=1)
    windows.append({
        'train_start': min_date,
        'train_end': train_end,
        'test_start': start,
        'test_end': test_end,
    })
    start += pd.Timedelta(days=STEP)

print(f"生成 {len(windows)} 个滚动窗口")

# 模型配置
model_configs = {}

if LGB_AVAILABLE:
    model_configs['LightGBM'] = {
        'clf': lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        ),
        'reg': lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
    }

model_configs['RandomForest'] = {
    'clf': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1),
    'reg': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)
}

# 逐窗口训练
window_results = []

for wi, w in enumerate(windows):
    train_mask = (df_model['trade_date'] >= w['train_start']) & (df_model['trade_date'] <= w['train_end'])
    test_mask = (df_model['trade_date'] >= w['test_start']) & (df_model['trade_date'] <= w['test_end'])

    train_df = df_model[train_mask].copy()
    test_df = df_model[test_mask].copy()

    if len(train_df) < 50 or len(test_df) < 5:
        continue

    X_train = train_df[feature_cols].values
    y_train_clf = train_df['target_direction'].values
    y_train_reg = train_df['target_next_day_return'].values

    X_test = test_df[feature_cols].values
    y_test_clf = test_df['target_direction'].values
    y_test_reg = test_df['target_next_day_return'].values

    valid_train = ~np.isnan(y_train_reg)
    valid_test = ~np.isnan(y_test_reg)

    X_train_reg = X_train[valid_train]
    y_train_reg_clean = y_train_reg[valid_train]
    X_test_reg = X_test[valid_test]
    y_test_reg_clean = y_test_reg[valid_test]

    scaler_w = StandardScaler()
    X_train_scaled = scaler_w.fit_transform(X_train)
    X_test_scaled = scaler_w.transform(X_test)

    window_pred = {
        'window': wi + 1,
        'test_dates': test_df['trade_date'].values,
        'test_symbols': test_df['symbol'].values,
        'y_test_clf': y_test_clf,
        'y_test_reg': y_test_reg,
        'test_df': test_df[['trade_date', 'symbol', 'close', 'target_next_day_return', 'target_direction',
                            'ma_bullish', 'ma_bearish', 'std_20d', 'atr_14', 'rsi_14', 'market_trend']].copy()
    }

    for model_name, models in model_configs.items():
        try:
            models['clf'].fit(X_train, y_train_clf)
            pred_clf = models['clf'].predict(X_test)
            prob_clf = models['clf'].predict_proba(X_test)[:, 1]
            window_pred[f'{model_name}_pred'] = pred_clf
            window_pred[f'{model_name}_prob'] = prob_clf
        except Exception as e:
            print(f"  窗口{wi+1} {model_name}分类失败: {e}")

        if len(X_train_reg) > 10 and len(X_test_reg) > 0:
            try:
                models['reg'].fit(X_train_reg, y_train_reg_clean)
                pred_reg = models['reg'].predict(X_test_reg)
                window_pred[f'{model_name}_reg'] = pred_reg
            except Exception as e:
                print(f"  窗口{wi+1} {model_name}回归失败: {e}")

    window_results.append(window_pred)
    print(f"  窗口 {wi+1}/{len(windows)} 完成 | 训练{len(train_df)}条 测试{len(test_df)}条")

print(f"\n完成 {len(window_results)} 个窗口的训练与预测")

# ==================== 4. 多模型融合 + 趋势过滤 + Kelly仓位 ====================
print("\n" + "=" * 70)
print("【4. 多策略融合: 分类+回归+趋势+波动率 + Kelly仓位】")
print("=" * 70)

# 汇总所有窗口的预测
fusion_data = []

for wr in window_results:
    probs = []
    for model_name in model_configs.keys():
        key = f'{model_name}_prob'
        if key in wr:
            probs.append(wr[key])

    if len(probs) >= 1:
        avg_prob = np.mean(probs, axis=0)
        test_df_w = wr['test_df'].copy()
        test_df_w['fusion_prob'] = avg_prob
        test_df_w['fusion_pred'] = (avg_prob > 0.5).astype(int)

        # 回归预测
        reg_preds = []
        for model_name in model_configs.keys():
            key = f'{model_name}_reg'
            if key in wr:
                reg_preds.append(wr[key])
        if len(reg_preds) > 0:
            test_df_w['fusion_reg'] = np.mean(reg_preds, axis=0)
        else:
            test_df_w['fusion_reg'] = 0

        fusion_data.append(test_df_w)

fusion_df = pd.concat(fusion_data, ignore_index=True)
print(f"融合数据: {len(fusion_df)} 条")

# --- 策略1: 基础融合模型 ---
fusion_df['signal_base'] = (fusion_df['fusion_prob'] > 0.5).astype(int)

# --- 策略2: 趋势过滤 ---
# MA多头排列时降低阈值(0.45)，空头时提高阈值(0.55)
def trend_adjusted_signal(row):
    if row['ma_bullish']:
        return 1 if row['fusion_prob'] > 0.45 else 0
    elif row['ma_bearish']:
        return 1 if row['fusion_prob'] > 0.60 else 0
    else:
        return 1 if row['fusion_prob'] > 0.50 else 0

fusion_df['signal_trend'] = fusion_df.apply(trend_adjusted_signal, axis=1)

# --- 策略3: 动态阈值 (基于近期胜率自适应) ---
# 计算滚动窗口内的模型表现，动态调整阈值
fusion_df = fusion_df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

for symbol in fusion_df['symbol'].unique():
    mask = fusion_df['symbol'] == symbol
    sym_data = fusion_df.loc[mask].copy()

    # 滚动20日胜率
    correct = (sym_data['fusion_pred'] == sym_data['target_direction']).astype(float)
    sym_data['rolling_accuracy'] = correct.rolling(20, min_periods=5).mean()

    # 动态阈值: 胜率高时更激进(0.45)，胜率低时更保守(0.55)
    def dynamic_threshold(acc):
        if pd.isna(acc):
            return 0.50
        return 0.50 - (acc - 0.5) * 0.2  # acc=0.6 -> 0.48, acc=0.4 -> 0.52

    sym_data['dynamic_threshold'] = sym_data['rolling_accuracy'].apply(dynamic_threshold)
    sym_data['signal_dynamic'] = (sym_data['fusion_prob'] > sym_data['dynamic_threshold']).astype(int)

    fusion_df.loc[mask, 'rolling_accuracy'] = sym_data['rolling_accuracy'].values
    fusion_df.loc[mask, 'dynamic_threshold'] = sym_data['dynamic_threshold'].values
    fusion_df.loc[mask, 'signal_dynamic'] = sym_data['signal_dynamic'].values

# --- 策略4: Kelly公式仓位管理 ---
# 计算每个symbol的近期胜率和盈亏比
kelly_data = []

for symbol in fusion_df['symbol'].unique():
    sym_df = fusion_df[fusion_df['symbol'] == symbol].sort_values('trade_date').copy()

    # 滚动计算胜率和盈亏比
    window = 20
    positions = []

    for i in range(len(sym_df)):
        if i < window:
            positions.append(0.5)  # 初始仓位50%
            continue

        recent = sym_df.iloc[max(0, i-window):i]
        wins = recent[recent['fusion_pred'] == recent['target_direction']]
        losses = recent[recent['fusion_pred'] != recent['target_direction']]

        p = len(wins) / len(recent) if len(recent) > 0 else 0.5

        win_returns = wins['target_next_day_return'].values
        loss_returns = losses['target_next_day_return'].values

        avg_win = np.mean(win_returns[win_returns > 0]) if len(win_returns[win_returns > 0]) > 0 else 0.01
        avg_loss = abs(np.mean(loss_returns[loss_returns < 0])) if len(loss_returns[loss_returns < 0]) > 0 else 0.01

        b = avg_win / avg_loss if avg_loss > 0 else 1

        # Kelly公式: f = (p*b - q) / b
        q = 1 - p
        kelly_f = (p * b - q) / b if b > 0 else 0
        kelly_f = max(0, min(1, kelly_f))  # 限制在0-1

        # 半Kelly (更保守)
        half_kelly = kelly_f * 0.5
        positions.append(half_kelly)

    sym_df['kelly_position'] = positions
    kelly_data.append(sym_df)

fusion_df = pd.concat(kelly_data, ignore_index=True)

# Kelly信号: 仓位 > 0 则买入
fusion_df['signal_kelly'] = (fusion_df['kelly_position'] > 0).astype(int)

# --- 策略5: 综合策略 (趋势过滤 + Kelly仓位) ---
fusion_df['signal_combined'] = fusion_df['signal_trend']
fusion_df['combined_position'] = fusion_df['signal_trend'] * fusion_df['kelly_position']

print(f"\n各策略信号统计:")
for s in ['signal_base', 'signal_trend', 'signal_dynamic', 'signal_kelly', 'signal_combined']:
    print(f"  {s}: 买入比例 {fusion_df[s].mean()*100:.1f}%")

# ==================== 5. 增强回测引擎 ====================
print("\n" + "=" * 70)
print("【5. 增强回测引擎 (多策略对比)】")
print("=" * 70)

FEE_RATE = 0.0001
SLIPPAGE = 0.0001
STOP_LOSS_ATR = 2.0

strategies_v3 = {}

for symbol in fusion_df['symbol'].unique():
    sym_data = fusion_df[fusion_df['symbol'] == symbol].sort_values('trade_date').reset_index(drop=True)
    if len(sym_data) < 10:
        continue

    actual_returns = sym_data['target_next_day_return'].values
    dates = sym_data['trade_date'].values
    atrs = sym_data['atr_14'].fillna(0.02).values
    n = len(actual_returns)

    # 基准
    benchmark_cum = np.cumprod(1 + actual_returns)

    # 策略定义: (信号列, 仓位列, 策略名)
    strategy_configs = [
        ('signal_base', None, '基础融合'),
        ('signal_trend', None, '趋势过滤'),
        ('signal_dynamic', None, '动态阈值'),
        ('signal_kelly', 'kelly_position', 'Kelly仓位'),
        ('signal_combined', 'combined_position', '综合策略'),
    ]

    results = {'dates': dates, 'actual': actual_returns, 'benchmark': benchmark_cum}

    for sig_col, pos_col, name in strategy_configs:
        signals = sym_data[sig_col].values.astype(float)
        positions = sym_data[pos_col].values if pos_col else signals

        cum = [1.0]
        in_position = False
        entry_price = 0

        for i in range(n):
            signal = signals[i]
            position = positions[i]
            daily_ret = actual_returns[i]
            atr = atrs[i]

            if signal == 1 and not in_position:
                in_position = True
                entry_price = 1.0
                cost = FEE_RATE + SLIPPAGE
                cum[-1] *= (1 - cost)
            elif signal == 0 and in_position:
                in_position = False
                cost = FEE_RATE + SLIPPAGE
                cum[-1] *= (1 - cost)

            if in_position:
                if daily_ret < -STOP_LOSS_ATR * atr:
                    daily_ret = -STOP_LOSS_ATR * atr
                    in_position = False

                ret = daily_ret * position
                cum.append(cum[-1] * (1 + ret))
            else:
                cum.append(cum[-1])

        results[f's_{name}'] = np.array(cum[1:])

    strategies_v3[symbol] = results

# ==================== 6. 回测结果汇总 ====================
print("\n>>> 多策略回测结果:\n")
print(f"{'ETF代码':<12} {'策略':<14} {'总收益率':<10} {'年化收益率':<10} {'最大回撤':<10} {'夏普比率':<8} {'胜率':<8} {'交易次数':<8}")
print("-" * 95)

all_returns_v3 = []

for symbol, data in strategies_v3.items():
    for s_name, s_key in [
        ('基准(持有)', 'benchmark'),
        ('基础融合', 's_基础融合'),
        ('趋势过滤', 's_趋势过滤'),
        ('动态阈值', 's_动态阈值'),
        ('Kelly仓位', 's_Kelly仓位'),
        ('综合策略', 's_综合策略'),
    ]:
        cum = data[s_key]
        total_ret = cum[-1] - 1
        n_days = len(cum)
        annual_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 and total_ret > -1 else -1

        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_dd = np.min(drawdown)

        daily_rets = np.diff(cum) / cum[:-1]
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0

        win_rate = np.sum(daily_rets > 0) / len(daily_rets) if len(daily_rets) > 0 else 0
        trades = np.sum(np.diff((daily_rets != 0).astype(int)) == 1) + 1

        print(f"{symbol:<12} {s_name:<14} {total_ret*100:>8.2f}% {annual_ret*100:>8.2f}% {max_dd*100:>8.2f}% {sharpe:>6.2f} {win_rate*100:>6.1f}% {trades:>6}")

        if s_name != '基准(持有)':
            all_returns_v3.append({
                'symbol': symbol, 'strategy': s_name,
                'total_return': total_ret, 'annual_return': annual_ret,
                'max_drawdown': max_dd, 'sharpe': sharpe,
                'win_rate': win_rate, 'trades': trades
            })

# ==================== 7. 可视化 ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Phase 2+3: 多策略融合 + Kelly仓位 + 趋势过滤', fontsize=14, fontweight='bold')

# 子图1: 各ETF累计收益对比 (综合策略 vs 基准)
ax1 = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (symbol, data) in enumerate(strategies_v3.items()):
    ax1.plot(data['dates'], data['benchmark'] - 1, '--', alpha=0.4, color=colors[i % len(colors)])
    ax1.plot(data['dates'], data['s_综合策略'] - 1, '-', color=colors[i % len(colors)], linewidth=2, label=symbol)
ax1.set_title('累计收益率: 综合策略 vs 基准')
ax1.set_xlabel('日期')
ax1.set_ylabel('累计收益率')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)

# 子图2: 各策略平均表现
ax2 = axes[0, 1]
if all_returns_v3:
    summary = pd.DataFrame(all_returns_v3).groupby('strategy')[['total_return', 'annual_return', 'max_drawdown', 'sharpe']].mean()
    x_pos = np.arange(len(summary))
    width = 0.2
    ax2.bar(x_pos - width, summary['total_return'] * 100, width, label='总收益率(%)', color='steelblue')
    ax2.bar(x_pos, summary['annual_return'] * 100, width, label='年化收益率(%)', color='seagreen')
    ax2.bar(x_pos + width, summary['max_drawdown'] * 100, width, label='最大回撤(%)', color='coral')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(summary.index, rotation=15, ha='right', fontsize=9)
    ax2.set_title('策略平均表现对比')
    ax2.set_ylabel('百分比 (%)')
    ax2.legend(fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 子图3: Kelly仓位变化
ax3 = axes[0, 2]
sample_symbol = list(strategies_v3.keys())[0]
sample_data = fusion_df[fusion_df['symbol'] == sample_symbol].sort_values('trade_date')
ax3.plot(sample_data['trade_date'], sample_data['kelly_position'], color='steelblue', linewidth=1.5)
ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='半Kelly基准')
ax3.set_title(f'{sample_symbol} Kelly仓位变化')
ax3.set_xlabel('日期')
ax3.set_ylabel('仓位比例')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4: 动态阈值变化
ax4 = axes[1, 0]
ax4.plot(sample_data['trade_date'], sample_data['dynamic_threshold'], color='green', linewidth=1.5, label='动态阈值')
ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='固定阈值0.5')
ax4.set_title(f'{sample_symbol} 动态阈值变化')
ax4.set_xlabel('日期')
ax4.set_ylabel('阈值')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 子图5: 各ETF策略超额收益热力图
ax5 = axes[1, 1]
if all_returns_v3:
    pivot_df = pd.DataFrame(all_returns_v3).pivot_table(
        index='symbol', columns='strategy', values='total_return'
    )
    im = ax5.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto')
    ax5.set_xticks(range(len(pivot_df.columns)))
    ax5.set_xticklabels(pivot_df.columns, rotation=45, ha='right', fontsize=9)
    ax5.set_yticks(range(len(pivot_df.index)))
    ax5.set_yticklabels(pivot_df.index, fontsize=9)
    ax5.set_title('各ETF策略总收益率热力图')
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            ax5.text(j, i, f'{pivot_df.values[i,j]*100:.1f}%',
                    ha='center', va='center', color='white' if abs(pivot_df.values[i,j]) > 0.05 else 'black',
                    fontsize=8)

# 子图6: 策略夏普比率对比
ax6 = axes[1, 2]
if all_returns_v3:
    sharpe_df = pd.DataFrame(all_returns_v3).groupby('strategy')['sharpe'].mean()
    colors_sharpe = ['steelblue'] * len(sharpe_df)
    colors_sharpe[sharpe_df.idxmax()] = 'coral'
    ax6.bar(sharpe_df.index, sharpe_df.values, color=colors_sharpe)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax6.set_title('策略平均夏普比率')
    ax6.set_ylabel('夏普比率')
    ax6.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(r'C:/Users/HY/PycharmProjects/QuanTrade/phase23_strategy_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存: C:/Users/HY/PycharmProjects/QuanTrade/phase23_strategy_comparison.png")

# ==================== 8. 保存结果到数据库 ====================
conn = sqlite3.connect(db_path)

# 保存融合预测结果
fusion_df[['trade_date', 'symbol', 'target_direction', 'target_next_day_return',
            'fusion_prob', 'fusion_pred', 'fusion_reg',
            'signal_base', 'signal_trend', 'signal_dynamic', 'signal_kelly', 'signal_combined',
            'kelly_position', 'combined_position', 'dynamic_threshold',
            'ma_bullish', 'ma_bearish', 'market_trend']].to_sql(
    'strategy_signals_v3', conn, if_exists='replace', index=False)

# 保存回测结果
if all_returns_v3:
    pd.DataFrame(all_returns_v3).to_sql('backtest_results_v3', conn, if_exists='replace', index=False)

conn.close()
print(f"策略信号已保存: strategy_signals_v3")
print(f"回测结果已保存: backtest_results_v3")

# ==================== 9. 各ETF详细分析 ====================
print("\n" + "=" * 70)
print("【9. 各ETF详细回测分析】")
print("=" * 70)

for symbol, data in strategies_v3.items():
    print(f"\n>>> {symbol}")
    print(f"  测试期交易天数: {len(data['actual'])}")
    print(f"  基准总收益:       {(data['benchmark'][-1]-1)*100:+.2f}%")
    print(f"  基础融合:         {(data['s_基础融合'][-1]-1)*100:+.2f}%")
    print(f"  趋势过滤:         {(data['s_趋势过滤'][-1]-1)*100:+.2f}%")
    print(f"  动态阈值:         {(data['s_动态阈值'][-1]-1)*100:+.2f}%")
    print(f"  Kelly仓位:        {(data['s_Kelly仓位'][-1]-1)*100:+.2f}%")
    print(f"  综合策略:         {(data['s_综合策略'][-1]-1)*100:+.2f}%")

    excess = (data['s_综合策略'][-1] - data['benchmark'][-1]) * 100
    print(f"  综合策略超额:     {excess:+.2f}%")

print("\n" + "=" * 70)
print("【Phase 2+3 改进完成】")
print("=" * 70)
print(f"\n改进总结:")
print(f"  - 特征数量: 66 → {len(feature_cols)} 个")
print(f"  - 策略体系: 基础融合 → 5种策略 (趋势过滤/动态阈值/Kelly/综合)")
print(f"  - 仓位管理: 满仓/空仓 → Kelly公式动态仓位 (0-100%)")
print(f"  - 止损机制: ATR动态止损 ({STOP_LOSS_ATR}x ATR)")
print(f"  - 成本模拟: 手续费{FEE_RATE*100:.2f}% + 滑点{SLIPPAGE*100:.2f}%")
