"""
Feature Engineering v4 — FinSTaR-inspired enhancements
- 120-day long-window features (drawdown, vol regime, trend, support/resistance, event detection)
- Assessment state labels (deterministic computation)
- Preserves all original 45+ compact features
- Saves to features_v4 table
"""
import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'

# ==================== 1. Load all data ====================
print("=" * 70)
print("【Feature Engineering v4 — Loading data】")
print("=" * 70)

conn = sqlite3.connect(DB_PATH)
df_all = pd.read_sql_query(
    "SELECT * FROM daily_prices ORDER BY symbol, trade_date", conn
)
conn.close()

df_all['trade_date'] = pd.to_datetime(df_all['trade_date'], format='mixed')
df_all = df_all.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

print(f"Total records: {len(df_all)}")
print(f"Symbols: {df_all['symbol'].unique().tolist()}")
for sym, cnt in df_all['symbol'].value_counts().sort_index().items():
    print(f"  {sym}: {cnt} records")


# ==================== 2. Original compact features ====================
def calc_original_features(g):
    """Original 45+ features from phase23_compact"""
    g = g.sort_values('trade_date').copy()
    n = len(g)
    if n < 20:
        return g

    # Returns
    g['return_1d'] = g['close'].pct_change()
    for w in [2, 3, 5, 10]:
        g[f'return_{w}d'] = g['close'].pct_change(w)

    # Moving averages
    for w in [3, 5, 10, 20]:
        g[f'ma_{w}'] = g['close'].rolling(w).mean()
        g[f'ma_dist_{w}'] = (g['close'] - g[f'ma_{w}']) / g[f'ma_{w}']
        g[f'ema_{w}'] = g['close'].ewm(span=w, adjust=False).mean()

    g['ma_bullish'] = (g['close'] > g['ma_5']) & (g['ma_5'] > g['ma_10']) & (g['ma_10'] > g['ma_20'])
    g['ma_bearish'] = (g['close'] < g['ma_5']) & (g['ma_5'] < g['ma_10']) & (g['ma_10'] < g['ma_20'])

    # Bollinger (10-day)
    ma10 = g['close'].rolling(10).mean()
    std10 = g['close'].rolling(10).std()
    g['boll_upper'] = ma10 + 2 * std10
    g['boll_lower'] = ma10 - 2 * std10
    g['boll_position'] = (g['close'] - g['boll_lower']) / (g['boll_upper'] - g['boll_lower'])

    # RSI (14-day)
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    g['rsi_14'] = 100 - (100 / (1 + rs))
    g['rsi_oversold'] = (g['rsi_14'] < 30).astype(int)
    g['rsi_overbought'] = (g['rsi_14'] > 70).astype(int)

    # MACD
    ema_12 = g['close'].ewm(span=12, adjust=False).mean()
    ema_26 = g['close'].ewm(span=26, adjust=False).mean()
    g['macd_dif'] = ema_12 - ema_26
    g['macd_dea'] = g['macd_dif'].ewm(span=9, adjust=False).mean()
    g['macd_hist'] = g['macd_dif'] - g['macd_dea']

    # KDJ
    low_min = g['low'].rolling(9).min()
    high_max = g['high'].rolling(9).max()
    rsv = (g['close'] - low_min) / (high_max - low_min) * 100
    g['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    g['kdj_d'] = g['kdj_k'].ewm(com=2, adjust=False).mean()
    g['kdj_j'] = 3 * g['kdj_k'] - 2 * g['kdj_d']

    # Volatility
    for w in [5, 10]:
        g[f'std_{w}d'] = g['return_1d'].rolling(w).std()
    g['atr_14'] = (g['high'] - g['low']).rolling(14).mean() / g['close']

    # Volume
    g['volume_ma_5'] = g['volume'].rolling(5).mean()
    g['volume_ratio'] = g['volume'] / g['volume_ma_5']
    g['amount_ma5'] = g['amount'].rolling(5).mean()
    g['amount_ratio'] = g['amount'] / g['amount_ma5']

    # Price position (20-day)
    high20 = g['high'].rolling(20).max()
    low20 = g['low'].rolling(20).min()
    g['price_position_20'] = (g['close'] - low20) / (high20 - low20)

    # Trend slope (10-day)
    def linear_slope(x):
        if len(x) < 5 or np.all(np.isnan(x)):
            return np.nan
        xi = np.arange(len(x))
        mask = ~np.isnan(x)
        if mask.sum() < 5:
            return np.nan
        return np.polyfit(xi[mask], x[mask], 1)[0] / np.mean(x[mask])
    g['trend_slope_10'] = g['close'].rolling(10).apply(linear_slope, raw=True)

    # OBV
    obv = [0]
    for i in range(1, len(g)):
        if g['close'].iloc[i] > g['close'].iloc[i-1]:
            obv.append(obv[-1] + g['volume'].iloc[i])
        elif g['close'].iloc[i] < g['close'].iloc[i-1]:
            obv.append(obv[-1] - g['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    g['obv'] = obv
    g['obv_ma10'] = g['obv'].rolling(10).mean()
    g['obv_ratio'] = g['obv'] / g['obv_ma10']

    # Price behavior
    g['body_pct'] = (g['close'] - g['open']) / g['open']
    g['upper_shadow'] = (g['high'] - g[['close', 'open']].max(axis=1)) / g['close']
    g['lower_shadow'] = (g[['close', 'open']].min(axis=1) - g['low']) / g['close']

    # Time features
    g['dayofweek'] = g['trade_date'].dt.dayofweek
    g['month'] = g['trade_date'].dt.month
    g['is_week_end'] = (g['dayofweek'] == 4).astype(int)

    return g


# ==================== 3. New 120-day long-window features (FinSTaR inspired) ====================
def calc_long_window_features(g):
    """
    120-day long window features inspired by FinSTaR paper:
    - drawdown_120d: drawdown from 120-day peak
    - vol_regime_ratio: recent(20d) vol / long(120d) vol
    - trend_120d_return: 120-day cumulative return
    - support_proximity_60d: distance to 60-day low
    - resistance_proximity_60d: distance to 60-day high
    - event_zscore_20d: z-score of daily return vs 20-day window
    - post_event_momentum_5d: direction after |z|>2.5 event
    - drawdown_recovery_prob: historical probability of recovery after >5% drawdown
    """
    g = g.sort_values('trade_date').copy()
    n = len(g)
    if n < 120:
        # Pad with NaN for short series
        g['drawdown_120d'] = np.nan
        g['vol_regime_ratio'] = np.nan
        g['trend_120d_return'] = np.nan
        g['support_proximity_60d'] = np.nan
        g['resistance_proximity_60d'] = np.nan
        g['event_zscore_20d'] = np.nan
        g['post_event_momentum_5d'] = np.nan
        g['drawdown_recovery_prob'] = np.nan
        return g

    # 1. Drawdown from 120-day peak (Assessment task: Drawdown)
    peak_120 = g['close'].rolling(120, min_periods=60).max()
    g['drawdown_120d'] = (g['close'] - peak_120) / peak_120

    # 2. Volatility regime ratio (Assessment task: Volatility Regime)
    ret = g['close'].pct_change()
    vol_20 = ret.rolling(20, min_periods=10).std() * np.sqrt(252)  # annualized
    vol_120 = ret.rolling(120, min_periods=60).std() * np.sqrt(252)
    g['vol_regime_ratio'] = vol_20 / vol_120.replace(0, np.nan)

    # 3. 120-day trend return (Assessment task: Trend Direction)
    price_120_ago = g['close'].shift(119)
    g['trend_120d_return'] = (g['close'] - price_120_ago) / price_120_ago

    # 4. Support/Resistance proximity (Prediction task: Support/Resistance)
    high_60 = g['high'].rolling(60, min_periods=30).max()
    low_60 = g['low'].rolling(60, min_periods=30).min()
    g['resistance_proximity_60d'] = (high_60 - g['close']) / g['close']
    g['support_proximity_60d'] = (g['close'] - low_60) / g['close']

    # 5. Event detection z-score (Prediction task: Event Response)
    mean_20 = ret.rolling(20, min_periods=10).mean()
    std_20 = ret.rolling(20, min_periods=10).std()
    g['event_zscore_20d'] = (ret - mean_20) / std_20.replace(0, np.nan)

    # 6. Post-event momentum (5-day forward return after |z|>2.5)
    is_event = g['event_zscore_20d'].abs() > 2.5
    fwd_ret_5 = g['close'].shift(-5) / g['close'] - 1
    g['post_event_momentum_5d'] = np.where(is_event, np.sign(fwd_ret_5), 0)

    # 7. Drawdown recovery probability (historical stat within window)
    # For each day with drawdown > 5%, compute probability that price recovers >3% within 20 days
    dd = g['drawdown_120d']
    recovery_prob = []
    for i in range(len(g)):
        if i < 120 or pd.isna(dd.iloc[i]) or dd.iloc[i] > -0.05:
            recovery_prob.append(np.nan)
            continue
        # Look back in available history for similar drawdowns
        window_start = max(0, i - 120)
        similar_dd = []
        for j in range(window_start, i):
            if not pd.isna(dd.iloc[j]) and dd.iloc[j] <= -0.05:
                # Check if recovered >3% within 20 days
                if j + 20 < len(g):
                    peak = g['close'].iloc[j:j+1].values[0]  # current price at j
                    # Actually recovery means price goes up 3% from the trough
                    trough = g['close'].iloc[j]
                    max_future = g['close'].iloc[j+1:min(j+21, len(g))].max()
                    recovered = (max_future - trough) / trough > 0.03
                    similar_dd.append(1 if recovered else 0)
        if similar_dd:
            recovery_prob.append(np.mean(similar_dd))
        else:
            recovery_prob.append(np.nan)
    g['drawdown_recovery_prob'] = recovery_prob

    # Additional long-window features
    # 8. 60-day price position (like 20-day but longer)
    high_60 = g['high'].rolling(60, min_periods=30).max()
    low_60 = g['low'].rolling(60, min_periods=30).min()
    g['price_position_60'] = (g['close'] - low_60) / (high_60 - low_60)

    # 9. 60-day return
    g['return_60d'] = g['close'].pct_change(60)

    # 10. Volatility of volatility (vol clustering indicator)
    g['vol_of_vol_20'] = vol_20.rolling(20, min_periods=10).std()

    # 11. Max drawdown depth in last 60 days
    rolling_peak = g['close'].rolling(60, min_periods=30).max()
    g['max_dd_60d'] = (g['close'] - rolling_peak) / rolling_peak

    # 12. Time since peak (days since 120-day high)
    is_peak = g['close'] == g['close'].rolling(120, min_periods=60).max()
    g['days_since_peak'] = is_peak.iloc[::-1].cumsum().iloc[::-1]  # rough approximation
    # Better: count days since last peak
    days_since = []
    last_peak_idx = -1
    for i in range(len(g)):
        if is_peak.iloc[i]:
            last_peak_idx = i
        days_since.append(i - last_peak_idx if last_peak_idx >= 0 else np.nan)
    g['days_since_peak'] = days_since

    return g


# ==================== 4. Assessment state labels (deterministic computation) ====================
def calc_assessment_states(g):
    """
    Assessment states — deterministic labels computed directly from observable data.
    These become features for the Prediction module.
    """
    g = g.copy()

    # Trend state: 5-class based on 120-day cumulative return
    r120 = g['trend_120d_return']
    conditions = [
        (r120 > 0.20, 'strong_uptrend'),
        (r120 > 0.05, 'mild_uptrend'),
        (r120 > -0.05, 'sideways'),
        (r120 > -0.20, 'mild_downtrend'),
        (r120 <= -0.20, 'strong_downtrend')
    ]
    g['trend_state'] = 'sideways'
    for cond, val in conditions:
        g.loc[cond, 'trend_state'] = val

    # Volatility state: 3-class based on vol_regime_ratio
    vr = g['vol_regime_ratio']
    g['vol_state'] = 'normal'
    g.loc[vr < 0.6, 'vol_state'] = 'low'
    g.loc[vr > 1.6, 'vol_state'] = 'high'

    # Drawdown state: 4-class based on drawdown_120d
    dd = g['drawdown_120d']
    g['drawdown_state'] = 'near_peak'
    g.loc[dd <= -0.03, 'drawdown_state'] = 'pullback'
    g.loc[dd <= -0.10, 'drawdown_state'] = 'correction'
    g.loc[dd <= -0.20, 'drawdown_state'] = 'severe_decline'

    # Support/Resistance state: 3-class
    res_prox = g['resistance_proximity_60d']
    sup_prox = g['support_proximity_60d']
    g['sr_state'] = 'middle'
    g.loc[res_prox < 0.03, 'sr_state'] = 'near_resistance'
    g.loc[sup_prox < 0.03, 'sr_state'] = 'near_support'

    # Event state: binary
    g['event_state'] = (g['event_zscore_20d'].abs() > 2.5).astype(int)

    # Encode states as numeric for ML
    trend_map = {'strong_downtrend': -2, 'mild_downtrend': -1, 'sideways': 0,
                 'mild_uptrend': 1, 'strong_uptrend': 2}
    vol_map = {'low': -1, 'normal': 0, 'high': 1}
    dd_map = {'near_peak': 0, 'pullback': 1, 'correction': 2, 'severe_decline': 3}
    sr_map = {'near_support': -1, 'middle': 0, 'near_resistance': 1}

    g['trend_state_num'] = g['trend_state'].map(trend_map)
    g['vol_state_num'] = g['vol_state'].map(vol_map)
    g['drawdown_state_num'] = g['drawdown_state'].map(dd_map)
    g['sr_state_num'] = g['sr_state'].map(sr_map)

    return g


# ==================== 5. Multi-horizon targets (for Prediction module) ====================
def calc_targets(g):
    """Multi-horizon future return directions for three-scenario training"""
    g = g.copy()
    for h in [5, 10, 20]:
        g[f'target_return_{h}d'] = g['close'].shift(-h) / g['close'] - 1
        g[f'target_direction_{h}d'] = (g[f'target_return_{h}d'] > 0).astype(int)

    # Three-scenario labels (for 10-day horizon)
    ret_10 = g['target_return_10d']
    g['scenario_label_10d'] = 'base'  # base = moderate move / continuation
    g.loc[ret_10 < -0.03, 'scenario_label_10d'] = 'adverse'  # significant decline
    g.loc[ret_10 > 0.05, 'scenario_label_10d'] = 'favorable'  # significant rally

    # Also for 5-day
    ret_5 = g['target_return_5d']
    g['scenario_label_5d'] = 'base'
    g.loc[ret_5 < -0.02, 'scenario_label_5d'] = 'adverse'
    g.loc[ret_5 > 0.03, 'scenario_label_5d'] = 'favorable'

    return g


# ==================== 6. Run feature engineering ====================
print("\n" + "=" * 70)
print("【Computing features for all symbols...】")
print("=" * 70)

all_features = []
for symbol, group in df_all.groupby('symbol'):
    print(f"  Processing {symbol} ({len(group)} records)...")
    g = calc_original_features(group)
    g = calc_long_window_features(g)
    g = calc_assessment_states(g)
    g = calc_targets(g)
    all_features.append(g)

df_features = pd.concat(all_features, ignore_index=True)

# ==================== 7. Save to database ====================
print("\n" + "=" * 70)
print("【Saving to features_v4 table...】")
print("=" * 70)

# Select columns to save
save_cols = ['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']

# Original features
orig_feature_cols = [
    'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d',
    'ma_dist_3', 'ma_dist_5', 'ma_dist_10', 'ma_dist_20',
    'ma_bullish', 'ma_bearish',
    'boll_position',
    'rsi_14', 'rsi_oversold', 'rsi_overbought',
    'macd_dif', 'macd_dea', 'macd_hist',
    'kdj_k', 'kdj_d', 'kdj_j',
    'std_5d', 'std_10d', 'atr_14',
    'volume_ratio', 'amount_ratio',
    'price_position_20', 'trend_slope_10',
    'obv_ratio',
    'body_pct', 'upper_shadow', 'lower_shadow',
    'dayofweek', 'month', 'is_week_end'
]

# New long-window features
new_feature_cols = [
    'drawdown_120d', 'vol_regime_ratio', 'trend_120d_return',
    'support_proximity_60d', 'resistance_proximity_60d',
    'event_zscore_20d', 'post_event_momentum_5d', 'drawdown_recovery_prob',
    'price_position_60', 'return_60d', 'vol_of_vol_20', 'max_dd_60d', 'days_since_peak'
]

# Assessment states
assessment_cols = [
    'trend_state', 'vol_state', 'drawdown_state', 'sr_state', 'event_state',
    'trend_state_num', 'vol_state_num', 'drawdown_state_num', 'sr_state_num'
]

# Targets
target_cols = [
    'target_return_5d', 'target_direction_5d',
    'target_return_10d', 'target_direction_10d',
    'target_return_20d', 'target_direction_20d',
    'scenario_label_5d', 'scenario_label_10d'
]

all_cols = save_cols + orig_feature_cols + new_feature_cols + assessment_cols + target_cols
all_cols = [c for c in all_cols if c in df_features.columns]

df_save = df_features[all_cols].copy()

# Convert boolean to int
for c in df_save.columns:
    if df_save[c].dtype == bool:
        df_save[c] = df_save[c].astype(int)

conn = sqlite3.connect(DB_PATH)

# Drop and recreate table
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS features_v4")

# Build CREATE TABLE
def sql_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return 'INTEGER'
    elif pd.api.types.is_float_dtype(dtype):
        return 'REAL'
    else:
        return 'TEXT'

create_sql = "CREATE TABLE features_v4 (\n"
create_sql += "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
for col in all_cols:
    create_sql += f"    {col} {sql_type(df_save[col].dtype)},\n"
create_sql = create_sql.rstrip(',\n') + "\n)"
cursor.execute(create_sql)

# Insert data
df_save.to_sql('features_v4', conn, if_exists='append', index=False)

conn.commit()
conn.close()

print(f"\nSaved {len(df_save)} records to features_v4")
print(f"  Columns: {len(all_cols)}")
print(f"  Original features: {len(orig_feature_cols)}")
print(f"  New long-window features: {len(new_feature_cols)}")
print(f"  Assessment states: {len(assessment_cols)}")
print(f"  Targets: {len(target_cols)}")

# Print sample assessment states
print("\n【Assessment state distribution (sample)】")
sample = df_save[df_save['symbol'] == '000002.SZ'].dropna(subset=['trend_state'])
if len(sample) > 0:
    print(f"  Trend states: {sample['trend_state'].value_counts().to_dict()}")
    print(f"  Vol states: {sample['vol_state'].value_counts().to_dict()}")
    print(f"  Drawdown states: {sample['drawdown_state'].value_counts().to_dict()}")
    print(f"  S/R states: {sample['sr_state'].value_counts().to_dict()}")

print("\n" + "=" * 70)
print("【Feature Engineering v4 COMPLETE】")
print("=" * 70)
