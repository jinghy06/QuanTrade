"""
Train Model v4 — Assessment/Prediction Separation + Three-Scenario Output
- OOD split: 4 large-cap stocks (training) vs 5 ETFs (testing)
- Assessment: deterministic state labels (already computed in features_v4)
- Prediction: LightGBM + RF multi-class classifier (3 scenarios)
- Saves models to models/v4/ and predictions to predictions_v4 table
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

print("=" * 70)
print("【Train Model v4 — Assessment/Prediction Separation】")
print("=" * 70)

DB_PATH = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
MODEL_DIR = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/models/v4'
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== 1. Load features_v4 ====================
print("\n【1. Loading features_v4...】")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM features_v4 ORDER BY symbol, trade_date", conn)
conn.close()

df['trade_date'] = pd.to_datetime(df['trade_date'])
print(f"Total records: {len(df)}")

# ==================== 2. OOD Split ====================
TRAIN_STOCKS = ['000002.SZ', '000333.SZ', '000568.SZ', '000651.SZ',
                '600519.SH', '300750.SZ', '601318.SH', '600036.SH',
                '000858.SZ', '002594.SZ', '600900.SH', '601012.SH',
                '600276.SH', '000725.SZ']
TEST_ETFS = ['562500.SH', '588200.SH', '588790.SH', '159382.SZ', '159241.SZ']

df_train = df[df['symbol'].isin(TRAIN_STOCKS)].copy()
df_test = df[df['symbol'].isin(TEST_ETFS)].copy()

print(f"\n【2. OOD Split】")
print(f"Training stocks: {TRAIN_STOCKS}")
print(f"  Records: {len(df_train)}")
print(f"  Date range: {df_train['trade_date'].min().date()} ~ {df_train['trade_date'].max().date()}")
print(f"Testing ETFs: {TEST_ETFS}")
print(f"  Records: {len(df_test)}")
print(f"  Date range: {df_test['trade_date'].min().date()} ~ {df_test['trade_date'].max().date()}")

# ==================== 3. Feature columns ====================
# Original numeric features
orig_features = [
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
new_features = [
    'drawdown_120d', 'vol_regime_ratio', 'trend_120d_return',
    'support_proximity_60d', 'resistance_proximity_60d',
    'event_zscore_20d', 'post_event_momentum_5d', 'drawdown_recovery_prob',
    'price_position_60', 'return_60d', 'vol_of_vol_20', 'max_dd_60d', 'days_since_peak'
]

# Assessment state numeric features (deterministic computation)
assessment_features = [
    'trend_state_num', 'vol_state_num', 'drawdown_state_num', 'sr_state_num', 'event_state'
]

# News sentiment features
sentiment_features = [
    'sentiment_1d', 'sentiment_3d', 'sentiment_7d',
    'major_events_1d', 'major_events_3d',
    'news_count_1d', 'news_count_3d'
]

feature_cols = orig_features + new_features + assessment_features + sentiment_features
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"\nFeature columns ({len(feature_cols)}):")
print(f"  Original: {len(orig_features)}")
print(f"  New long-window: {len(new_features)}")
print(f"  Assessment states: {len(assessment_features)}")
print(f"  Sentiment features: {len([c for c in sentiment_features if c in df.columns])}")

# ==================== 4. Target: Three-scenario label (10-day horizon) ====================
print("\n【4. Three-scenario target distribution】")

# Clean data
df_train_clean = df_train.dropna(subset=feature_cols + ['scenario_label_10d']).copy()
df_test_clean = df_test.dropna(subset=feature_cols + ['scenario_label_10d']).copy()

print(f"Train after dropna: {len(df_train_clean)}")
print(f"Test after dropna: {len(df_test_clean)}")

print("\nTrain scenario distribution:")
print(df_train_clean['scenario_label_10d'].value_counts())
print("\nTest scenario distribution:")
print(df_test_clean['scenario_label_10d'].value_counts())

# Encode scenario labels
scenario_map = {'adverse': 0, 'base': 1, 'favorable': 2}
df_train_clean['scenario_target'] = df_train_clean['scenario_label_10d'].map(scenario_map)
df_test_clean['scenario_target'] = df_test_clean['scenario_label_10d'].map(scenario_map)

# Also binary target for comparison
df_train_clean['binary_target'] = df_train_clean['target_direction_10d']
df_test_clean['binary_target'] = df_test_clean['target_direction_10d']

X_train = df_train_clean[feature_cols].values
y_scenario_train = df_train_clean['scenario_target'].values
y_binary_train = df_train_clean['binary_target'].values

X_test = df_test_clean[feature_cols].values
y_scenario_test = df_test_clean['scenario_target'].values
y_binary_test = df_test_clean['binary_target'].values

# ==================== 5. Train RandomForest (Three-scenario) ====================
print("\n" + "=" * 70)
print("【5. Training RandomForest — Three-Scenario Classification】")
print("=" * 70)

rf_scenario = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_scenario.fit(X_train, y_scenario_train)

# Predict probabilities
rf_scenario_proba = rf_scenario.predict_proba(X_test)
rf_scenario_pred = rf_scenario.predict(X_test)

print("\nRF Three-Scenario Test Results:")
print(classification_report(y_scenario_test, rf_scenario_pred,
                            target_names=['adverse', 'base', 'favorable']))

# ==================== 6. Train LightGBM (Three-scenario) ====================
print("\n" + "=" * 70)
print("【6. Training LightGBM — Three-Scenario Classification】")
print("=" * 70)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM not available, skipping.")

if LGB_AVAILABLE:
    lgb_scenario = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    lgb_scenario.fit(X_train, y_scenario_train)

    lgb_scenario_proba = lgb_scenario.predict_proba(X_test)
    lgb_scenario_pred = lgb_scenario.predict(X_test)

    print("\nLGBM Three-Scenario Test Results:")
    print(classification_report(y_scenario_test, lgb_scenario_pred,
                                target_names=['adverse', 'base', 'favorable']))

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf_scenario.feature_importances_,
        'lgb_importance': lgb_scenario.feature_importances_
    }).sort_values('lgb_importance', ascending=False)
    print("\nTop 15 features (LGBM importance):")
    print(importance.head(15).to_string(index=False))
else:
    lgb_scenario = None
    lgb_scenario_proba = None
    lgb_scenario_pred = None
    importance = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf_scenario.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    print("\nTop 15 features (RF importance):")
    print(importance.head(15).to_string(index=False))

# ==================== 7. Fusion: Average probabilities ====================
print("\n" + "=" * 70)
print("【7. Fusion Model — Average Probabilities】")
print("=" * 70)

if LGB_AVAILABLE:
    fusion_proba = (rf_scenario_proba + lgb_scenario_proba) / 2
else:
    fusion_proba = rf_scenario_proba

fusion_pred = np.argmax(fusion_proba, axis=1)

print("\nFusion Three-Scenario Test Results:")
print(classification_report(y_scenario_test, fusion_pred,
                            target_names=['adverse', 'base', 'favorable']))

# ==================== 8. Binary comparison (up/down) ====================
print("\n" + "=" * 70)
print("【8. Binary Classification Comparison (Up/Down)】")
print("=" * 70)

rf_binary = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=10,
    min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1
)
rf_binary.fit(X_train, y_binary_train)
rf_binary_pred = rf_binary.predict(X_test)

print("\nRF Binary Test Results:")
print(f"  Accuracy: {accuracy_score(y_binary_test, rf_binary_pred):.4f}")
print(classification_report(y_binary_test, rf_binary_pred, target_names=['down', 'up']))

if LGB_AVAILABLE:
    lgb_binary = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, class_weight='balanced', random_state=42, verbose=-1
    )
    lgb_binary.fit(X_train, y_binary_train)
    lgb_binary_pred = lgb_binary.predict(X_test)
    print("\nLGBM Binary Test Results:")
    print(f"  Accuracy: {accuracy_score(y_binary_test, lgb_binary_pred):.4f}")

# ==================== 9. Save models ====================
print("\n" + "=" * 70)
print("【9. Saving models...】")
print("=" * 70)

# Save RF scenario model
with open(os.path.join(MODEL_DIR, 'rf_scenario.pkl'), 'wb') as f:
    pickle.dump(rf_scenario, f)
print(f"  Saved: rf_scenario.pkl")

# Save RF binary model
with open(os.path.join(MODEL_DIR, 'rf_binary.pkl'), 'wb') as f:
    pickle.dump(rf_binary, f)
print(f"  Saved: rf_binary.pkl")

if LGB_AVAILABLE:
    with open(os.path.join(MODEL_DIR, 'lgb_scenario.pkl'), 'wb') as f:
        pickle.dump(lgb_scenario, f)
    print(f"  Saved: lgb_scenario.pkl")
    with open(os.path.join(MODEL_DIR, 'lgb_binary.pkl'), 'wb') as f:
        pickle.dump(lgb_binary, f)
    print(f"  Saved: lgb_binary.pkl")

# Save feature list
with open(os.path.join(MODEL_DIR, 'feature_cols.json'), 'w') as f:
    json.dump(feature_cols, f)
print(f"  Saved: feature_cols.json")

# Save scenario weights
scenario_weights = {
    'adverse_weight': 0.0,   # if adverse prob > 0.4, go to cash/hedge
    'base_weight': 0.5,      # normal position
    'favorable_weight': 1.0, # if favorable prob > 0.3, full position
    'threshold_adverse': 0.4,
    'threshold_favorable': 0.3
}
with open(os.path.join(MODEL_DIR, 'scenario_weights.json'), 'w') as f:
    json.dump(scenario_weights, f)
print(f"  Saved: scenario_weights.json")

# ==================== 10. Save predictions to database ====================
print("\n【10. Saving predictions to predictions_v4...】")

# Build prediction dataframe
df_pred = df_test_clean[['trade_date', 'symbol', 'close', 'target_return_10d',
                         'target_direction_10d', 'scenario_label_10d']].copy()

df_pred['rf_adverse_proba'] = rf_scenario_proba[:, 0]
df_pred['rf_base_proba'] = rf_scenario_proba[:, 1]
df_pred['rf_favorable_proba'] = rf_scenario_proba[:, 2]
df_pred['rf_scenario_pred'] = rf_scenario_pred

df_pred['fusion_adverse_proba'] = fusion_proba[:, 0]
df_pred['fusion_base_proba'] = fusion_proba[:, 1]
df_pred['fusion_favorable_proba'] = fusion_proba[:, 2]
df_pred['fusion_scenario_pred'] = fusion_pred

df_pred['rf_binary_pred'] = rf_binary_pred

if LGB_AVAILABLE:
    df_pred['lgb_adverse_proba'] = lgb_scenario_proba[:, 0]
    df_pred['lgb_base_proba'] = lgb_scenario_proba[:, 1]
    df_pred['lgb_favorable_proba'] = lgb_scenario_proba[:, 2]
    df_pred['lgb_scenario_pred'] = lgb_scenario_pred
    df_pred['lgb_binary_pred'] = lgb_binary_pred

# Add assessment states
df_pred['trend_state'] = df_test_clean['trend_state'].values
df_pred['vol_state'] = df_test_clean['vol_state'].values
df_pred['drawdown_state'] = df_test_clean['drawdown_state'].values
df_pred['sr_state'] = df_test_clean['sr_state'].values
df_pred['event_state'] = df_test_clean['event_state'].values

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS predictions_v4")

df_pred.to_sql('predictions_v4', conn, index=False)
conn.commit()
conn.close()

print(f"  Saved {len(df_pred)} predictions to predictions_v4")

# ==================== 11. Summary ====================
print("\n" + "=" * 70)
print("【Train Model v4 COMPLETE】")
print("=" * 70)
print(f"Models saved to: {MODEL_DIR}")
print(f"Predictions saved to: predictions_v4 ({len(df_pred)} records)")
print(f"\nOOD Performance Summary:")
print(f"  RF Binary Accuracy: {accuracy_score(y_binary_test, rf_binary_pred):.4f}")
print(f"  RF 3-Scenario Accuracy: {accuracy_score(y_scenario_test, rf_scenario_pred):.4f}")
if LGB_AVAILABLE:
    print(f"  LGBM Binary Accuracy: {accuracy_score(y_binary_test, lgb_binary_pred):.4f}")
    print(f"  LGBM 3-Scenario Accuracy: {accuracy_score(y_scenario_test, lgb_scenario_pred):.4f}")
print(f"  Fusion 3-Scenario Accuracy: {accuracy_score(y_scenario_test, fusion_pred):.4f}")
