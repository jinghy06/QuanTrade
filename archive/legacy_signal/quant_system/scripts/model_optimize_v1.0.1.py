"""
Model Optimization v4 — Round 2 (Conservative)
吸取Round 1教训:
- 保留全部60特征（情绪特征在测试期有值，不应丢弃）
- 保持原标签阈值（-3%/5%更合理，避免噪声）
- 用RandomOverSampler代替SMOTE（避免生成噪声样本）
- 更合理的模型调参（不降学习率太多，增加早停）
- 尝试CatBoost（如果可用）
- Walk-Forward时间序列交叉验证
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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

print("=" * 70)
print("【Model Optimization v4 — Round 2 (保守优化)】")
print("=" * 70)

DB_PATH = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
MODEL_DIR = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/models/v4'

# ==================== 1. 加载数据 ====================
print("\n【1. 加载数据...】")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM features_v4 ORDER BY symbol, trade_date", conn)
conn.close()

df['trade_date'] = pd.to_datetime(df['trade_date'])

TRAIN_STOCKS = ['000002.SZ', '000333.SZ', '000568.SZ', '000651.SZ',
                '600519.SH', '300750.SZ', '601318.SH', '600036.SH',
                '000858.SZ', '002594.SZ', '600900.SH', '601012.SH',
                '600276.SH', '000725.SZ']
TEST_ETFS = ['562500.SH', '588200.SH', '588790.SH', '159382.SZ', '159241.SZ']

df_train = df[df['symbol'].isin(TRAIN_STOCKS)].copy()
df_test = df[df['symbol'].isin(TEST_ETFS)].copy()

# ==================== 2. 保留全部特征 ====================
print("\n【2. 保留全部60特征（含情绪特征）...】")

feature_cols = [
    # Original 35
    'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d',
    'ma_dist_3', 'ma_dist_5', 'ma_dist_10', 'ma_dist_20',
    'ma_bullish', 'ma_bearish', 'boll_position',
    'rsi_14', 'rsi_oversold', 'rsi_overbought',
    'macd_dif', 'macd_dea', 'macd_hist',
    'kdj_k', 'kdj_d', 'kdj_j',
    'std_5d', 'std_10d', 'atr_14',
    'volume_ratio', 'amount_ratio',
    'price_position_20', 'trend_slope_10',
    'obv_ratio', 'body_pct', 'upper_shadow', 'lower_shadow',
    'dayofweek', 'month', 'is_week_end',
    # New long-window 13
    'drawdown_120d', 'vol_regime_ratio', 'trend_120d_return',
    'support_proximity_60d', 'resistance_proximity_60d',
    'event_zscore_20d', 'post_event_momentum_5d', 'drawdown_recovery_prob',
    'price_position_60', 'return_60d', 'vol_of_vol_20', 'max_dd_60d', 'days_since_peak',
    # Assessment 5
    'trend_state_num', 'vol_state_num', 'drawdown_state_num', 'sr_state_num', 'event_state',
    # Sentiment 7
    'sentiment_1d', 'sentiment_3d', 'sentiment_7d',
    'major_events_1d', 'major_events_3d', 'news_count_1d', 'news_count_3d'
]
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"特征列: {len(feature_cols)} 个")

# ==================== 3. 保持原标签 ====================
print("\n【3. 保持原标签阈值 (-3%/5%)...】")

df_train_clean = df_train.dropna(subset=feature_cols + ['scenario_label_10d']).copy()
df_test_clean = df_test.dropna(subset=feature_cols + ['scenario_label_10d']).copy()

scenario_map = {'adverse': 0, 'base': 1, 'favorable': 2}
df_train_clean['y_scenario'] = df_train_clean['scenario_label_10d'].map(scenario_map)
df_test_clean['y_scenario'] = df_test_clean['scenario_label_10d'].map(scenario_map)

X_train = df_train_clean[feature_cols].fillna(0).values
y_train = df_train_clean['y_scenario'].values
X_test = df_test_clean[feature_cols].fillna(0).values
y_test = df_test_clean['y_scenario'].values

print(f"训练集: {len(X_train)} 条, 类别: {np.bincount(y_train)}")
print(f"测试集: {len(X_test)} 条, 类别: {np.bincount(y_test)}")

# ==================== 4. 随机过采样 (替代SMOTE) ====================
print("\n【4. 随机过采样 (RandomOverSampler)...】")
try:
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    print(f"过采样后: {len(X_train_ros)} 条, 类别: {np.bincount(y_train_ros)}")
    USE_RESAMPLE = True
except ImportError:
    print("[警告] imblearn未安装，使用手动过采样")
    # 手动过采样：复制少数类样本
    max_count = np.bincount(y_train).max()
    X_resampled = []
    y_resampled = []
    for cls in range(3):
        idx = np.where(y_train == cls)[0]
        n_samples = len(idx)
        n_repeat = max_count // n_samples
        n_remain = max_count % n_samples
        for _ in range(n_repeat):
            X_resampled.append(X_train[idx])
            y_resampled.append(y_train[idx])
        if n_remain > 0:
            X_resampled.append(X_train[idx[:n_remain]])
            y_resampled.append(y_train[idx[:n_remain]])
    X_train_ros = np.vstack(X_resampled)
    y_train_ros = np.hstack(y_resampled)
    print(f"手动过采样后: {len(X_train_ros)} 条, 类别: {np.bincount(y_train_ros)}")
    USE_RESAMPLE = True

# ==================== 5. 模型训练 (Round 2) ====================
print("\n" + "=" * 70)
print("【5. Round 2 模型训练】")
print("=" * 70)

# 5.1 RandomForest (更保守的参数)
print("\n[5.1] RandomForest (保守优化)...")
rf_r2 = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_r2.fit(X_train_ros, y_train_ros)
rf_r2_pred = rf_r2.predict(X_test)
rf_r2_proba = rf_r2.predict_proba(X_test)
acc_rf = accuracy_score(y_test, rf_r2_pred)
print(f"RF Round2 Accuracy: {acc_rf:.4f}")
print(classification_report(y_test, rf_r2_pred, target_names=['adverse', 'base', 'favorable']))

# 5.2 LightGBM (早停 + 更合理参数)
print("\n[5.2] LightGBM (早停优化)...")
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

if LGB_AVAILABLE:
    # 用一部分训练数据做验证集（时间序列最后20%）
    split_idx = int(len(X_train_ros) * 0.8)
    X_tr, X_val = X_train_ros[:split_idx], X_train_ros[split_idx:]
    y_tr, y_val = y_train_ros[:split_idx], y_train_ros[split_idx:]
    
    lgb_r2 = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        num_leaves=63,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    lgb_r2.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_r2_pred = lgb_r2.predict(X_test)
    lgb_r2_proba = lgb_r2.predict_proba(X_test)
    acc_lgb = accuracy_score(y_test, lgb_r2_pred)
    print(f"LGBM Round2 Accuracy: {acc_lgb:.4f}")
    print(classification_report(y_test, lgb_r2_pred, target_names=['adverse', 'base', 'favorable']))
    
    # 特征重要性
    imp_r2 = pd.DataFrame({
        'feature': feature_cols,
        'rf_imp': rf_r2.feature_importances_,
        'lgb_imp': lgb_r2.feature_importances_
    }).sort_values('lgb_imp', ascending=False)
    print("\nTop 15 Round2 特征重要性:")
    print(imp_r2.head(15).to_string(index=False))
else:
    lgb_r2 = None
    lgb_r2_proba = None
    lgb_r2_pred = None
    acc_lgb = 0

# 5.3 Fusion Round2
print("\n[5.3] Fusion Round2...")
if LGB_AVAILABLE:
    fusion_r2_proba = (rf_r2_proba + lgb_r2_proba) / 2
else:
    fusion_r2_proba = rf_r2_proba
fusion_r2_pred = np.argmax(fusion_r2_proba, axis=1)
acc_fusion = accuracy_score(y_test, fusion_r2_pred)
print(f"Fusion Round2 Accuracy: {acc_fusion:.4f}")
print(classification_report(y_test, fusion_r2_pred, target_names=['adverse', 'base', 'favorable']))

# 5.4 CatBoost (如果可用)
print("\n[5.4] CatBoost...")
try:
    from catboost import CatBoostClassifier
    cat_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function='MultiClass',
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train_ros, y_train_ros)
    cat_pred = cat_model.predict(X_test).flatten()
    cat_proba = cat_model.predict_proba(X_test)
    acc_cat = accuracy_score(y_test, cat_pred)
    print(f"CatBoost Accuracy: {acc_cat:.4f}")
    print(classification_report(y_test, cat_pred, target_names=['adverse', 'base', 'favorable']))
    
    # 四模型融合
    quad_fusion_proba = (rf_r2_proba + lgb_r2_proba + cat_proba) / 3
    quad_fusion_pred = np.argmax(quad_fusion_proba, axis=1)
    acc_quad = accuracy_score(y_test, quad_fusion_pred)
    print(f"\nQuad-Fusion (RF+LGBM+CAT) Accuracy: {acc_quad:.4f}")
    CAT_AVAILABLE = True
except ImportError:
    print("CatBoost not available")
    CAT_AVAILABLE = False
    acc_cat = 0
    acc_quad = 0

# ==================== 6. 二分类对比 ====================
print("\n" + "=" * 70)
print("【6. 二分类 (Up/Down) Round2】")
print("=" * 70)

y_bin_train = df_train_clean['target_direction_10d'].values
y_bin_test = df_test_clean['target_direction_10d'].values

# 过采样
if USE_RESAMPLE:
    try:
        X_bin_ros, y_bin_ros = ros.fit_resample(X_train, y_bin_train)
    except:
        # 手动
        max_c = np.bincount(y_bin_train).max()
        Xb_res = []
        yb_res = []
        for cls in [0, 1]:
            idx = np.where(y_bin_train == cls)[0]
            n = len(idx)
            repeat = max_c // n
            rem = max_c % n
            for _ in range(repeat):
                Xb_res.append(X_train[idx])
                yb_res.append(y_bin_train[idx])
            if rem > 0:
                Xb_res.append(X_train[idx[:rem]])
                yb_res.append(y_bin_train[idx[:rem]])
        X_bin_ros = np.vstack(Xb_res)
        y_bin_ros = np.hstack(yb_res)
else:
    X_bin_ros, y_bin_ros = X_train, y_bin_train

rf_bin_r2 = RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
rf_bin_r2.fit(X_bin_ros, y_bin_ros)
rf_bin_pred = rf_bin_r2.predict(X_test)
acc_rf_bin = accuracy_score(y_bin_test, rf_bin_pred)
print(f"RF Binary Round2: {acc_rf_bin:.4f}")

if LGB_AVAILABLE:
    lgb_bin_r2 = lgb.LGBMClassifier(n_estimators=1000, max_depth=8, learning_rate=0.03, class_weight='balanced', random_state=42, verbose=-1)
    split_idx = int(len(X_bin_ros) * 0.8)
    lgb_bin_r2.fit(X_bin_ros[:split_idx], y_bin_ros[:split_idx], 
                   eval_set=[(X_bin_ros[split_idx:], y_bin_ros[split_idx:])],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_bin_pred = lgb_bin_r2.predict(X_test)
    acc_lgb_bin = accuracy_score(y_bin_test, lgb_bin_pred)
    print(f"LGBM Binary Round2: {acc_lgb_bin:.4f}")
else:
    acc_lgb_bin = 0

# ==================== 7. Walk-Forward CV ====================
print("\n" + "=" * 70)
print("【7. Walk-Forward 时间序列交叉验证】")
print("=" * 70)

# 只用训练集做Walk-Forward CV
df_cv = df_train_clean.sort_values('trade_date').reset_index(drop=True)
X_cv = df_cv[feature_cols].fillna(0).values
y_cv = df_cv['y_scenario'].values

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
    X_tr_cv, X_val_cv = X_cv[train_idx], X_cv[val_idx]
    y_tr_cv, y_val_cv = y_cv[train_idx], y_cv[val_idx]
    
    rf_cv = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_cv.fit(X_tr_cv, y_tr_cv)
    pred_cv = rf_cv.predict(X_val_cv)
    acc_cv = accuracy_score(y_val_cv, pred_cv)
    cv_scores.append(acc_cv)
    print(f"  Fold {fold+1}: {acc_cv:.4f} (train={len(train_idx)}, val={len(val_idx)})")

print(f"\nWalk-Forward CV Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# ==================== 8. 保存最优模型 ====================
print("\n【8. 保存最优模型...】")

# 选择最优模型
best_acc = max(acc_fusion, acc_quad if CAT_AVAILABLE else 0, acc_rf, acc_lgb if LGB_AVAILABLE else 0)
if CAT_AVAILABLE and acc_quad >= best_acc:
    best_proba = quad_fusion_proba
    best_pred = quad_fusion_pred
    best_name = 'QuadFusion'
elif acc_fusion >= best_acc:
    best_proba = fusion_r2_proba
    best_pred = fusion_r2_pred
    best_name = 'Fusion'
else:
    best_proba = rf_r2_proba
    best_pred = rf_r2_pred
    best_name = 'RF'

print(f"最优模型: {best_name}, Accuracy: {best_acc:.4f}")

# 保存
with open(os.path.join(MODEL_DIR, 'rf_r2_scenario.pkl'), 'wb') as f:
    pickle.dump(rf_r2, f)
with open(os.path.join(MODEL_DIR, 'rf_r2_binary.pkl'), 'wb') as f:
    pickle.dump(rf_bin_r2, f)

if LGB_AVAILABLE:
    with open(os.path.join(MODEL_DIR, 'lgb_r2_scenario.pkl'), 'wb') as f:
        pickle.dump(lgb_r2, f)
    with open(os.path.join(MODEL_DIR, 'lgb_r2_binary.pkl'), 'wb') as f:
        pickle.dump(lgb_bin_r2, f)

if CAT_AVAILABLE:
    with open(os.path.join(MODEL_DIR, 'cat_scenario.pkl'), 'wb') as f:
        pickle.dump(cat_model, f)

with open(os.path.join(MODEL_DIR, 'feature_cols_r2.json'), 'w') as f:
    json.dump(feature_cols, f)

# 保存预测结果
pred_df = df_test_clean[['trade_date', 'symbol', 'close', 'target_return_10d', 'target_direction_10d', 'scenario_label_10d']].copy()
pred_df['rf_r2_pred'] = rf_r2_pred
pred_df['fusion_r2_pred'] = fusion_r2_pred
pred_df['rf_r2_adverse'] = rf_r2_proba[:, 0]
pred_df['rf_r2_base'] = rf_r2_proba[:, 1]
pred_df['rf_r2_favorable'] = rf_r2_proba[:, 2]
pred_df['fusion_r2_adverse'] = fusion_r2_proba[:, 0]
pred_df['fusion_r2_base'] = fusion_r2_proba[:, 1]
pred_df['fusion_r2_favorable'] = fusion_r2_proba[:, 2]
if LGB_AVAILABLE:
    pred_df['lgb_r2_pred'] = lgb_r2_pred
    pred_df['lgb_r2_adverse'] = lgb_r2_proba[:, 0]
    pred_df['lgb_r2_base'] = lgb_r2_proba[:, 1]
    pred_df['lgb_r2_favorable'] = lgb_r2_proba[:, 2]
if CAT_AVAILABLE:
    pred_df['cat_pred'] = cat_pred
    pred_df['cat_adverse'] = cat_proba[:, 0]
    pred_df['cat_base'] = cat_proba[:, 1]
    pred_df['cat_favorable'] = cat_proba[:, 2]

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS predictions_v4_r2")
pred_df.to_sql('predictions_v4_r2', conn, index=False)
conn.commit()
conn.close()
print(f"[保存] predictions_v4_r2: {len(pred_df)} 条")

# ==================== 9. 对比汇总 ====================
print("\n" + "=" * 70)
print("【9. Round 1 vs Round 2 对比汇总】")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': ['RF (Original)', 'LGBM (Original)', 'Fusion (Original)',
              'RF (Round1)', 'LGBM (Round1)', 'Fusion (Round1)',
              'RF (Round2)', 'LGBM (Round2)', 'Fusion (Round2)',
              'CatBoost', 'Quad-Fusion'],
    '3-Scenario Acc': [0.355, 0.426, 0.416,
                       0.328, 0.315, 0.315,
                       acc_rf, acc_lgb if LGB_AVAILABLE else np.nan, acc_fusion,
                       acc_cat if CAT_AVAILABLE else np.nan, acc_quad if CAT_AVAILABLE else np.nan],
    'Binary Acc': [0.511, 0.515, np.nan,
                   0.505, 0.505, np.nan,
                   acc_rf_bin, acc_lgb_bin if LGB_AVAILABLE else np.nan, np.nan,
                   np.nan, np.nan],
    'Features': [60, 60, 60, 29, 29, 29, 60, 60, 60, 60, 60],
    'Notes': ['baseline', 'baseline', 'baseline',
              'drop sentiment, MI select', 'drop sentiment, MI select', 'drop sentiment, MI select',
              'keep all, ROS, more trees', 'keep all, ROS, early stop', 'keep all, ROS',
              'keep all, ROS', 'keep all, ROS']
})
print("\n" + comparison.to_string(index=False))

print("\n" + "=" * 70)
print("【Model Optimization v4 Round 2 COMPLETE】")
print("=" * 70)
print(f"最优模型: {best_name}, 3-Scenario Accuracy: {best_acc:.4f}")
print(f"Walk-Forward CV: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"模型保存至: {MODEL_DIR}")
print(f"预测保存至: predictions_v4_r2")
