"""
Scenario Engine v4 — AGGRESSIVE Tuned Version
进一步激进调优: 解决牛市中仍然过于保守的问题
核心调整:
1. base_weight: 0.6 → 0.80 (基准情景保持80%高仓位)
2. adverse_weight: 0.30 → 0.50 (不利时也只减到50%，不空仓)
3. min_position: 0.20 → 0.50 (最低保留半仓)
4. smooth_scale: 0.5 → 0.8 (概率加权更激进)
5. up_trend_boost: 0.10 → 0.15 (上涨趋势额外+15%)
6. threshold_adverse: 0.35 → 0.45 (更难触发adverse减仓)
7. threshold_favorable: 0.25 → 0.20 (更容易触发favorable加仓)
"""
import json
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
MODEL_DIR = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/models/v4'
TEST_ETFS = ['562500.SH', '588200.SH', '588790.SH', '159382.SZ', '159241.SZ']

# ===== 激进调优配置 =====
AGGRESSIVE_WEIGHTS = {
    "adverse_weight": 0.50,      # 不利时保留50%仓位
    "base_weight": 0.80,         # 基准情景80%高仓位
    "favorable_weight": 1.0,   # 满仓
    "threshold_adverse": 0.45,   # 更难触发adverse
    "threshold_favorable": 0.20, # 更容易触发favorable
    "smooth_mode": True,
    "smooth_scale": 0.8,         # 概率加权更激进
    "min_position": 0.50,        # 最低半仓
    "up_trend_boost": 0.15       # 上涨趋势额外+15%
}

print("=" * 70)
print("【Scenario Engine v4 AGGRESSIVE — 激进调优版】")
print("=" * 70)

# 加载模型
rf_scenario_path = os.path.join(MODEL_DIR, 'rf_scenario.pkl')
lgb_scenario_path = os.path.join(MODEL_DIR, 'lgb_scenario.pkl')
feature_cols_path = os.path.join(MODEL_DIR, 'feature_cols.json')

with open(feature_cols_path, 'r', encoding='utf-8') as f:
    FEATURE_COLS = json.load(f)
with open(rf_scenario_path, 'rb') as f:
    rf_scenario = pickle.load(f)
with open(lgb_scenario_path, 'rb') as f:
    lgb_scenario = pickle.load(f)

rf_binary_path = os.path.join(MODEL_DIR, 'rf_binary.pkl')
lgb_binary_path = os.path.join(MODEL_DIR, 'lgb_binary.pkl')
rf_binary = None
lgb_binary = None
if os.path.exists(rf_binary_path):
    with open(rf_binary_path, 'rb') as f:
        rf_binary = pickle.load(f)
if os.path.exists(lgb_binary_path):
    with open(lgb_binary_path, 'rb') as f:
        lgb_binary = pickle.load(f)

# 读取数据
conn = sqlite3.connect(DB_PATH)
placeholders = ','.join(["'" + s + "'" for s in TEST_ETFS])
df_features = pd.read_sql_query(
    f"SELECT * FROM features_v4 WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date",
    conn
)
conn.close()

df_features = df_features.drop_duplicates(subset=['symbol', 'trade_date'], keep='last')
df = df_features.copy()
df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')

# 二分类预测
if rf_binary is not None and lgb_binary is not None:
    X_bin = df[FEATURE_COLS].fillna(0).values
    df['rf_binary_pred'] = rf_binary.predict(X_bin).astype(int)
    df['lgb_binary_pred'] = lgb_binary.predict(X_bin).astype(int)
else:
    conn = sqlite3.connect(DB_PATH)
    df_pred = pd.read_sql_query(
        f"SELECT trade_date, symbol, rf_binary_pred, lgb_binary_pred FROM predictions_v4 WHERE symbol IN ({placeholders})",
        conn
    )
    conn.close()
    df_pred = df_pred.drop_duplicates(subset=['symbol', 'trade_date'], keep='last')
    df_pred['trade_date'] = pd.to_datetime(df_pred['trade_date'], format='mixed')
    df = pd.merge(df, df_pred, on=['symbol', 'trade_date'], how='left')
    df['rf_binary_pred'] = df['rf_binary_pred'].fillna(0).astype(int)
    df['lgb_binary_pred'] = df['lgb_binary_pred'].fillna(0).astype(int)

# 生成三情景概率
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
for c in missing_cols:
    df[c] = 0.0
X = df[FEATURE_COLS].fillna(0).values
rf_proba = rf_scenario.predict_proba(X)
lgb_proba = lgb_scenario.predict_proba(X)
fusion_proba = (rf_proba + lgb_proba) / 2.0

idx_adverse, idx_base, idx_favorable = 0, 1, 2
df['fusion_adverse_proba'] = fusion_proba[:, idx_adverse]
df['fusion_base_proba'] = fusion_proba[:, idx_base]
df['fusion_favorable_proba'] = fusion_proba[:, idx_favorable]

# 激进决策规则
def scenario_decision_aggressive(row):
    p_adv = row['fusion_adverse_proba']
    p_base = row['fusion_base_proba']
    p_fav = row['fusion_favorable_proba']
    binary_up = int(row.get('rf_binary_pred', 1))
    
    # 概率加权平滑仓位 (更激进)
    position = AGGRESSIVE_WEIGHTS['base_weight'] + (p_fav - p_adv) * AGGRESSIVE_WEIGHTS['smooth_scale']
    
    if binary_up == 1:
        position += AGGRESSIVE_WEIGHTS['up_trend_boost']
    else:
        position -= AGGRESSIVE_WEIGHTS['up_trend_boost'] * 0.5  # 下跌时减得少一点
    
    # 硬边界
    position = max(AGGRESSIVE_WEIGHTS['min_position'], min(1.0, position))
    
    # 主导情景
    probs = [('adverse', p_adv), ('base', p_base), ('favorable', p_fav)]
    probs.sort(key=lambda x: x[1], reverse=True)
    decision = probs[0][0]
    
    # 信号方向
    if position < 0.3:
        signal_dir = 0
    elif binary_up == 1:
        signal_dir = 1
    else:
        signal_dir = -1
    
    return pd.Series({
        'scenario_decision': decision,
        'position_size': round(max(0.0, min(1.0, float(position))), 3),
        'signal_direction': int(signal_dir)
    })

print("[计算] 应用激进版三情景决策规则...")
print(f"[配置] adverse_weight={AGGRESSIVE_WEIGHTS['adverse_weight']}, base_weight={AGGRESSIVE_WEIGHTS['base_weight']}")
print(f"[配置] min_position={AGGRESSIVE_WEIGHTS['min_position']}, smooth_scale={AGGRESSIVE_WEIGHTS['smooth_scale']}")

decision_df = df.apply(scenario_decision_aggressive, axis=1)
df = pd.concat([df, decision_df], axis=1)

# 保存到数据库 (覆盖原表)
output_cols = [
    'trade_date', 'symbol', 'close',
    'trend_state', 'vol_state', 'drawdown_state', 'sr_state', 'event_state',
    'fusion_adverse_proba', 'fusion_base_proba', 'fusion_favorable_proba',
    'scenario_decision', 'position_size', 'signal_direction',
    'rf_binary_pred', 'lgb_binary_pred'
]
for c in output_cols:
    if c not in df.columns:
        df[c] = None

df_out = df[output_cols].copy()
df_out['trade_date'] = pd.to_datetime(df_out['trade_date'])
df_out['event_state'] = df_out['event_state'].fillna(0).astype(int)
df_out['rf_binary_pred'] = df_out['rf_binary_pred'].fillna(-1).astype(int)
df_out['lgb_binary_pred'] = df_out['lgb_binary_pred'].fillna(-1).astype(int)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS scenario_signals_v4")
cursor.execute("""
CREATE TABLE scenario_signals_v4 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TIMESTAMP,
    symbol TEXT,
    close REAL,
    trend_state TEXT,
    vol_state TEXT,
    drawdown_state TEXT,
    sr_state TEXT,
    event_state INTEGER,
    fusion_adverse_proba REAL,
    fusion_base_proba REAL,
    fusion_favorable_proba REAL,
    scenario_decision TEXT,
    position_size REAL,
    signal_direction INTEGER,
    rf_binary_pred INTEGER,
    lgb_binary_pred INTEGER
)
""")
conn.commit()
df_out.to_sql('scenario_signals_v4', conn, if_exists='append', index=False)
conn.commit()
conn.close()

print(f"[保存] scenario_signals_v4: {len(df_out)} 条记录")

# 汇总
print("\n" + "=" * 70)
print("【激进调优版 — 最新信号与仓位分布】")
print("=" * 70)

for sym in TEST_ETFS:
    sym_df = df_out[df_out['symbol'] == sym].sort_values('trade_date')
    if len(sym_df) == 0:
        continue
    latest = sym_df.iloc[-1]
    avg_pos = sym_df['position_size'].mean()
    
    print(f"\n>>> {sym}")
    print(f"  最新: {latest['trade_date'].strftime('%Y-%m-%d')} | 决策={latest['scenario_decision']} | 仓位={latest['position_size']:.2f}")
    print(f"  历史平均仓位: {avg_pos:.2f} (调优版约0.35, 激进版目标>0.6)")
    
    pos_dist = {
        '空仓(<0.3)': (sym_df['position_size'] < 0.3).sum(),
        '轻仓(0.3-0.6)': ((sym_df['position_size'] >= 0.3) & (sym_df['position_size'] < 0.6)).sum(),
        '半仓(0.6-0.8)': ((sym_df['position_size'] >= 0.6) & (sym_df['position_size'] < 0.8)).sum(),
        '满仓(>=0.8)': (sym_df['position_size'] >= 0.8).sum()
    }
    print(f"  仓位分布: {pos_dist}")

print("\n" + "=" * 70)
print("【Scenario Engine v4 AGGRESSIVE 完成】")
print("=" * 70)
print("核心激进调优:")
print("  1. adverse_weight: 0.30 → 0.50 (不利时保留半仓)")
print("  2. base_weight: 0.60 → 0.80 (基准情景80%仓位)")
print("  3. min_position: 0.20 → 0.50 (最低半仓，彻底避免踏空)")
print("  4. smooth_scale: 0.5 → 0.8 (概率加权更激进)")
print("  5. up_trend_boost: 0.10 → 0.15 (上涨趋势额外+15%)")
print("  6. threshold_adverse: 0.35 → 0.45 (更难触发减仓)")
print("  7. threshold_favorable: 0.25 → 0.20 (更容易触发满仓)")
print("=" * 70)
