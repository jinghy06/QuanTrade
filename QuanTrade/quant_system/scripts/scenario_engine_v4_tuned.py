"""
Scenario Engine v4 — Tuned Version (Less Conservative)
调优目标: 解决三情景策略过于保守的问题
核心调整:
1. 降低adverse阈值 (0.4 → 0.35)
2. adverse时不再空仓，而是减仓到0.3 (保留部分敞口)
3. 引入概率加权平滑仓位，避免硬阈值跳变
4. base情景仓位提升到0.6 (原0.5)
5. 当binary_up==1时，即使adverse概率中等也保持部分仓位
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

# ==================== 配置 ====================
DB_PATH = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
MODEL_DIR = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/models/v4'
TEST_ETFS = ['562500.SH', '588200.SH', '588790.SH', '159382.SZ', '159241.SZ']

# ===== 调优后的权重配置 =====
TUNED_WEIGHTS = {
    "adverse_weight": 0.30,      # 原0.0 → 0.3 (adverse时不空仓，保留30%)
    "base_weight": 0.60,           # 原0.5 → 0.6 (base时更积极)
    "favorable_weight": 1.0,     # 保持1.0
    "threshold_adverse": 0.35,   # 原0.4 → 0.35 (更容易触发adverse减仓)
    "threshold_favorable": 0.25,   # 原0.3 → 0.25 (更容易触发favorable加仓)
    "smooth_mode": True,         # 新增: 概率加权平滑
    "smooth_scale": 0.5,         # 平滑系数
    "min_position": 0.20,        # 最低仓位20% (避免完全踏空)
    "up_trend_boost": 0.10       # 上涨趋势时额外+10%仓位
}

print("=" * 70)
print("【Scenario Engine v4 TUNED — 调优版三情景决策引擎】")
print("=" * 70)

# ==================== 1. 检查模型文件 ====================
rf_scenario_path = os.path.join(MODEL_DIR, 'rf_scenario.pkl')
lgb_scenario_path = os.path.join(MODEL_DIR, 'lgb_scenario.pkl')
feature_cols_path = os.path.join(MODEL_DIR, 'feature_cols.json')

for p in [rf_scenario_path, lgb_scenario_path, feature_cols_path]:
    if not os.path.exists(p):
        print(f"[错误] 模型文件不存在: {p}")
        sys.exit(1)

print(f"[OK] 所有模型文件已找到")

# ==================== 2. 加载模型 ====================
with open(feature_cols_path, 'r', encoding='utf-8') as f:
    FEATURE_COLS = json.load(f)

with open(rf_scenario_path, 'rb') as f:
    rf_scenario = pickle.load(f)
with open(lgb_scenario_path, 'rb') as f:
    lgb_scenario = pickle.load(f)

# 加载二分类模型
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

print(f"[OK] 模型加载完成")

# ==================== 3. 读取数据 ====================
conn = sqlite3.connect(DB_PATH)
placeholders = ','.join([f"'{s}'" for s in TEST_ETFS])
df_features = pd.read_sql_query(
    f"SELECT * FROM features_v4 WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date",
    conn
)
conn.close()

if df_features.empty:
    print("[错误] features_v4 中没有测试ETF数据")
    sys.exit(1)

df_features = df_features.drop_duplicates(subset=['symbol', 'trade_date'], keep='last')
df = df_features.copy()
df['trade_date'] = pd.to_datetime(df['trade_date'])

print(f"[数据] features_v4: {len(df)} 条记录")

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
    df_pred['trade_date'] = pd.to_datetime(df_pred['trade_date'])
    df = pd.merge(df, df_pred, on=['symbol', 'trade_date'], how='left')
    df['rf_binary_pred'] = df['rf_binary_pred'].fillna(0).astype(int)
    df['lgb_binary_pred'] = df['lgb_binary_pred'].fillna(0).astype(int)

# ==================== 4. 生成三情景概率 ====================
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
for c in missing_cols:
    df[c] = 0.0

X = df[FEATURE_COLS].fillna(0).values
rf_proba = rf_scenario.predict_proba(X)
lgb_proba = lgb_scenario.predict_proba(X)
fusion_proba = (rf_proba + lgb_proba) / 2.0

idx_adverse, idx_base, idx_favorable = 0, 1, 2

df['rf_adverse_proba'] = rf_proba[:, idx_adverse]
df['rf_base_proba'] = rf_proba[:, idx_base]
df['rf_favorable_proba'] = rf_proba[:, idx_favorable]
df['lgb_adverse_proba'] = lgb_proba[:, idx_adverse]
df['lgb_base_proba'] = lgb_proba[:, idx_base]
df['lgb_favorable_proba'] = lgb_proba[:, idx_favorable]
df['fusion_adverse_proba'] = fusion_proba[:, idx_adverse]
df['fusion_base_proba'] = fusion_proba[:, idx_base]
df['fusion_favorable_proba'] = fusion_proba[:, idx_favorable]

print("[OK] 三情景概率生成完成")

# ==================== 5. 调优后的三情景决策规则 ====================
def scenario_decision_tuned(row):
    """
    调优版决策规则 — 解决过于保守问题:
    1. 概率加权平滑仓位 (smooth_mode)
    2. adverse时保留min_position最低仓位
    3. 降低favorable阈值到0.25
    4. 上涨趋势额外boost
    """
    p_adv = row['fusion_adverse_proba']
    p_base = row['fusion_base_proba']
    p_fav = row['fusion_favorable_proba']
    binary_up = int(row.get('rf_binary_pred', 1))
    
    # --- 方法A: 概率加权平滑仓位 ---
    if TUNED_WEIGHTS['smooth_mode']:
        # 基础仓位 = base_weight + (favorable - adverse) * scale
        position = TUNED_WEIGHTS['base_weight'] + (p_fav - p_adv) * TUNED_WEIGHTS['smooth_scale']
        
        # 根据binary方向微调
        if binary_up == 1:
            position += TUNED_WEIGHTS['up_trend_boost']
        else:
            position -= TUNED_WEIGHTS['up_trend_boost']
        
        # 硬边界保护
        position = max(TUNED_WEIGHTS['min_position'], min(1.0, position))
        
        # 确定主导情景 (用于展示)
        probs = [('adverse', p_adv), ('base', p_base), ('favorable', p_fav)]
        probs.sort(key=lambda x: x[1], reverse=True)
        decision = probs[0][0]
        
    else:
        # --- 方法B: 改进的硬阈值 (不那么激进空仓) ---
        triggered = []
        if p_fav > TUNED_WEIGHTS['threshold_favorable']:
            triggered.append(('favorable', p_fav, TUNED_WEIGHTS['favorable_weight']))
        if p_adv > TUNED_WEIGHTS['threshold_adverse']:
            triggered.append(('adverse', p_adv, TUNED_WEIGHTS['adverse_weight']))
        if not triggered or (p_base >= p_adv and p_base >= p_fav):
            triggered.append(('base', p_base, TUNED_WEIGHTS['base_weight']))
        
        triggered.sort(key=lambda x: x[1], reverse=True)
        decision, _, position = triggered[0]
        
        # binary方向修正
        if binary_up == 1:
            if decision == 'adverse':
                position = max(position, TUNED_WEIGHTS['min_position'])
            elif decision == 'favorable':
                position = TUNED_WEIGHTS['favorable_weight']
        else:
            if decision == 'favorable':
                position = TUNED_WEIGHTS['base_weight']
    
    # 信号方向
    if position < 0.15:
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

print("[计算] 应用调优版三情景决策规则...")
print(f"[配置] adverse_weight={TUNED_WEIGHTS['adverse_weight']}, base_weight={TUNED_WEIGHTS['base_weight']}")
print(f"[配置] threshold_adverse={TUNED_WEIGHTS['threshold_adverse']}, threshold_favorable={TUNED_WEIGHTS['threshold_favorable']}")
print(f"[配置] smooth_mode={TUNED_WEIGHTS['smooth_mode']}, min_position={TUNED_WEIGHTS['min_position']}")

decision_df = df.apply(scenario_decision_tuned, axis=1)
df = pd.concat([df, decision_df], axis=1)

# ==================== 6. 保存到数据库 (覆盖原表) ====================
output_cols = [
    'trade_date', 'symbol', 'close',
    'trend_state', 'vol_state', 'drawdown_state', 'sr_state', 'event_state',
    'rf_adverse_proba', 'rf_base_proba', 'rf_favorable_proba',
    'lgb_adverse_proba', 'lgb_base_proba', 'lgb_favorable_proba',
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
create_sql = """
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
    rf_adverse_proba REAL,
    rf_base_proba REAL,
    rf_favorable_proba REAL,
    lgb_adverse_proba REAL,
    lgb_base_proba REAL,
    lgb_favorable_proba REAL,
    fusion_adverse_proba REAL,
    fusion_base_proba REAL,
    fusion_favorable_proba REAL,
    scenario_decision TEXT,
    position_size REAL,
    signal_direction INTEGER,
    rf_binary_pred INTEGER,
    lgb_binary_pred INTEGER
)
"""
cursor.execute(create_sql)
conn.commit()

df_out.to_sql('scenario_signals_v4', conn, if_exists='append', index=False)
conn.commit()
conn.close()

print(f"[保存] scenario_signals_v4: {len(df_out)} 条记录已写入数据库")

# ==================== 7. 汇总对比 (原配置 vs 调优配置) ====================
print("\n" + "=" * 70)
print("【调优前后对比 — 最新信号 (2026-06-05)】")
print("=" * 70)

for sym in TEST_ETFS:
    sym_df = df_out[df_out['symbol'] == sym].sort_values('trade_date')
    if len(sym_df) == 0:
        continue
    latest = sym_df.iloc[-1]
    avg_pos = sym_df['position_size'].mean()
    
    print(f"\n>>> {sym}")
    print(f"  最新日期: {latest['trade_date'].strftime('%Y-%m-%d')}")
    print(f"  融合概率: 不利={latest['fusion_adverse_proba']:.3f} 基准={latest['fusion_base_proba']:.3f} 有利={latest['fusion_favorable_proba']:.3f}")
    print(f"  调优后决策: {latest['scenario_decision']} | 仓位={latest['position_size']:.2f} | 方向={latest['signal_direction']}")
    print(f"  历史平均仓位: {avg_pos:.2f} (原配置约 0.0~0.5)")
    
    dec_counts = sym_df['scenario_decision'].value_counts()
    print(f"  历史决策分布: {dict(dec_counts)}")
    
    # 仓位分布统计
    pos_dist = {
        '空仓(<0.2)': (sym_df['position_size'] < 0.2).sum(),
        '轻仓(0.2-0.5)': ((sym_df['position_size'] >= 0.2) & (sym_df['position_size'] < 0.5)).sum(),
        '半仓(0.5-0.8)': ((sym_df['position_size'] >= 0.5) & (sym_df['position_size'] < 0.8)).sum(),
        '满仓(>=0.8)': (sym_df['position_size'] >= 0.8).sum()
    }
    print(f"  仓位分布: {pos_dist}")

print("\n" + "=" * 70)
print("【Scenario Engine v4 TUNED 完成】")
print("=" * 70)
print("核心调优:")
print("  1. adverse_weight: 0.0 → 0.30 (不利时保留30%仓位)")
print("  2. base_weight: 0.5 → 0.60 (基准情景更积极)")
print("  3. threshold_adverse: 0.4 → 0.35 (阈值降低)")
print("  4. threshold_favorable: 0.3 → 0.25 (更容易触发加仓)")
print("  5. smooth_mode: True (概率加权平滑仓位)")
print("  6. min_position: 0.20 (最低保留20%仓位，避免完全踏空)")
print("  7. up_trend_boost: +10% (上涨趋势额外加仓)")
print("=" * 70)
