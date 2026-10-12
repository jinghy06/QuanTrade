"""
Scenario Engine v4 — 三情景决策引擎
- 从 features_v4 读取测试ETF数据
- 加载 rf_scenario.pkl / lgb_scenario.pkl
- 生成三情景概率并应用决策规则
- 输出 scenario_signals_v4 表
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

print("=" * 70)
print("【Scenario Engine v4 — 三情景决策引擎】")
print("=" * 70)

# ==================== 1. 检查模型文件 ====================
rf_scenario_path = os.path.join(MODEL_DIR, 'rf_scenario.pkl')
lgb_scenario_path = os.path.join(MODEL_DIR, 'lgb_scenario.pkl')
feature_cols_path = os.path.join(MODEL_DIR, 'feature_cols.json')
weights_path = os.path.join(MODEL_DIR, 'scenario_weights.json')

for p in [rf_scenario_path, lgb_scenario_path, feature_cols_path, weights_path]:
    if not os.path.exists(p):
        print(f"[错误] 模型文件不存在: {p}")
        sys.exit(1)

print(f"[OK] 所有模型文件已找到")

# ==================== 2. 加载模型与配置 ====================
with open(feature_cols_path, 'r', encoding='utf-8') as f:
    FEATURE_COLS = json.load(f)

with open(weights_path, 'r', encoding='utf-8') as f:
    weights = json.load(f)

adverse_weight = weights.get('adverse_weight', 0.0)
base_weight = weights.get('base_weight', 0.5)
favorable_weight = weights.get('favorable_weight', 1.0)
threshold_adverse = weights.get('threshold_adverse', 0.4)
threshold_favorable = weights.get('threshold_favorable', 0.3)

print(f"[配置] 权重: adverse={adverse_weight}, base={base_weight}, favorable={favorable_weight}")
print(f"[配置] 阈值: adverse>{threshold_adverse}, favorable>{threshold_favorable}")

with open(rf_scenario_path, 'rb') as f:
    rf_scenario = pickle.load(f)
with open(lgb_scenario_path, 'rb') as f:
    lgb_scenario = pickle.load(f)

# 同时加载二分类模型（用于生成完整的对比信号）
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

print(f"[OK] 模型加载完成: RF={type(rf_scenario).__name__}, LGB={type(lgb_scenario).__name__}")
if rf_binary:
    print(f"[OK] 二分类模型加载完成: RF_binary={type(rf_binary).__name__}")
if lgb_binary:
    print(f"[OK] 二分类模型加载完成: LGB_binary={type(lgb_binary).__name__}")

# ==================== 3. 读取数据 ====================
conn = sqlite3.connect(DB_PATH)

# 读取 features_v4 中测试ETF的数据
placeholders = ','.join([f"'{s}'" for s in TEST_ETFS])
df_features = pd.read_sql_query(
    f"SELECT * FROM features_v4 WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date",
    conn
)

conn.close()

if df_features.empty:
    print("[错误] features_v4 中没有测试ETF数据")
    sys.exit(1)

print(f"[数据] features_v4: {len(df_features)} 条记录")

# 去重：同一symbol+date保留最新（若存在重复）
df_features = df_features.drop_duplicates(subset=['symbol', 'trade_date'], keep='last')
df = df_features.copy()
df['trade_date'] = pd.to_datetime(df['trade_date'])

# 用二分类模型生成预测（确保所有行都有值）
if rf_binary is not None and lgb_binary is not None:
    X_bin = df[FEATURE_COLS].fillna(0).values
    df['rf_binary_pred'] = rf_binary.predict(X_bin).astype(int)
    df['lgb_binary_pred'] = lgb_binary.predict(X_bin).astype(int)
    print(f"[OK] 二分类预测已生成")
else:
    # 回退：从 predictions_v4 读取
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
    print(f"[数据] predictions_v4: {len(df_pred)} 条记录（用于二分类对比）")

# ==================== 4. 生成三情景概率 ====================
# 确保特征列存在
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    print(f"[警告] 缺失特征列: {missing_cols}")
    # 用0填充缺失列
    for c in missing_cols:
        df[c] = 0.0

X = df[FEATURE_COLS].fillna(0).values

# RF 概率
rf_proba = rf_scenario.predict_proba(X)
# LGB 概率
lgb_proba = lgb_scenario.predict_proba(X)

# 融合概率 (简单平均)
fusion_proba = (rf_proba + lgb_proba) / 2.0

# 情景顺序: 假设 classes_ = [0,1,2] 对应 [adverse, base, favorable]
# 但需根据模型实际classes确认
rf_classes = list(rf_scenario.classes_)
lgb_classes = list(lgb_scenario.classes_)
print(f"[模型] RF classes: {rf_classes}")
print(f"[模型] LGB classes: {lgb_classes}")

# 假设 0=adverse, 1=base, 2=favorable (与训练时一致)
idx_adverse = 0
idx_base = 1
idx_favorable = 2

df['rf_adverse_proba'] = rf_proba[:, idx_adverse]
df['rf_base_proba'] = rf_proba[:, idx_base]
df['rf_favorable_proba'] = rf_proba[:, idx_favorable]

df['lgb_adverse_proba'] = lgb_proba[:, idx_adverse]
df['lgb_base_proba'] = lgb_proba[:, idx_base]
df['lgb_favorable_proba'] = lgb_proba[:, idx_favorable]

df['fusion_adverse_proba'] = fusion_proba[:, idx_adverse]
df['fusion_base_proba'] = fusion_proba[:, idx_base]
df['fusion_favorable_proba'] = fusion_proba[:, idx_favorable]

print(f"[OK] 三情景概率生成完成")

# ==================== 5. 三情景决策规则 ====================
def scenario_decision(row):
    """
    核心决策规则:
    - Base概率 > 0.5 → 维持方向, 仓位=base_weight
    - Adverse概率 > threshold_adverse → 减仓/对冲, 仓位=adverse_weight
    - Favorable概率 > threshold_favorable → 加仓, 仓位=favorable_weight
    - 多条件触发时取概率最高情景对应的仓位
    - 结合binary_pred方向:
      * 预测up但adverse概率高 → 仓位=0 (观望)
      * 预测up且favorable概率高 → 仓位=1.0
    """
    p_adv = row['fusion_adverse_proba']
    p_base = row['fusion_base_proba']
    p_fav = row['fusion_favorable_proba']

    # 默认: base情景
    decision = 'base'
    position = base_weight

    # 检查各情景触发条件
    triggered = []
    if p_base > 0.5:
        triggered.append(('base', p_base, base_weight))
    if p_adv > threshold_adverse:
        triggered.append(('adverse', p_adv, adverse_weight))
    if p_fav > threshold_favorable:
        triggered.append(('favorable', p_fav, favorable_weight))

    if triggered:
        # 取概率最高的情景
        triggered.sort(key=lambda x: x[1], reverse=True)
        decision, _, position = triggered[0]

    # 结合二分类方向进行修正
    # 使用 rf_binary_pred 作为方向参考 (1=up, 0=down)
    binary_up = row.get('rf_binary_pred', 1)
    if pd.isna(binary_up):
        binary_up = 1
    else:
        binary_up = int(binary_up)

    if binary_up == 1:
        # 预测上涨
        if decision == 'adverse' and p_adv > threshold_adverse:
            # 预测up但adverse概率高 → 观望
            position = 0.0
        elif decision == 'favorable' and p_fav > threshold_favorable:
            # 预测up且favorable概率高 → 满仓
            position = 1.0
    else:
        # 预测下跌
        if decision == 'favorable':
            # 预测down但favorable概率高 → 降低仓位，不反向做多
            position = base_weight
        elif decision == 'adverse':
            # 预测down且adverse概率高 → 空仓/对冲
            position = adverse_weight

    # 信号方向
    if position == 0.0:
        signal_dir = 0  # neutral
    elif binary_up == 1:
        signal_dir = 1  # long
    else:
        signal_dir = -1  # short/hedge

    return pd.Series({
        'scenario_decision': decision,
        'position_size': max(0.0, min(1.0, float(position))),
        'signal_direction': int(signal_dir)
    })

print("[计算] 应用三情景决策规则...")
decision_df = df.apply(scenario_decision, axis=1)
df = pd.concat([df, decision_df], axis=1)

# ==================== 6. 整理输出表 ====================
output_cols = [
    'trade_date', 'symbol', 'close',
    'trend_state', 'vol_state', 'drawdown_state', 'sr_state', 'event_state',
    'rf_adverse_proba', 'rf_base_proba', 'rf_favorable_proba',
    'fusion_adverse_proba', 'fusion_base_proba', 'fusion_favorable_proba',
    'scenario_decision', 'position_size', 'signal_direction',
    'rf_binary_pred', 'lgb_binary_pred'
]

# 确保所有列存在
for c in output_cols:
    if c not in df.columns:
        df[c] = None

df_out = df[output_cols].copy()

# 转换类型
df_out['trade_date'] = pd.to_datetime(df_out['trade_date'])
df_out['event_state'] = df_out['event_state'].fillna(0).astype(int)
df_out['rf_binary_pred'] = df_out['rf_binary_pred'].fillna(-1).astype(int)
df_out['lgb_binary_pred'] = df_out['lgb_binary_pred'].fillna(-1).astype(int)

# ==================== 7. 保存到数据库 ====================
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 删除旧表
cursor.execute("DROP TABLE IF EXISTS scenario_signals_v4")

# 创建新表
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

# 插入数据
for _, row in df_out.iterrows():
    cursor.execute("""
        INSERT INTO scenario_signals_v4 (
            trade_date, symbol, close,
            trend_state, vol_state, drawdown_state, sr_state, event_state,
            rf_adverse_proba, rf_base_proba, rf_favorable_proba,
            fusion_adverse_proba, fusion_base_proba, fusion_favorable_proba,
            scenario_decision, position_size, signal_direction,
            rf_binary_pred, lgb_binary_pred
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row['trade_date'].strftime('%Y-%m-%d') if pd.notna(row['trade_date']) else None,
        row['symbol'], row['close'],
        row['trend_state'], row['vol_state'], row['drawdown_state'], row['sr_state'], row['event_state'],
        row['rf_adverse_proba'], row['rf_base_proba'], row['rf_favorable_proba'],
        row['fusion_adverse_proba'], row['fusion_base_proba'], row['fusion_favorable_proba'],
        row['scenario_decision'], row['position_size'], row['signal_direction'],
        row['rf_binary_pred'], row['lgb_binary_pred']
    ))

conn.commit()
conn.close()

print(f"[保存] scenario_signals_v4: {len(df_out)} 条记录已写入数据库")

# ==================== 8. 打印汇总 ====================
print("\n" + "=" * 70)
print("【三情景信号汇总】")
print("=" * 70)

for sym in TEST_ETFS:
    sym_df = df_out[df_out['symbol'] == sym].sort_values('trade_date')
    if len(sym_df) == 0:
        continue
    latest = sym_df.iloc[-1]
    print(f"\n>>> {sym} | 最新日期: {latest['trade_date'].strftime('%Y-%m-%d')}")
    print(f"  收盘: {latest['close']:.3f}")
    print(f"  评估: {latest['trend_state']} | {latest['vol_state']} | {latest['drawdown_state']} | {latest['sr_state']}")
    print(f"  RF概率: 不利={latest['rf_adverse_proba']:.3f} 基准={latest['rf_base_proba']:.3f} 有利={latest['rf_favorable_proba']:.3f}")
    print(f"  融合概率: 不利={latest['fusion_adverse_proba']:.3f} 基准={latest['fusion_base_proba']:.3f} 有利={latest['fusion_favorable_proba']:.3f}")
    print(f"  决策: {latest['scenario_decision']} | 仓位={latest['position_size']:.2f} | 方向={latest['signal_direction']}")
    print(f"  二分类对比: RF={latest['rf_binary_pred']} LGB={latest['lgb_binary_pred']}")

    # 统计分布
    dec_counts = sym_df['scenario_decision'].value_counts()
    print(f"  历史决策分布: {dict(dec_counts)}")

print("\n" + "=" * 70)
print("【Scenario Engine v4 完成】")
print("=" * 70)
