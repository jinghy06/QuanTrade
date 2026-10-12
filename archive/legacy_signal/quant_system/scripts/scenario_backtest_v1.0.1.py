"""
Scenario Engine + Backtest v4 Round 2 — Using Optimized Models
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

# 激进调优配置
AGGRESSIVE_WEIGHTS = {
    "adverse_weight": 0.50,
    "base_weight": 0.80,
    "favorable_weight": 1.0,
    "threshold_adverse": 0.45,
    "threshold_favorable": 0.20,
    "smooth_mode": True,
    "smooth_scale": 0.8,
    "min_position": 0.50,
    "up_trend_boost": 0.15
}

print("=" * 70)
print("【Scenario Engine + Backtest v4 Round 2 — Optimized】")
print("=" * 70)

# ==================== 1. 加载Round 2优化模型 ====================
rf_path = os.path.join(MODEL_DIR, 'rf_r2_scenario.pkl')
lgb_path = os.path.join(MODEL_DIR, 'lgb_r2_scenario.pkl')
feat_path = os.path.join(MODEL_DIR, 'feature_cols_r2.json')

for p in [rf_path, feat_path]:
    if not os.path.exists(p):
        print(f"[错误] 模型文件不存在: {p}")
        sys.exit(1)

with open(feat_path, 'r') as f:
    FEATURE_COLS = json.load(f)

with open(rf_path, 'rb') as f:
    rf_model = pickle.load(f)

lgb_model = None
if os.path.exists(lgb_path):
    with open(lgb_path, 'rb') as f:
        lgb_model = pickle.load(f)
    print("[OK] 加载模型: RF + LGBM (Round 2)")
else:
    print("[OK] 加载模型: RF (Round 2)")

# ==================== 2. 读取数据 ====================
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

# 确保特征列存在
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
for c in missing_cols:
    df[c] = 0.0

X = df[FEATURE_COLS].fillna(0).values

# ==================== 3. 生成预测 ====================
rf_proba = rf_model.predict_proba(X)
rf_pred = rf_model.predict(X)

if lgb_model is not None:
    lgb_proba = lgb_model.predict_proba(X)
    lgb_pred = lgb_model.predict(X)
    fusion_proba = (rf_proba + lgb_proba) / 2
    print("[OK] 融合RF+LGBM概率")
else:
    lgb_proba = None
    lgb_pred = None
    fusion_proba = rf_proba

fusion_pred = np.argmax(fusion_proba, axis=1)

idx_adverse, idx_base, idx_favorable = 0, 1, 2

df['rf_adverse_proba'] = rf_proba[:, idx_adverse]
df['rf_base_proba'] = rf_proba[:, idx_base]
df['rf_favorable_proba'] = rf_proba[:, idx_favorable]
df['fusion_adverse_proba'] = fusion_proba[:, idx_adverse]
df['fusion_base_proba'] = fusion_proba[:, idx_base]
df['fusion_favorable_proba'] = fusion_proba[:, idx_favorable]

if lgb_model is not None:
    df['lgb_adverse_proba'] = lgb_proba[:, idx_adverse]
    df['lgb_base_proba'] = lgb_proba[:, idx_base]
    df['lgb_favorable_proba'] = lgb_proba[:, idx_favorable]

# ==================== 4. 三情景决策（激进版） ====================
def scenario_decision_aggressive(row):
    p_adv = row['fusion_adverse_proba']
    p_base = row['fusion_base_proba']
    p_fav = row['fusion_favorable_proba']
    
    position = AGGRESSIVE_WEIGHTS['base_weight'] + (p_fav - p_adv) * AGGRESSIVE_WEIGHTS['smooth_scale']
    
    # 根据binary方向微调（用fusion_pred作为方向参考）
    binary_up = 1 if row.get('fusion_pred', 1) in [1, 2] else 0
    if binary_up == 1:
        position += AGGRESSIVE_WEIGHTS['up_trend_boost']
    else:
        position -= AGGRESSIVE_WEIGHTS['up_trend_boost'] * 0.5
    
    position = max(AGGRESSIVE_WEIGHTS['min_position'], min(1.0, position))
    
    probs = [('adverse', p_adv), ('base', p_base), ('favorable', p_fav)]
    probs.sort(key=lambda x: x[1], reverse=True)
    decision = probs[0][0]
    
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

print("[计算] 应用Round 2优化模型 + 激进决策规则...")
decision_df = df.apply(scenario_decision_aggressive, axis=1)
df = pd.concat([df, decision_df], axis=1)

# ==================== 5. 保存信号 ====================
output_cols = [
    'trade_date', 'symbol', 'close',
    'trend_state', 'vol_state', 'drawdown_state', 'sr_state', 'event_state',
    'fusion_adverse_proba', 'fusion_base_proba', 'fusion_favorable_proba',
    'scenario_decision', 'position_size', 'signal_direction'
]
for c in output_cols:
    if c not in df.columns:
        df[c] = None

df_out = df[output_cols].copy()
df_out['trade_date'] = pd.to_datetime(df_out['trade_date'])
df_out['event_state'] = df_out['event_state'].fillna(0).astype(int)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS scenario_signals_v4_r2")
cursor.execute("""
CREATE TABLE scenario_signals_v4_r2 (
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
    signal_direction INTEGER
)
""")
conn.commit()
df_out.to_sql('scenario_signals_v4_r2', conn, if_exists='append', index=False)
conn.commit()
conn.close()

print(f"[保存] scenario_signals_v4_r2: {len(df_out)} 条")

# ==================== 6. 回测 ====================
print("\n【6. Round 2 回测...】")

conn = sqlite3.connect(DB_PATH)
df_price = pd.read_sql_query(
    f"SELECT * FROM daily_prices WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date",
    conn
)
df_price['trade_date'] = pd.to_datetime(df_price['trade_date'], format='mixed')
df_price = df_price.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
conn.close()

def backtest_strategy(df_price, df_signals, strategy_name, position_col='position_size',
                      fee_rate=0.0001, slippage=0.0001, initial_capital=1_000_000):
    results = []
    daily_records = []
    
    for symbol in TEST_ETFS:
        price_df = df_price[df_price['symbol'] == symbol].sort_values('trade_date').reset_index(drop=True)
        sig_df = df_signals[df_signals['symbol'] == symbol].sort_values('trade_date').reset_index(drop=True)
        
        if len(price_df) < 10 or len(sig_df) < 5:
            continue
        
        merged = pd.merge(price_df, sig_df[['trade_date', position_col, 'signal_direction']], 
                         on='trade_date', how='left')
        merged[position_col] = merged[position_col].fillna(0)
        merged['return'] = merged['close'].pct_change()
        
        capital = initial_capital
        position = 0.0
        nav_history = []
        
        for i in range(1, len(merged)):
            row = merged.iloc[i]
            target_pos = float(row[position_col]) if not pd.isna(row[position_col]) else 0.0
            effective_pos = target_pos
            
            if abs(effective_pos - position) > 0.01:
                trade_value = abs(effective_pos - position) * capital
                fee = trade_value * (fee_rate + slippage)
                capital -= fee
                position = effective_pos
            
            daily_return = row['return'] if not pd.isna(row['return']) else 0
            capital *= (1 + daily_return * position)
            
            nav_history.append({
                'trade_date': row['trade_date'],
                'symbol': symbol,
                'strategy': strategy_name,
                'nav': capital,
                'position': position,
                'signal': 1 if position > 0 else 0,
                'daily_return': daily_return * position,
                'close': row['close']
            })
        
        # 基准
        benchmark_nav = initial_capital
        for i in range(1, len(merged)):
            row = merged.iloc[i]
            daily_return = row['return'] if not pd.isna(row['return']) else 0
            benchmark_nav *= (1 + daily_return)
        
        nav_series = pd.Series([n['nav'] for n in nav_history])
        if len(nav_series) > 0:
            total_return = nav_series.iloc[-1] / initial_capital - 1
            days = len(nav_series)
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            cummax = nav_series.cummax()
            drawdown = (nav_series - cummax) / cummax
            max_drawdown = drawdown.min()
            daily_rets = pd.Series([n['daily_return'] for n in nav_history])
            sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
            win_rate = (daily_rets > 0).mean()
            benchmark_total = benchmark_nav / initial_capital - 1
            
            results.append({
                'symbol': symbol,
                'strategy': strategy_name,
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'benchmark_return': benchmark_total,
                'trading_days': days
            })
        
        # 基准历史
        benchmark_nav = initial_capital
        for i in range(1, len(merged)):
            row = merged.iloc[i]
            daily_return = row['return'] if not pd.isna(row['return']) else 0
            benchmark_nav *= (1 + daily_return)
            daily_records.append({
                'trade_date': row['trade_date'],
                'symbol': symbol,
                'strategy': '基准(持有)',
                'nav': benchmark_nav,
                'position': 1.0,
                'signal': 1,
                'daily_return': daily_return,
                'close': row['close']
            })
        
        daily_records.extend(nav_history)
    
    return pd.DataFrame(results), pd.DataFrame(daily_records)

# 三情景Round2策略
res_r2, daily_r2 = backtest_strategy(df_price, df_out, '三情景Round2')

# 三情景Round2 + Kelly
df_kelly = df_out.copy()
df_kelly['kelly_pos'] = (2 * df_kelly['fusion_favorable_proba'] - 1).clip(0, 1)
df_kelly['position_size'] = (df_kelly['position_size'] * 0.5 + df_kelly['kelly_pos'] * 0.5).clip(0.2, 1.0)
res_r2_kelly, daily_r2_kelly = backtest_strategy(df_price, df_kelly, '三情景Round2+Kelly')

# 汇总
all_results = pd.concat([res_r2, res_r2_kelly], ignore_index=True)
all_daily = pd.concat([daily_r2, daily_r2_kelly], ignore_index=True)

print("\n【Round 2 回测结果】")
for sym in TEST_ETFS:
    sym_df = all_results[all_results['symbol'] == sym]
    if len(sym_df) == 0:
        continue
    print(f"\n>>> {sym}")
    print(sym_df[['strategy', 'total_return', 'annual_return', 'max_drawdown', 'sharpe', 'win_rate', 'benchmark_return']].to_string(index=False))

print("\n【平均表现】")
avg_df = all_results.groupby('strategy').agg({
    'total_return': 'mean',
    'benchmark_return': 'mean',
    'max_drawdown': 'mean',
    'sharpe': 'mean',
    'win_rate': 'mean'
}).round(4).reset_index()
print(avg_df.to_string(index=False))

# 保存
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS backtest_results_v4_r2")
cursor.execute("""
CREATE TABLE backtest_results_v4_r2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    strategy TEXT,
    total_return REAL,
    annual_return REAL,
    max_drawdown REAL,
    sharpe REAL,
    win_rate REAL,
    benchmark_return REAL,
    trading_days INTEGER
)
""")
conn.commit()
all_results.to_sql('backtest_results_v4_r2', conn, if_exists='append', index=False)

cursor.execute("DROP TABLE IF EXISTS backtest_daily_v4_r2")
cursor.execute("""
CREATE TABLE backtest_daily_v4_r2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TIMESTAMP,
    symbol TEXT,
    strategy TEXT,
    nav REAL,
    position REAL,
    signal INTEGER,
    daily_return REAL,
    close REAL
)
""")
conn.commit()
all_daily.to_sql('backtest_daily_v4_r2', conn, if_exists='append', index=False)
conn.commit()
conn.close()

print(f"\n[保存] backtest_results_v4_r2: {len(all_results)} 条")
print(f"[保存] backtest_daily_v4_r2: {len(all_daily)} 条")

print("\n" + "=" * 70)
print("【Scenario Engine + Backtest v4 Round 2 COMPLETE】")
print("=" * 70)
