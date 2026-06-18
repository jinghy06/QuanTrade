#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成v5预测信号 - 基于训练好的v5模型
对测试集(2024-2025)和5只ETF生成预测
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'
MODEL_DIR = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\models\v5'

ETF_LIST = ['562500.SH', '588200.SH', '588790.SH', '159382.SZ', '159241.SZ']

def load_models():
    rf_s = pickle.load(open(f'{MODEL_DIR}/rf_v5_scenario.pkl', 'rb'))
    lgb_s = pickle.load(open(f'{MODEL_DIR}/lgb_v5_scenario.pkl', 'rb'))
    rf_b = pickle.load(open(f'{MODEL_DIR}/rf_v5_binary.pkl', 'rb'))
    lgb_b = pickle.load(open(f'{MODEL_DIR}/lgb_v5_binary.pkl', 'rb'))
    with open(f'{MODEL_DIR}/feature_cols_v5.json', 'r') as f:
        feature_cols = json.load(f)
    return rf_s, lgb_s, rf_b, lgb_b, feature_cols

def generate_predictions():
    print("=" * 70)
    print("生成v5预测信号")
    print("=" * 70)
    
    # 加载模型
    print("\n[1/3] 加载模型...")
    rf_s, lgb_s, rf_b, lgb_b, feature_cols = load_models()
    
    # 加载特征数据
    print("[2/3] 加载特征数据...")
    conn = sqlite3.connect(DB_PATH)
    
    # 对测试集(2024-2025)和ETF生成预测
    df = pd.read_sql_query("""
        SELECT * FROM features_v5 
        WHERE trade_date >= '2024-01-01'
        ORDER BY symbol, trade_date
    """, conn)
    conn.close()
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    print(f"  数据: {len(df)}条, {df['symbol'].nunique()}只股票")
    
    # 确保特征列存在
    available_cols = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(df.columns)
    if missing:
        print(f"  警告: 缺失特征 {missing}")
    
    # 处理NaN
    X = df[available_cols].fillna(0)
    
    # 生成预测
    print("[3/3] 生成预测...")
    
    # 三情景概率
    rf_s_proba = rf_s.predict_proba(X)
    lgb_s_proba = lgb_s.predict_proba(X)
    fusion_proba = (rf_s_proba + lgb_s_proba) / 2
    
    # 获取类别顺序
    classes = rf_s.classes_
    
    # 构建预测结果
    pred_df = pd.DataFrame({
        'trade_date': df['trade_date'],
        'symbol': df['symbol'],
        'close': df['close'],
        'target_return_10d': df['target_return_10d'],
        'target_direction_10d': df['target_direction_10d'],
        'scenario_label_10d': df['scenario_label_10d'],
        'rf_v5_pred': rf_s.predict(X),
        'lgb_v5_pred': lgb_s.predict(X),
        'fusion_v5_pred': classes[np.argmax(fusion_proba, axis=1)],
        'rf_v5_adverse': rf_s_proba[:, list(classes).index('adverse')] if 'adverse' in classes else 0,
        'rf_v5_base': rf_s_proba[:, list(classes).index('base')] if 'base' in classes else 0,
        'rf_v5_favorable': rf_s_proba[:, list(classes).index('favorable')] if 'favorable' in classes else 0,
        'lgb_v5_adverse': lgb_s_proba[:, list(classes).index('adverse')] if 'adverse' in classes else 0,
        'lgb_v5_base': lgb_s_proba[:, list(classes).index('base')] if 'base' in classes else 0,
        'lgb_v5_favorable': lgb_s_proba[:, list(classes).index('favorable')] if 'favorable' in classes else 0,
        'fusion_v5_adverse': fusion_proba[:, list(classes).index('adverse')] if 'adverse' in classes else 0,
        'fusion_v5_base': fusion_proba[:, list(classes).index('base')] if 'base' in classes else 0,
        'fusion_v5_favorable': fusion_proba[:, list(classes).index('favorable')] if 'favorable' in classes else 0,
        'rf_v5_binary_pred': rf_b.predict(X),
        'lgb_v5_binary_pred': lgb_b.predict(X),
    })
    
    # 保存到数据库
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS predictions_v5")
    pred_df.to_sql('predictions_v5', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX idx_p5_date ON predictions_v5(trade_date)")
    conn.execute("CREATE INDEX idx_p5_symbol ON predictions_v5(symbol)")
    conn.commit()
    conn.close()
    
    print(f"\n预测完成! 保存 {len(pred_df)}条到 predictions_v5")
    
    # 统计
    print(f"\n  预测分布:")
    for label, cnt in pred_df['fusion_v5_pred'].value_counts().items():
        print(f"    {label}: {cnt} ({cnt/len(pred_df)*100:.1f}%)")
    
    # ETF部分
    etf_pred = pred_df[pred_df['symbol'].isin(ETF_LIST)]
    print(f"\n  ETF预测: {len(etf_pred)}条")
    for sym in ETF_LIST:
        sym_df = etf_pred[etf_pred['symbol'] == sym]
        if not sym_df.empty:
            print(f"    {sym}: {len(sym_df)}条")

if __name__ == '__main__':
    generate_predictions()
