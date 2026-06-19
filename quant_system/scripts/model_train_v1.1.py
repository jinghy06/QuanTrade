#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练 v5 - 基于 features_v5（107,248条，20年数据）
时间划分: 训练2005-2022 | 验证2023 | 测试2024-2025
沿用Round 2优化: 保留全部特征 + 手动随机过采样 + 增加树数量
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'
MODEL_DIR = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\models\v5'

# 特征列（排除非特征列）
EXCLUDE_COLS = ['trade_date', 'symbol', 'target_return_10d', 'target_direction_10d',
                'scenario_label_10d', 'trend_state', 'vol_state', 'drawdown_state', 'sr_state']

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM features_v5", conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

def time_split(df):
    """时间划分: 训练2005-2022 | 验证2023 | 测试2024-2025"""
    train = df[df['trade_date'] < '2023-01-01']
    val = df[(df['trade_date'] >= '2023-01-01') & (df['trade_date'] < '2024-01-01')]
    test = df[df['trade_date'] >= '2024-01-01']
    return train, val, test

def manual_oversample(X, y, scenario_y, random_state=42):
    """手动随机过采样 - 三情景平衡"""
    np.random.seed(random_state)
    
    # 找到每个类别的索引
    classes, counts = np.unique(scenario_y, return_counts=True)
    max_count = max(counts)
    
    new_X, new_y, new_scenario = [], [], []
    
    for cls in classes:
        idx = np.where(scenario_y == cls)[0]
        n_samples = len(idx)
        
        # 保留原始样本
        new_X.append(X.iloc[idx])
        new_y.append(y.iloc[idx])
        new_scenario.append(scenario_y.iloc[idx])
        
        # 随机过采样到最大数量
        if n_samples < max_count:
            n_needed = max_count - n_samples
            sampled_idx = np.random.choice(idx, size=n_needed, replace=True)
            new_X.append(X.iloc[sampled_idx])
            new_y.append(y.iloc[sampled_idx])
            new_scenario.append(scenario_y.iloc[sampled_idx])
    
    X_bal = pd.concat(new_X, ignore_index=True)
    y_bal = pd.concat(new_y, ignore_index=True)
    s_bal = pd.concat(new_scenario, ignore_index=True)
    
    # 打乱顺序
    shuffle_idx = np.random.permutation(len(X_bal))
    return X_bal.iloc[shuffle_idx].reset_index(drop=True), \
           y_bal.iloc[shuffle_idx].reset_index(drop=True), \
           s_bal.iloc[shuffle_idx].reset_index(drop=True)

def train_models(X_train, y_train_dir, y_train_scenario, X_val, y_val_dir, y_val_scenario, feature_cols):
    """训练所有模型"""
    
    # 1. 手动过采样
    print("\n  手动随机过采样...")
    X_bal, y_bal_dir, y_bal_scenario = manual_oversample(X_train, y_train_dir, y_train_scenario)
    print(f"    平衡后: {len(X_bal)}条")
    for cls, cnt in zip(*np.unique(y_bal_scenario, return_counts=True)):
        print(f"      {cls}: {cnt}")
    
    # 2. RF 3-scenario
    print("\n  训练 RF 3-scenario...")
    rf_scenario = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_scenario.fit(X_bal[feature_cols], y_bal_scenario)
    
    # 3. LGBM 3-scenario
    print("  训练 LGBM 3-scenario...")
    lgb_scenario = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=8,
        num_leaves=64, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
        verbose=-1
    )
    lgb_scenario.fit(X_bal[feature_cols], y_bal_scenario,
                     eval_set=[(X_val[feature_cols], y_val_scenario)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # 4. RF Binary
    print("  训练 RF Binary...")
    rf_binary = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_binary.fit(X_bal[feature_cols], y_bal_dir)
    
    # 5. LGBM Binary
    print("  训练 LGBM Binary...")
    lgb_binary = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=8,
        num_leaves=64, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
        verbose=-1
    )
    lgb_binary.fit(X_bal[feature_cols], y_bal_dir,
                   eval_set=[(X_val[feature_cols], y_val_dir)],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    
    return rf_scenario, lgb_scenario, rf_binary, lgb_binary

def evaluate_models(models, X_test, y_test_dir, y_test_scenario, feature_cols):
    """评估模型"""
    rf_s, lgb_s, rf_b, lgb_b = models
    
    print(f"\n{'='*70}")
    print("测试集评估 (2024-2025)")
    print(f"{'='*70}")
    
    # RF 3-scenario
    rf_s_pred = rf_s.predict(X_test[feature_cols])
    rf_s_acc = accuracy_score(y_test_scenario, rf_s_pred)
    print(f"\nRF 3-scenario: {rf_s_acc*100:.2f}%")
    print(classification_report(y_test_scenario, rf_s_pred, digits=3))
    
    # LGBM 3-scenario
    lgb_s_pred = lgb_s.predict(X_test[feature_cols])
    lgb_s_acc = accuracy_score(y_test_scenario, lgb_s_pred)
    print(f"\nLGBM 3-scenario: {lgb_s_acc*100:.2f}%")
    print(classification_report(y_test_scenario, lgb_s_pred, digits=3))
    
    # Fusion 3-scenario
    rf_s_proba = rf_s.predict_proba(X_test[feature_cols])
    lgb_s_proba = lgb_s.predict_proba(X_test[feature_cols])
    fusion_proba = (rf_s_proba + lgb_s_proba) / 2
    fusion_pred = rf_s.classes_[np.argmax(fusion_proba, axis=1)]
    fusion_acc = accuracy_score(y_test_scenario, fusion_pred)
    print(f"\nFusion 3-scenario: {fusion_acc*100:.2f}%")
    
    # RF Binary
    rf_b_pred = rf_b.predict(X_test[feature_cols])
    rf_b_acc = accuracy_score(y_test_dir, rf_b_pred)
    print(f"\nRF Binary: {rf_b_acc*100:.2f}%")
    
    # LGBM Binary
    lgb_b_pred = lgb_b.predict(X_test[feature_cols])
    lgb_b_acc = accuracy_score(y_test_dir, lgb_b_pred)
    print(f"\nLGBM Binary: {lgb_b_acc*100:.2f}%")
    
    return {
        'rf_scenario_acc': rf_s_acc,
        'lgb_scenario_acc': lgb_s_acc,
        'fusion_scenario_acc': fusion_acc,
        'rf_binary_acc': rf_b_acc,
        'lgb_binary_acc': lgb_b_acc,
    }

def save_models(models, feature_cols, metrics):
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    rf_s, lgb_s, rf_b, lgb_b = models
    
    pickle.dump(rf_s, open(f'{MODEL_DIR}/rf_v5_scenario.pkl', 'wb'))
    pickle.dump(lgb_s, open(f'{MODEL_DIR}/lgb_v5_scenario.pkl', 'wb'))
    pickle.dump(rf_b, open(f'{MODEL_DIR}/rf_v5_binary.pkl', 'wb'))
    pickle.dump(lgb_b, open(f'{MODEL_DIR}/lgb_v5_binary.pkl', 'wb'))
    
    with open(f'{MODEL_DIR}/feature_cols_v5.json', 'w') as f:
        json.dump(feature_cols, f)
    
    with open(f'{MODEL_DIR}/metrics_v5.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n模型已保存到 {MODEL_DIR}")

def main():
    print("=" * 70)
    print("模型训练 v5 - 基于 features_v5 (107,248条, 20年)")
    print("=" * 70)
    
    print("\n[1/4] 加载数据...")
    df = load_data()
    feature_cols = get_feature_cols(df)
    print(f"  特征数: {len(feature_cols)}")
    
    print("\n[2/4] 时间划分...")
    train, val, test = time_split(df)
    print(f"  训练集: {len(train)}条 ({train['trade_date'].min().date()}~{train['trade_date'].max().date()})")
    print(f"  验证集: {len(val)}条 ({val['trade_date'].min().date()}~{val['trade_date'].max().date()})")
    print(f"  测试集: {len(test)}条 ({test['trade_date'].min().date()}~{test['trade_date'].max().date()})")
    
    # 目标变量
    y_train_dir = train['target_direction_10d']
    y_train_scenario = train['scenario_label_10d']
    y_val_dir = val['target_direction_10d']
    y_val_scenario = val['scenario_label_10d']
    y_test_dir = test['target_direction_10d']
    y_test_scenario = test['scenario_label_10d']
    
    print(f"\n  训练集目标分布:")
    for label, cnt in y_train_scenario.value_counts().items():
        print(f"    {label}: {cnt} ({cnt/len(y_train_scenario)*100:.1f}%)")
    
    print("\n[3/4] 训练模型...")
    models = train_models(train, y_train_dir, y_train_scenario, val, y_val_dir, y_val_scenario, feature_cols)
    
    print("\n[4/4] 评估与保存...")
    metrics = evaluate_models(models, test, y_test_dir, y_test_scenario, feature_cols)
    save_models(models, feature_cols, metrics)
    
    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"  Fusion 3-scenario: {metrics['fusion_scenario_acc']*100:.2f}%")
    print(f"  LGBM Binary: {metrics['lgb_binary_acc']*100:.2f}%")

if __name__ == '__main__':
    main()
