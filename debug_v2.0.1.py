"""调试v2.0.1信号"""
import sqlite3
import pandas as pd
import numpy as np
import importlib.util

# 动态导入带点号的模块
spec = importlib.util.spec_from_file_location("v201", "run_etf_system_v2.0.1.py")
v201 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v201)

from sentiment_engine import MultiFactorEngine

conn = sqlite3.connect("QuanTrade/quant_system/data/quant.db")

# 加载数据
etf_df = pd.read_sql("SELECT * FROM etf_daily_prices WHERE symbol='515070' ORDER BY trade_date", conn)
benchmark_df = pd.read_sql("SELECT * FROM etf_daily_prices WHERE symbol='510300' ORDER BY trade_date", conn)
gold_df = pd.read_sql("SELECT * FROM gold_daily_prices ORDER BY date", conn)
conn.close()

# 转换格式
etf_df['trade_date'] = pd.to_datetime(etf_df['trade_date'])
etf_df = etf_df.set_index('trade_date')

benchmark_df['trade_date'] = pd.to_datetime(benchmark_df['trade_date'])
benchmark_df = benchmark_df.set_index('trade_date')

gold_df['date'] = pd.to_datetime(gold_df['date'])
gold_df = gold_df.set_index('date')

print("=" * 60)
print("调试v2.0.1信号")
print("=" * 60)

# Layer 1: ML预测
features = v201.calculate_features(etf_df)
labels = v201.create_labels(etf_df['close'])

print(f"\n[Layer 1] ML模型:")
print(f"  特征形状: {features.shape}")

model, acc = v201.train_ml_model(features, labels, 'lightgbm')
print(f"  模型准确率: {acc:.2%}")

# 预测最后一天
X = features.iloc[[-1]].dropna()
if len(X) > 0:
    ml_prob = model.predict(X)[0]
    print(f"  最后一天ML概率: {ml_prob:.4f}")
else:
    ml_prob = 0.5
    print("  无法预测")

# Layer 2: 热点赛道得分
sector_score = v201.calculate_sector_score(etf_df, benchmark_df)
print(f"\n[Layer 2] 热点赛道得分: {sector_score:.4f}")

# Layer 4: 多因子
multi_factor = MultiFactorEngine()
factors = multi_factor.calculate_all(etf_df, gold_df=gold_df)
factor_score = factors['combined'].iloc[-1] if len(factors) > 0 else 0
print(f"\n[Layer 4] 多因子得分: {factor_score:.4f}")

# 综合得分
final_score = ml_prob * 0.4 + sector_score * 0.3 + factor_score * 0.3
print(f"\n[综合] 最终得分: {final_score:.4f}")
print(f"  ML概率(0.4): {ml_prob * 0.4:.4f}")
print(f"  赛道得分(0.3): {sector_score * 0.3:.4f}")
print(f"  多因子(0.3): {factor_score * 0.3:.4f}")

print(f"\n[结论]")
print(f"  阈值: 0.5")
print(f"  最终得分: {final_score:.4f}")
print(f"  是否交易: {'是' if final_score > 0.5 else '否'}")
