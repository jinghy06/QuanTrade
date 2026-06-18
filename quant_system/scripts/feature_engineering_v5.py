#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程 v5 - 基于 daily_prices_v5（2005年至今，22只股票，~11万条）
与v4相同的特征体系，但数据量扩展约14倍
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

def load_prices_v5():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT trade_date, symbol, open, high, low, close, volume, amount, pct_change, turnover
        FROM daily_prices_v5
        ORDER BY symbol, trade_date
    """, conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def compute_features(df):
    """计算全部特征（与v4一致）"""
    print(f"  原始数据: {len(df)}条, {df['symbol'].nunique()}只股票")
    
    # 按股票分组计算
    all_features = []
    
    for symbol in sorted(df['symbol'].unique()):
        s = df[df['symbol'] == symbol].copy().sort_values('trade_date')
        if len(s) < 130:
            continue
        
        # 基础价格特征
        s['returns_1d'] = s['close'].pct_change()
        s['returns_5d'] = s['close'].pct_change(5)
        s['returns_10d'] = s['close'].pct_change(10)
        s['returns_20d'] = s['close'].pct_change(20)
        s['returns_60d'] = s['close'].pct_change(60)
        s['returns_120d'] = s['close'].pct_change(120)
        
        # 波动率
        s['volatility_20d'] = s['returns_1d'].rolling(20).std()
        s['volatility_60d'] = s['returns_1d'].rolling(60).std()
        s['volatility_120d'] = s['returns_1d'].rolling(120).std()
        
        # ATR
        s['tr'] = np.maximum(s['high'] - s['low'],
                             np.maximum(abs(s['high'] - s['close'].shift(1)),
                                        abs(s['low'] - s['close'].shift(1))))
        s['atr_14'] = s['tr'].rolling(14).mean()
        s['atr_20'] = s['tr'].rolling(20).mean()
        s['atr_60'] = s['tr'].rolling(60).mean()
        
        # 均线
        for window in [5, 10, 20, 60, 120]:
            s[f'ma_{window}'] = s['close'].rolling(window).mean()
            s[f'ma_ratio_{window}'] = s['close'] / s[f'ma_{window}']
        
        # 成交量
        s['volume_ma_20'] = s['volume'].rolling(20).mean()
        s['volume_ratio'] = s['volume'] / s['volume_ma_20']
        s['volume_ma_60'] = s['volume'].rolling(60).mean()
        
        # 价格位置
        s['high_20d'] = s['high'].rolling(20).max()
        s['low_20d'] = s['low'].rolling(20).min()
        s['price_position_20d'] = (s['close'] - s['low_20d']) / (s['high_20d'] - s['low_20d'] + 1e-10)
        
        s['high_60d'] = s['high'].rolling(60).max()
        s['low_60d'] = s['low'].rolling(60).min()
        s['price_position_60d'] = (s['close'] - s['low_60d']) / (s['high_60d'] - s['low_60d'] + 1e-10)
        
        # 回撤
        s['cummax_120d'] = s['close'].cummax()
        s['drawdown_120d'] = (s['close'] - s['cummax_120d']) / s['cummax_120d']
        s['days_since_peak'] = (s['trade_date'] - s.loc[s['close'].cummax().idxmax(), 'trade_date']).dt.days
        
        # 长窗口特征
        s['trend_120d_return'] = s['close'].pct_change(120)
        s['vol_regime_ratio'] = s['volatility_20d'] / s['volatility_60d'].replace(0, np.nan)
        s['drawdown_recovery_prob'] = 1.0 / (1.0 + np.exp(-s['drawdown_120d'] * 10))
        
        # 支撑阻力
        s['support_proximity_60d'] = (s['close'] - s['low_60d']) / (s['high_60d'] - s['low_60d'] + 1e-10)
        s['resistance_proximity_60d'] = (s['high_60d'] - s['close']) / (s['high_60d'] - s['low_60d'] + 1e-10)
        
        # 事件特征
        s['event_zscore_20d'] = (s['returns_1d'] - s['returns_1d'].rolling(20).mean()) / s['returns_1d'].rolling(20).std().replace(0, np.nan)
        
        # Assessment状态
        s['trend_state'] = np.where(s['returns_60d'] > 0.05, 'uptrend',
                                    np.where(s['returns_60d'] < -0.05, 'downtrend', 'sideways'))
        s['vol_state'] = np.where(s['volatility_20d'] > s['volatility_20d'].rolling(60).quantile(0.8), 'high_vol',
                                  np.where(s['volatility_20d'] < s['volatility_20d'].rolling(60).quantile(0.2), 'low_vol', 'normal_vol'))
        s['drawdown_state'] = np.where(s['drawdown_120d'] < -0.15, 'deep_dd',
                                       np.where(s['drawdown_120d'] < -0.05, 'moderate_dd', 'shallow_dd'))
        s['sr_state'] = np.where(s['price_position_60d'] > 0.8, 'near_resistance',
                                 np.where(s['price_position_60d'] < 0.2, 'near_support', 'mid_range'))
        
        # 目标变量
        s['target_return_10d'] = s['close'].shift(-10) / s['close'] - 1
        s['target_direction_10d'] = (s['target_return_10d'] > 0).astype(int)
        
        # 三情景标签
        def label_scenario(r):
            if pd.isna(r):
                return 'unknown'
            elif r > 0.05:
                return 'favorable'
            elif r < -0.05:
                return 'adverse'
            else:
                return 'base'
        s['scenario_label_10d'] = s['target_return_10d'].apply(label_scenario)
        
        all_features.append(s)
    
    result = pd.concat(all_features, ignore_index=True)
    
    # 删除NaN过多的行（需要120日历史数据）
    result = result.dropna(subset=['returns_120d', 'target_return_10d'])
    
    return result

def main():
    print("=" * 70)
    print("特征工程 v5 - 基于 daily_prices_v5 (~11万条, 2005-2025)")
    print("=" * 70)
    
    print("\n[1/3] 加载价格数据...")
    prices = load_prices_v5()
    
    print(f"\n[2/3] 计算特征...")
    features = compute_features(prices)
    
    print(f"\n[3/3] 保存到数据库...")
    conn = sqlite3.connect(DB_PATH)
    
    # 保存到features_v5
    conn.execute("DROP TABLE IF EXISTS features_v5")
    features.to_sql('features_v5', conn, if_exists='replace', index=False)
    
    # 创建索引
    conn.execute("CREATE INDEX idx_f5_date ON features_v5(trade_date)")
    conn.execute("CREATE INDEX idx_f5_symbol ON features_v5(symbol)")
    conn.execute("CREATE INDEX idx_f5_sym_date ON features_v5(symbol, trade_date)")
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*70}")
    print(f"特征工程完成!")
    print(f"  总记录数: {len(features)}")
    print(f"  股票数量: {features['symbol'].nunique()}")
    print(f"  时间范围: {features['trade_date'].min().date()} ~ {features['trade_date'].max().date()}")
    print(f"  特征列数: {len(features.columns)}")
    
    # 目标分布
    print(f"\n  目标分布:")
    for label, cnt in features['scenario_label_10d'].value_counts().items():
        print(f"    {label}: {cnt} ({cnt/len(features)*100:.1f}%)")

if __name__ == '__main__':
    main()
