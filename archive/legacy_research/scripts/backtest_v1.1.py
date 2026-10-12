#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v5模型回测 - 基于训练集股票2024-2025年预测
使用趋势跟踪加仓 + 股灾预判策略
"""

import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

# 策略参数
SCENARIO_POSITIONS = {'adverse': 0.60, 'base': 0.90, 'favorable': 1.00}

TREND_CONFIG = {
    'trend_threshold_strong': 0.15,
    'trend_threshold_moderate': 0.08,
    'ma_bullish_required': True,
    'add_position_step': 0.20,
    'add_position_max': 1.00,
    'add_position_cooldown': 3,
}

CRASH_CONFIG = {
    'vol_spike_ratio': 3.0,
    'drawdown_threshold': 0.20,
    'liquidity_dryup_ratio': 0.3,
    'yellow_alert_reduction': 0.20,
    'orange_alert_reduction': 0.50,
    'red_alert_reduction': 1.00,
    'recovery_required_days': 5,
}

def load_data():
    conn = sqlite3.connect(DB_PATH)
    
    # 加载v5预测
    pred_df = pd.read_sql_query("""
        SELECT trade_date, symbol, close, fusion_v5_adverse as adverse,
               fusion_v5_base as base, fusion_v5_favorable as favorable,
               target_return_10d, scenario_label_10d
        FROM predictions_v5
        ORDER BY symbol, trade_date
    """, conn)
    
    # 加载价格数据（用于计算趋势指标）
    price_df = pd.read_sql_query("""
        SELECT trade_date, symbol, close, high, low, volume
        FROM daily_prices_v5
        WHERE trade_date >= '2024-01-01'
        ORDER BY symbol, trade_date
    """, conn)
    
    conn.close()
    
    pred_df['trade_date'] = pd.to_datetime(pred_df['trade_date'])
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    
    return pred_df, price_df

def calculate_indicators(price_df):
    df = price_df.copy().sort_values(['symbol', 'trade_date'])
    
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        s = df.loc[mask].copy()
        
        s['return_120d'] = s['close'].pct_change(120)
        s['ma_20'] = s['close'].rolling(20).mean()
        s['ma_60'] = s['close'].rolling(60).mean()
        s['ma_120'] = s['close'].rolling(120).mean()
        s['ma_bullish'] = (s['ma_20'] > s['ma_60']) & (s['ma_60'] > s['ma_120'])
        
        s['high_120d'] = s['high'].rolling(120).max()
        s['drawdown_from_high'] = (s['close'] - s['high_120d']) / s['high_120d']
        
        s['tr'] = np.maximum(s['high'] - s['low'],
                             np.maximum(abs(s['high'] - s['close'].shift(1)),
                                        abs(s['low'] - s['close'].shift(1))))
        s['atr_14'] = s['tr'].rolling(14).mean()
        s['atr_20'] = s['tr'].rolling(20).mean()
        s['vol_spike'] = s['atr_20'] / s['atr_14']
        
        s['volume_ma_20'] = s['volume'].rolling(20).mean()
        s['liquidity_ratio'] = s['volume'] / s['volume_ma_20']
        
        s['momentum_20d'] = s['close'].pct_change(20)
        s['momentum_60d'] = s['close'].pct_change(60)
        
        cols_to_update = ['return_120d', 'ma_bullish', 'drawdown_from_high', 'vol_spike',
                          'liquidity_ratio', 'momentum_20d', 'momentum_60d']
        for col in cols_to_update:
            df.loc[mask, col] = s[col].values
    
    return df

def calculate_alerts(row):
    alerts = 0
    if not pd.isna(row.get('vol_spike')) and row['vol_spike'] > CRASH_CONFIG['vol_spike_ratio']:
        alerts += 1
    if not pd.isna(row.get('drawdown_from_high')) and row['drawdown_from_high'] < -CRASH_CONFIG['drawdown_threshold']:
        alerts += 1
    if not pd.isna(row.get('liquidity_ratio')) and row['liquidity_ratio'] < CRASH_CONFIG['liquidity_dryup_ratio']:
        alerts += 1
    if not pd.isna(row.get('momentum_20d')) and row['momentum_20d'] < -0.20:
        alerts += 1
    if not pd.isna(row.get('momentum_60d')) and row['momentum_60d'] < -0.30:
        alerts += 1
    return alerts

def get_crash_reduction(alerts):
    if alerts >= 3: return CRASH_CONFIG['red_alert_reduction']
    elif alerts == 2: return CRASH_CONFIG['orange_alert_reduction']
    elif alerts == 1: return CRASH_CONFIG['yellow_alert_reduction']
    return 0.0

def backtest(pred_df, price_df):
    price_ind = calculate_indicators(price_df)
    merged = pred_df.merge(price_ind, on=['trade_date', 'symbol'], how='inner')
    merged = merged.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    
    results = []
    daily_records = []
    
    for symbol in merged['symbol'].unique():
        sym_data = merged[merged['symbol'] == symbol].copy()
        if len(sym_data) < 20:
            continue
        
        cash = 1.0
        position = 0.0
        nav = 1.0
        last_add_date = None
        alert_active = False
        consecutive_safe_days = 0
        peak_nav = 1.0
        max_dd = 0.0
        
        for idx in range(len(sym_data)):
            row = sym_data.iloc[idx]
            date = row['trade_date']
            close = row['close_y'] if 'close_y' in row else row['close_x']
            
            # 前一日持仓市值
            if idx > 0:
                prev_close = sym_data.iloc[idx-1]['close_y'] if 'close_y' in sym_data.iloc[idx-1] else sym_data.iloc[idx-1]['close_x']
                position_value = position * (close / prev_close)
            else:
                position_value = 0.0
            
            # 股灾预警
            alerts = calculate_alerts(row)
            crash_reduction = get_crash_reduction(alerts)
            
            if alerts > 0:
                if not alert_active:
                    alert_active = True
                consecutive_safe_days = 0
            else:
                consecutive_safe_days += 1
                if alert_active and consecutive_safe_days >= CRASH_CONFIG['recovery_required_days']:
                    alert_active = False
            
            # 三情景基础仓位
            probs = {'adverse': row.get('adverse', 0), 'base': row.get('base', 0), 'favorable': row.get('favorable', 0)}
            scenario = max(probs, key=probs.get)
            base_position = SCENARIO_POSITIONS.get(scenario, 0.0)
            
            # 趋势加仓
            trend_position = base_position
            trend_reason = "无加仓"
            
            if not alert_active:
                return_120d = row.get('return_120d')
                ma_bullish = row.get('ma_bullish', False)
                
                if not pd.isna(return_120d):
                    if return_120d > TREND_CONFIG['trend_threshold_strong']:
                        if not TREND_CONFIG['ma_bullish_required'] or ma_bullish:
                            new_pos = min(base_position + TREND_CONFIG['add_position_step'], TREND_CONFIG['add_position_max'])
                            if new_pos > base_position:
                                trend_position = new_pos
                                trend_reason = f"强趋势加仓({return_120d:.1%})"
                                last_add_date = date
                    elif return_120d > TREND_CONFIG['trend_threshold_moderate']:
                        if not TREND_CONFIG['ma_bullish_required'] or ma_bullish:
                            new_pos = min(base_position + TREND_CONFIG['add_position_step'] * 0.5, TREND_CONFIG['add_position_max'])
                            if new_pos > base_position:
                                trend_position = new_pos
                                trend_reason = f"中等趋势加仓({return_120d:.1%})"
                                last_add_date = date
            
            # 股灾减仓
            final_position = trend_position
            crash_reason = "无预警"
            if alert_active:
                final_position = trend_position * (1 - crash_reduction)
                if crash_reduction >= 1.0: crash_reason = f"红色预警({alerts}指标):空仓"
                elif crash_reduction >= 0.5: crash_reason = f"橙色预警({alerts}指标):减仓50%"
                elif crash_reduction >= 0.2: crash_reason = f"黄色预警({alerts}指标):减仓20%"
            
            final_position = max(0.0, min(1.0, final_position))
            
            # 执行调仓
            current_total = cash + position_value
            target_value = current_total * final_position
            
            if idx > 0:
                trade_value = target_value - position_value
                if trade_value > 0:
                    cash -= trade_value
                    position_value = target_value
                elif trade_value < 0:
                    cash -= trade_value
                    position_value = target_value
            else:
                position_value = current_total * final_position
                cash = current_total - position_value
            
            position = final_position
            nav = cash + position_value
            
            if nav > peak_nav:
                peak_nav = nav
            dd = (nav - peak_nav) / peak_nav
            if dd < max_dd:
                max_dd = dd
            
            daily_records.append({
                'trade_date': date, 'symbol': symbol, 'strategy': 'v5_trend_crash',
                'nav': nav, 'position': position, 'base_position': base_position,
                'scenario': scenario, 'alerts': alerts, 'alert_active': alert_active,
                'trend_reason': trend_reason, 'crash_reason': crash_reason,
                'close': close
            })
        
        # 计算收益指标
        if len(sym_data) > 1:
            total_return = nav - 1.0
            start_date = sym_data['trade_date'].iloc[0]
            end_date = sym_data['trade_date'].iloc[-1]
            years = (end_date - start_date).days / 365.25
            annual_return = (nav ** (1/years) - 1) if years > 0 and nav > 0 else 0
            benchmark_return = (sym_data['close_y'].iloc[-1] / sym_data['close_y'].iloc[0]) - 1 if 'close_y' in sym_data else 0
            
            daily_returns = []
            for j in range(1, len(daily_records)):
                if daily_records[j]['symbol'] == symbol:
                    prev_nav = daily_records[j-1]['nav']
                    curr_nav = daily_records[j]['nav']
                    if prev_nav > 0:
                        daily_returns.append((curr_nav - prev_nav) / prev_nav)
            
            if daily_returns:
                sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
                win_rate = sum(1 for r in daily_returns if r > 0) / len(daily_returns)
            else:
                sharpe = 0
                win_rate = 0
            
            results.append({
                'symbol': symbol, 'strategy': 'v5_trend_crash',
                'total_return': total_return, 'annual_return': annual_return,
                'max_drawdown': max_dd, 'sharpe': sharpe, 'win_rate': win_rate,
                'benchmark_return': benchmark_return,
                'excess_return': total_return - benchmark_return,
                'start_date': start_date, 'end_date': end_date,
                'final_nav': nav,
                'avg_position': np.mean([d['position'] for d in daily_records if d['symbol'] == symbol]),
                'alert_days': sum(1 for d in daily_records if d['symbol'] == symbol and d['alert_active']),
            })
    
    return pd.DataFrame(results), pd.DataFrame(daily_records)

def main():
    print("=" * 70)
    print("v5模型回测 - 训练集股票2024-2025")
    print("=" * 70)
    
    pred_df, price_df = load_data()
    print(f"\n预测数据: {len(pred_df)}条, {pred_df['symbol'].nunique()}只股票")
    print(f"价格数据: {len(price_df)}条, {price_df['symbol'].nunique()}只股票")
    
    results_df, daily_df = backtest(pred_df, price_df)
    
    print(f"\n{'='*70}")
    print("回测结果")
    print(f"{'='*70}")
    
    for _, row in results_df.iterrows():
        print(f"\n  {row['symbol']}")
        print(f"   总收益率:     {row['total_return']*100:>8.2f}%")
        print(f"   年化收益率:   {row['annual_return']*100:>8.2f}%")
        print(f"   最大回撤:     {row['max_drawdown']*100:>8.2f}%")
        print(f"   夏普比率:     {row['sharpe']:>8.2f}")
        print(f"   胜率:         {row['win_rate']*100:>8.1f}%")
        print(f"   基准收益:     {row['benchmark_return']*100:>8.2f}%")
        print(f"   超额收益:     {row['excess_return']*100:>8.2f}%")
        print(f"   平均仓位:     {row['avg_position']*100:>8.1f}%")
        print(f"   预警天数:     {row['alert_days']:>8d}")
    
    avg_return = results_df['total_return'].mean()
    avg_annual = results_df['annual_return'].mean()
    avg_dd = results_df['max_drawdown'].mean()
    avg_sharpe = results_df['sharpe'].mean()
    avg_excess = results_df['excess_return'].mean()
    
    print(f"\n{'='*70}")
    print("平均表现")
    print(f"{'='*70}")
    print(f"   平均总收益率:   {avg_return*100:>8.2f}%")
    print(f"   平均年化收益:   {avg_annual*100:>8.2f}%")
    print(f"   平均最大回撤:   {avg_dd*100:>8.2f}%")
    print(f"   平均夏普比率:   {avg_sharpe:>8.2f}")
    print(f"   平均超额收益:   {avg_excess*100:>8.2f}%")
    
    # 保存
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS backtest_results_v5")
    conn.execute("DROP TABLE IF EXISTS backtest_daily_v5")
    results_df.to_sql('backtest_results_v5', conn, if_exists='replace', index=False)
    daily_df.to_sql('backtest_daily_v5', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    
    print(f"\n已保存到 backtest_results_v5, backtest_daily_v5")

if __name__ == '__main__':
    main()
