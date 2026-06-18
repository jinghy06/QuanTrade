#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势跟踪加仓 + 股灾预判 策略回测引擎 v4_r3
============================================
核心创新:
1. 趋势跟踪动态加仓: 基于趋势强度在模型预测基础上动态调整仓位
2. 股灾多指标预警: 波动率突增/快速回撤/估值极端/流动性枯竭 四级预警
3. 渐进式仓位管理: 避免一次性大幅调仓，减少冲击成本

股灾历史特征 (2007/2015):
- 2007年10月: 上证指数6124点见顶，PE>60倍，随后12个月跌73%
- 2015年6月: 上证指数5178点见顶，杠杆资金占比极高，随后3个月跌45%
共同特征: 高估值 + 高波动 + 快速下跌 + 流动性枯竭
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

# 趋势跟踪加仓参数
TREND_CONFIG = {
    'trend_lookback': 120,           # 趋势观察窗口
    'trend_threshold_strong': 0.15,  # 强趋势阈值 (120日收益率>15%)
    'trend_threshold_moderate': 0.08, # 中等趋势阈值 (120日收益率>8%)
    'ma_bullish_required': True,      # 需要均线多头排列
    'add_position_step': 0.20,       # 每次加仓幅度 (20%)
    'add_position_max': 1.00,        # 最大仓位
    'add_position_cooldown': 3,      # 加仓冷却期(交易日)
}

# 股灾预警参数
CRASH_CONFIG = {
    # 预警指标阈值 (提高阈值，减少牛市误报)
    'vol_spike_ratio': 3.0,          # ATR20/ATR14 > 3.0 视为波动率突增
    'drawdown_threshold': 0.20,      # 从120日高点回撤>20%
    'valuation_extreme_pe': 95,      # PE历史分位>95%
    'valuation_extreme_pb': 95,      # PB历史分位>95%
    'liquidity_dryup_ratio': 0.3,    # 成交量/20日均量 < 0.3
    
    # 预警等级对应减仓比例
    'yellow_alert_reduction': 0.20,  # 1个指标触发: 减仓20%
    'orange_alert_reduction': 0.50,  # 2个指标触发: 减仓50%
    'red_alert_reduction': 1.00,     # 3个+指标触发: 强制空仓
    
    # 预警冷却期
    'alert_cooldown_days': 10,       # 预警后至少维持10日
    'recovery_required_days': 5,     # 恢复需要连续5日无预警
}

# 三情景基础仓位 (提高基础仓位)
SCENARIO_POSITIONS = {
    'adverse': 0.60,
    'base': 0.90,
    'favorable': 1.00,
}

# ETF列表
ETF_LIST = ['562500.SH', '588200.SH', '588790.SH', '159382.SZ', '159241.SZ']


def load_data():
    """加载预测信号和原始价格数据"""
    conn = sqlite3.connect(DB_PATH)
    
    # 加载Round 2预测信号 - 使用fusion_r2的概率列
    pred_df = pd.read_sql_query("""
        SELECT 
            trade_date, symbol,
            fusion_r2_adverse as adverse,
            fusion_r2_base as base,
            fusion_r2_favorable as favorable,
            target_return_10d as predicted_return_20d
        FROM predictions_v4_r2
        WHERE symbol IN ({})
    """.format(','.join([f"'{s}'" for s in ETF_LIST])), conn)
    
    # 加载原始价格数据（用于计算趋势指标和股灾预警）
    price_df = pd.read_sql_query("""
        SELECT trade_date, symbol, close, high, low, volume, open
        FROM daily_prices
        WHERE symbol IN ({})
        ORDER BY symbol, trade_date
    """.format(','.join([f"'{s}'" for s in ETF_LIST])), conn)
    
    conn.close()
    
    pred_df['trade_date'] = pd.to_datetime(pred_df['trade_date'], format='mixed')
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'], format='mixed')
    
    return pred_df, price_df


def calculate_trend_indicators(price_df):
    """计算趋势跟踪所需指标"""
    df = price_df.copy().sort_values(['symbol', 'trade_date'])
    
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        s = df.loc[mask].copy()
        
        # 120日收益率 (趋势强度)
        s['return_120d'] = s['close'].pct_change(120)
        
        # 均线排列: 20日 > 60日 > 120日 (多头排列)
        s['ma_20'] = s['close'].rolling(20).mean()
        s['ma_60'] = s['close'].rolling(60).mean()
        s['ma_120'] = s['close'].rolling(120).mean()
        s['ma_bullish'] = (s['ma_20'] > s['ma_60']) & (s['ma_60'] > s['ma_120'])
        
        # 120日高点及回撤
        s['high_120d'] = s['high'].rolling(120).max()
        s['drawdown_from_high'] = (s['close'] - s['high_120d']) / s['high_120d']
        
        # ATR (用于波动率预警)
        s['tr'] = np.maximum(
            s['high'] - s['low'],
            np.maximum(
                abs(s['high'] - s['close'].shift(1)),
                abs(s['low'] - s['close'].shift(1))
            )
        )
        s['atr_14'] = s['tr'].rolling(14).mean()
        s['atr_20'] = s['tr'].rolling(20).mean()
        s['vol_spike'] = s['atr_20'] / s['atr_14']
        
        # 成交量趋势 (流动性)
        s['volume_ma_20'] = s['volume'].rolling(20).mean()
        s['liquidity_ratio'] = s['volume'] / s['volume_ma_20']
        
        # 20日动量 (短期趋势)
        s['momentum_20d'] = s['close'].pct_change(20)
        
        # 60日动量 (中期趋势)
        s['momentum_60d'] = s['close'].pct_change(60)
        
        df.loc[mask, s.columns.difference(['symbol', 'trade_date', 'close', 'high', 'low', 'volume', 'open'])] = \
            s[s.columns.difference(['symbol', 'trade_date', 'close', 'high', 'low', 'volume', 'open'])]
    
    return df


def calculate_crash_alerts(row):
    """计算股灾预警指标，返回触发的预警数量"""
    alerts = 0
    
    # 1. 波动率突增: ATR20/ATR14 > 2.0
    if not pd.isna(row.get('vol_spike')) and row['vol_spike'] > CRASH_CONFIG['vol_spike_ratio']:
        alerts += 1
    
    # 2. 快速回撤: 从120日高点回撤>15%
    if not pd.isna(row.get('drawdown_from_high')) and row['drawdown_from_high'] < -CRASH_CONFIG['drawdown_threshold']:
        alerts += 1
    
    # 3. 流动性枯竭: 成交量/20日均量 < 0.5
    if not pd.isna(row.get('liquidity_ratio')) and row['liquidity_ratio'] < CRASH_CONFIG['liquidity_dryup_ratio']:
        alerts += 1
    
    # 4. 短期暴跌: 20日跌幅>20% (股灾中的快速下跌)
    if not pd.isna(row.get('momentum_20d')) and row['momentum_20d'] < -0.20:
        alerts += 1
    
    # 5. 中期趋势恶化: 60日跌幅>30%
    if not pd.isna(row.get('momentum_60d')) and row['momentum_60d'] < -0.30:
        alerts += 1
    
    return alerts


def get_crash_reduction(alerts):
    """根据预警数量确定减仓比例"""
    if alerts >= 3:
        return CRASH_CONFIG['red_alert_reduction']
    elif alerts == 2:
        return CRASH_CONFIG['orange_alert_reduction']
    elif alerts == 1:
        return CRASH_CONFIG['yellow_alert_reduction']
    return 0.0


def get_trend_addition(row, current_position, last_add_date, current_date):
    """
    趋势跟踪加仓逻辑
    返回: (新仓位, 是否加仓, 加仓原因)
    """
    cfg = TREND_CONFIG
    
    # 检查冷却期
    if last_add_date is not None:
        days_since_add = (current_date - last_add_date).days
        # 简化: 假设交易日约250天/年，冷却期按交易日算
        if days_since_add < cfg['add_position_cooldown'] * 1.4:  # 1.4系数转换日历日到交易日
            return current_position, False, "冷却期中"
    
    # 检查趋势条件
    return_120d = row.get('return_120d')
    ma_bullish = row.get('ma_bullish', False)
    
    if pd.isna(return_120d):
        return current_position, False, "趋势数据不足"
    
    # 强趋势: 120日收益>20% + 均线多头排列
    if return_120d > cfg['trend_threshold_strong']:
        if not cfg['ma_bullish_required'] or ma_bullish:
            new_pos = min(current_position + cfg['add_position_step'], cfg['add_position_max'])
            if new_pos > current_position:
                return new_pos, True, f"强趋势加仓(120日收益{return_120d:.1%})"
    
    # 中等趋势: 120日收益>10% + 均线多头排列
    elif return_120d > cfg['trend_threshold_moderate']:
        if not cfg['ma_bullish_required'] or ma_bullish:
            new_pos = min(current_position + cfg['add_position_step'] * 0.5, cfg['add_position_max'])
            if new_pos > current_position:
                return new_pos, True, f"中等趋势加仓(120日收益{return_120d:.1%})"
    
    return current_position, False, "趋势条件不满足"


def backtest_trend_crash_aware(pred_df, price_df):
    """趋势跟踪+股灾预判 回测主函数"""
    
    # 计算技术指标
    price_with_indicators = calculate_trend_indicators(price_df)
    
    # 合并预测和价格数据 (pred_df已经是宽格式)
    merged = pred_df.merge(price_with_indicators, on=['trade_date', 'symbol'], how='inner')
    merged = merged.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    
    results = []
    daily_records = []
    
    for symbol in ETF_LIST:
        sym_data = merged[merged['symbol'] == symbol].copy()
        if len(sym_data) < 20:
            continue
        
        # 初始化
        cash = 1.0
        position = 0.0  # 当前持仓比例
        nav = 1.0
        last_add_date = None
        alert_active = False
        alert_start_date = None
        alert_level = 0
        consecutive_safe_days = 0
        
        trades = []
        peak_nav = 1.0
        max_dd = 0.0
        
        for idx in range(len(sym_data)):
            row = sym_data.iloc[idx]
            date = row['trade_date']
            close = row['close']
            
            # 前一日持仓市值
            if idx > 0:
                prev_close = sym_data.iloc[idx-1]['close']
                position_value = position_value * (close / prev_close)
            else:
                position_value = 0.0  # 首日初始持仓市值为0
            
            # ===== 步骤1: 计算股灾预警 =====
            alerts = calculate_crash_alerts(row)
            crash_reduction = get_crash_reduction(alerts)
            
            # 预警状态管理
            if alerts > 0:
                if not alert_active:
                    alert_active = True
                    alert_start_date = date
                    alert_level = alerts
                else:
                    alert_level = max(alert_level, alerts)
                consecutive_safe_days = 0
            else:
                consecutive_safe_days += 1
                # 需要连续5日安全才解除预警
                if alert_active and consecutive_safe_days >= CRASH_CONFIG['recovery_required_days']:
                    alert_active = False
                    alert_level = 0
            
            # ===== 步骤2: 三情景基础仓位 =====
            probs = {
                'adverse': row.get('adverse', 0),
                'base': row.get('base', 0),
                'favorable': row.get('favorable', 0)
            }
            
            if all(pd.isna(v) for v in probs.values()):
                base_position = 0.0
                scenario = 'unknown'
            else:
                scenario = max(probs, key=probs.get)
                base_position = SCENARIO_POSITIONS.get(scenario, 0.0)
            
            # ===== 步骤3: 趋势跟踪加仓 =====
            trend_position = base_position
            trend_reason = "无加仓"
            
            if not alert_active:  # 预警期间不加仓
                new_pos, added, reason = get_trend_addition(
                    row, base_position, last_add_date, date
                )
                if added:
                    trend_position = new_pos
                    trend_reason = reason
                    last_add_date = date
            
            # ===== 步骤4: 股灾预警减仓 =====
            final_position = trend_position
            crash_reason = "无预警"
            
            if alert_active:
                final_position = trend_position * (1 - crash_reduction)
                if crash_reduction >= 1.0:
                    crash_reason = f"🔴红色预警({alerts}指标): 强制空仓"
                elif crash_reduction >= 0.6:
                    crash_reason = f"🟠橙色预警({alerts}指标): 减仓60%"
                elif crash_reduction >= 0.3:
                    crash_reason = f"🟡黄色预警({alerts}指标): 减仓30%"
            
            # 仓位限制
            final_position = max(0.0, min(1.0, final_position))
            
            # ===== 步骤5: 执行调仓 =====
            current_total = cash + position_value
            target_value = current_total * final_position
            
            if idx > 0:
                # 计算需要买卖的金额
                trade_value = target_value - position_value
                
                # 简化: 假设无交易成本
                if trade_value > 0:
                    # 买入
                    cash -= trade_value
                    position_value = target_value
                elif trade_value < 0:
                    # 卖出
                    cash -= trade_value  # trade_value是负数
                    position_value = target_value
            else:
                # 首日建仓
                position_value = current_total * final_position
                cash = current_total - position_value
            
            position = final_position
            nav = cash + position_value
            
            # 更新峰值和回撤
            if nav > peak_nav:
                peak_nav = nav
            dd = (nav - peak_nav) / peak_nav
            if dd < max_dd:
                max_dd = dd
            
            # 记录交易
            if idx > 0:
                prev_pos = daily_records[-1]['position'] if daily_records else 0
                if abs(final_position - prev_pos) > 0.01:
                    trades.append({
                        'date': date,
                        'action': 'BUY' if final_position > prev_pos else 'SELL',
                        'old_position': prev_pos,
                        'new_position': final_position,
                        'reason': f"{trend_reason} | {crash_reason}"
                    })
            
            daily_records.append({
                'trade_date': date,
                'symbol': symbol,
                'strategy': 'trend_crash_aware',
                'nav': nav,
                'position': position,
                'base_position': base_position,
                'trend_position': trend_position,
                'scenario': scenario,
                'alerts': alerts,
                'alert_active': alert_active,
                'crash_reduction': crash_reduction,
                'trend_reason': trend_reason,
                'crash_reason': crash_reason,
                'return_120d': row.get('return_120d'),
                'ma_bullish': row.get('ma_bullish', False),
                'drawdown_from_high': row.get('drawdown_from_high'),
                'vol_spike': row.get('vol_spike'),
                'momentum_20d': row.get('momentum_20d'),
                'close': close
            })
        
        # 计算策略收益指标
        if len(sym_data) > 1:
            total_return = nav - 1.0
            start_date = sym_data['trade_date'].iloc[0]
            end_date = sym_data['trade_date'].iloc[-1]
            years = (end_date - start_date).days / 365.25
            annual_return = (nav ** (1/years) - 1) if years > 0 and nav > 0 else 0
            
            # 基准收益 (买入持有)
            benchmark_return = (sym_data['close'].iloc[-1] / sym_data['close'].iloc[0]) - 1
            
            # 计算日收益率序列
            daily_returns = []
            for j in range(1, len(sym_data)):
                prev_nav = daily_records[j-1]['nav'] if j-1 < len(daily_records) else 1.0
                curr_nav = daily_records[j]['nav'] if j < len(daily_records) else nav
                if prev_nav > 0:
                    daily_returns.append((curr_nav - prev_nav) / prev_nav)
            
            if daily_returns:
                sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
                win_rate = sum(1 for r in daily_returns if r > 0) / len(daily_returns)
            else:
                sharpe = 0
                win_rate = 0
            
            results.append({
                'symbol': symbol,
                'strategy': 'trend_crash_aware',
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_dd,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'benchmark_return': benchmark_return,
                'excess_return': total_return - benchmark_return,
                'num_trades': len(trades),
                'start_date': start_date,
                'end_date': end_date,
                'final_nav': nav,
                'avg_position': np.mean([d['position'] for d in daily_records if d['symbol'] == symbol]),
                'alert_days': sum(1 for d in daily_records if d['symbol'] == symbol and d['alert_active']),
                'trend_add_days': sum(1 for d in daily_records if d['symbol'] == symbol and '加仓' in d['trend_reason'])
            })
    
    return pd.DataFrame(results), pd.DataFrame(daily_records)


def run_backtest():
    """运行回测并保存结果"""
    print("=" * 80)
    print("趋势跟踪加仓 + 股灾预判 策略回测 v4_r3")
    print("=" * 80)
    
    pred_df, price_df = load_data()
    print(f"\n数据加载完成:")
    print(f"   预测信号: {len(pred_df)} 条")
    print(f"   价格数据: {len(price_df)} 条")
    
    results_df, daily_df = backtest_trend_crash_aware(pred_df, price_df)
    
    print(f"\n{'='*80}")
    print("回测结果汇总")
    print(f"{'='*80}")
    
    for _, row in results_df.iterrows():
        print(f"\n  {row['symbol']}")
        print(f"   总收益率:     {row['total_return']*100:>8.2f}%")
        print(f"   年化收益率:   {row['annual_return']*100:>8.2f}%")
        print(f"   最大回撤:     {row['max_drawdown']*100:>8.2f}%")
        print(f"   夏普比率:     {row['sharpe']:>8.2f}")
        print(f"   胜率:         {row['win_rate']*100:>8.1f}%")
        print(f"   基准收益:     {row['benchmark_return']*100:>8.2f}%")
        print(f"   超额收益:     {row['excess_return']*100:>8.2f}%")
        print(f"   交易次数:     {row['num_trades']:>8d}")
        print(f"   平均仓位:     {row['avg_position']*100:>8.1f}%")
        print(f"   预警天数:     {row['alert_days']:>8d}")
        print(f"   加仓天数:     {row['trend_add_days']:>8d}")
    
    # 平均表现
    avg_return = results_df['total_return'].mean()
    avg_annual = results_df['annual_return'].mean()
    avg_dd = results_df['max_drawdown'].mean()
    avg_sharpe = results_df['sharpe'].mean()
    avg_excess = results_df['excess_return'].mean()
    
    print(f"\n{'='*80}")
    print("5只ETF平均表现")
    print(f"{'='*80}")
    print(f"   平均总收益率:   {avg_return*100:>8.2f}%")
    print(f"   平均年化收益:   {avg_annual*100:>8.2f}%")
    print(f"   平均最大回撤:   {avg_dd*100:>8.2f}%")
    print(f"   平均夏普比率:   {avg_sharpe:>8.2f}")
    print(f"   平均超额收益:   {avg_excess*100:>8.2f}%")
    
    # 保存到数据库
    conn = sqlite3.connect(DB_PATH)
    
    # 删除旧数据
    conn.execute("DELETE FROM backtest_results_v4_r3 WHERE strategy='trend_crash_aware'")
    conn.execute("DELETE FROM backtest_daily_v4_r3 WHERE strategy='trend_crash_aware'")
    
    # 保存结果
    results_df.to_sql('backtest_results_v4_r3', conn, if_exists='append', index=False)
    daily_df.to_sql('backtest_daily_v4_r3', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    
    print(f"\n结果已保存到数据库表: backtest_results_v4_r3, backtest_daily_v4_r3")
    
    return results_df, daily_df


if __name__ == '__main__':
    run_backtest()
