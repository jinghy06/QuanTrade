"""
修复回测结果 - 重新计算基准和Kelly策略，修复NaN问题
"""
import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("【修复回测结果 - 基准/Kelly NaN修复】")
print("=" * 70)

db_path = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
conn = sqlite3.connect(db_path)

# 读取策略信号
df = pd.read_sql_query('SELECT * FROM strategy_signals_v3 ORDER BY trade_date, symbol', conn)
df['trade_date'] = pd.to_datetime(df['trade_date'])
conn.close()

print(f"数据量: {len(df)} 条, ETF: {df['symbol'].nunique()} 只")

# 回测参数
FEE_RATE = 0.0001
SLIPPAGE = 0.0001
STOP_LOSS_ATR = 2.0

def run_backtest(sym_data, signal_col, position_col=None):
    """运行单策略回测"""
    sym_data = sym_data.sort_values('trade_date').reset_index(drop=True)
    actual = sym_data['target_next_day_return'].fillna(0).values
    n = len(actual)
    
    if n == 0:
        return None
    
    # 基准: 满仓持有
    benchmark_cum = np.cumprod(1 + actual)
    
    # 策略
    if position_col and position_col in sym_data.columns:
        positions = sym_data[position_col].fillna(0).values
    else:
        positions = sym_data[signal_col].fillna(0).values.astype(float)
    
    signals = sym_data[signal_col].fillna(0).values.astype(float)
    atrs = sym_data['atr_14'].fillna(0.02).values if 'atr_14' in sym_data.columns else np.full(n, 0.02)
    
    cum = [1.0]
    in_pos = False
    
    for i in range(n):
        signal = signals[i]
        position = positions[i]
        daily_ret = actual[i]
        atr = atrs[i]
        
        if signal == 1 and not in_pos:
            in_pos = True
            cost = FEE_RATE + SLIPPAGE
            cum[-1] *= (1 - cost)
        elif signal == 0 and in_pos:
            in_pos = False
            cost = FEE_RATE + SLIPPAGE
            cum[-1] *= (1 - cost)
        
        if in_pos:
            if daily_ret < -STOP_LOSS_ATR * atr:
                daily_ret = -STOP_LOSS_ATR * atr
                in_pos = False
            ret = daily_ret * position
            cum.append(cum[-1] * (1 + ret))
        else:
            cum.append(cum[-1])
    
    strategy_cum = np.array(cum[1:])
    
    # 计算指标
    total_ret = strategy_cum[-1] - 1 if len(strategy_cum) > 0 else 0
    n_days = len(strategy_cum)
    annual_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 and total_ret > -1 else -1
    
    peak = np.maximum.accumulate(strategy_cum)
    drawdown = (strategy_cum - peak) / peak
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
    
    daily_rets = np.diff(strategy_cum) / strategy_cum[:-1] if len(strategy_cum) > 1 else np.array([0])
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
    win_rate = np.sum(daily_rets > 0) / len(daily_rets) if len(daily_rets) > 0 else 0
    
    return {
        'total_return': total_ret,
        'annual_return': annual_ret,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'benchmark_return': benchmark_cum[-1] - 1 if len(benchmark_cum) > 0 else 0,
        'benchmark_cum': benchmark_cum,
        'strategy_cum': strategy_cum,
    }

# 运行所有策略回测
all_results = []
strategy_defs = [
    ('基准(持有)', 'signal_base', None),  # 满仓持有，信号始终为1
    ('基础融合', 'signal_base', None),
    ('趋势过滤', 'signal_trend', None),
    ('动态阈值', 'signal_dynamic', None),
    ('Kelly仓位', 'signal_kelly', 'kelly_position'),
    ('综合策略', 'signal_combined', 'combined_position'),
]

print(f"\n{'ETF':<12} {'策略':<12} {'总收益':<10} {'年化':<10} {'最大回撤':<10} {'夏普':<8} {'胜率':<8} {'基准收益':<10}")
print("-" * 95)

for symbol in df['symbol'].unique():
    sym_data = df[df['symbol'] == symbol].copy()
    
    for s_name, sig_col, pos_col in strategy_defs:
        if s_name == '基准(持有)':
            # 基准: 始终满仓
            result = run_backtest(sym_data, 'signal_base', None)
            if result:
                # 修正: 基准不需要信号过滤，直接满仓
                actual = sym_data['target_next_day_return'].fillna(0).values
                benchmark_cum = np.cumprod(1 + actual)
                result['strategy_cum'] = benchmark_cum
                result['total_return'] = benchmark_cum[-1] - 1
                n_days = len(benchmark_cum)
                result['annual_return'] = (1 + result['total_return']) ** (252 / n_days) - 1 if n_days > 0 and result['total_return'] > -1 else -1
                daily_rets = np.diff(benchmark_cum) / benchmark_cum[:-1] if len(benchmark_cum) > 1 else np.array([0])
                result['sharpe'] = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
                result['win_rate'] = np.sum(daily_rets > 0) / len(daily_rets) if len(daily_rets) > 0 else 0
                peak = np.maximum.accumulate(benchmark_cum)
                drawdown = (benchmark_cum - peak) / peak
                result['max_drawdown'] = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            result = run_backtest(sym_data, sig_col, pos_col)
        
        if result:
            print(f"{symbol:<12} {s_name:<12} {result['total_return']*100:>8.2f}% {result['annual_return']*100:>8.2f}% {result['max_drawdown']*100:>8.2f}% {result['sharpe']:>6.2f} {result['win_rate']*100:>6.1f}% {result['benchmark_return']*100:>8.2f}%")
            
            all_results.append({
                'symbol': symbol,
                'strategy': s_name,
                'total_return': result['total_return'],
                'annual_return': result['annual_return'],
                'max_drawdown': result['max_drawdown'],
                'sharpe': result['sharpe'],
                'win_rate': result['win_rate'],
                'benchmark_return': result['benchmark_return'],
            })

# 保存到数据库
conn = sqlite3.connect(db_path)
results_df = pd.DataFrame(all_results)
results_df.to_sql('backtest_results_v3', conn, if_exists='replace', index=False)
conn.close()

print(f"\n✅ 回测结果已修复并保存到 backtest_results_v3 ({len(results_df)} 条)")

# 汇总
print("\n" + "=" * 70)
print("【策略表现汇总】")
print("=" * 70)
summary = results_df.groupby('strategy').agg({
    'total_return': 'mean',
    'annual_return': 'mean',
    'max_drawdown': 'mean',
    'sharpe': 'mean',
    'win_rate': 'mean',
}).round(4)
print(summary)
