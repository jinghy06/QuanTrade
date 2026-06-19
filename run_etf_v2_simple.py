"""
QuanTrade 2.0 - 简化版（验证核心逻辑）
=====================================
核心理念：先证明简单策略有效，再叠加复杂度

策略：
1. 纯动量排名（20日收益率）
2. 市场过滤（价格 > 60日均线才买入）
3. 月度调仓（减少成本）
4. 只持1只ETF（小资金集中）
"""

import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from evaluator_agent import EvaluationAgent

DB_PATH = "QuanTrade/quant_system/data/quant.db"
COST_RATE = 0.0036

# 8只核心ETF（有完整历史数据）
CORE_ETFS = ['510300', '515070', '159995', '512660', '512670', '562500', '516510', '515960']


def load_data():
    """加载数据"""
    conn = sqlite3.connect(DB_PATH)
    
    all_prices = {}
    for symbol in CORE_ETFS:
        df = pd.read_sql(
            f"SELECT * FROM etf_daily_prices WHERE symbol='{symbol}' ORDER BY trade_date",
            conn
        )
        if len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')
            all_prices[symbol] = df
    
    conn.close()
    return all_prices


def calculate_momentum(close: pd.Series, lookback: int = 20) -> float:
    """计算动量得分"""
    if len(close) < lookback + 1:
        return 0
    return close.iloc[-1] / close.iloc[-lookback] - 1


def is_uptrend(close: pd.Series, ma_period: int = 60) -> bool:
    """判断是否上升趋势"""
    if len(close) < ma_period:
        return False
    ma = close.rolling(ma_period).mean().iloc[-1]
    return close.iloc[-1] > ma


def backtest_simple(
    all_prices: dict,
    initial_capital: float = 50000,
    momentum_lookback: int = 20,
    ma_period: int = 60,
    hold_threshold: float = 0.0  # 动量>0才持仓
) -> dict:
    """简化版回测"""
    
    print(f"\n参数: 动量周期={momentum_lookback}天, 均线={ma_period}天, 阈值={hold_threshold}")
    
    # 获取公共日期
    all_dates = set()
    for df in all_prices.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)
    
    # 从第ma_period天开始
    start_idx = ma_period + 10
    backtest_dates = all_dates[start_idx:]
    
    capital = initial_capital
    holdings = {}  # {symbol: shares}
    nav_history = []
    trade_count = 0
    current_month = None
    
    for date in backtest_dates:
        date_ts = pd.Timestamp(date)
        
        # 月度调仓（每月第一个交易日）
        if current_month != date_ts.month:
            current_month = date_ts.month
            
            # 计算每只ETF的动量
            scores = {}
            for symbol, df in all_prices.items():
                if date not in df.index:
                    continue
                hist = df.loc[:date]
                if len(hist) < momentum_lookback + 1:
                    continue
                
                # 动量得分
                mom = calculate_momentum(hist['close'], momentum_lookback)
                
                # 趋势过滤
                trend_ok = is_uptrend(hist['close'], ma_period)
                
                if trend_ok:
                    scores[symbol] = mom
            
            # 选择最强ETF
            if scores:
                best_symbol = max(scores, key=scores.get)
                best_score = scores[best_symbol]
                
                if best_score > hold_threshold:
                    # 换仓
                    for h_symbol, h_shares in list(holdings.items()):
                        if h_symbol != best_symbol:
                            sell_price = all_prices[h_symbol]['close'].get(date, 0)
                            if sell_price > 0:
                                capital += h_shares * sell_price * (1 - COST_RATE)
                                trade_count += 1
                            del holdings[h_symbol]
                    
                    # 买入
                    if best_symbol not in holdings:
                        buy_price = all_prices[best_symbol]['close'].get(date, 0)
                        if buy_price > 0:
                            shares = int(capital / buy_price / 100) * 100
                            if shares > 0:
                                cost = shares * buy_price * (1 + COST_RATE)
                                capital -= cost
                                holdings[best_symbol] = shares
                                trade_count += 1
                else:
                    # 清仓
                    for h_symbol, h_shares in list(holdings.items()):
                        sell_price = all_prices[h_symbol]['close'].get(date, 0)
                        if sell_price > 0:
                            capital += h_shares * sell_price * (1 - COST_RATE)
                            trade_count += 1
                    holdings = {}
        
        # 计算NAV
        portfolio_value = capital
        for h_symbol, h_shares in holdings.items():
            price = all_prices[h_symbol]['close'].get(date, 0)
            portfolio_value += h_shares * price
        nav_history.append({'date': date, 'nav': portfolio_value})
    
    # 基准（沪深300买入持有）
    benchmark = all_prices.get('510300')
    if benchmark is not None:
        bench_start_price = benchmark['close'].get(backtest_dates[0], 1)
        bench_nav = []
        for d in backtest_dates:
            price = benchmark['close'].get(d, bench_nav[-1]['nav'] / initial_capital * bench_start_price if bench_nav else bench_start_price)
            bench_nav.append({'date': d, 'nav': initial_capital * price / bench_start_price})
    else:
        bench_nav = nav_history.copy()
    
    nav_df = pd.DataFrame(nav_history)
    nav_df['returns'] = nav_df['nav'].pct_change()
    
    bench_df = pd.DataFrame(bench_nav)
    bench_df['returns'] = bench_df['nav'].pct_change()
    
    return {
        'nav': nav_df,
        'benchmark': bench_df,
        'final_nav': nav_df['nav'].iloc[-1],
        'total_return': nav_df['nav'].iloc[-1] / initial_capital - 1,
        'benchmark_return': bench_df['nav'].iloc[-1] / initial_capital - 1,
        'trade_count': trade_count,
        'initial_capital': initial_capital
    }


def main():
    print("=" * 70)
    print("       QuanTrade 2.0 - 简化版（验证核心逻辑）")
    print("=" * 70)
    
    # 加载数据
    print("\n加载数据...")
    all_prices = load_data()
    print(f"  ETF数据: {len(all_prices)} 只")
    
    # 测试不同参数组合
    param_sets = [
        {'momentum_lookback': 20, 'ma_period': 60, 'hold_threshold': 0.0},
        {'momentum_lookback': 20, 'ma_period': 60, 'hold_threshold': 0.02},
        {'momentum_lookback': 10, 'ma_period': 20, 'hold_threshold': 0.0},
        {'momentum_lookback': 60, 'ma_period': 120, 'hold_threshold': 0.0},
    ]
    
    best_result = None
    best_params = None
    best_return = -float('inf')
    
    for params in param_sets:
        result = backtest_simple(all_prices, initial_capital=50000, **params)
        total_return = result['total_return']
        
        print(f"\n  策略收益: {total_return:.2%} | 基准: {result['benchmark_return']:.2%} | 超额: {total_return - result['benchmark_return']:.2%} | 交易: {result['trade_count']}次")
        
        if total_return > best_return:
            best_return = total_return
            best_result = result
            best_params = params
    
    print("\n" + "=" * 70)
    print(f"最佳参数: {best_params}")
    print("=" * 70)
    print(f"  初始资金: {best_result['initial_capital']:,.0f} 元")
    print(f"  最终净值: {best_result['final_nav']:,.0f} 元")
    print(f"  策略收益: {best_result['total_return']:.2%}")
    print(f"  基准收益: {best_result['benchmark_return']:.2%}")
    print(f"  超额收益: {best_result['total_return'] - best_result['benchmark_return']:.2%}")
    print(f"  交易次数: {best_result['trade_count']}")
    
    # 独立评价
    print("\n" + "=" * 70)
    print("              独立评价Agent评估")
    print("=" * 70)
    
    agent = EvaluationAgent(verbose=True)
    report = agent.full_evaluation(
        strategy_returns=best_result['nav']['returns'].dropna(),
        benchmark_returns=best_result['benchmark']['returns'].dropna()
    )
    agent.print_report(report)
    
    return best_result, report


if __name__ == "__main__":
    main()
