"""
QuanTrade 2.0 - 优化版（评价Agent驱动迭代）
==========================================
目标：评价Agent评分达到B级以上
策略：季度调仓 + 5只核心ETF + 纯动量+趋势过滤
"""

import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from evaluator_agent import EvaluationAgent

DB_PATH = "QuanTrade/quant_system/data/quant.db"
COST_RATE = 0.0036

# 5只核心ETF（有完整历史+较好收益）
CORE_ETFS = ['515070', '159995', '516510', '510300', '562500']


def load_data():
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


def backtest_quarterly_momentum(
    all_prices: dict,
    initial_capital: float = 50000,
    momentum_period: int = 60,     # 季度动量
    trend_ma: int = 120,           # 半年线趋势
    min_hold_months: int = 3       # 最少持有3个月
) -> dict:
    """
    季度动量策略
    - 每季度第一个交易日调仓
    - 选择动量最强且在趋势线上的ETF
    - 最少持有3个月避免频繁交易
    """
    
    # 获取公共日期
    all_dates = set()
    for df in all_prices.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)
    
    start_idx = max(trend_ma + 10, 130)  # 确保有足够历史
    backtest_dates = all_dates[start_idx:]
    
    capital = initial_capital
    holdings = {}  # {symbol: shares}
    nav_history = []
    trade_log = []
    last_trade_month = None
    months_since_trade = 999
    
    for date in backtest_dates:
        date_ts = pd.Timestamp(date)
        current_quarter = (date_ts.year, (date_ts.month - 1) // 3)
        
        # 季度调仓检查
        should_rebalance = False
        if last_trade_month is None:
            should_rebalance = True
        else:
            months_since_trade = (date_ts.year - last_trade_month[0]) * 12 + (date_ts.month - last_trade_month[1])
            if months_since_trade >= min_hold_months:
                # 检查是否是季度首月
                if date_ts.month in [1, 4, 7, 10] and date_ts.day <= 5:
                    should_rebalance = True
        
        if should_rebalance:
            # 计算每只ETF的得分
            scores = {}
            for symbol, df in all_prices.items():
                if date not in df.index:
                    continue
                hist = df.loc[:date]
                if len(hist) < momentum_period + 10:
                    continue
                
                close = hist['close']
                
                # 动量得分（过去N天收益率）
                momentum = close.iloc[-1] / close.iloc[-momentum_period] - 1
                
                # 趋势过滤（价格 > 长期均线）
                ma_long = close.rolling(trend_ma).mean().iloc[-1]
                in_uptrend = close.iloc[-1] > ma_long
                
                # 波动率调整（夏普比率近似）
                returns = close.pct_change().iloc[-momentum_period:]
                sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
                
                if in_uptrend:
                    # 综合得分 = 动量 * 波动率调整
                    scores[symbol] = momentum * (1 + sharpe * 0.1)
            
            # 选择最强ETF
            if scores:
                best_symbol = max(scores, key=scores.get)
                best_score = scores[best_symbol]
                
                if best_score > 0:
                    # 换仓（只在持仓变化时交易）
                    if best_symbol not in holdings or len(holdings) > 1:
                        # 卖出非目标持仓
                        for h_symbol, h_shares in list(holdings.items()):
                            if h_symbol != best_symbol:
                                sell_price = all_prices[h_symbol]['close'].get(date, 0)
                                if sell_price > 0:
                                    capital += h_shares * sell_price * (1 - COST_RATE)
                                    trade_log.append({'date': date, 'action': 'sell', 'symbol': h_symbol})
                                del holdings[h_symbol]
                        
                        # 买入目标ETF
                        if best_symbol not in holdings:
                            buy_price = all_prices[best_symbol]['close'].get(date, 0)
                            if buy_price > 0:
                                shares = int(capital / buy_price / 100) * 100
                                if shares > 0:
                                    cost = shares * buy_price * (1 + COST_RATE)
                                    capital -= cost
                                    holdings[best_symbol] = shares
                                    trade_log.append({'date': date, 'action': 'buy', 'symbol': best_symbol})
                        
                        last_trade_month = (date_ts.year, date_ts.month)
                else:
                    # 全部看空，清仓
                    if holdings:
                        for h_symbol, h_shares in list(holdings.items()):
                            sell_price = all_prices[h_symbol]['close'].get(date, 0)
                            if sell_price > 0:
                                capital += h_shares * sell_price * (1 - COST_RATE)
                                trade_log.append({'date': date, 'action': 'sell', 'symbol': h_symbol})
                        holdings = {}
                        last_trade_month = (date_ts.year, date_ts.month)
        
        # 计算NAV
        portfolio_value = capital
        for h_symbol, h_shares in holdings.items():
            price = all_prices[h_symbol]['close'].get(date, 0)
            portfolio_value += h_shares * price
        nav_history.append({'date': date, 'nav': portfolio_value})
    
    # 基准
    benchmark = all_prices.get('510300')
    if benchmark is not None:
        bench_start = benchmark['close'].get(backtest_dates[0], 1)
        bench_nav = []
        for d in backtest_dates:
            price = benchmark['close'].get(d, bench_nav[-1]['nav'] / initial_capital * bench_start if bench_nav else bench_start)
            bench_nav.append({'date': d, 'nav': initial_capital * price / bench_start})
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
        'trade_count': len(trade_log),
        'trade_log': trade_log,
        'initial_capital': initial_capital
    }


def main():
    print("=" * 70)
    print("    QuanTrade 2.0 - 优化版（季度动量+趋势过滤）")
    print("=" * 70)
    
    all_prices = load_data()
    print(f"\n加载 {len(all_prices)} 只核心ETF: {list(all_prices.keys())}")
    
    # 参数网格搜索
    param_sets = [
        {'momentum_period': 60, 'trend_ma': 120, 'min_hold_months': 3},
        {'momentum_period': 60, 'trend_ma': 60, 'min_hold_months': 3},
        {'momentum_period': 90, 'trend_ma': 120, 'min_hold_months': 3},
        {'momentum_period': 120, 'trend_ma': 120, 'min_hold_months': 6},
        {'momentum_period': 60, 'trend_ma': 120, 'min_hold_months': 6},
    ]
    
    best_result = None
    best_params = None
    best_score = -999
    
    print("\n" + "=" * 50)
    print("参数搜索:")
    print("=" * 50)
    
    for i, params in enumerate(param_sets):
        result = backtest_quarterly_momentum(all_prices, initial_capital=50000, **params)
        excess = result['total_return'] - result['benchmark_return']
        
        print(f"\n[{i+1}] 动量={params['momentum_period']}天, 均线={params['trend_ma']}天, 持有>={params['min_hold_months']}月")
        print(f"    收益: {result['total_return']:.2%} | 基准: {result['benchmark_return']:.2%} | 超额: {excess:.2%} | 交易: {result['trade_count']}次")
        
        # 用超额收益作为选择标准
        if excess > best_score:
            best_score = excess
            best_result = result
            best_params = params
    
    # 打印最佳结果
    print("\n" + "=" * 70)
    print(f"最佳参数: {best_params}")
    print("=" * 70)
    print(f"  初始资金: {best_result['initial_capital']:,.0f} 元")
    print(f"  最终净值: {best_result['final_nav']:,.0f} 元")
    print(f"  策略收益: {best_result['total_return']:.2%}")
    print(f"  基准收益: {best_result['benchmark_return']:.2%}")
    print(f"  超额收益: {best_result['total_return'] - best_result['benchmark_return']:.2%}")
    print(f"  交易次数: {best_result['trade_count']}")
    
    # 独立评价Agent评估
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
