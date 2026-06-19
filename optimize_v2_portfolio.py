"""
QuanTrade 2.0 - 组合ETF + 波动率目标
=====================================
目标：提高夏普比率到1.0+
方法：
1. 持有前3名ETF（分散风险）
2. 波动率目标（高波动时降低仓位）
3. 等权再平衡（减少单票风险）
"""
import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from evaluator_agent import EvaluationAgent

DB_PATH = "QuanTrade/quant_system/data/quant.db"
COST_RATE = 0.0036
ALL_ETFS = ['159995', '510300', '512660', '512670', '515070', '515960', '516510', '562500']


def load_data():
    conn = sqlite3.connect(DB_PATH)
    all_prices = {}
    for symbol in ALL_ETFS:
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


def backtest_portfolio(
    all_prices,
    momentum_period=90,
    trend_ma=120,
    min_hold_months=6,
    top_n=3,                    # 持有前3名
    vol_target=0.15,            # 目标波动率15%
    vol_lookback=20             # 波动率计算周期
):
    """组合ETF策略"""
    all_dates = set()
    for df in all_prices.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    start_idx = 130
    backtest_dates = all_dates[start_idx:]

    capital = 50000
    holdings = {}  # {symbol: shares}
    nav_history = []
    last_trade_date = None
    trade_count = 0

    for date in backtest_dates:
        date_ts = pd.Timestamp(date)

        # 季度调仓
        should_rebalance = False
        if last_trade_date is None:
            should_rebalance = True
        else:
            months_since = (date_ts.year - last_trade_date.year) * 12 + (date_ts.month - last_trade_date.month)
            if months_since >= min_hold_months and date_ts.month in [1, 4, 7, 10] and date_ts.day <= 5:
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

                # 动量
                momentum = close.iloc[-1] / close.iloc[-momentum_period] - 1

                # 趋势
                ma_long = close.rolling(trend_ma).mean().iloc[-1]
                in_uptrend = close.iloc[-1] > ma_long

                if in_uptrend and momentum > 0:
                    scores[symbol] = momentum

            # 选择前N名
            if len(scores) >= top_n:
                top_symbols = sorted(scores, key=scores.get, reverse=True)[:top_n]

                # 计算组合波动率
                returns_list = []
                for s in top_symbols:
                    hist = all_prices[s].loc[:date]['close']
                    ret = hist.pct_change().iloc[-vol_lookback:]
                    returns_list.append(ret)
                returns_df = pd.DataFrame(returns_list).T
                port_vol = returns_df.mean(axis=1).std() * np.sqrt(252)

                # 波动率调整仓位
                vol_ratio = min(vol_target / (port_vol + 1e-8), 1.0)

                # 卖出非目标持仓
                for h_symbol, h_shares in list(holdings.items()):
                    if h_symbol not in top_symbols:
                        sell_price = all_prices[h_symbol]['close'].get(date, 0)
                        if sell_price > 0:
                            capital += h_shares * sell_price * (1 - COST_RATE)
                            trade_count += 1
                        del holdings[h_symbol]

                # 等权买入目标ETF
                per_etf_capital = capital * vol_ratio / top_n
                for symbol in top_symbols:
                    if symbol not in holdings:
                        buy_price = all_prices[symbol]['close'].get(date, 0)
                        if buy_price > 0:
                            shares = int(per_etf_capital / buy_price / 100) * 100
                            if shares > 0:
                                cost = shares * buy_price * (1 + COST_RATE)
                                capital -= cost
                                holdings[symbol] = shares
                                trade_count += 1

                last_trade_date = date_ts

        # 计算NAV
        portfolio_value = capital
        for h_symbol, h_shares in holdings.items():
            price = all_prices[h_symbol]['close'].get(date, 0)
            portfolio_value += h_shares * price
        nav_history.append({'date': date, 'nav': portfolio_value})

    # 基准
    benchmark = all_prices.get('510300')
    bench_start = benchmark['close'].get(backtest_dates[0], 1)
    bench_nav = []
    for d in backtest_dates:
        price = benchmark['close'].get(d, bench_nav[-1]['nav'] / 50000 * bench_start if bench_nav else bench_start)
        bench_nav.append({'date': d, 'nav': 50000 * price / bench_start})

    nav_df = pd.DataFrame(nav_history)
    nav_df['returns'] = nav_df['nav'].pct_change()
    bench_df = pd.DataFrame(bench_nav)
    bench_df['returns'] = bench_df['nav'].pct_change()

    return nav_df, bench_df, trade_count


def main():
    print("=" * 70)
    print("    QuanTrade 2.0 - 组合ETF + 波动率目标")
    print("=" * 70)

    all_prices = load_data()
    print(f"\n加载 {len(all_prices)} 只ETF")

    # 测试不同参数
    param_sets = [
        {'momentum_period': 90, 'trend_ma': 120, 'min_hold_months': 6, 'top_n': 3, 'vol_target': 0.15},
        {'momentum_period': 90, 'trend_ma': 120, 'min_hold_months': 6, 'top_n': 2, 'vol_target': 0.15},
        {'momentum_period': 90, 'trend_ma': 120, 'min_hold_months': 6, 'top_n': 3, 'vol_target': 0.12},
        {'momentum_period': 120, 'trend_ma': 120, 'min_hold_months': 6, 'top_n': 3, 'vol_target': 0.15},
    ]

    best_score = 0
    best_report = None
    best_params = None

    for i, params in enumerate(param_sets):
        nav_df, bench_df, trade_count = backtest_portfolio(all_prices, **params)

        strategy_return = nav_df['nav'].iloc[-1] / 50000 - 1
        benchmark_return = bench_df['nav'].iloc[-1] / 50000 - 1

        agent = EvaluationAgent(verbose=False)
        report = agent.full_evaluation(
            strategy_returns=nav_df['returns'].dropna(),
            benchmark_returns=bench_df['returns'].dropna()
        )

        print(f"\n[{i+1}] top_n={params['top_n']}, vol_target={params['vol_target']}, 动量={params['momentum_period']}天")
        print(f"    收益: {strategy_return:.2%} | 基准: {benchmark_return:.2%} | 超额: {strategy_return - benchmark_return:.2%} | 交易: {trade_count}次")
        print(f"    评分: {report.overall_score}/100 ({report.grade}) | 夏普: {report.metrics.get('sharpe_ratio', 0):.2f} | 回撤: {report.metrics.get('max_drawdown', 0):.2%}")

        if report.overall_score > best_score:
            best_score = report.overall_score
            best_report = report
            best_params = params

    print("\n" + "=" * 70)
    print(f"最佳参数: {best_params}")
    print(f"最佳评分: {best_score}/100 ({best_report.grade})")
    print("=" * 70)

    agent = EvaluationAgent(verbose=True)
    agent.print_report(best_report)

    return best_report


if __name__ == "__main__":
    main()
