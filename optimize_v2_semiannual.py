"""
QuanTrade 2.0 - 半年度调仓+保守仓位
===================================
尝试：
1. 半年度调仓（1月/7月）
2. 保守仓位（只用80%资金）
3. 更长动量周期
"""
import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from evaluator_agent import EvaluationAgent

DB_PATH = "QuanTrade/quant_system/data/quant.db"
COST_RATE = 0.0036
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


def backtest_semiannual(
    all_prices,
    momentum_period=120,
    trend_ma=120,
    position_ratio=0.8,  # 保守仓位80%
    rebalance_months=[1, 7]  # 1月和7月调仓
):
    """半年度调仓策略"""
    all_dates = set()
    for df in all_prices.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    start_idx = 130
    backtest_dates = all_dates[start_idx:]

    capital = 50000
    invested = 0
    holdings = {}
    nav_history = []
    trade_count = 0
    last_rebalance_month = None

    for date in backtest_dates:
        date_ts = pd.Timestamp(date)

        # 半年度调仓
        should_rebalance = False
        if last_rebalance_month is None:
            should_rebalance = True
        elif date_ts.month in rebalance_months and date_ts.day <= 5:
            if last_rebalance_month != (date_ts.year, date_ts.month):
                should_rebalance = True

        if should_rebalance:
            scores = {}
            for symbol, df in all_prices.items():
                if date not in df.index:
                    continue
                hist = df.loc[:date]
                if len(hist) < momentum_period + 10:
                    continue
                close = hist['close']

                # 长期动量
                momentum = close.iloc[-1] / close.iloc[-momentum_period] - 1

                # 趋势过滤
                ma_long = close.rolling(trend_ma).mean().iloc[-1]
                in_uptrend = close.iloc[-1] > ma_long

                if in_uptrend and momentum > 0:
                    scores[symbol] = momentum

            # 卖出全部持仓
            for h_symbol, h_shares in list(holdings.items()):
                sell_price = all_prices[h_symbol]['close'].get(date, 0)
                if sell_price > 0:
                    capital += h_shares * sell_price * (1 - COST_RATE)
                    trade_count += 1
            holdings = {}

            # 买入最佳ETF
            if scores:
                best_symbol = max(scores, key=scores.get)
                buy_price = all_prices[best_symbol]['close'].get(date, 0)
                if buy_price > 0:
                    # 保守仓位：只用position_ratio比例的资金
                    available = capital * position_ratio
                    shares = int(available / buy_price / 100) * 100
                    if shares > 0:
                        cost = shares * buy_price * (1 + COST_RATE)
                        capital -= cost
                        holdings[best_symbol] = shares
                        trade_count += 1

            last_rebalance_month = (date_ts.year, date_ts.month)

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
    print("    QuanTrade 2.0 - 半年度调仓+保守仓位")
    print("=" * 70)

    all_prices = load_data()
    print(f"\n加载 {len(all_prices)} 只核心ETF")

    # 测试不同参数
    param_sets = [
        {'momentum_period': 120, 'trend_ma': 120, 'position_ratio': 0.8, 'rebalance_months': [1, 7]},
        {'momentum_period': 120, 'trend_ma': 120, 'position_ratio': 0.7, 'rebalance_months': [1, 7]},
        {'momentum_period': 180, 'trend_ma': 120, 'position_ratio': 0.8, 'rebalance_months': [1, 7]},
        {'momentum_period': 120, 'trend_ma': 120, 'position_ratio': 0.8, 'rebalance_months': [1, 4, 7, 10]},
        {'momentum_period': 90, 'trend_ma': 120, 'position_ratio': 0.8, 'rebalance_months': [1, 7]},
    ]

    best_score = 0
    best_report = None
    best_params = None
    best_nav = None
    best_bench = None

    for i, params in enumerate(param_sets):
        nav_df, bench_df, trade_count = backtest_semiannual(all_prices, **params)

        strategy_return = nav_df['nav'].iloc[-1] / 50000 - 1
        benchmark_return = bench_df['nav'].iloc[-1] / 50000 - 1

        agent = EvaluationAgent(verbose=False)
        report = agent.full_evaluation(
            strategy_returns=nav_df['returns'].dropna(),
            benchmark_returns=bench_df['returns'].dropna()
        )

        print(f"\n[{i+1}] 动量={params['momentum_period']}天, 仓位={params['position_ratio']:.0%}, 调仓={params['rebalance_months']}月")
        print(f"    收益: {strategy_return:.2%} | 基准: {benchmark_return:.2%} | 超额: {strategy_return - benchmark_return:.2%} | 交易: {trade_count}次")
        print(f"    评分: {report.overall_score}/100 ({report.grade})")
        print(f"    夏普: {report.metrics.get('sharpe_ratio', 0):.2f} | 回撤: {report.metrics.get('max_drawdown', 0):.2%} | 过拟合: {report.metrics.get('overfit_score', 0):.2f}")

        if report.overall_score > best_score:
            best_score = report.overall_score
            best_report = report
            best_params = params
            best_nav = nav_df
            best_bench = bench_df

    print("\n" + "=" * 70)
    print(f"最佳参数: {best_params}")
    print(f"最佳评分: {best_score}/100 ({best_report.grade})")
    print("=" * 70)

    agent = EvaluationAgent(verbose=True)
    agent.print_report(best_report)

    return best_report


if __name__ == "__main__":
    main()
