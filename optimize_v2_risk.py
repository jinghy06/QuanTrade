"""
QuanTrade 2.0 - 带风险管理的优化版
=================================
添加：
1. 最大回撤控制（回撤超过阈值时减仓）
2. 波动率目标（根据波动率调整仓位）
3. 市场状态检测（熊市时降低仓位）
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


def backtest_with_risk(
    all_prices,
    momentum_period=90,
    trend_ma=120,
    min_hold_months=6,
    max_drawdown_threshold=-0.15,  # 回撤超过15%时减仓
    volatility_target=0.15,        # 目标波动率15%
    market_ma_period=60            # 市场状态判断均线
):
    """带风险管理的回测"""
    all_dates = set()
    for df in all_prices.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    start_idx = 130
    backtest_dates = all_dates[start_idx:]

    capital = 50000
    holdings = {}  # {symbol: (shares, entry_price)}
    nav_history = []
    last_trade_date = None
    trade_count = 0
    peak_nav = initial_capital = 50000
    position_ratio = 1.0  # 仓位比例

    for date in backtest_dates:
        date_ts = pd.Timestamp(date)

        # 计算当前NAV
        portfolio_value = capital
        for h_symbol, (h_shares, _) in holdings.items():
            price = all_prices[h_symbol]['close'].get(date, 0)
            portfolio_value += h_shares * price

        # 更新峰值
        if portfolio_value > peak_nav:
            peak_nav = portfolio_value

        # 风险管理：回撤控制
        current_drawdown = (portfolio_value - peak_nav) / peak_nav
        if current_drawdown < max_drawdown_threshold:
            # 回撤过大，减仓到50%
            if position_ratio > 0.5:
                position_ratio = 0.5
                # 卖出一半持仓
                for h_symbol, (h_shares, entry_price) in list(holdings.items()):
                    sell_shares = h_shares // 2
                    if sell_shares > 0:
                        sell_price = all_prices[h_symbol]['close'].get(date, 0)
                        if sell_price > 0:
                            capital += sell_shares * sell_price * (1 - COST_RATE)
                            holdings[h_symbol] = (h_shares - sell_shares, entry_price)
                            trade_count += 1
        elif current_drawdown > -0.05:
            # 回撤恢复，恢复正常仓位
            position_ratio = 1.0

        # 风险管理：市场状态检测
        benchmark = all_prices.get('510300')
        if benchmark is not None and date in benchmark.index:
            bench_hist = benchmark.loc[:date]['close']
            if len(bench_hist) >= market_ma_period:
                bench_ma = bench_hist.rolling(market_ma_period).mean().iloc[-1]
                market_uptrend = bench_hist.iloc[-1] > bench_ma
                if not market_uptrend:
                    position_ratio = min(position_ratio, 0.7)  # 熊市降低仓位

        # 季度调仓
        should_rebalance = False
        if last_trade_date is None:
            should_rebalance = True
        else:
            months_since = (date_ts.year - last_trade_date.year) * 12 + (date_ts.month - last_trade_date.month)
            if months_since >= min_hold_months and date_ts.month in [1, 4, 7, 10] and date_ts.day <= 5:
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
                momentum = close.iloc[-1] / close.iloc[-momentum_period] - 1
                ma_long = close.rolling(trend_ma).mean().iloc[-1]
                in_uptrend = close.iloc[-1] > ma_long

                # 波动率调整
                returns = close.pct_change().iloc[-momentum_period:]
                vol = returns.std() * np.sqrt(252)
                vol_adj = min(volatility_target / (vol + 1e-8), 1.5)

                if in_uptrend and momentum > 0:
                    scores[symbol] = momentum * vol_adj

            if scores:
                best_symbol = max(scores, key=scores.get)
                if best_symbol not in holdings or len(holdings) > 1:
                    # 卖出非目标持仓
                    for h_symbol, (h_shares, _) in list(holdings.items()):
                        if h_symbol != best_symbol:
                            sell_price = all_prices[h_symbol]['close'].get(date, 0)
                            if sell_price > 0:
                                capital += h_shares * sell_price * (1 - COST_RATE)
                                trade_count += 1
                            del holdings[h_symbol]

                    # 买入目标ETF
                    if best_symbol not in holdings:
                        buy_price = all_prices[best_symbol]['close'].get(date, 0)
                        if buy_price > 0:
                            # 根据position_ratio调整买入量
                            available_capital = capital * position_ratio
                            shares = int(available_capital / buy_price / 100) * 100
                            if shares > 0:
                                cost = shares * buy_price * (1 + COST_RATE)
                                capital -= cost
                                holdings[best_symbol] = (shares, buy_price)
                                trade_count += 1
                    last_trade_date = date_ts
            elif holdings:
                for h_symbol, (h_shares, _) in list(holdings.items()):
                    sell_price = all_prices[h_symbol]['close'].get(date, 0)
                    if sell_price > 0:
                        capital += h_shares * sell_price * (1 - COST_RATE)
                        trade_count += 1
                holdings = {}
                last_trade_date = date_ts

        # 计算NAV
        portfolio_value = capital
        for h_symbol, (h_shares, _) in holdings.items():
            price = all_prices[h_symbol]['close'].get(date, 0)
            portfolio_value += h_shares * price
        nav_history.append({'date': date, 'nav': portfolio_value})

    # 基准
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
    print("    QuanTrade 2.0 - 带风险管理的优化版")
    print("=" * 70)

    all_prices = load_data()
    print(f"\n加载 {len(all_prices)} 只核心ETF")

    # 测试不同风险管理参数
    risk_params = [
        {'max_drawdown_threshold': -0.15, 'volatility_target': 0.15, 'market_ma_period': 60},
        {'max_drawdown_threshold': -0.10, 'volatility_target': 0.12, 'market_ma_period': 60},
        {'max_drawdown_threshold': -0.20, 'volatility_target': 0.15, 'market_ma_period': 60},
    ]

    best_grade = 'F'
    best_score = 0
    best_report = None

    for i, risk_p in enumerate(risk_params):
        nav_df, bench_df, trade_count = backtest_with_risk(
            all_prices,
            momentum_period=90,
            trend_ma=120,
            min_hold_months=6,
            **risk_p
        )

        strategy_return = nav_df['nav'].iloc[-1] / 50000 - 1
        benchmark_return = bench_df['nav'].iloc[-1] / 50000 - 1

        agent = EvaluationAgent(verbose=False)
        report = agent.full_evaluation(
            strategy_returns=nav_df['returns'].dropna(),
            benchmark_returns=bench_df['returns'].dropna()
        )

        print(f"\n[{i+1}] 回撤阈值={risk_p['max_drawdown_threshold']:.0%}, 波动率目标={risk_p['volatility_target']:.0%}")
        print(f"    收益: {strategy_return:.2%} | 基准: {benchmark_return:.2%} | 交易: {trade_count}次")
        print(f"    评分: {report.overall_score}/100 ({report.grade}) | 夏普: {report.metrics.get('sharpe_ratio', 0):.2f} | 最大回撤: {report.metrics.get('max_drawdown', 0):.2%}")

        if report.overall_score > best_score:
            best_score = report.overall_score
            best_grade = report.grade
            best_report = report
            best_nav = nav_df
            best_bench = bench_df

    print("\n" + "=" * 70)
    print(f"最佳评分: {best_score}/100 ({best_grade})")
    print("=" * 70)

    agent = EvaluationAgent(verbose=True)
    agent.print_report(best_report)

    return best_report


if __name__ == "__main__":
    main()
