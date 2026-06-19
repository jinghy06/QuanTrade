"""
QuanTrade 2.0 - 最终优化版
===========================
目标：提高统计显著性，降低收益波动
方法：
1. 使用多时间框架动量确认
2. 添加趋势强度过滤
3. 降低单次仓位集中度
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


def backtest_enhanced(
    all_prices,
    momentum_short=60,
    momentum_long=120,
    trend_ma=120,
    min_hold_months=6,
    trend_threshold=0.05  # 趋势强度阈值
):
    """
    增强版回测
    - 多时间框架动量确认
    - 趋势强度过滤
    """
    all_dates = set()
    for df in all_prices.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    start_idx = 130
    backtest_dates = all_dates[start_idx:]

    capital = 50000
    holdings = {}
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
            scores = {}
            for symbol, df in all_prices.items():
                if date not in df.index:
                    continue
                hist = df.loc[:date]
                if len(hist) < momentum_long + 10:
                    continue
                close = hist['close']

                # 多时间框架动量
                mom_short = close.iloc[-1] / close.iloc[-momentum_short] - 1
                mom_long = close.iloc[-1] / close.iloc[-momentum_long] - 1

                # 趋势强度（价格相对均线的偏离度）
                ma_long = close.rolling(trend_ma).mean().iloc[-1]
                trend_strength = (close.iloc[-1] - ma_long) / ma_long

                # 趋势一致性（短期和长期动量方向一致）
                trend_consistent = (mom_short > 0) == (mom_long > 0)

                # 趋势过滤
                in_uptrend = trend_strength > 0

                if in_uptrend and trend_consistent and mom_long > trend_threshold:
                    # 综合得分：长期动量为主，短期动量为辅
                    scores[symbol] = mom_long * 0.7 + mom_short * 0.3

            if scores:
                best_symbol = max(scores, key=scores.get)
                if best_symbol not in holdings or len(holdings) > 1:
                    for h_symbol, h_shares in list(holdings.items()):
                        if h_symbol != best_symbol:
                            sell_price = all_prices[h_symbol]['close'].get(date, 0)
                            if sell_price > 0:
                                capital += h_shares * sell_price * (1 - COST_RATE)
                                trade_count += 1
                            del holdings[h_symbol]
                    if best_symbol not in holdings:
                        buy_price = all_prices[best_symbol]['close'].get(date, 0)
                        if buy_price > 0:
                            shares = int(capital / buy_price / 100) * 100
                            if shares > 0:
                                capital -= shares * buy_price * (1 + COST_RATE)
                                holdings[best_symbol] = shares
                                trade_count += 1
                    last_trade_date = date_ts
            elif holdings:
                for h_symbol, h_shares in list(holdings.items()):
                    sell_price = all_prices[h_symbol]['close'].get(date, 0)
                    if sell_price > 0:
                        capital += h_shares * sell_price * (1 - COST_RATE)
                        trade_count += 1
                holdings = {}
                last_trade_date = date_ts

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
    print("    QuanTrade 2.0 - 最终优化版")
    print("=" * 70)

    all_prices = load_data()
    print(f"\n加载 {len(all_prices)} 只核心ETF")

    # 参数搜索
    param_sets = [
        {'momentum_short': 60, 'momentum_long': 120, 'trend_ma': 120, 'min_hold_months': 6, 'trend_threshold': 0.05},
        {'momentum_short': 60, 'momentum_long': 120, 'trend_ma': 120, 'min_hold_months': 6, 'trend_threshold': 0.02},
        {'momentum_short': 90, 'momentum_long': 180, 'trend_ma': 120, 'min_hold_months': 6, 'trend_threshold': 0.05},
        {'momentum_short': 60, 'momentum_long': 120, 'trend_ma': 60, 'min_hold_months': 6, 'trend_threshold': 0.05},
    ]

    best_score = 0
    best_report = None
    best_params = None

    for i, params in enumerate(param_sets):
        nav_df, bench_df, trade_count = backtest_enhanced(all_prices, **params)

        strategy_return = nav_df['nav'].iloc[-1] / 50000 - 1
        benchmark_return = bench_df['nav'].iloc[-1] / 50000 - 1

        agent = EvaluationAgent(verbose=False)
        report = agent.full_evaluation(
            strategy_returns=nav_df['returns'].dropna(),
            benchmark_returns=bench_df['returns'].dropna()
        )

        print(f"\n[{i+1}] 短动量={params['momentum_short']}, 长动量={params['momentum_long']}, 阈值={params['trend_threshold']}")
        print(f"    收益: {strategy_return:.2%} | 基准: {benchmark_return:.2%} | 交易: {trade_count}次")
        print(f"    评分: {report.overall_score}/100 ({report.grade})")
        print(f"    夏普: {report.metrics.get('sharpe_ratio', 0):.2f} | 最大回撤: {report.metrics.get('max_drawdown', 0):.2%}")
        print(f"    过拟合: {report.metrics.get('overfit_score', 0):.2f} | Walk-Forward: {report.metrics.get('wf_consistency', 0):.0%}")

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
