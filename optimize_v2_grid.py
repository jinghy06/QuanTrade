"""
QuanTrade 2.0 - 网格搜索最优参数
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


def backtest(all_prices, momentum_period=90, trend_ma=120, min_hold_months=6):
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
                if in_uptrend and momentum > 0:
                    scores[symbol] = momentum

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
    print("    QuanTrade 2.0 - 网格搜索")
    print("=" * 70)

    all_prices = load_data()
    print(f"\n加载 {len(all_prices)} 只ETF")

    # 网格搜索
    momentum_list = [60, 75, 90, 105, 120]
    ma_list = [60, 90, 120, 150, 180]
    hold_list = [3, 6, 9, 12]

    best_score = 0
    best_params = None
    best_report = None
    results = []

    total = len(momentum_list) * len(ma_list) * len(hold_list)
    count = 0

    for mom in momentum_list:
        for ma in ma_list:
            for hold in hold_list:
                count += 1
                nav_df, bench_df, tc = backtest(all_prices, mom, ma, hold)
                sr = nav_df['nav'].iloc[-1] / 50000 - 1
                br = bench_df['nav'].iloc[-1] / 50000 - 1

                agent = EvaluationAgent(verbose=False)
                report = agent.full_evaluation(nav_df['returns'].dropna(), bench_df['returns'].dropna())

                results.append({
                    'momentum': mom,
                    'ma': ma,
                    'hold': hold,
                    'return': sr,
                    'excess': sr - br,
                    'score': report.overall_score,
                    'grade': report.grade,
                    'sharpe': report.metrics.get('sharpe_ratio', 0),
                    'drawdown': report.metrics.get('max_drawdown', 0)
                })

                if report.overall_score > best_score:
                    best_score = report.overall_score
                    best_params = (mom, ma, hold)
                    best_report = report

                if count % 10 == 0:
                    print(f"  进度: {count}/{total}")

    # 打印结果
    print("\n" + "=" * 70)
    print("网格搜索结果:")
    print("=" * 70)

    # 按评分排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    print("\n前10名:")
    for i, row in results_df.head(10).iterrows():
        print(f"  动量={row['momentum']:.0f}, 均线={row['ma']:.0f}, 持有>={row['hold']:.0f}月 | 收益={row['return']:.2%} | 评分={row['score']:.0f}({row['grade']}) | 夏普={row['sharpe']:.2f}")

    print(f"\n最佳参数: 动量={best_params[0]}, 均线={best_params[1]}, 持有>={best_params[2]}月")
    print(f"最佳评分: {best_score}/100 ({best_report.grade})")

    # 打印详细报告
    agent = EvaluationAgent(verbose=True)
    agent.print_report(best_report)

    return best_report


if __name__ == "__main__":
    main()
