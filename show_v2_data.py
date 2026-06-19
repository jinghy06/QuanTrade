"""显示QuanTrade 2.0完整数据"""
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


def backtest(all_prices, momentum_period=90, trend_ma=60, min_hold_months=6):
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
    trades = []

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
                                trades.append({'date': date, 'action': 'SELL', 'symbol': h_symbol})
                            del holdings[h_symbol]
                    if best_symbol not in holdings:
                        buy_price = all_prices[best_symbol]['close'].get(date, 0)
                        if buy_price > 0:
                            shares = int(capital / buy_price / 100) * 100
                            if shares > 0:
                                capital -= shares * buy_price * (1 + COST_RATE)
                                holdings[best_symbol] = shares
                                trade_count += 1
                                trades.append({'date': date, 'action': 'BUY', 'symbol': best_symbol})
                    last_trade_date = date_ts
            elif holdings:
                for h_symbol, h_shares in list(holdings.items()):
                    sell_price = all_prices[h_symbol]['close'].get(date, 0)
                    if sell_price > 0:
                        capital += h_shares * sell_price * (1 - COST_RATE)
                        trade_count += 1
                        trades.append({'date': date, 'action': 'SELL', 'symbol': h_symbol})
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
    return nav_df, bench_df, trade_count, trades


def main():
    all_prices = load_data()
    nav_df, bench_df, trade_count, trades = backtest(all_prices)

    strategy_return = nav_df['nav'].iloc[-1] / 50000 - 1
    benchmark_return = bench_df['nav'].iloc[-1] / 50000 - 1
    returns = nav_df['returns'].dropna()

    print("=" * 70)
    print("           QuanTrade 2.0 完整数据报告")
    print("=" * 70)

    # 1. 基础收益
    print("\n【1. 基础收益数据】")
    print(f"  初始资金:     50,000 元")
    print(f"  最终净值:     {nav_df['nav'].iloc[-1]:,.0f} 元")
    print(f"  策略总收益:   {strategy_return:.2%}")
    print(f"  基准总收益:   {benchmark_return:.2%}")
    print(f"  超额收益:     {strategy_return - benchmark_return:.2%}")
    print(f"  回测天数:     {len(nav_df)} 天")
    print(f"  回测年数:     {len(nav_df)/252:.1f} 年")

    # 2. 风险指标
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    sharpe = (returns.mean() - 0.02/252) / returns.std() * np.sqrt(252)
    annual_return = (1 + strategy_return) ** (252 / len(nav_df)) - 1
    annual_vol = returns.std() * np.sqrt(252)

    print("\n【2. 风险指标】")
    print(f"  年化收益率:   {annual_return:.2%}")
    print(f"  年化波动率:   {annual_vol:.2%}")
    print(f"  夏普比率:     {sharpe:.2f}")
    print(f"  最大回撤:     {drawdown.min():.2%}")

    # 3. 交易统计
    win_days = (returns > 0).sum()
    lose_days = (returns < 0).sum()
    win_rate = win_days / (win_days + lose_days)

    print("\n【3. 交易统计】")
    print(f"  交易次数:     {trade_count} 次")
    print(f"  盈利天数:     {win_days} 天")
    print(f"  亏损天数:     {lose_days} 天")
    print(f"  胜率(日):     {win_rate:.2%}")
    print(f"  平均日收益:   {returns.mean():.4%}")
    print(f"  最大日盈利:   {returns.max():.2%}")
    print(f"  最大日亏损:   {returns.min():.2%}")

    # 4. 交易记录
    print("\n【4. 交易记录】")
    for t in trades:
        print(f"  {t['date'].strftime('%Y-%m-%d')} | {t['action']:4s} | {t['symbol']}")

    # 5. 评价Agent
    print("\n【5. 独立评价Agent评估】")
    agent = EvaluationAgent(verbose=False)
    report = agent.full_evaluation(
        strategy_returns=returns,
        benchmark_returns=bench_df['returns'].dropna()
    )

    print(f"  综合评分:     {report.overall_score}/100")
    print(f"  等级:         {report.grade}")
    print(f"  判断:         {report.verdict}")
    print(f"  过拟合评分:   {report.metrics['overfit_score']:.2f}")
    print(f"  Walk-Forward: {report.metrics['wf_consistency']:.0%}")
    print(f"  统计显著性:   {'是' if report.metrics['is_significant'] else '否'} (p={report.metrics['p_value']:.4f})")
    print(f"  信息比率:     {report.metrics['information_ratio']:.3f}")
    print(f"  t统计量:      {report.metrics['t_statistic']:.3f}")

    # 6. 警告
    print("\n【6. 评价Agent警告】")
    for w in report.warnings:
        print(f"  {w}")

    # 7. 分年度收益
    print("\n【7. 分年度收益】")
    nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year
    for year in sorted(nav_df['year'].unique()):
        year_data = nav_df[nav_df['year'] == year]
        year_return = year_data['nav'].iloc[-1] / year_data['nav'].iloc[0] - 1
        print(f"  {year}年: {year_return:.2%}")

    # 8. ETF池
    print("\n【8. ETF池】")
    for symbol in ALL_ETFS:
        df = all_prices.get(symbol)
        if df is not None:
            total_ret = df['close'].iloc[-1] / df['close'].iloc[0] - 1
            print(f"  {symbol}: {len(df)}条数据, 买入持有收益: {total_ret:.2%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
