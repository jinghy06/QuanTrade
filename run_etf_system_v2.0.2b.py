"""
QuanTrade 2.0.2b - 纯动量热点赛道策略（简化版）
====================================================
核心改动：
1. 去掉ML模型（熊市训练→牛市测试不匹配）
2. 纯动量热点选股：选中长期动量最强的5-10只ETF
3. 满仓轮动，不保留现金
4. 基准=24只ETF等权买入持有
5. 融入用户策略：大跌建仓、分批、情绪高点减仓
6. 三因子融入：情绪/政治/政策作为仓位调整系数
"""

import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from evaluator_agent import EvaluationAgent

# ============================================================
# 配置
# ============================================================
DB_PATH = "QuanTrade/quant_system/data/quant.db"
COST_RATE = 0.0036
INITIAL_CAPITAL = 50000

ETF_POOL = [
    '562500', '515070', '159995', '159550', '516510',
    '512660', '512670', '515960',
    '515790', '516160', '561160', '159790',
    '512010', '159928', '512690', '515170',
    '512480', '588000', '159915', '513180',
    '512880', '512800', '512200',
]

BENCHMARK_SYMBOL = '510300'
GOLD_SYMBOL = '518880'

EMOTION_ETFS = ['562500', '515070', '159995', '159550', '516510', '512660', '512670', '515960', '588000', '159915', '513180', '512480']
FUNDAMENTAL_ETFS = ['512010', '159928', '512690', '515170', '515790', '516160', '561160', '159790', '512880', '512800', '512200']


# ============================================================
# 数据加载
# ============================================================
def load_all_data():
    conn = sqlite3.connect(DB_PATH)
    all_prices = {}
    for symbol in ETF_POOL + [BENCHMARK_SYMBOL]:
        df = pd.read_sql(f"SELECT * FROM etf_daily_prices WHERE symbol='{symbol}' ORDER BY trade_date", conn)
        if len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')
            all_prices[symbol] = df
    gold_df = pd.read_sql("SELECT * FROM gold_daily_prices ORDER BY date", conn)
    gold_df['date'] = pd.to_datetime(gold_df['date'])
    gold_df = gold_df.set_index('date')
    conn.close()
    return all_prices, gold_df


# ============================================================
# 热点赛道得分（纯动量版）
# ============================================================
def calculate_momentum_score(etf_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> float:
    """
    纯动量热点得分：
    - 120日动量 (40%) — 最重要，捕捉长期趋势
    - 60日动量 (30%) — 中期趋势
    - 20日相对强弱 vs 沪深300 (20%) — 相对表现
    - 趋势强度 (10%) — 均线排列
    """
    if len(etf_df) < 120:
        return -999
    
    close = etf_df['close']
    
    # 120日动量 (40%)
    mom_120 = close.iloc[-1] / close.iloc[-120] - 1
    score_120 = np.clip(mom_120 / 0.5, -1, 1)  # 50%涨幅=满分
    
    # 60日动量 (30%)
    mom_60 = close.iloc[-1] / close.iloc[-60] - 1
    score_60 = np.clip(mom_60 / 0.3, -1, 1)
    
    # 20日相对强弱 vs 沪深300 (20%)
    score_relative = 0
    if benchmark_df is not None and len(benchmark_df) > 120:
        bench = benchmark_df['close'].reindex(etf_df.index, method='ffill')
        etf_mom_20 = close.iloc[-1] / close.iloc[-20] - 1
        bench_mom_20 = bench.iloc[-1] / bench.iloc[-20] - 1
        score_relative = np.clip((etf_mom_20 - bench_mom_20) / 0.2, -1, 1)
    
    # 趋势强度 (10%)
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    trend_score = 0.4 * (1 if ma5 > ma20 else 0) + 0.3 * (1 if ma20 > ma60 else 0) + 0.3 * (1 if close.iloc[-1] > ma5 else 0)
    
    return score_120 * 0.40 + score_60 * 0.30 + score_relative * 0.20 + trend_score * 0.10


# ============================================================
# 三因子
# ============================================================
def calculate_three_factors(benchmark_df: pd.DataFrame, gold_df: pd.DataFrame) -> dict:
    if benchmark_df is None or len(benchmark_df) < 60:
        return {'sentiment': 0, 'geopolitical': 0, 'policy': 0, 'combined': 0}
    
    close = benchmark_df['close']
    returns = close.pct_change()
    
    # 情绪：波动率+动量
    vol_20 = returns.rolling(20).std().iloc[-1]
    vol_ma60 = returns.rolling(60).std().iloc[-1]
    vol_signal = -0.5 if vol_20 > vol_ma60 * 1.5 else (0.5 if vol_20 < vol_ma60 * 0.7 else 0)
    mom_5 = close.pct_change(5).iloc[-1]
    mom_60 = close.pct_change(60).iloc[-1]
    mom_signal = 0.5 if mom_5 > 0 and mom_60 > 0 else (-0.5 if mom_5 < 0 and mom_60 < 0 else 0)
    sentiment = np.clip((vol_signal + mom_signal) / 2, -1, 1)
    
    # 地缘：黄金走势代理
    geo = 0
    if gold_df is not None and len(gold_df) > 60:
        g_close = gold_df['close']
        g_mom_5 = g_close.pct_change(5).iloc[-1]
        g_mom_20 = g_close.pct_change(20).iloc[-1]
        g_ma20 = g_close.rolling(20).mean().iloc[-1]
        if g_mom_5 > 0.02 and g_mom_20 > 0.05 and g_close.iloc[-1] > g_ma20:
            geo = -0.8
        elif g_mom_5 > 0.02:
            geo = -0.4
        elif g_mom_5 < -0.02:
            geo = 0.4
    
    # 政策：趋势+市场宽度
    ma120 = close.rolling(120).mean().iloc[-1]
    trend_score = np.clip((close.iloc[-1] - ma120) / ma120 * 5, -1, 1)
    high_60 = close.rolling(60).max()
    low_60 = close.rolling(60).min()
    position = (close.iloc[-1] - low_60.iloc[-1]) / (high_60.iloc[-1] - low_60.iloc[-1] + 1e-8)
    breadth_score = 0.5 if position > 0.8 else (-0.5 if position < 0.2 else 0)
    policy = np.clip((trend_score + breadth_score) / 2, -1, 1)
    
    combined = sentiment * 0.4 + geo * 0.3 + policy * 0.3
    return {'sentiment': sentiment, 'geopolitical': geo, 'policy': policy, 'combined': np.clip(combined, -1, 1)}


# ============================================================
# 情绪泡沫检测
# ============================================================
def detect_bubble(etf_df: pd.DataFrame) -> dict:
    if len(etf_df) < 60:
        return {'is_bubble': False, 'bubble_score': 0}
    
    close = etf_df['close']
    volume = etf_df.get('volume', pd.Series(0, index=etf_df.index))
    
    return_60d = close.iloc[-1] / close.iloc[-60] - 1
    vol_ma20 = volume.rolling(20).mean().iloc[-1] if volume.sum() > 0 else 1
    vol_current = volume.iloc[-1] if volume.sum() > 0 else 0
    turnover_spike = vol_current / (vol_ma20 + 1e-8) > 3
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    
    vol_5d = close.pct_change().rolling(5).std().iloc[-1]
    vol_60d = close.pct_change().rolling(60).std().iloc[-1]
    vol_spike = vol_5d / (vol_60d + 1e-8) > 2
    
    bubble_score = 0
    if return_60d > 0.5: bubble_score += 0.3
    if turnover_spike: bubble_score += 0.2
    if rsi > 80: bubble_score += 0.3
    if vol_spike: bubble_score += 0.2
    
    return {'is_bubble': bubble_score >= 0.5, 'bubble_score': bubble_score, 'rsi': rsi, 'return_60d': return_60d}


# ============================================================
# 等权基准
# ============================================================
def calculate_equal_weight_benchmark(all_prices: dict, dates: list) -> pd.DataFrame:
    start_date = dates[0]
    valid = [s for s in ETF_POOL if s in all_prices and start_date in all_prices[s].index]
    if not valid:
        return pd.DataFrame({'date': dates, 'nav': [INITIAL_CAPITAL] * len(dates)})
    
    per_etf = INITIAL_CAPITAL / len(valid)
    holdings = {}
    capital = INITIAL_CAPITAL
    
    for s in valid:
        p = all_prices[s]['close'].get(start_date, 0)
        if p > 0:
            shares = int(per_etf / p / 100) * 100
            if shares > 0:
                capital -= shares * p * (1 + COST_RATE)
                holdings[s] = shares
    
    nav = []
    for d in dates:
        v = capital
        for s, sh in holdings.items():
            if s in all_prices and d in all_prices[s].index:
                v += sh * all_prices[s]['close'].get(d, 0)
        nav.append({'date': d, 'nav': v})
    return pd.DataFrame(nav)


# ============================================================
# 主回测
# ============================================================
def main():
    print("=" * 70)
    print("    QuanTrade 2.0.2b - 纯动量热点赛道策略")
    print("=" * 70)
    
    all_prices, gold_df = load_all_data()
    benchmark_df = all_prices.get(BENCHMARK_SYMBOL)
    
    print(f"\n[数据] ETF: {len([s for s in ETF_POOL if s in all_prices])}/{len(ETF_POOL)} 只")
    print(f"       基准: {BENCHMARK_SYMBOL}, {len(benchmark_df)} 条")
    print(f"       黄金: {len(gold_df)} 条")
    
    all_dates = sorted(set().union(*[all_prices[s].index for s in ETF_POOL if s in all_prices]))
    train_end = pd.Timestamp('2023-12-31')
    test_dates = [d for d in all_dates if d > train_end]
    print(f"       测试期: {test_dates[0].date()} ~ {test_dates[-1].date()}, {len(test_dates)} 天")
    
    # 等权基准
    print("\n" + "=" * 70)
    print("[基准] 等权买入持有")
    print("=" * 70)
    bench_nav = calculate_equal_weight_benchmark(all_prices, test_dates)
    bench_ret = bench_nav['nav'].iloc[-1] / INITIAL_CAPITAL - 1
    print(f"  基准收益: {bench_ret:.2%}")
    
    # 策略回测
    print("\n" + "=" * 70)
    print("[策略回测] 纯动量热点赛道 + 三因子")
    print("=" * 70)
    
    capital = INITIAL_CAPITAL
    holdings = {}  # {symbol: shares}
    nav_history = []
    trade_count = 0
    last_rebalance = None
    
    # 月度调仓日
    rebalance_dates = []
    last_m = None
    for d in test_dates:
        if last_m != (d.year, d.month):
            rebalance_dates.append(d)
            last_m = (d.year, d.month)
    
    for i, date in enumerate(test_dates):
        date_ts = pd.Timestamp(date)
        
        # 市场状态
        market_return = 0
        if benchmark_df is not None and i > 0:
            prev = test_dates[i-1]
            if prev in benchmark_df.index and date in benchmark_df.index:
                market_return = benchmark_df['close'].get(date, 0) / benchmark_df['close'].get(prev, 1) - 1
        
        # 三因子
        bench_hist = benchmark_df.loc[:date] if benchmark_df is not None else None
        gold_hist = gold_df.loc[:date] if gold_df is not None else None
        factors = calculate_three_factors(bench_hist, gold_hist)
        
        # 调仓判断：月度 或 情绪高点 或 大跌
        should_rebalance = False
        if last_rebalance is None:
            should_rebalance = True
        elif date in rebalance_dates:
            should_rebalance = True
        elif factors['sentiment'] > 0.8 and holdings:
            should_rebalance = True
            print(f"  {date.date()} 情绪高点触发")
        elif market_return < -0.03 and not holdings:
            should_rebalance = True
            print(f"  {date.date()} 大跌触发建仓")
        
        if should_rebalance:
            # 计算所有ETF动量得分
            scores = {}
            for symbol in ETF_POOL:
                if symbol not in all_prices or date not in all_prices[symbol].index:
                    continue
                hist = all_prices[symbol].loc[:date]
                if len(hist) < 120:
                    continue
                score = calculate_momentum_score(hist, benchmark_df)
                if score > -900:
                    scores[symbol] = score
            
            if scores:
                # 选前N只（根据市场环境调整）
                # 牛市（情绪>0.3）多选10只，震荡/熊市选5只
                n_top = 10 if factors['sentiment'] > 0.3 else 5
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_etfs = sorted_scores[:n_top]
                
                # 清仓不在前N的
                for s in list(holdings.keys()):
                    if s not in [x[0] for x in top_etfs]:
                        p = all_prices[s]['close'].get(date, 0)
                        if p > 0:
                            capital += holdings[s] * p * (1 - COST_RATE)
                            trade_count += 1
                            del holdings[s]
                
                # 检查减仓：泡沫/情绪高点/目标收益
                for s, shares in list(holdings.items()):
                    p = all_prices[s]['close'].get(date, 0)
                    if p <= 0: continue
                    
                    # 估算成本（简单平均）
                    bubble = detect_bubble(all_prices[s].loc[:date])
                    
                    reduce = 0.0
                    # 情绪高点减仓
                    if factors['sentiment'] > 0.8:
                        if s in EMOTION_ETFS and bubble['is_bubble']:
                            reduce = 0.60  # 泡沫情绪ETF多减
                            print(f"    [泡沫预警] {s} 减仓60%")
                        elif s in EMOTION_ETFS:
                            reduce = 0.40
                        else:
                            reduce = 0.20
                    
                    # 达到目标收益减仓（涨30%减仓30%）
                    # 简化为：如果60日涨幅>30%且情绪>0.5，减仓
                    if bubble['return_60d'] > 0.30 and factors['sentiment'] > 0.5:
                        reduce = max(reduce, 0.30)
                    
                    if reduce > 0:
                        sell_shares = int(shares * reduce / 100) * 100
                        if sell_shares > 0:
                            capital += sell_shares * p * (1 - COST_RATE)
                            holdings[s] -= sell_shares
                            if holdings[s] <= 0:
                                del holdings[s]
                            trade_count += 1
                            print(f"    {date.date()} 减仓 {s} {reduce:.0%} (RSI{bubble['rsi']:.0f}, 60日{bubble['return_60d']:.1%})")
                
                # 买入（等权分配到前N只）
                valid_top = [s for s, _ in top_etfs if s not in holdings or holdings[s] == 0]
                
                # 股票仓位：牛市90%（情绪>0.3），其他70%
                stock_ratio = 0.90 if factors['sentiment'] > 0.3 else 0.70
                # 恐慌时降低仓位
                if factors['geopolitical'] < -0.5:
                    stock_ratio = max(0.30, stock_ratio - 0.30)
                
                per_etf_cap = capital * stock_ratio / len(top_etfs)
                
                for symbol, score in top_etfs:
                    p = all_prices[symbol]['close'].get(date, 0)
                    if p <= 0: continue
                    
                    # 已持有：检查是否加仓（上升潮）
                    if symbol in holdings and holdings[symbol] > 0:
                        hist = all_prices[symbol].loc[:date]
                        close = hist['close']
                        if len(close) >= 3:
                            upsurge = close.pct_change(3).iloc[-1] > 0.05
                            ma20 = close.rolling(20).mean().iloc[-1]
                            above_ma20 = close.iloc[-1] > ma20 * 1.02
                            if upsurge or above_ma20:
                                # 计算当前仓位，如果不足目标则加仓
                                current_value = holdings[symbol] * p
                                total_value = capital + sum(holdings[s] * all_prices[s]['close'].get(date, 0) for s in holdings)
                                target_value = total_value * stock_ratio / len(top_etfs)
                                if current_value < target_value * 0.8:
                                    add_amount = min(target_value - current_value, capital * 0.15)
                                    add_shares = int(add_amount / p / 100) * 100
                                    if add_shares > 0:
                                        capital -= add_shares * p * (1 + COST_RATE)
                                        holdings[symbol] += add_shares
                                        trade_count += 1
                                        print(f"    {date.date()} 加仓 {symbol} {add_shares}股 @ {p:.3f} (上升潮)")
                    else:
                        # 未持有：建仓
                        # 用户策略：大跌建仓（如果当天大跌）
                        hist = all_prices[symbol].loc[:date]
                        etf_return = hist['close'].pct_change().iloc[-1] if len(hist) > 1 else 0
                        
                        if market_return < -0.03 and etf_return < -0.05:
                            # 试探建仓（小仓位）
                            probe_cap = per_etf_cap * 0.3
                            shares = int(probe_cap / p / 100) * 100
                            if shares > 0:
                                capital -= shares * p * (1 + COST_RATE)
                                holdings[symbol] = shares
                                trade_count += 1
                                print(f"    {date.date()} 试探建仓 {symbol} {shares}股 @ {p:.3f} (大跌: 沪指{market_return:.2%}, ETF{etf_return:.2%})")
                        else:
                            # 正常建仓
                            shares = int(per_etf_cap / p / 100) * 100
                            if shares > 0:
                                capital -= shares * p * (1 + COST_RATE)
                                holdings[symbol] = shares
                                trade_count += 1
                                print(f"    {date.date()} 建仓 {symbol} {shares}股 @ {p:.3f} (得分{score:.3f})")
                
                last_rebalance = date_ts
        
        # NAV
        nav = capital
        for s, sh in holdings.items():
            if s in all_prices and date in all_prices[s].index:
                nav += sh * all_prices[s]['close'].get(date, 0)
        nav_history.append({'date': date, 'nav': nav})
    
    # 结果
    nav_df = pd.DataFrame(nav_history)
    nav_df['returns'] = nav_df['nav'].pct_change()
    bench_nav['returns'] = bench_nav['nav'].pct_change()
    
    strategy_ret = nav_df['nav'].iloc[-1] / INITIAL_CAPITAL - 1
    bench_ret = bench_nav['nav'].iloc[-1] / INITIAL_CAPITAL - 1
    excess = strategy_ret - bench_ret
    
    cummax = nav_df['nav'].cummax()
    max_dd = ((nav_df['nav'] - cummax) / cummax).min()
    
    print("\n" + "=" * 70)
    print("                    回测结果")
    print("=" * 70)
    print(f"  策略收益: {strategy_ret:.2%}")
    print(f"  基准收益: {bench_ret:.2%}")
    print(f"  超额收益: {excess:.2%}")
    print(f"  最大回撤: {max_dd:.2%}")
    print(f"  交易次数: {trade_count}")
    
    # 评价
    print("\n" + "=" * 70)
    print("              独立评价Agent评估")
    print("=" * 70)
    
    agent = EvaluationAgent(verbose=True)
    common = nav_df['date'].isin(bench_nav['date'])
    s_ret = nav_df.loc[common, 'returns'].dropna()
    b_ret = bench_nav.set_index('date')['returns'].reindex(nav_df.loc[common, 'date']).dropna()
    
    report = agent.full_evaluation(s_ret, b_ret)
    agent.print_report(report)
    
    return {
        'strategy_ret': strategy_ret, 'bench_ret': bench_ret, 'excess': excess,
        'max_dd': max_dd, 'trade_count': trade_count, 'report': report
    }


if __name__ == "__main__":
    main()
