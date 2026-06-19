"""
QuanTrade 2.0.2d - 半年调仓+持有期限版
=========================================
核心改动：
1. 半年调仓（进一步降低交易频率，减少成本侵蚀）
2. 持有最低期限3个月（避免底部卖出错失反弹）
3. 选前10只动量最强ETF，满仓持有
4. 事件驱动：情绪高点减仓30-50%、大跌加仓
5. 基准=24只ETF等权买入持有
6. 三因子融入：情绪/政治/政策作为仓位调整
7. 情绪泡沫识别：区分泡沫ETF和业绩ETF
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

# 用户策略参数
HOLD_MIN_MONTHS = 3          # 持有最低期限3个月
MAX_POSITIONS = 10           # 最多持仓10只（实际买入7-8只，取决于市场状态）
STOCK_RATIO_BULL = 0.98      # 牛市股票仓位98%（几乎满仓）
STOCK_RATIO_NORMAL = 0.98    # 正常股票仓位98%（几乎满仓）
STOCK_RATIO_PANIC = 0.50   # 恐慌股票仓位50%
EMOTION_REDUCE_BUBBLE = 0.50  # 泡沫情绪ETF减仓50%
EMOTION_REDUCE_EMOTION = 0.30  # 情绪ETF减仓30%
EMOTION_REDUCE_FUND = 0.15   # 业绩ETF减仓15%
PROFIT_REDUCE = 0.25         # 达到目标收益减仓25%
PROFIT_TARGET = 0.25         # 目标收益25%


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
    动量得分计算
    
    双模式：
    - 正常市场：选长期动量最强的（趋势跟踪）
    - 底部市场（120日跌幅>15%）：选超跌最深的（反弹潜力）
    """
    if len(etf_df) < 120:
        return -999
    
    close = etf_df['close']
    
    mom_120 = close.iloc[-1] / close.iloc[-120] - 1
    mom_60 = close.iloc[-1] / close.iloc[-60] - 1
    
    # 底部判断：120日跌幅超过15%视为底部
    is_bottom = mom_120 < -0.15
    
    if is_bottom:
        # 底部模式：选超跌+开始反弹的ETF
        # 120日跌幅越大（越超跌）得分越高 — 这是反弹潜力
        score_oversold = np.clip(-mom_120 / 0.30, -1, 1)  # 跌幅>30%得满分
        
        # 60日跌幅越大（越超跌）得分越高
        score_60 = np.clip(-mom_60 / 0.20, -1, 1)  # 60日跌幅>20%得满分
        
        # 20日相对强弱：比市场跌得更多 = 反弹潜力更大
        score_relative = 0
        if benchmark_df is not None and len(benchmark_df) > 20:
            bench = benchmark_df['close'].reindex(etf_df.index, method='ffill')
            bench_mom_20 = bench.iloc[-1] / bench.iloc[-20] - 1
            etf_mom_20 = close.iloc[-1] / close.iloc[-20] - 1
            # 比市场多跌 = 得分更高（跌幅差越大，反弹潜力越大）
            score_relative = np.clip((bench_mom_20 - etf_mom_20) / 0.20, -1, 1)
        
        # 趋势：短期均线是否开始向上（确认反弹开始）
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1] if len(etf_df) >= 10 else ma5
        trend_score = 0.6 * (1 if ma5 > ma10 else 0) + 0.4 * (1 if close.iloc[-1] > ma5 else 0)
        
        return score_oversold * 0.40 + score_60 * 0.30 + score_relative * 0.20 + trend_score * 0.10
    else:
        # 正常模式：纯动量热点（趋势跟踪）
        score_120 = np.clip(mom_120 / 0.5, -1, 1)
        score_60 = np.clip(mom_60 / 0.3, -1, 1)
        
        score_relative = 0
        if benchmark_df is not None and len(benchmark_df) > 120:
            bench = benchmark_df['close'].reindex(etf_df.index, method='ffill')
            etf_mom_20 = close.iloc[-1] / close.iloc[-20] - 1
            bench_mom_20 = bench.iloc[-1] / bench.iloc[-20] - 1
            score_relative = np.clip((etf_mom_20 - bench_mom_20) / 0.2, -1, 1)
        
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
    
    vol_20 = returns.rolling(20).std().iloc[-1]
    vol_ma60 = returns.rolling(60).std().iloc[-1]
    vol_signal = -0.5 if vol_20 > vol_ma60 * 1.5 else (0.5 if vol_20 < vol_ma60 * 0.7 else 0)
    mom_5 = close.pct_change(5).iloc[-1]
    mom_60 = close.pct_change(60).iloc[-1]
    mom_signal = 0.5 if mom_5 > 0 and mom_60 > 0 else (-0.5 if mom_5 < 0 and mom_60 < 0 else 0)
    sentiment = np.clip((vol_signal + mom_signal) / 2, -1, 1)
    
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
        return {'is_bubble': False, 'bubble_score': 0, 'rsi': 50, 'return_60d': 0}
    
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
    print("    QuanTrade 2.0.2c - 季度调仓+持有期限版")
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
    print("[策略回测] 季度调仓 + 热点赛道 + 三因子")
    print("=" * 70)
    
    capital = INITIAL_CAPITAL
    holdings = {}  # {symbol: {'shares': int, 'entry_date': pd.Timestamp}}
    nav_history = []
    trade_count = 0
    last_rebalance = None
    
    # 半年调仓日（6月、12月）
    rebalance_months = [6, 12]
    
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
        
        # 调仓判断：只在初始建仓、情绪高点或大跌时调仓
        trigger_reason = None
        should_rebalance = False
        if last_rebalance is None:
            should_rebalance = True
            trigger_reason = 'initial'
        elif factors['sentiment'] > 0.6 and holdings:
            should_rebalance = True
            trigger_reason = 'emotion_high'
            print(f"  {date.date()} 情绪高点触发减仓")
        elif market_return < -0.03:
            should_rebalance = True
            trigger_reason = 'dip'
            print(f"  {date.date()} 大跌触发加仓")
        
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
            
            if not scores:
                last_rebalance = date_ts
                continue
            
            # 选前N只
            n_top = MAX_POSITIONS
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_etfs = sorted_scores[:n_top]
            
            # 1. 卖出/调仓逻辑（仅初始建仓和定期调仓）
            if trigger_reason == 'initial':
                # 初始建仓：卖出所有不在前N的
                for s in list(holdings.keys()):
                    if s not in [x[0] for x in top_etfs]:
                        p = all_prices[s]['close'].get(date, 0)
                        if p > 0:
                            capital += holdings[s]['shares'] * p * (1 - COST_RATE)
                            trade_count += 1
                            del holdings[s]
                            print(f"    {date.date()} 卖出 {s} (不在前{n_top})")
            
            # 2. 减仓：情绪高点/泡沫/目标收益（仅情绪高点触发）
            if trigger_reason == 'emotion_high':
                for s in list(holdings.keys()):
                    p = all_prices[s]['close'].get(date, 0)
                    if p <= 0: continue
                    
                    bubble = detect_bubble(all_prices[s].loc[:date])
                    
                    reduce = 0.0
                    # 情绪高点减仓
                    if factors['sentiment'] > 0.6:
                        if s in EMOTION_ETFS and bubble['is_bubble']:
                            reduce = EMOTION_REDUCE_BUBBLE
                            print(f"    [泡沫预警] {s}")
                        elif s in EMOTION_ETFS:
                            reduce = EMOTION_REDUCE_EMOTION
                        else:
                            reduce = EMOTION_REDUCE_FUND
                    
                    # 目标收益减仓（涨25%+且情绪>0.5）
                    entry_price = holdings[s].get('entry_price', p)
                    profit_pct = (p - entry_price) / entry_price
                    if profit_pct > PROFIT_TARGET and factors['sentiment'] > 0.5:
                        reduce = max(reduce, PROFIT_REDUCE)
                    
                    if reduce > 0:
                        sell_shares = int(holdings[s]['shares'] * reduce / 100) * 100
                        if sell_shares > 0 and sell_shares < holdings[s]['shares']:
                            capital += sell_shares * p * (1 - COST_RATE)
                            holdings[s]['shares'] -= sell_shares
                            trade_count += 1
                            print(f"    {date.date()} 减仓 {s} {reduce:.0%} (RSI{bubble['rsi']:.0f}, 60日{bubble['return_60d']:.1%}, 盈亏{profit_pct:.1%})")
            
            # 3. 股票仓位
            stock_ratio = STOCK_RATIO_BULL if factors['sentiment'] > 0.3 else STOCK_RATIO_NORMAL
            if factors['geopolitical'] < -0.5:
                stock_ratio = max(STOCK_RATIO_PANIC, stock_ratio - 0.30)
            
            # 4. 买入/建仓
            stock_value = sum(holdings[s]['shares'] * all_prices[s]['close'].get(date, 0) for s in holdings if s in all_prices and date in all_prices[s].index)
            total_value = capital + stock_value
            target_stock_value = total_value * stock_ratio
            
            # 大跌触发：加仓跌幅最大的现有持仓（不买入新ETF）
            if trigger_reason == 'dip':
                # 计算现有持仓的跌幅，优先加仓跌最多的
                holding_drops = []
                for s in holdings:
                    p = all_prices[s]['close'].get(date, 0)
                    if p <= 0: continue
                    entry_price = holdings[s].get('entry_price', p)
                    drop_pct = (p - entry_price) / entry_price
                    holding_drops.append((s, drop_pct, p))
                
                # 按跌幅排序（跌最多的优先加仓）
                holding_drops.sort(key=lambda x: x[1])
                
                # 加仓目标：将股票仓位提升到目标
                for s, drop_pct, p in holding_drops:
                    if target_stock_value <= stock_value:
                        break
                    gap = target_stock_value - stock_value
                    buy_shares = int(gap * 0.3 / p / 100) * 100  # 分3批，每次加30%缺口
                    if buy_shares >= 100 and capital >= buy_shares * p * (1 + COST_RATE):
                        capital -= buy_shares * p * (1 + COST_RATE)
                        holdings[s]['shares'] += buy_shares
                        stock_value += buy_shares * p
                        trade_count += 1
                        print(f"    {date.date()} 加仓 {s} {buy_shares}股 @ {p:.3f} (跌幅{drop_pct:.1%})")
            else:
                # 正常买入/建仓：按目标权重分配
                per_etf_target = target_stock_value / len(top_etfs)
                
                for symbol, score in top_etfs:
                    p = all_prices[symbol]['close'].get(date, 0)
                    if p <= 0: continue
                    
                    current_shares = holdings.get(symbol, {}).get('shares', 0)
                    current_value = current_shares * p
                    
                    if current_value < per_etf_target:
                        gap = per_etf_target - current_value
                        buy_shares = int(gap / p / 100) * 100
                        if buy_shares >= 100 and capital >= buy_shares * p * (1 + COST_RATE):
                            capital -= buy_shares * p * (1 + COST_RATE)
                            if symbol not in holdings:
                                holdings[symbol] = {'shares': 0, 'entry_date': date, 'entry_price': p}
                            holdings[symbol]['shares'] += buy_shares
                            trade_count += 1
                            print(f"    {date.date()} 买入 {symbol} {buy_shares}股 @ {p:.3f} (得分{score:.3f})")
                
                if current_value < per_etf_target * 0.8:
                    # 需要买入
                    need_value = per_etf_target - current_value
                    # 限制单次买入不超过总资金的15%（分批建仓）
                    max_buy = total_value * 0.15
                    buy_value = min(need_value, max_buy, capital * 0.95)
                    
                    shares = int(buy_value / p / 100) * 100
                    if shares > 0:
                        capital -= shares * p * (1 + COST_RATE)
                        if symbol in holdings:
                            # 更新平均成本
                            old_shares = holdings[symbol]['shares']
                            old_cost = holdings[symbol].get('entry_price', p) * old_shares
                            new_cost = (old_cost + shares * p) / (old_shares + shares)
                            holdings[symbol]['shares'] += shares
                            holdings[symbol]['entry_price'] = new_cost
                            action = "加仓"
                        else:
                            holdings[symbol] = {'shares': shares, 'entry_date': date_ts, 'entry_price': p}
                            action = "建仓"
                        
                        trade_count += 1
                        
                        # 判断类型
                        hist = all_prices[symbol].loc[:date]
                        etf_return = hist['close'].pct_change().iloc[-1] if len(hist) > 1 else 0
                        if market_return < -0.03 and etf_return < -0.05:
                            print(f"    {date.date()} 抄底{action} {symbol} {shares}股 @ {p:.3f} (沪指{market_return:.2%}, ETF{etf_return:.2%})")
                        else:
                            print(f"    {date.date()} {action} {symbol} {shares}股 @ {p:.3f} (得分{score:.3f})")
                
                elif current_value > per_etf_target * 1.5:
                    # 超出目标太多，再平衡卖出
                    excess = current_value - per_etf_target
                    sell_shares = int(excess / p / 100) * 100
                    if sell_shares > 0 and sell_shares < current_shares:
                        capital += sell_shares * p * (1 - COST_RATE)
                        holdings[symbol]['shares'] -= sell_shares
                        trade_count += 1
                        print(f"    {date.date()} 再平衡卖出 {symbol} {sell_shares}股 @ {p:.3f}")
            
            # 打印当前持仓
            if holdings:
                total_portfolio = capital + sum(holdings[s]['shares'] * all_prices[s]['close'].get(date, 0) for s in holdings)
                stock_pct = sum(holdings[s]['shares'] * all_prices[s]['close'].get(date, 0) for s in holdings) / total_portfolio * 100 if total_portfolio > 0 else 0
                held = ', '.join([f"{s}({info['shares']}股,{(date-info['entry_date']).days}天)" for s, info in holdings.items()])
                print(f"  {date.date()} 持仓: 股票{stock_pct:.0f}% 现金{capital/total_portfolio*100:.0f}% | {held}")
            
            last_rebalance = date_ts
        
        # NAV
        nav = capital
        for s, info in holdings.items():
            if s in all_prices and date in all_prices[s].index:
                nav += info['shares'] * all_prices[s]['close'].get(date, 0)
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
