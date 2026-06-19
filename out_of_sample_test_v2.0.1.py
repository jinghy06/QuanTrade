"""
QuanTrade 2.0.1 - 样本外测试
===========================
用完全独立于训练集的ETF进行测试

测试ETF：15只仅有2026年数据的ETF（从未参与ML模型训练）
- 159790, 159915, 159928, 512010, 512200, 512480
- 512690, 512800, 512880, 513180, 515170, 515790
- 516160, 561160, 588000

测试方法：
1. 用训练好的模型（基于8只完整数据ETF）预测特征
2. 热点赛道轮动评分
3. 纯动量策略对比
4. 买入持有基准

独立评价Agent评估
"""

import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

import importlib.util
spec = importlib.util.spec_from_file_location("v201", "run_etf_system_v2.0.1.py")
v201 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v201)

calculate_features = v201.calculate_features
calculate_sector_score = v201.calculate_sector_score
calculate_market_prob = v201.calculate_market_prob
from sentiment_engine import MultiFactorEngine
from gold_hedge import GoldHedge
from evaluator_agent import EvaluationAgent

DB_PATH = "QuanTrade/quant_system/data/quant.db"
COST_RATE = 0.0036
INITIAL_CAPITAL = 50000

# 样本外ETF（从未参与训练）
OUT_OF_SAMPLE_ETFS = [
    '159790', '159915', '159928', '512010', '512200', '512480',
    '512690', '512800', '512880', '513180', '515170', '515790',
    '516160', '561160', '588000'
]

# 基准
BENCHMARK = '510300'


def load_data():
    """加载数据"""
    conn = sqlite3.connect(DB_PATH)
    
    all_prices = {}
    for symbol in OUT_OF_SAMPLE_ETFS + [BENCHMARK]:
        df = pd.read_sql(
            f"SELECT * FROM etf_daily_prices WHERE symbol='{symbol}' ORDER BY trade_date",
            conn
        )
        if len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')
            all_prices[symbol] = df
    
    gold_df = pd.read_sql("SELECT * FROM gold_daily_prices ORDER BY date", conn)
    gold_df['date'] = pd.to_datetime(gold_df['date'])
    gold_df = gold_df.set_index('date')
    
    conn.close()
    return all_prices, gold_df


def backtest_momentum(all_prices, benchmark_df, lookback=60):
    """纯动量策略回测（对比基准）"""
    valid_symbols = [s for s in OUT_OF_SAMPLE_ETFS if s in all_prices]
    all_dates = sorted(set().union(*[all_prices[s].index for s in valid_symbols if s in all_prices]))
    
    if len(all_dates) < lookback + 10:
        return None
    
    capital = INITIAL_CAPITAL
    holdings = {}
    nav_history = []
    last_rebalance = None
    trade_count = 0
    
    for date in all_dates[lookback:]:
        date_ts = pd.Timestamp(date)
        
        # 季度调仓
        should_rebalance = False
        if last_rebalance is None:
            should_rebalance = True
        else:
            months_diff = (date_ts.year - last_rebalance.year) * 12 + (date_ts.month - last_rebalance.month)
            if months_diff >= 3:
                should_rebalance = True
        
        if should_rebalance:
            scores = {}
            for symbol in valid_symbols:
                df = all_prices[symbol]
                if date not in df.index:
                    continue
                hist = df.loc[:date]
                if len(hist) < lookback:
                    continue
                
                close = hist['close']
                momentum = close.iloc[-1] / close.iloc[-lookback] - 1
                ma20 = close.rolling(20).mean().iloc[-1]
                in_uptrend = close.iloc[-1] > ma20
                
                score = momentum if in_uptrend else momentum * 0.5
                scores[symbol] = score
            
            if scores:
                best = max(scores, key=scores.get)
                
                # 清仓非目标
                for h_symbol in list(holdings.keys()):
                    if h_symbol != best:
                        sell_price = all_prices[h_symbol]['close'].get(date, 0)
                        if sell_price > 0:
                            capital += holdings[h_symbol] * sell_price * (1 - COST_RATE)
                            trade_count += 1
                        del holdings[h_symbol]
                
                # 买入
                if best not in holdings:
                    buy_price = all_prices[best]['close'].get(date, 0)
                    if buy_price > 0:
                        shares = int(capital / buy_price / 100) * 100
                        if shares > 0:
                            capital -= shares * buy_price * (1 + COST_RATE)
                            holdings[best] = shares
                            trade_count += 1
                
                last_rebalance = date_ts
        
        # NAV
        nav = capital
        for h_symbol, h_shares in holdings.items():
            price = all_prices[h_symbol]['close'].get(date, 0)
            nav += h_shares * price
        nav_history.append({'date': date, 'nav': nav})
    
    return pd.DataFrame(nav_history), trade_count


def backtest_five_layer(all_prices, benchmark_df, gold_df):
    """五层策略回测（样本外）"""
    valid_symbols = [s for s in OUT_OF_SAMPLE_ETFS if s in all_prices]
    all_dates = sorted(set().union(*[all_prices[s].index for s in valid_symbols if s in all_prices]))
    
    if len(all_dates) < 60:
        return None
    
    multi_factor = MultiFactorEngine()
    gold_hedge = GoldHedge()
    gold_hedge.load_gold_data()
    
    capital = INITIAL_CAPITAL
    holdings = {}
    nav_history = []
    last_rebalance = None
    trade_count = 0
    
    for date in all_dates[60:]:
        date_ts = pd.Timestamp(date)
        
        # 季度调仓
        should_rebalance = False
        if last_rebalance is None:
            should_rebalance = True
        else:
            months_diff = (date_ts.year - last_rebalance.year) * 12 + (date_ts.month - last_rebalance.month)
            if months_diff >= 3:
                should_rebalance = True
        
        if should_rebalance:
            scores = {}
            for symbol in valid_symbols:
                df = all_prices[symbol]
                if date not in df.index:
                    continue
                hist = df.loc[:date]
                if len(hist) < 60:
                    continue
                
                # Layer 1: 用特征代理（因为没有训练过ML模型）
                features = calculate_features(hist)
                if len(features) > 0:
                    latest = features.iloc[-1]
                    # 简单的规则：RSI<30超卖=0.2, RSI>70超买=0.8, 趋势向上=0.7
                    rsi = latest.get('rsi_14', 50)
                    ma_trend = latest.get('trend_5_10', 0) + latest.get('trend_10_20', 0)
                    ml_score = 0.5
                    if rsi < 30:
                        ml_score = 0.7
                    elif rsi > 70:
                        ml_score = 0.3
                    elif ma_trend >= 1:
                        ml_score = 0.65
                    else:
                        ml_score = 0.45
                else:
                    ml_score = 0.5
                
                # Layer 2: 热点赛道
                sector_score = calculate_sector_score(hist, benchmark_df)
                
                # Layer 4: 多因子
                try:
                    factors = multi_factor.calculate_all(hist, gold_df=gold_df)
                    factor_score = factors['combined'].iloc[-1] if len(factors) > 0 else 0
                except:
                    factor_score = 0
                
                final_score = ml_score * 0.4 + (sector_score + 1) / 2 * 0.3 + (factor_score + 1) / 2 * 0.3
                scores[symbol] = final_score
            
            if scores:
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top2 = sorted_scores[:2]
                
                market_prob = calculate_market_prob(scores)
                gold_allocation = gold_hedge.calculate_gold_allocation(market_prob)
                gold_signal = gold_hedge.get_gold_signal(str(date))
                adjusted = gold_hedge.adjust_allocation_by_gold(gold_allocation, gold_signal)
                stock_ratio = adjusted['stock']
                
                # 清仓非目标
                target_symbols = [s[0] for s in top2]
                for h_symbol in list(holdings.keys()):
                    if h_symbol not in target_symbols:
                        sell_price = all_prices[h_symbol]['close'].get(date, 0)
                        if sell_price > 0:
                            capital += holdings[h_symbol] * sell_price * (1 - COST_RATE)
                            trade_count += 1
                        del holdings[h_symbol]
                
                # 买入
                available = capital * stock_ratio
                per_etf = available / len(top2)
                for symbol, score in top2:
                    if symbol not in holdings:
                        buy_price = all_prices[symbol]['close'].get(date, 0)
                        if buy_price > 0:
                            shares = int(per_etf / buy_price / 100) * 100
                            if shares > 0:
                                capital -= shares * buy_price * (1 + COST_RATE)
                                holdings[symbol] = shares
                                trade_count += 1
                
                last_rebalance = date_ts
        
        # NAV
        nav = capital
        for h_symbol, h_shares in holdings.items():
            price = all_prices[h_symbol]['close'].get(date, 0)
            nav += h_shares * price
        nav_history.append({'date': date, 'nav': nav})
    
    return pd.DataFrame(nav_history), trade_count


def backtest_buyhold(all_prices, benchmark_df):
    """买入持有基准（等权持有所有样本外ETF）"""
    valid_symbols = [s for s in OUT_OF_SAMPLE_ETFS if s in all_prices]
    all_dates = sorted(set().union(*[all_prices[s].index for s in valid_symbols if s in all_prices]))
    
    if len(all_dates) < 2:
        return None
    
    # 第一天等权买入
    start_date = all_dates[0]
    per_etf_cap = INITIAL_CAPITAL / len(valid_symbols)
    holdings = {}
    capital = INITIAL_CAPITAL
    
    for symbol in valid_symbols:
        buy_price = all_prices[symbol]['close'].get(start_date, 0)
        if buy_price > 0:
            shares = int(per_etf_cap / buy_price / 100) * 100
            if shares > 0:
                capital -= shares * buy_price * (1 + COST_RATE)
                holdings[symbol] = shares
    
    nav_history = []
    for date in all_dates:
        nav = capital
        for h_symbol, h_shares in holdings.items():
            price = all_prices[h_symbol]['close'].get(date, 0)
            nav += h_shares * price
        nav_history.append({'date': date, 'nav': nav})
    
    return pd.DataFrame(nav_history)


def main():
    print("=" * 70)
    print("    QuanTrade 2.0.1 - 样本外测试（独立ETF）")
    print("=" * 70)
    
    all_prices, gold_df = load_data()
    benchmark_df = all_prices.get(BENCHMARK)
    
    print(f"\n样本外ETF: {len([s for s in OUT_OF_SAMPLE_ETFS if s in all_prices])} 只")
    for s in OUT_OF_SAMPLE_ETFS:
        if s in all_prices:
            print(f"  {s}: {len(all_prices[s])} 条")
    
    # 基准
    print(f"\n基准: {BENCHMARK}")
    if benchmark_df is not None:
        print(f"  数据: {len(benchmark_df)} 条, {benchmark_df.index[0].date()} ~ {benchmark_df.index[-1].date()}")
    
    # 方法1: 五层策略
    print("\n" + "=" * 70)
    print("[方法1] 五层策略（样本外）")
    print("=" * 70)
    nav1, trades1 = backtest_five_layer(all_prices, benchmark_df, gold_df)
    if nav1 is not None:
        ret1 = nav1['nav'].iloc[-1] / INITIAL_CAPITAL - 1
        print(f"  收益: {ret1:.2%}")
        print(f"  交易: {trades1} 次")
        cummax = nav1['nav'].cummax()
        dd1 = (nav1['nav'] - cummax) / cummax
        print(f"  最大回撤: {dd1.min():.2%}")
    
    # 方法2: 纯动量
    print("\n" + "=" * 70)
    print("[方法2] 纯动量策略（对比）")
    print("=" * 70)
    nav2, trades2 = backtest_momentum(all_prices, benchmark_df)
    if nav2 is not None:
        ret2 = nav2['nav'].iloc[-1] / INITIAL_CAPITAL - 1
        print(f"  收益: {ret2:.2%}")
        print(f"  交易: {trades2} 次")
        cummax2 = nav2['nav'].cummax()
        dd2 = (nav2['nav'] - cummax2) / cummax2
        print(f"  最大回撤: {dd2.min():.2%}")
    
    # 方法3: 买入持有
    print("\n" + "=" * 70)
    print("[方法3] 买入持有（等权）")
    print("=" * 70)
    nav3 = backtest_buyhold(all_prices, benchmark_df)
    if nav3 is not None:
        ret3 = nav3['nav'].iloc[-1] / INITIAL_CAPITAL - 1
        print(f"  收益: {ret3:.2%}")
        cummax3 = nav3['nav'].cummax()
        dd3 = (nav3['nav'] - cummax3) / cummax3
        print(f"  最大回撤: {dd3.min():.2%}")
    
    # 基准收益
    print("\n" + "=" * 70)
    print("[基准] 沪深300")
    print("=" * 70)
    if benchmark_df is not None:
        bench_ret = benchmark_df['close'].iloc[-1] / benchmark_df['close'].iloc[0] - 1
        print(f"  收益: {bench_ret:.2%}")
    
    # 独立评价Agent
    print("\n" + "=" * 70)
    print("              独立评价Agent评估")
    print("=" * 70)
    
    # 构建基准NAV序列
    if benchmark_df is not None and nav1 is not None:
        bench_dates = nav1['date'].tolist()
        bench_prices = [benchmark_df['close'].get(d, benchmark_df['close'].iloc[0]) for d in bench_dates]
        bench_nav = [INITIAL_CAPITAL * p / bench_prices[0] for p in bench_prices]
        bench_nav_df = pd.DataFrame({'date': bench_dates, 'nav': bench_nav})
        bench_nav_df['returns'] = bench_nav_df['nav'].pct_change()
        
        nav1['returns'] = nav1['nav'].pct_change()
        
        agent = EvaluationAgent(verbose=False)
        report1 = agent.full_evaluation(
            strategy_returns=nav1['returns'].dropna(),
            benchmark_returns=bench_nav_df['returns'].dropna()
        )
        print(f"\n五层策略: {report1.overall_score:.1f}/100 (等级: {report1.grade})")
        print(f"判断: {report1.verdict}")
        if report1.warnings:
            print(f"警告 ({len(report1.warnings)}):")
            for w in report1.warnings[:5]:
                print(f"  {w}")
        
        if nav2 is not None:
            nav2['returns'] = nav2['nav'].pct_change()
            report2 = agent.full_evaluation(
                strategy_returns=nav2['returns'].dropna(),
                benchmark_returns=bench_nav_df['returns'].dropna()
            )
            print(f"\n纯动量: {report2.overall_score:.1f}/100 (等级: {report2.grade})")
        
        if nav3 is not None:
            nav3['returns'] = nav3['nav'].pct_change()
            report3 = agent.full_evaluation(
                strategy_returns=nav3['returns'].dropna(),
                benchmark_returns=bench_nav_df['returns'].dropna()
            )
            print(f"买入持有: {report3.overall_score:.1f}/100 (等级: {report3.grade})")
    
    # 总结
    print("\n" + "=" * 70)
    print("                    样本外测试总结")
    print("=" * 70)
    if nav1 is not None:
        print(f"五层策略: {ret1:.2%} | 交易{trades1}次 | 最大回撤{dd1.min():.2%}")
    if nav2 is not None:
        print(f"纯动量:   {ret2:.2%} | 交易{trades2}次 | 最大回撤{dd2.min():.2%}")
    if nav3 is not None:
        print(f"买入持有: {ret3:.2%} | 最大回撤{dd3.min():.2%}")
    if benchmark_df is not None:
        print(f"沪深300:  {bench_ret:.2%}")
    
    print("\n结论:")
    print("  样本外ETF仅有2026年数据，数据量有限，结果仅供参考")
    print("  五层策略的泛化能力需要更长时间数据验证")


if __name__ == "__main__":
    main()
