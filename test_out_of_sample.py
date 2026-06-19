"""
样本外测试 - 用不在训练集的ETF验证策略
测试ETF：159550（算力ETF，237条数据，不在核心8只中）
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

from sentiment_engine import MultiFactorEngine
from gold_hedge import GoldHedge
from evaluator_agent import EvaluationAgent

DB_PATH = "QuanTrade/quant_system/data/quant.db"
COST_RATE = 0.0036


def main():
    print("=" * 70)
    print("    样本外测试 - 159550（算力ETF）")
    print("=" * 70)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 加载测试ETF（不在训练集中）
    test_df = pd.read_sql("SELECT * FROM etf_daily_prices WHERE symbol='159550' ORDER BY trade_date", conn)
    benchmark_df = pd.read_sql("SELECT * FROM etf_daily_prices WHERE symbol='510300' ORDER BY trade_date", conn)
    gold_df = pd.read_sql("SELECT * FROM gold_daily_prices ORDER BY date", conn)
    conn.close()
    
    # 转换格式
    test_df['trade_date'] = pd.to_datetime(test_df['trade_date'])
    test_df = test_df.set_index('trade_date')
    
    benchmark_df['trade_date'] = pd.to_datetime(benchmark_df['trade_date'])
    benchmark_df = benchmark_df.set_index('trade_date')
    
    gold_df['date'] = pd.to_datetime(gold_df['date'])
    gold_df = gold_df.set_index('date')
    
    print(f"\n测试ETF: 159550（算力ETF）")
    print(f"数据量: {len(test_df)} 条")
    print(f"时间范围: {test_df.index[0].date()} ~ {test_df.index[-1].date()}")
    
    # ============================================================
    # 方法1: 用训练好的模型直接预测
    # ============================================================
    print("\n" + "=" * 70)
    print("[方法1] 用训练好的模型直接预测")
    print("=" * 70)
    
    # 加载训练好的模型（用515070的模型作为代表）
    conn = sqlite3.connect(DB_PATH)
    train_df = pd.read_sql("SELECT * FROM etf_daily_prices WHERE symbol='515070' ORDER BY trade_date", conn)
    conn.close()
    
    train_df['trade_date'] = pd.to_datetime(train_df['trade_date'])
    train_df = train_df.set_index('trade_date')
    
    # 训练模型
    features = v201.calculate_features(train_df)
    labels = v201.create_labels(train_df['close'])
    model, acc = v201.train_ml_model(features, labels, 'lightgbm')
    print(f"训练模型（515070）准确率: {acc:.2%}")
    
    # 用训练好的模型预测测试ETF
    test_features = v201.calculate_features(test_df)
    
    # 回测
    capital = 50000
    holdings = 0
    nav_history = []
    trade_count = 0
    
    for i in range(60, len(test_df)):
        date = test_df.index[i]
        
        # ML预测
        X = test_features.iloc[[i]].dropna()
        if len(X) > 0:
            ml_prob = model.predict(X)[0]
        else:
            ml_prob = 0.5
        
        # 热点赛道得分
        bench_hist = benchmark_df.loc[:date]
        sector_score = v201.calculate_sector_score(test_df.loc[:date], bench_hist)
        
        # 多因子
        multi_factor = MultiFactorEngine()
        factors = multi_factor.calculate_all(test_df.loc[:date], gold_df=gold_df)
        factor_score = factors['combined'].iloc[-1] if len(factors) > 0 else 0
        
        # 综合得分
        final_score = ml_prob * 0.4 + sector_score * 0.3 + factor_score * 0.3
        
        # 交易逻辑
        price = test_df['close'].iloc[i]
        
        if final_score > 0.3 and holdings == 0:
            # 买入
            shares = int(capital / price / 100) * 100
            if shares > 0:
                capital -= shares * price * (1 + COST_RATE)
                holdings = shares
                trade_count += 1
        
        elif final_score < 0.2 and holdings > 0:
            # 卖出
            capital += holdings * price * (1 - COST_RATE)
            holdings = 0
            trade_count += 1
        
        # 计算NAV
        nav = capital + holdings * price
        nav_history.append({'date': date, 'nav': nav})
    
    # 基准
    bench_start = benchmark_df['close'].get(test_df.index[60], 1)
    bench_nav = []
    for d in test_df.index[60:]:
        price = benchmark_df['close'].get(d, bench_nav[-1]['nav'] / 50000 * bench_start if bench_nav else bench_start)
        bench_nav.append({'date': d, 'nav': 50000 * price / bench_start})
    
    nav_df = pd.DataFrame(nav_history)
    nav_df['returns'] = nav_df['nav'].pct_change()
    bench_df_result = pd.DataFrame(bench_nav)
    bench_df_result['returns'] = bench_df_result['nav'].pct_change()
    
    strategy_return = nav_df['nav'].iloc[-1] / 50000 - 1
    benchmark_return = bench_df_result['nav'].iloc[-1] / 50000 - 1
    
    print(f"\n方法1结果:")
    print(f"  策略收益: {strategy_return:.2%}")
    print(f"  基准收益: {benchmark_return:.2%}")
    print(f"  超额收益: {strategy_return - benchmark_return:.2%}")
    print(f"  交易次数: {trade_count}")
    
    # ============================================================
    # 方法2: 纯动量策略（对比）
    # ============================================================
    print("\n" + "=" * 70)
    print("[方法2] 纯动量策略（对比）")
    print("=" * 70)
    
    capital = 50000
    holdings = 0
    nav_history2 = []
    trade_count2 = 0
    last_trade_date = None
    
    for i in range(60, len(test_df)):
        date = test_df.index[i]
        date_ts = pd.Timestamp(date)
        
        # 季度调仓
        should_rebalance = False
        if last_trade_date is None:
            should_rebalance = True
        else:
            months_since = (date_ts.year - last_trade_date.year) * 12 + (date_ts.month - last_trade_date.month)
            if months_since >= 3:
                should_rebalance = True
        
        if should_rebalance:
            # 计算动量
            close = test_df['close'].iloc[:i+1]
            momentum = close.iloc[-1] / close.iloc[-60] - 1 if len(close) > 60 else 0
            ma60 = close.rolling(60).mean().iloc[-1]
            in_uptrend = close.iloc[-1] > ma60
            
            price = test_df['close'].iloc[i]
            
            if in_uptrend and momentum > 0:
                if holdings == 0:
                    shares = int(capital / price / 100) * 100
                    if shares > 0:
                        capital -= shares * price * (1 + COST_RATE)
                        holdings = shares
                        trade_count2 += 1
            else:
                if holdings > 0:
                    capital += holdings * price * (1 - COST_RATE)
                    holdings = 0
                    trade_count2 += 1
            
            last_trade_date = date_ts
        
        nav = capital + holdings * test_df['close'].iloc[i]
        nav_history2.append({'date': date, 'nav': nav})
    
    nav_df2 = pd.DataFrame(nav_history2)
    nav_df2['returns'] = nav_df2['nav'].pct_change()
    strategy_return2 = nav_df2['nav'].iloc[-1] / 50000 - 1
    
    print(f"\n方法2结果:")
    print(f"  策略收益: {strategy_return2:.2%}")
    print(f"  基准收益: {benchmark_return:.2%}")
    print(f"  超额收益: {strategy_return2 - benchmark_return:.2%}")
    print(f"  交易次数: {trade_count2}")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 70)
    print("                    总结")
    print("=" * 70)
    print(f"\n测试ETF: 159550（算力ETF）- 不在训练集中")
    print(f"数据量: {len(test_df)} 条")
    print(f"\n方法1（5层ML）: {strategy_return:.2%} | 交易{trade_count}次")
    print(f"方法2（纯动量）: {strategy_return2:.2%} | 交易{trade_count2}次")
    print(f"基准（沪深300）: {benchmark_return:.2%}")
    
    # 评价Agent
    print("\n" + "=" * 70)
    print("              独立评价Agent评估")
    print("=" * 70)
    
    agent = EvaluationAgent(verbose=False)
    report = agent.full_evaluation(
        strategy_returns=nav_df['returns'].dropna(),
        benchmark_returns=bench_df_result['returns'].dropna()
    )
    
    print(f"方法1（5层ML）: {report.overall_score}/100 ({report.grade})")
    
    report2 = agent.full_evaluation(
        strategy_returns=nav_df2['returns'].dropna(),
        benchmark_returns=bench_df_result['returns'].dropna()
    )
    
    print(f"方法2（纯动量）: {report2.overall_score}/100 ({report2.grade})")
    
    return report, report2


if __name__ == "__main__":
    main()
