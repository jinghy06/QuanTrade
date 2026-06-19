"""分析策略问题"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

def main():
    conn = sqlite3.connect(DB_PATH)

    # 1. 分析预测准确性
    print("=" * 70)
    print("1. 预测准确性分析")
    print("=" * 70)

    pred_df = pd.read_sql_query('SELECT * FROM predictions_exp_baseline', conn)
    features_df = pd.read_sql_query(
        "SELECT trade_date, symbol, target_return_10d FROM features_v5 WHERE trade_date >= '2024-01-01'",
        conn
    )

    # 合并预测和实际收益
    merged = pred_df.merge(features_df, on=['trade_date', 'symbol'], how='inner')

    # 计算预测准确性
    def check_prediction(row):
        if row['prediction'] == 'favorable' and row['target_return_10d'] > 0.05:
            return 'correct'
        elif row['prediction'] == 'adverse' and row['target_return_10d'] < -0.05:
            return 'correct'
        elif row['prediction'] == 'base' and -0.05 <= row['target_return_10d'] <= 0.05:
            return 'correct'
        else:
            return 'wrong'

    merged['prediction_correct'] = merged.apply(check_prediction, axis=1)
    accuracy = (merged['prediction_correct'] == 'correct').mean() * 100
    print(f"预测准确率: {accuracy:.2f}%")

    # 分析各类别的准确性
    for scenario in ['favorable', 'base', 'adverse']:
        mask = merged['prediction'] == scenario
        if mask.sum() > 0:
            scenario_acc = (merged.loc[mask, 'prediction_correct'] == 'correct').mean() * 100
            print(f"  {scenario} 准确率: {scenario_acc:.2f}% (样本数: {mask.sum()})")

    # 2. 分析预测分布
    print("\n" + "=" * 70)
    print("2. 预测分布分析")
    print("=" * 70)
    print(pred_df['prediction'].value_counts())

    # 3. 分析实际收益分布
    print("\n" + "=" * 70)
    print("3. 实际收益分布")
    print("=" * 70)
    r = features_df['target_return_10d']
    print(f"  mean: {r.mean()*100:.2f}%")
    print(f"  std: {r.std()*100:.2f}%")
    print(f"  favorable (>5%): {(r > 0.05).sum()} ({(r > 0.05).mean()*100:.1f}%)")
    print(f"  base (-5%~5%): {((r >= -0.05) & (r <= 0.05)).sum()} ({((r >= -0.05) & (r <= 0.05)).mean()*100:.1f}%)")
    print(f"  adverse (<-5%): {(r < -0.05).sum()} ({(r < -0.05).mean()*100:.1f}%)")

    # 4. 分析回测结果
    print("\n" + "=" * 70)
    print("4. 回测结果分析")
    print("=" * 70)
    bt_df = pd.read_sql_query('SELECT * FROM backtest_results_exp_baseline', conn)
    print(f"平均总收益: {bt_df['total_return'].mean()*100:.2f}%")
    print(f"平均超额收益: {bt_df['excess_return'].mean()*100:.2f}%")
    print(f"平均夏普比率: {bt_df['sharpe'].mean():.2f}")
    print(f"平均最大回撤: {bt_df['max_drawdown'].mean()*100:.2f}%")
    print(f"平均仓位: {bt_df['avg_position'].mean()*100:.1f}%")

    # 5. 分析市场环境
    print("\n" + "=" * 70)
    print("5. 市场环境分析")
    print("=" * 70)
    price_df = pd.read_sql_query(
        "SELECT trade_date, symbol, close FROM daily_prices_v5 WHERE trade_date >= '2024-01-01'",
        conn
    )
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    benchmark = price_df.groupby('trade_date')['close'].mean()
    benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
    daily_returns = benchmark.pct_change().dropna()
    annual_vol = daily_returns.std() * np.sqrt(252) * 100
    print(f"基准收益（等权持有）: {benchmark_return:.2f}%")
    print(f"年化波动率: {annual_vol:.2f}%")

    # 6. 分析策略仓位
    print("\n" + "=" * 70)
    print("6. 策略仓位分析")
    print("=" * 70)
    daily_df = pd.read_sql_query('SELECT * FROM backtest_daily_exp_baseline', conn)
    position_dist = daily_df['position'].value_counts().sort_index()
    print("仓位分布:")
    for pos, count in position_dist.items():
        print(f"  {pos:.2f}: {count} ({count/len(daily_df)*100:.1f}%)")

    conn.close()

if __name__ == '__main__':
    main()
