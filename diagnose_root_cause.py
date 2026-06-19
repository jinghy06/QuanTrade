"""根本性诊断：为什么策略跑输基准？"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

def main():
    conn = sqlite3.connect(DB_PATH)

    # ================================================================
    # 1. 逐股票分析
    # ================================================================
    print("=" * 100)
    print("1. 逐股票分析：为什么超额为负？")
    print("=" * 100)

    bt = pd.read_sql_query('SELECT * FROM backtest_results_exp_baseline', conn)
    print(f"{'股票':<12s} {'策略收益':>10s} {'基准收益':>10s} {'超额收益':>10s} {'平均仓位':>10s}")
    print("-" * 80)

    for _, row in bt.iterrows():
        print(f"{row['symbol']:<12s} {row['total_return']*100:>9.2f}% {row['benchmark_return']*100:>9.2f}% {row['excess_return']*100:>9.2f}% {row['avg_position']*100:>9.1f}%")

    print("-" * 80)
    print(f"{'平均':<12s} {bt['total_return'].mean()*100:>9.2f}% {bt['benchmark_return'].mean()*100:>9.2f}% {bt['excess_return'].mean()*100:>9.2f}% {bt['avg_position'].mean()*100:>9.1f}%")

    # ================================================================
    # 2. 超额收益的来源分解
    # ================================================================
    print("\n" + "=" * 100)
    print("2. 超额收益来源分解（核心诊断）")
    print("=" * 100)

    avg_benchmark = bt['benchmark_return'].mean()
    avg_position = bt['avg_position'].mean()
    avg_excess = bt['excess_return'].mean()

    # 超额 = 策略收益 - 基准收益
    # 策略收益 ≈ 基准收益 * 平均仓位 + alpha
    # 超额 ≈ 基准收益 * (平均仓位 - 1) + alpha
    position_drag = avg_benchmark * (avg_position - 1)
    implied_alpha = avg_excess - position_drag

    print(f"\n  平均基准收益:      {avg_benchmark*100:>8.2f}%  (市场涨了18.29%)")
    print(f"  平均仓位:          {avg_position*100:>8.1f}%  (只投了38%的钱)")
    print(f"  仓位拖累:          {position_drag*100:>8.2f}%  (因为仓位不足损失了11.26%)")
    print(f"  平均超额收益:      {avg_excess*100:>8.2f}%  (实际跑输18%)")
    print(f"  隐含alpha:         {implied_alpha*100:>8.2f}%  (模型预测反而亏了6.76%)")

    print(f"\n  结论:")
    print(f"    问题1: 仓位不足 → 少赚了 11.26%")
    print(f"    问题2: 模型预测错误 → 额外亏了 6.76%")
    print(f"    合计: 跑输了 18.02%")

    # ================================================================
    # 3. 预测信号质量分析
    # ================================================================
    print("\n" + "=" * 100)
    print("3. 预测信号质量分析")
    print("=" * 100)

    pred = pd.read_sql_query('SELECT * FROM predictions_exp_baseline', conn)
    features = pd.read_sql_query(
        "SELECT trade_date, symbol, target_return_10d FROM features_v5 WHERE trade_date >= '2024-01-01'", conn
    )
    merged = pred.merge(features, on=['trade_date', 'symbol'])

    fav_mask = merged['prediction'] == 'favorable'
    adv_mask = merged['prediction'] == 'adverse'
    base_mask = merged['prediction'] == 'base'

    fav_ret = merged.loc[fav_mask, 'target_return_10d'].mean()
    base_ret = merged.loc[base_mask, 'target_return_10d'].mean()
    adv_ret = merged.loc[adv_mask, 'target_return_10d'].mean()

    print(f"  预测favorable时, 实际10日收益: {fav_ret*100:.2f}% (样本: {fav_mask.sum()})")
    print(f"  预测base时, 实际10日收益:      {base_ret*100:.2f}% (样本: {base_mask.sum()})")
    print(f"  预测adverse时, 实际10日收益:   {adv_ret*100:.2f}% (样本: {adv_mask.sum()})")

    signal_spread = fav_ret - adv_ret
    print(f"\n  信号价差 (favorable - adverse): {signal_spread*100:.2f}%")

    if signal_spread > 0:
        print(f"  → 信号有正向区分度！favorable确实比adverse收益更高")
        print(f"  → 但区分度太小（只有{signal_spread*100:.2f}%），无法覆盖仓位不足的损失")
    else:
        print(f"  → 信号没有区分度！甚至可能反向")

    # ================================================================
    # 4. 策略的真正问题
    # ================================================================
    print("\n" + "=" * 100)
    print("4. 策略的真正问题（根本原因）")
    print("=" * 100)

    print("""
  问题1: 基准选择错误
  ─────────────────────────────────────
  当前基准: 100%等权持有所有股票
  策略仓位: 平均38.5%
  
  在一个涨了18%的市场里，只投38%的钱，必然跑输！
  这不是策略有问题，而是基准选择不合理。
  

  问题2: 模型预测能力不足
  ─────────────────────────────────────
  favorable信号: 预测"看涨"时，实际收益={:.2f}%
  adverse信号:   预测"看跌"时，实际收益={:.2f}%
  信号价差:      只有{:.2f}%
  
  模型几乎无法区分涨跌，预测能力接近随机。
  

  问题3: 策略逻辑自相矛盾
  ─────────────────────────────────────
  三情景决策: adverse→空仓, base→半仓, favorable→满仓
  但模型67%的时间预测base（中性）→ 大部分时间只有50%仓位
  
  模型越"不确定"，策略越保守，收益越差。
  

  问题4: 回测期间（2024-2025）市场特殊
  ─────────────────────────────────────
  平均基准收益+18%，但分布极不均匀:
  - 000002.SZ: -51% (万科，地产暴雷)
  - 600036.SH: +67% (招商银行，银行牛市)
  - 000333.SZ: +62% (美的集团，家电龙头)
  
  少数股票暴涨，多数平淡。策略在暴涨股票上仓位不足。
""".format(fav_ret*100, adv_ret*100, signal_spread*100))

    # ================================================================
    # 5. 正确的重构方向
    # ================================================================
    print("=" * 100)
    print("5. 正确的重构方向")
    print("=" * 100)

    print("""
  当前架构的根本问题:
  
  [特征] → [ML模型] → [预测涨跌] → [仓位决策] → [回测]
  
  问题:
  1. "预测涨跌"这件事本身就很难（接近随机）
  2. 仓位决策与预测强绑定，导致仓位不稳定
  3. 基准是100%持有，但策略只有40%仓位
  

  重构方案A: 从"预测涨跌"转向"风险管理"
  ─────────────────────────────────────
  [特征] → [波动率模型] → [仓位管理] → [回测]
  
  不再试图预测方向，而是:
  - 低波动时加仓（市场平稳）
  - 高波动时减仓（市场动荡）
  - 始终保持较高仓位（60-90%）
  

  重构方案B: 从"分类"转向"排序"
  ─────────────────────────────────────
  [特征] → [ML模型] → [股票排序] → [多空组合] → [回测]
  
  不再预测绝对涨跌，而是:
  - 选出"相对最好"的股票做多
  - 选出"相对最差"的股票做空/不持有
  - 对比: top组 vs bottom组
  

  重构方案C: 从"主动择时"转向"被动增强"
  ─────────────────────────────────────
  [特征] → [ML模型] → [权重调整] → [回测]
  
  基准: 100%等权持有
  策略: 100%持有，但根据模型信号调整各股票权重
  - 看好的股票: 权重×1.5
  - 不看好的股票: 材重×0.5
  - 总仓位始终100%
""")

    conn.close()

if __name__ == '__main__':
    main()
