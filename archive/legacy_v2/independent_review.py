"""
QuanTrade 独立审查脚本
=======================
不依赖项目内置的 EvaluationAgent，以外部独立视角评估策略质量。
"""
import sys
import importlib.util
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

DB_PATH = Path('QuanTrade/quant_system/data/quant.db')
RESULTS_DIR = Path('review_results')
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def annual_return(daily_rets):
    return (1 + daily_rets.mean()) ** 252 - 1

def annual_vol(daily_rets):
    return daily_rets.std() * np.sqrt(252)

def sharpe(daily_rets, risk_free=0.0):
    v = annual_vol(daily_rets)
    if v == 0:
        return np.nan
    return (annual_return(daily_rets) - risk_free) / v

def sortino(daily_rets, target=0.0):
    downside = daily_rets[daily_rets < target]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    downside_std = downside.std() * np.sqrt(252)
    return (annual_return(daily_rets) - target) / downside_std

def max_drawdown(nav):
    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    return dd.min()

def calmar(daily_rets, nav):
    dd = max_drawdown(nav)
    if dd == 0:
        return np.nan
    return annual_return(daily_rets) / abs(dd)

def win_rate(daily_rets):
    return (daily_rets > 0).mean()

def profit_loss_ratio(daily_rets):
    wins = daily_rets[daily_rets > 0]
    losses = daily_rets[daily_rets < 0]
    if len(losses) == 0 or losses.mean() == 0:
        return np.nan
    return wins.mean() / abs(losses.mean())

def calc_alpha_beta(strategy_rets, benchmark_rets):
    aligned = pd.concat([strategy_rets, benchmark_rets], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan, np.nan
    x = aligned.iloc[:, 1].values
    y = aligned.iloc[:, 0].values
    slope, intercept, _, _, _ = stats.linregress(x, y)
    # 日 alpha 年化
    annual_alpha = intercept * 252
    return annual_alpha, slope

def information_ratio(strategy_rets, benchmark_rets):
    aligned = pd.concat([strategy_rets, benchmark_rets], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    if diff.std() == 0:
        return np.nan
    return diff.mean() / diff.std() * np.sqrt(252)

def t_test_significance(strategy_rets, benchmark_rets=None):
    """检验策略收益是否显著大于0（或显著大于基准）"""
    s = strategy_rets.dropna()
    if benchmark_rets is not None:
        aligned = pd.concat([s, benchmark_rets], axis=1).dropna()
        if len(aligned) < 30:
            return np.nan, np.nan
        diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        if diff.std() == 0:
            return np.nan, np.nan
        t, p = stats.ttest_1samp(diff, 0)
        return t, p / 2  # 单尾
    else:
        if s.std() == 0:
            return np.nan, np.nan
        t, p = stats.ttest_1samp(s, 0)
        return t, p / 2

def apply_costs(daily_rets, turnover_per_year, cost_per_trade):
    """按年换手率和单边成本扣除成本"""
    daily_cost = turnover_per_year * cost_per_trade / 252
    return daily_rets - daily_cost

def rolling_window_consistency(nav, benchmark_nav, window=252):
    """滚动窗口胜率：策略跑赢基准的比例"""
    aligned = pd.concat([nav, benchmark_nav], axis=1).dropna()
    aligned.columns = ['strategy', 'benchmark']
    wins = 0
    total = 0
    for i in range(window, len(aligned)):
        s_ret = aligned['strategy'].iloc[i] / aligned['strategy'].iloc[i - window] - 1
        b_ret = aligned['benchmark'].iloc[i] / aligned['benchmark'].iloc[i - window] - 1
        total += 1
        if s_ret > b_ret:
            wins += 1
    return wins / total if total > 0 else np.nan

def load_equal_weight_etf(start_date, end_date):
    """计算等权 ETF 组合（不含基准 510300）的每日收益"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT trade_date, symbol, close FROM etf_daily_prices WHERE trade_date BETWEEN ? AND ?",
        conn, params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    )
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed', dayfirst=False)
    df = df[df['symbol'] != '510300']
    pivot = df.pivot(index='trade_date', columns='symbol', values='close')
    rets = pivot.pct_change()
    # 等权，忽略缺失
    equal_weight = rets.mean(axis=1, skipna=True)
    return equal_weight

# ---------------------------------------------------------------------------
# 运行回测并获取 NAV
# ---------------------------------------------------------------------------
def run_backtest_module(module_path, func_name='main'):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = mod
    spec.loader.exec_module(mod)
    result = getattr(mod, func_name)()
    nav = result['nav_df'].set_index('date')
    bench = result['bench_nav'].set_index('date')
    return nav, bench, result

# ---------------------------------------------------------------------------
# 主审查流程
# ---------------------------------------------------------------------------
def review_version(name, module_path, estimated_turnover=2.0):
    print(f"\n{'='*80}")
    print(f"  独立审查: {name}")
    print(f"{'='*80}")

    nav, bench, raw = run_backtest_module(module_path)
    nav.index = pd.to_datetime(nav.index)
    bench.index = pd.to_datetime(bench.index)

    s_rets = nav['returns'].dropna()
    b_rets = bench['returns'].dropna()

    # 对齐
    aligned = pd.concat([s_rets, b_rets], axis=1).dropna()
    aligned.columns = ['strategy', 'benchmark']
    s_rets = aligned['strategy']
    b_rets = aligned['benchmark']

    # 等权 ETF 基准
    ew_rets = load_equal_weight_etf(nav.index.min(), nav.index.max())
    ew_rets = ew_rets.reindex(s_rets.index).dropna()

    # 基础指标
    metrics = {
        'start_date': s_rets.index.min().strftime('%Y-%m-%d'),
        'end_date': s_rets.index.max().strftime('%Y-%m-%d'),
        'trading_days': len(s_rets),
        'total_return': nav['nav'].iloc[-1] / nav['nav'].iloc[0] - 1,
        'benchmark_total_return': bench['nav'].iloc[-1] / bench['nav'].iloc[0] - 1,
        'equal_weight_total_return': (1 + ew_rets).prod() - 1 if len(ew_rets) > 0 else np.nan,
        'annual_return': annual_return(s_rets),
        'benchmark_annual_return': annual_return(b_rets),
        'equal_weight_annual_return': annual_return(ew_rets) if len(ew_rets) > 0 else np.nan,
        'annual_volatility': annual_vol(s_rets),
        'sharpe': sharpe(s_rets),
        'sortino': sortino(s_rets),
        'max_drawdown': max_drawdown(nav['nav']),
        'calmar': calmar(s_rets, nav['nav']),
        'win_rate': win_rate(s_rets),
        'profit_loss_ratio': profit_loss_ratio(s_rets),
        'skewness': s_rets.skew(),
        'kurtosis': s_rets.kurtosis(),
        'alpha_vs_300': calc_alpha_beta(s_rets, b_rets)[0],
        'beta_vs_300': calc_alpha_beta(s_rets, b_rets)[1],
        'info_ratio_vs_300': information_ratio(s_rets, b_rets),
        'info_ratio_vs_equal_weight': information_ratio(s_rets, ew_rets) if len(ew_rets) > 0 else np.nan,
        't_stat_vs_zero': t_test_significance(s_rets)[0],
        'p_value_vs_zero': t_test_significance(s_rets)[1],
        't_stat_vs_300': t_test_significance(s_rets, b_rets)[0],
        'p_value_vs_300': t_test_significance(s_rets, b_rets)[1],
        'rolling_1y_consistency_vs_300': rolling_window_consistency(nav['nav'], bench['nav'], 252),
        'rolling_1y_consistency_vs_ew': rolling_window_consistency(nav['nav'], (1 + ew_rets).cumprod(), 252) if len(ew_rets) > 0 else np.nan,
        'monthly_return_autocorr_1': s_rets.resample('ME').sum().autocorr(lag=1),
        'trade_count': raw['trade_count'],
    }

    # 成本敏感性（按估计年换手率）
    cost_sensitivity = {}
    for cost in [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]:
        net_rets = apply_costs(s_rets, estimated_turnover, cost)
        cost_sensitivity[f'cost_{cost*100:.2f}%'] = (1 + net_rets).prod() - 1
    metrics['cost_sensitivity'] = cost_sensitivity

    # 输出
    print(f"\n样本期: {metrics['start_date']} ~ {metrics['end_date']} ({metrics['trading_days']} 交易日)")
    print(f"策略总收益: {metrics['total_return']:.2%}")
    print(f"沪深300总收益: {metrics['benchmark_total_return']:.2%}")
    print(f"等权ETF总收益: {metrics['equal_weight_total_return']:.2%}")
    print(f"超额收益(vs 300): {metrics['total_return'] - metrics['benchmark_total_return']:.2%}")
    print(f"超额收益(vs 等权): {metrics['total_return'] - metrics['equal_weight_total_return']:.2%}")
    print(f"\n年化收益: {metrics['annual_return']:.2%}")
    print(f"年化波动: {metrics['annual_volatility']:.2%}")
    print(f"夏普比率: {metrics['sharpe']:.2f}")
    print(f"索提诺比率: {metrics['sortino']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"卡玛比率: {metrics['calmar']:.2f}")
    print(f"\nBeta(vs 300): {metrics['beta_vs_300']:.2f}")
    print(f"年化Alpha(vs 300): {metrics['alpha_vs_300']:.2%}")
    print(f"信息比率(vs 300): {metrics['info_ratio_vs_300']:.2f}")
    print(f"信息比率(vs 等权): {metrics['info_ratio_vs_equal_weight']:.2f}")
    print(f"\n日收益胜率: {metrics['win_rate']:.2%}")
    print(f"盈亏比: {metrics['profit_loss_ratio']:.2f}")
    print(f"收益偏度: {metrics['skewness']:.2f}")
    print(f"收益峰度: {metrics['kurtosis']:.2f}")
    print(f"\nt统计量(vs 0): {metrics['t_stat_vs_zero']:.3f}, p值: {metrics['p_value_vs_zero']:.4f}")
    print(f"t统计量(vs 300): {metrics['t_stat_vs_300']:.3f}, p值: {metrics['p_value_vs_300']:.4f}")
    print(f"滚动1年胜率(vs 300): {metrics['rolling_1y_consistency_vs_300']:.2%}")
    print(f"滚动1年胜率(vs 等权): {metrics['rolling_1y_consistency_vs_ew']:.2%}")
    print(f"月度收益自相关系数(滞后1月): {metrics['monthly_return_autocorr_1']:.3f}")
    print(f"\n成本敏感性（估计年换手 {estimated_turnover*100:.0f}%）:")
    for k, v in cost_sensitivity.items():
        flag = "PASS" if v > 0 else "FAIL"
        print(f"  {k:>8}: {v:>8.2%} [{flag}]")

    # 保存 NAV 和收益到 CSV
    out = pd.DataFrame({
        'strategy_nav': nav['nav'],
        'benchmark_nav': bench['nav'],
        'strategy_return': s_rets,
        'benchmark_return': b_rets,
    })
    out.to_csv(RESULTS_DIR / f'{name}_returns.csv')
    print(f"\n已保存: {RESULTS_DIR / f'{name}_returns.csv'}")

    return metrics

# ---------------------------------------------------------------------------
# 独立评级函数
# ---------------------------------------------------------------------------
def independent_grade(metrics, version_name):
    """
    基于以下原则给出独立评级：
    1. 统计显著性必须存在（p < 0.05）
    2. 真实成本下（0.1% 单边）必须为正
    3. 最大回撤不宜超过基准过多
    4. 信息比率应 > 0.5
    5. 滚动窗口一致性应 > 60%
    6. 样本期应跨越完整周期
    7. Beta 不宜过高（< 1.3）
    """
    score = 0
    fails = []

    # 1. 统计显著性 vs 沪深300 (20分)
    if metrics['p_value_vs_300'] < 0.05 and metrics['info_ratio_vs_300'] > 0.5:
        score += 20
    elif metrics['p_value_vs_300'] < 0.05:
        score += 12
        fails.append("对沪深300超额收益统计显著但信息比率偏低")
    else:
        fails.append("对沪深300超额收益统计不显著")

    # 2. 成本敏感性 (20分) - 基于年换手率假设
    cs = metrics['cost_sensitivity']
    if cs.get('cost_0.10%', -1) > 0:
        score += 20
    else:
        fails.append("0.1% 单边成本下收益为负")

    # 3. 风险调整后收益 (15分)
    if metrics['sharpe'] > 1.0 and metrics['max_drawdown'] > -0.30:
        score += 15
    elif metrics['sharpe'] > 0.5:
        score += 8
        fails.append("夏普比率偏低或回撤过大")
    else:
        fails.append("风险调整后收益差")

    # 4. 滚动一致性 (15分)
    if metrics['rolling_1y_consistency_vs_300'] >= 0.7:
        score += 15
    elif metrics['rolling_1y_consistency_vs_300'] >= 0.5:
        score += 8
        fails.append("滚动一致性一般")
    else:
        fails.append("滚动一致性很差")

    # 5. 等权基准对比 (10分)
    if metrics['info_ratio_vs_equal_weight'] > 0.5:
        score += 10
    elif metrics['info_ratio_vs_equal_weight'] > 0.3:
        score += 5
        fails.append("相对等权ETF信息比率偏低")
    else:
        fails.append("相对等权ETF无显著超额")

    # 6. 收益质量 (10分)
    if metrics['profit_loss_ratio'] > 1.0 and abs(metrics['skewness']) < 2 and metrics['kurtosis'] < 10:
        score += 10
    else:
        score += 4
        fails.append("收益分布存在肥尾或偏斜")

    # 7. 样本期与周期覆盖 (10分)
    n_years = metrics['trading_days'] / 252
    if n_years >= 5:
        score += 10
    elif n_years >= 3:
        score += 5
        fails.append("样本期较短，未跨越完整牛熊周期")
    else:
        fails.append("样本期过短，结论不可靠")

    # 评级
    if score >= 80:
        grade = 'A'
    elif score >= 65:
        grade = 'B'
    elif score >= 50:
        grade = 'C'
    elif score >= 35:
        grade = 'D'
    else:
        grade = 'F'

    return grade, score, fails

# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("="*80)
    print("  QuanTrade 独立审查（不依赖项目内置 EvaluationAgent）")
    print("="*80)

    m1 = review_version('v2.0.2c', Path('run_etf_system_v2.0.2c.py'), estimated_turnover=2.0)
    m2 = review_version('v2.2.0_final', Path('run_etf_system_v2.2.0_final.py'), estimated_turnover=4.0)

    g1, s1, f1 = independent_grade(m1, 'v2.0.2c')
    g2, s2, f2 = independent_grade(m2, 'v2.2.0_final')

    print("\n" + "="*80)
    print("  独立评级汇总")
    print("="*80)
    print(f"\nv2.0.2c:       {g1} ({s1}/100)")
    for fail in f1:
        print(f"  ! {fail}")
    print(f"\nv2.2.0_final:  {g2} ({s2}/100)")
    for fail in f2:
        print(f"  ! {fail}")

    # 写入汇总文件
    with open(RESULTS_DIR / 'independent_grade.txt', 'w', encoding='utf-8') as f:
        f.write("QuanTrade 独立审查评级\n")
        f.write("="*60 + "\n\n")
        f.write(f"v2.0.2c:       {g1} ({s1}/100)\n")
        for fail in f1:
            f.write(f"  ! {fail}\n")
        f.write(f"\nv2.2.0_final:  {g2} ({s2}/100)\n")
        for fail in f2:
            f.write(f"  ! {fail}\n")
    print(f"\n已保存评级汇总: {RESULTS_DIR / 'independent_grade.txt'}")
