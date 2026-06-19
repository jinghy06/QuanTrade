"""
QuanTrade 2.0 - 独立评价Agent (Anti-Overfitting Agent)
======================================================
核心原则：不能为了目标而达到目标
这个Agent独立于策略开发，专门负责：
1. 检测过拟合（训练集 vs 测试集表现差异）
2. 统计显著性检验（收益是否显著优于随机）
3. 多时间窗口验证（Walk-Forward Analysis）
4. 交易成本敏感性分析
5. 市场环境适应性检验
6. 与朴素基准策略对比

使用方式：
    agent = EvaluationAgent()
    report = agent.full_evaluation(strategy_returns, benchmark_returns, predictions, actual_labels)
    agent.print_report(report)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EvaluationReport:
    """评价报告数据结构"""
    grade: str  # A/B/C/D/F
    overall_score: float  # 0-100
    verdict: str  # 最终判断
    warnings: List[str]  # 警告列表
    metrics: Dict  # 详细指标
    
    def __str__(self):
        return f"Grade: {self.grade} | Score: {self.overall_score:.1f} | Verdict: {self.verdict}"


class EvaluationAgent:
    """
    独立评价Agent - 不参与策略开发，只负责客观评价
    
    设计原则：
    1. 保守主义：宁可错过机会，不可误判质量
    2. 多维度验证：单一指标不可靠，需要多角度印证
    3. 诚实反馈：不回避问题，不美化结果
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.warnings = []
        
    def log(self, msg: str):
        if self.verbose:
            print(f"[评价Agent] {msg}")
    
    def full_evaluation(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        predictions: Optional[pd.Series] = None,
        actual_labels: Optional[pd.Series] = None,
        train_test_split_ratio: float = 0.7,
        n_walk_forward_windows: int = 5
    ) -> EvaluationReport:
        """
        完整评价流程
        
        Args:
            strategy_returns: 策略每日收益率序列
            benchmark_returns: 基准每日收益率序列
            predictions: 模型预测信号（可选）
            actual_labels: 实际标签（可选）
            train_test_split_ratio: 训练集比例
            n_walk_forward_windows: Walk-Forward窗口数
            
        Returns:
            EvaluationReport: 完整评价报告
        """
        self.warnings = []
        metrics = {}
        
        self.log("=" * 60)
        self.log("开始全面评估 - 目标：诚实评价，防止过拟合")
        self.log("=" * 60)
        
        # 1. 基础收益分析
        self.log("\n[1/7] 基础收益分析...")
        basic_metrics = self._basic_return_analysis(strategy_returns, benchmark_returns)
        metrics.update(basic_metrics)
        
        # 2. 过拟合检测
        self.log("\n[2/7] 过拟合检测...")
        overfit_metrics = self._overfitting_detection(
            strategy_returns, benchmark_returns, train_test_split_ratio
        )
        metrics.update(overfit_metrics)
        
        # 3. Walk-Forward验证
        self.log("\n[3/7] Walk-Forward验证...")
        wf_metrics = self._walk_forward_validation(
            strategy_returns, benchmark_returns, n_walk_forward_windows
        )
        metrics.update(wf_metrics)
        
        # 4. 统计显著性检验
        self.log("\n[4/7] 统计显著性检验...")
        stat_metrics = self._statistical_tests(strategy_returns, benchmark_returns)
        metrics.update(stat_metrics)
        
        # 5. 市场环境适应性
        self.log("\n[5/7] 市场环境适应性检验...")
        regime_metrics = self._regime_analysis(strategy_returns, benchmark_returns)
        metrics.update(regime_metrics)
        
        # 6. 交易成本敏感性
        self.log("\n[6/7] 交易成本敏感性分析...")
        cost_metrics = self._cost_sensitivity(strategy_returns, benchmark_returns)
        metrics.update(cost_metrics)
        
        # 7. 预测能力检验（如果有预测数据）
        if predictions is not None and actual_labels is not None:
            self.log("\n[7/7] 预测能力检验...")
            pred_metrics = self._prediction_quality(predictions, actual_labels)
            metrics.update(pred_metrics)
        else:
            self.log("\n[7/7] 跳过预测能力检验（无预测数据）")
            metrics['prediction_quality'] = 'N/A'
        
        # 计算综合评分
        overall_score = self._calculate_score(metrics)
        grade = self._score_to_grade(overall_score)
        verdict = self._generate_verdict(overall_score, metrics)
        
        report = EvaluationReport(
            grade=grade,
            overall_score=overall_score,
            verdict=verdict,
            warnings=self.warnings,
            metrics=metrics
        )
        
        return report
    
    def _basic_return_analysis(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Dict:
        """基础收益分析"""
        # 累计收益
        strategy_cum = (1 + strategy_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()
        
        strategy_total = strategy_cum.iloc[-1] - 1
        benchmark_total = benchmark_cum.iloc[-1] - 1
        excess_return = strategy_total - benchmark_total
        
        # 年化收益
        n_years = len(strategy_returns) / 252
        strategy_annual = (1 + strategy_total) ** (1/n_years) - 1 if n_years > 0 else 0
        benchmark_annual = (1 + benchmark_total) ** (1/n_years) - 1 if n_years > 0 else 0
        
        # 夏普比率
        sharpe = self._calculate_sharpe(strategy_returns)
        benchmark_sharpe = self._calculate_sharpe(benchmark_returns)
        
        # 最大回撤
        max_dd = self._calculate_max_drawdown(strategy_returns)
        benchmark_dd = self._calculate_max_drawdown(benchmark_returns)
        
        # 胜率
        win_rate = (strategy_returns > 0).mean()
        
        # 盈亏比
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        profit_loss_ratio = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else float('inf')
        
        metrics = {
            'strategy_total_return': strategy_total,
            'benchmark_total_return': benchmark_total,
            'excess_return': excess_return,
            'strategy_annual_return': strategy_annual,
            'benchmark_annual_return': benchmark_annual,
            'sharpe_ratio': sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_dd,
            'benchmark_max_drawdown': benchmark_dd,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'n_trading_days': len(strategy_returns),
            'n_years': n_years
        }
        
        self.log(f"  策略总收益: {strategy_total:.2%}")
        self.log(f"  基准总收益: {benchmark_total:.2%}")
        self.log(f"  超额收益: {excess_return:.2%}")
        self.log(f"  夏普比率: {sharpe:.2f}")
        self.log(f"  最大回撤: {max_dd:.2%}")
        self.log(f"  胜率: {win_rate:.2%}")
        
        return metrics
    
    def _overfitting_detection(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series,
        split_ratio: float
    ) -> Dict:
        """
        过拟合检测 - 核心方法
        比较训练期和测试期的表现差异
        """
        n = len(strategy_returns)
        split_idx = int(n * split_ratio)
        
        train_strategy = strategy_returns[:split_idx]
        test_strategy = strategy_returns[split_idx:]
        train_benchmark = benchmark_returns[:split_idx]
        test_benchmark = benchmark_returns[split_idx:]
        
        # 训练期表现
        train_excess = train_strategy.mean() - train_benchmark.mean()
        test_excess = test_strategy.mean() - test_benchmark.mean()
        
        # 收益衰减率（当训练期超额极低时，不计算衰减率）
        if train_excess != 0 and abs(train_excess) > 0.0001:  # 至少日均0.01%才计算
            decay_rate = (train_excess - test_excess) / abs(train_excess)
        else:
            decay_rate = 0  # 训练期无超额，不算衰减
        
        # 夏普比率衰减（同样处理）
        train_sharpe = self._calculate_sharpe(train_strategy)
        test_sharpe = self._calculate_sharpe(test_strategy)
        if abs(train_sharpe) > 0.01:
            sharpe_decay = (train_sharpe - test_sharpe) / max(abs(train_sharpe), 0.01)
        else:
            sharpe_decay = 0
        
        # 过拟合评分（0-1，越高越可能过拟合）
        overfit_score = 0
        if decay_rate > 0.5:
            overfit_score += 0.3
            self.warnings.append(f"警告: 收益衰减严重: {decay_rate:.1%}")
        if sharpe_decay > 0.5:
            overfit_score += 0.3
            self.warnings.append(f"警告: 夏普比率衰减严重: {sharpe_decay:.1%}")
        # 只有当训练期和测试期超额都有显著差异时才判断过拟合
        if train_excess > 0.0001 and test_excess < -0.0001:
            overfit_score += 0.4
            self.warnings.append("严重警告: 训练期正超额，测试期负超额 - 强烈过拟合信号")
        
        metrics = {
            'train_excess_return': train_excess,
            'test_excess_return': test_excess,
            'return_decay_rate': decay_rate,
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'sharpe_decay_rate': sharpe_decay,
            'overfit_score': overfit_score,
            'is_overfit': overfit_score > 0.5
        }
        
        self.log(f"  训练期超额: {train_excess:.2%}")
        self.log(f"  测试期超额: {test_excess:.2%}")
        self.log(f"  收益衰减率: {decay_rate:.1%}")
        self.log(f"  过拟合评分: {overfit_score:.2f} (0=无过拟合, 1=严重过拟合)")
        
        if overfit_score > 0.5:
            self.log("  警告: 检测到过拟合风险!")
        
        return metrics
    
    def _walk_forward_validation(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series,
        n_windows: int
    ) -> Dict:
        """
        Walk-Forward验证 - 滚动窗口测试
        这是最严格的验证方式，模拟真实交易场景
        """
        n = len(strategy_returns)
        window_size = n // n_windows
        
        window_results = []
        
        for i in range(n_windows):
            start = i * window_size
            end = min((i + 1) * window_size, n)
            
            window_strategy = strategy_returns[start:end]
            window_benchmark = benchmark_returns[start:end]
            
            if len(window_strategy) < 20:
                continue
            
            window_excess = window_strategy.mean() - window_benchmark.mean()
            window_sharpe = self._calculate_sharpe(window_strategy)
            window_dd = self._calculate_max_drawdown(window_strategy)
            
            window_results.append({
                'window': i + 1,
                'excess_return': window_excess,
                'sharpe': window_sharpe,
                'max_drawdown': window_dd,
                'n_days': len(window_strategy)
            })
        
        if not window_results:
            self.warnings.append("[警告] Walk-Forward验证数据不足")
            return {'wf_n_positive_windows': 0, 'wf_consistency': 0}
        
        # 统计各窗口表现
        excess_returns = [w['excess_return'] for w in window_results]
        sharpes = [w['sharpe'] for w in window_results]
        
        n_positive = sum(1 for e in excess_returns if e > 0)
        consistency = n_positive / len(window_results)
        
        # 表现稳定性（变异系数）
        if np.mean(excess_returns) != 0:
            cv = np.std(excess_returns) / abs(np.mean(excess_returns))
        else:
            cv = float('inf')
        
        metrics = {
            'wf_n_windows': len(window_results),
            'wf_n_positive_windows': n_positive,
            'wf_consistency': consistency,
            'wf_mean_excess': np.mean(excess_returns),
            'wf_std_excess': np.std(excess_returns),
            'wf_cv': cv,
            'wf_mean_sharpe': np.mean(sharpes),
            'wf_details': window_results
        }
        
        self.log(f"  有效窗口数: {len(window_results)}")
        self.log(f"  正超额窗口: {n_positive}/{len(window_results)} ({consistency:.1%})")
        self.log(f"  平均超额收益: {np.mean(excess_returns):.2%}")
        self.log(f"  收益稳定性(CV): {cv:.2f}")
        
        if consistency < 0.5:
            self.warnings.append(f"[警告] Walk-Forward一致性低: {consistency:.1%}")
        if cv > 2:
            self.warnings.append(f"[警告] 收益波动过大(CV={cv:.2f})")
        
        return metrics
    
    def _statistical_tests(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Dict:
        """
        统计显著性检验
        检验策略收益是否显著优于基准
        """
        excess_returns = strategy_returns - benchmark_returns
        
        # t检验：超额收益是否显著大于0
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        
        # 单尾检验（我们关心的是是否显著大于0）
        one_tail_p = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        
        # 信息比率
        if excess_returns.std() > 0:
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            information_ratio = 0
        
        # 自相关检验（检查收益是否独立）
        if len(excess_returns) > 10:
            autocorr = excess_returns.autocorr(lag=1)
        else:
            autocorr = 0
        
        # Bootstrap置信区间
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(excess_returns, size=len(excess_returns), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_means, 5)
        ci_upper = np.percentile(bootstrap_means, 95)
        
        metrics = {
            't_statistic': t_stat,
            'p_value': one_tail_p,
            'is_significant': one_tail_p < 0.05,
            'information_ratio': information_ratio,
            'autocorrelation': autocorr,
            'bootstrap_ci_lower': ci_lower,
            'bootstrap_ci_upper': ci_upper,
            'bootstrap_ci_contains_zero': ci_lower <= 0 <= ci_upper
        }
        
        self.log(f"  t统计量: {t_stat:.3f}")
        self.log(f"  p值(单尾): {one_tail_p:.4f}")
        self.log(f"  统计显著(p<0.05): {'是' if one_tail_p < 0.05 else '否'}")
        self.log(f"  信息比率: {information_ratio:.3f}")
        self.log(f"  90%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        if one_tail_p >= 0.05:
            self.warnings.append("[警告] 超额收益统计不显著 - 可能是运气")
        if ci_lower <= 0 <= ci_upper:
            self.warnings.append("[警告] Bootstrap置信区间包含0 - 收益不确定")
        
        return metrics
    
    def _regime_analysis(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Dict:
        """
        市场环境适应性分析
        检验策略在不同市场环境下的表现
        """
        # 对齐索引
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]
        
        # 用基准收益划分市场环境
        benchmark_cum = (1 + benchmark_returns).cumprod()
        
        # 计算60日滚动收益
        rolling_return = benchmark_returns.rolling(60).mean()
        
        # 划分环境
        bull_mask = rolling_return > 0.001  # 牛市：日均收益 > 0.1%
        bear_mask = rolling_return < -0.001  # 熊市：日均收益 < -0.1%
        sideways_mask = ~bull_mask & ~bear_mask  # 震荡市
        
        results = {}
        for regime_name, mask in [("牛市", bull_mask), ("熊市", bear_mask), ("震荡市", sideways_mask)]:
            if mask.sum() < 20:
                continue
            
            regime_strategy = strategy_returns[mask]
            regime_benchmark = benchmark_returns[mask]
            
            excess = regime_strategy.mean() - regime_benchmark.mean()
            sharpe = self._calculate_sharpe(regime_strategy)
            
            results[f'{regime_name}_excess'] = excess
            results[f'{regime_name}_sharpe'] = sharpe
            results[f'{regime_name}_n_days'] = mask.sum()
            
            self.log(f"  {regime_name}: 超额={excess:.2%}, 夏普={sharpe:.2f}, 天数={mask.sum()}")
        
        # 检查是否只在单一环境有效
        regime_effective = sum(1 for k, v in results.items() if k.endswith('_excess') and v > 0)
        results['regime_robustness'] = regime_effective / max(len([k for k in results if k.endswith('_excess')]), 1)
        
        if results.get('regime_robustness', 0) < 0.5:
            self.warnings.append("[警告] 策略只在部分市场环境有效，鲁棒性不足")
        
        return results
        
        return results
    
    def _cost_sensitivity(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Dict:
        """
        交易成本敏感性分析
        检验策略对交易成本的敏感程度
        """
        # 假设不同交易成本水平
        cost_levels = [0.001, 0.002, 0.003, 0.005, 0.01]  # 0.1% ~ 1%
        
        # 估算换手率（用收益变化频率近似）
        turnover_estimate = (strategy_returns.diff().abs() > 0.001).mean()
        
        results = {'turnover_estimate': turnover_estimate}
        
        for cost in cost_levels:
            # 简化：假设每日都有换手
            adjusted_returns = strategy_returns - cost * turnover_estimate
            adjusted_total = (1 + adjusted_returns).cumprod().iloc[-1] - 1
            benchmark_total = (1 + benchmark_returns).cumprod().iloc[-1] - 1
            
            results[f'excess_at_{cost:.1%}_cost'] = adjusted_total - benchmark_total
        
        # 找到盈亏平衡成本
        # 当超额收益 = 成本 * 换手率时，策略失效
        base_excess = strategy_returns.mean() - benchmark_returns.mean()
        if turnover_estimate > 0:
            breakeven_cost = base_excess / turnover_estimate
        else:
            breakeven_cost = float('inf')
        
        results['breakeven_cost'] = breakeven_cost
        
        self.log(f"  估计换手率: {turnover_estimate:.2%}")
        self.log(f"  盈亏平衡成本: {breakeven_cost:.2%}")
        
        for cost in cost_levels:
            excess = results[f'excess_at_{cost:.1%}_cost']
            status = "OK" if excess > 0 else "FAIL"
            self.log(f"  成本{cost:.1%}时超额: {excess:.2%} {status}")
        
        if breakeven_cost < 0.003:
            self.warnings.append(f"[警告] 盈亏平衡成本过低({breakeven_cost:.2%})，策略对成本敏感")
        
        return results
    
    def _prediction_quality(
        self, 
        predictions: pd.Series, 
        actual_labels: pd.Series
    ) -> Dict:
        """
        预测质量检验
        检验模型预测的实际价值
        """
        # 对齐数据
        common_idx = predictions.index.intersection(actual_labels.index)
        pred = predictions.loc[common_idx]
        actual = actual_labels.loc[common_idx]
        
        # 准确率
        accuracy = (pred == actual).mean()
        
        # 精确率、召回率（假设1为正类）
        tp = ((pred == 1) & (actual == 1)).sum()
        fp = ((pred == 1) & (actual == 0)).sum()
        fn = ((pred == 0) & (actual == 1)).sum()
        tn = ((pred == 0) & (actual == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 随机基准准确率
        random_accuracy = actual.mean() * actual.mean() + (1 - actual.mean()) * (1 - actual.mean())
        
        # 技能得分（相对于随机的提升）
        skill_score = (accuracy - random_accuracy) / (1 - random_accuracy) if random_accuracy < 1 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'random_baseline_accuracy': random_accuracy,
            'skill_score': skill_score,
            'n_predictions': len(pred)
        }
        
        self.log(f"  准确率: {accuracy:.2%}")
        self.log(f"  随机基准: {random_accuracy:.2%}")
        self.log(f"  技能得分: {skill_score:.3f}")
        self.log(f"  精确率: {precision:.2%}, 召回率: {recall:.2%}, F1: {f1:.3f}")
        
        if skill_score < 0.05:
            self.warnings.append("[警告] 预测技能得分低，模型预测能力有限")
        
        return metrics
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算年化夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess = returns.mean() - risk_free_rate / 252
        return excess / returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        return drawdown.min()
    
    def _calculate_score(self, metrics: Dict) -> float:
        """
        计算综合评分（0-100）
        
        评分维度：
        - 超额收益：25分
        - 夏普比率：20分
        - 过拟合检测：25分（惩罚分）
        - Walk-Forward一致性：15分
        - 统计显著性：15分
        """
        score = 50  # 从50分开始
        
        # 超额收益（±15分）
        excess = metrics.get('excess_return', 0)
        if excess > 0.2:
            score += 15
        elif excess > 0.1:
            score += 10
        elif excess > 0.05:
            score += 5
        elif excess > 0:
            score += 2
        elif excess > -0.05:
            score -= 5
        else:
            score -= 15
        
        # 夏普比率（±10分）
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 2:
            score += 10
        elif sharpe > 1:
            score += 7
        elif sharpe > 0.5:
            score += 3
        elif sharpe > 0:
            score += 0
        else:
            score -= 10
        
        # 过拟合检测（-20到0分）
        overfit = metrics.get('overfit_score', 0)
        score -= int(overfit * 20)
        
        # Walk-Forward一致性（±10分）
        wf_consistency = metrics.get('wf_consistency', 0.5)
        if wf_consistency >= 0.8:
            score += 10
        elif wf_consistency >= 0.6:
            score += 5
        elif wf_consistency >= 0.4:
            score += 0
        else:
            score -= 10
        
        # 统计显著性（±5分）
        if metrics.get('is_significant', False):
            score += 5
        else:
            score -= 5
        
        return max(0, min(100, score))
    
    def _score_to_grade(self, score: float) -> str:
        """分数转等级"""
        if score >= 80:
            return 'A'
        elif score >= 65:
            return 'B'
        elif score >= 50:
            return 'C'
        elif score >= 35:
            return 'D'
        else:
            return 'F'
    
    def _generate_verdict(self, score: float, metrics: Dict) -> str:
        """生成最终判断"""
        warnings_count = len(self.warnings)
        
        if score >= 80 and warnings_count == 0:
            return "[OK] 策略质量优秀，可以考虑实盘验证"
        elif score >= 65 and warnings_count <= 1:
            return "[OK] 策略质量良好，建议进一步优化后实盘测试"
        elif score >= 50:
            return "[警告] 策略质量一般，存在明显问题需要解决"
        elif score >= 35:
            return "[失败] 策略质量较差，不建议实盘使用"
        else:
            return "[严重警告] 策略质量极差，需要重新设计"
    
    def print_report(self, report: EvaluationReport):
        """打印完整评价报告"""
        print("\n" + "=" * 70)
        print("                    QuanTrade 独立评价报告")
        print("=" * 70)
        
        print(f"\n[评分] 综合评分: {report.overall_score:.1f}/100")
        print(f"[等级] 等级: {report.grade}")
        print(f"[判断] 判断: {report.verdict}")
        
        if report.warnings:
            print(f"\n[警告] 警告 ({len(report.warnings)}):")
            for w in report.warnings:
                print(f"  {w}")
        else:
            print("\n[OK] 无警告")
        
        print("\n" + "-" * 70)
        print("详细指标:")
        print("-" * 70)
        
        key_metrics = [
            ('总收益', 'strategy_total_return', '%'),
            ('基准收益', 'benchmark_total_return', '%'),
            ('超额收益', 'excess_return', '%'),
            ('夏普比率', 'sharpe_ratio', '.2f'),
            ('最大回撤', 'max_drawdown', '%'),
            ('胜率', 'win_rate', '%'),
            ('过拟合评分', 'overfit_score', '.2f'),
            ('Walk-Forward一致性', 'wf_consistency', '%'),
            ('统计显著性', 'is_significant', 'bool'),
            ('信息比率', 'information_ratio', '.3f'),
        ]
        
        for name, key, fmt in key_metrics:
            if key in report.metrics:
                value = report.metrics[key]
                if fmt == '%':
                    print(f"  {name:20s}: {value:.2%}")
                elif fmt == 'bool':
                    print(f"  {name:20s}: {'是' if value else '否'}")
                else:
                    print(f"  {name:20s}: {value:{fmt}}")
        
        print("\n" + "=" * 70)
        
        # 防过拟合特别提示
        if report.metrics.get('is_overfit', False):
            print("\n[严重警告] 过拟合警告 [严重警告]")
            print("策略在训练期表现远好于测试期，这强烈暗示过拟合。")
            print("建议：")
            print("  1. 减少特征数量")
            print("  2. 增加正则化")
            print("  3. 使用更简单的模型")
            print("  4. 增加训练数据量")
            print("=" * 70)


def demo():
    """演示用法"""
    print("=" * 70)
    print("QuanTrade 独立评价Agent 演示")
    print("=" * 70)
    
    # 生成模拟数据
    np.random.seed(42)
    n_days = 1000
    
    # 模拟基准收益（市场）
    benchmark_returns = pd.Series(
        np.random.normal(0.0003, 0.015, n_days),
        index=pd.date_range('2020-01-01', periods=n_days, freq='B')
    )
    
    # 模拟策略收益（略优于基准）
    strategy_returns = benchmark_returns + pd.Series(
        np.random.normal(0.0002, 0.005, n_days),
        index=benchmark_returns.index
    )
    
    # 创建评价Agent
    agent = EvaluationAgent(verbose=True)
    
    # 运行完整评估
    report = agent.full_evaluation(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns
    )
    
    # 打印报告
    agent.print_report(report)
    
    return report


if __name__ == "__main__":
    demo()
