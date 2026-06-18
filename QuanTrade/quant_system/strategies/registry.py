"""
A股量化信号系统 - 策略注册中心

统一管理所有策略，支持:
- 注册/注销策略
- 批量运行所有策略
- 策略间冲突检测
- 信号投票聚合
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from strategies.base import BaseStrategy, SignalResult

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    策略注册中心
    对应计划书中的"策略组合管理"
    """

    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
        self._history: List[Dict] = []  # 信号历史

    # ============================================================
    # 注册管理
    # ============================================================

    def register(self, strategy: BaseStrategy) -> "StrategyRegistry":
        """
        注册策略（支持链式调用）

        Example:
            registry = StrategyRegistry()
            registry.register(MACrossStrategy(fast=5, slow=20))
                     .register(RSIStrategy(period=14, overbought=70))
        """
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("策略必须继承 BaseStrategy")

        self._strategies[strategy.name] = strategy
        logger.info("注册策略: %s (%s)", strategy.name, strategy.display_name)
        return self

    def unregister(self, name: str) -> bool:
        """注销策略"""
        if name in self._strategies:
            del self._strategies[name]
            logger.info("注销策略: %s", name)
            return True
        return False

    def get(self, name: str) -> Optional[BaseStrategy]:
        """获取指定策略"""
        return self._strategies.get(name)

    def has(self, name: str) -> bool:
        """是否已注册某策略"""
        return name in self._strategies

    def list_strategies(self) -> List[Dict]:
        """列出所有已注册策略信息"""
        return [s.get_info() for s in self._strategies.values()]

    def clear(self):
        """清空所有策略"""
        self._strategies.clear()
        logger.info("已清空所有策略")

    def items(self) -> Iterator[Tuple[str, BaseStrategy]]:
        """遍历所有策略"""
        yield from self._strategies.items()

    def __len__(self):
        return len(self._strategies)

    def __contains__(self, name: str):
        return name in self._strategies

    # ============================================================
    # 批量运行
    # ============================================================

    def run_all(
        self, df: pd.DataFrame, symbol: str = ""
    ) -> Dict[str, SignalResult]:
        """
        运行所有已注册策略，返回信号结果字典

        Args:
            df: K线数据
            symbol: 股票代码（用于日志）

        Returns:
            {策略名: SignalResult, ...}
        """
        results = {}
        triggered_count = 0

        for name, strategy in self._strategies.items():
            try:
                signal = strategy.generate_signal(df)
                results[name] = signal
                if signal.triggered:
                    triggered_count += 1
                    logger.info(
                        "[%s] %s 触发 %s (置信度:%.2f)",
                        name,
                        symbol,
                        signal.action,
                        signal.confidence,
                    )
            except Exception as e:
                logger.error("[%s] 策略运行失败: %s", name, e)
                results[name] = SignalResult(
                    strategy_name=name,
                    triggered=False,
                    rationale=f"运行异常: {e}",
                )

        # 记录历史
        self._history.append({
            "symbol": symbol,
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_strategies": len(self._strategies),
            "n_triggered": triggered_count,
            "results": {k: v.to_dict() for k, v in results.items()},
        })

        return results

    def aggregate_voting(
        self, results: Dict[str, SignalResult], method: str = "confidence_weighted"
    ) -> SignalResult:
        """
        多策略投票聚合

        Args:
            results: run_all 的输出
            method: 聚合方法
                - "majority": 简单多数票
                - "confidence_weighted": 按置信度加权
                - "unanimous": 全票通过（保守）

        Returns:
            聚合后的统一信号
        """
        triggered = {k: v for k, v in results.items() if v.triggered}

        if not triggered:
            return SignalResult(
                strategy_name="aggregate",
                triggered=False,
                rationale="无策略触发信号",
            )

        # 统计各方向票数
        buy_votes = [v for v in triggered.values() if v.action in ("buy", "light_buy")]
        sell_votes = [v for v in triggered.values() if v.action in ("sell", "light_sell")]

        if method == "unanimous":
            # 全票通过：所有触发策略方向必须一致
            if len(buy_votes) == len(triggered):
                action = "buy"
            elif len(sell_votes) == len(triggered):
                action = "sell"
            else:
                return SignalResult(
                    strategy_name="aggregate",
                    triggered=False,
                    rationale=f"策略方向不一致: 买{len(buy_votes)} vs 卖{len(sell_votes)}",
                )

        elif method == "confidence_weighted":
            # 按置信度加权投票
            buy_score = sum(v.confidence for v in buy_votes)
            sell_score = sum(v.confidence for v in sell_votes)

            if buy_score > sell_score:
                action = "buy" if len(buy_votes) > 1 else "light_buy"
            elif sell_score > buy_score:
                action = "sell" if len(sell_votes) > 1 else "light_sell"
            else:
                return SignalResult(
                    strategy_name="aggregate",
                    triggered=False,
                    rationale="加权投票平局，保持观望",
                )

        else:  # majority
            if len(buy_votes) > len(sell_votes):
                action = "buy"
            elif len(sell_votes) > len(buy_votes):
                action = "sell"
            else:
                return SignalResult(
                    strategy_name="aggregate",
                    triggered=False,
                    rationale=f"投票平局: 买{len(buy_votes)} vs 卖{len(sell_votes)}",
                )

        # 聚合参数
        all_confidences = [v.confidence for v in triggered.values()]
        avg_confidence = sum(all_confidences) / len(all_confidences)

        # 取触发策略中的目标价/止损/仓位的加权平均
        weights = [v.confidence for v in triggered.values() if v.target_price > 0]
        target_prices = [v.target_price for v in triggered.values() if v.target_price > 0]
        stop_losses = [v.stop_loss for v in triggered.values() if v.stop_loss > 0]
        positions = [v.position_pct for v in triggered.values() if v.position_pct > 0]

        avg_target = (
            sum(t * w for t, w in zip(target_prices, weights)) / sum(weights)
            if weights else 0
        )
        avg_stop = sum(stop_losses) / len(stop_losses) if stop_losses else 0
        avg_position = sum(positions) / len(positions) if positions else 0

        # 汇总理由
        rationales = [f"{v.strategy_name}: {v.rationale}" for v in triggered.values()]
        all_risks = []
        for v in triggered.values():
            all_risks.extend(v.risk_factors)

        return SignalResult(
            strategy_name=f"aggregate({method})",
            triggered=True,
            action=action,
            confidence=round(avg_confidence, 4),
            target_price=round(avg_target, 2) if avg_target > 0 else 0,
            stop_loss=round(avg_stop, 2) if avg_stop > 0 else 0,
            position_pct=round(avg_position, 4) if avg_position > 0 else 0,
            rationale=f"[{len(triggered)}个策略触发] " + "; ".join(rationales[:2]),
            risk_factors=list(set(all_risks)),
        )

    def check_conflicts(
        self, results: Dict[str, SignalResult]
    ) -> List[Dict]:
        """
        检测策略间冲突
        返回冲突列表，每项包含冲突策略和方向
        """
        conflicts = []
        buy_strategies = []
        sell_strategies = []

        for name, signal in results.items():
            if not signal.triggered:
                continue
            if signal.action in ("buy", "light_buy"):
                buy_strategies.append(name)
            elif signal.action in ("sell", "light_sell"):
                sell_strategies.append(name)

        if buy_strategies and sell_strategies:
            conflicts.append({
                "type": "direction_conflict",
                "buy": buy_strategies,
                "sell": sell_strategies,
                "severity": "high" if len(buy_strategies) == len(sell_strategies) else "medium",
            })

        return conflicts
