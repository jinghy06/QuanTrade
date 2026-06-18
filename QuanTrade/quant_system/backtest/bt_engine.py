"""
A股量化信号系统 - 回测引擎封装
主力回测库: bt (FinRL-X同款)
参考: pmorissette/bt
对应计划书 Phase 1: "回测2022-2024年，评估夏普比率、最大回撤、胜率"
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import bt
import ffn
import numpy as np
import pandas as pd

from config.settings import COMMISSION, INITIAL_CAPITAL, SLIPPAGE
from data.data_store import DataStore

logger = logging.getLogger(__name__)


class BTBacktester:
    """
    bt回测引擎封装
    将ML信号包装为bt.Algo，进行Portfolio级回测
    """

    def __init__(self):
        self.store = DataStore()
        self.results = None

    def prepare_data(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        准备回测所需的价格数据
        Returns: DataFrame with MultiIndex (date, symbol) -> close price
                 pivoted to (date, symbol) columns
        """
        prices = {}
        for symbol in symbols:
            df = self.store.get_kline(symbol, start_date, end_date)
            if not df.empty and "close" in df.columns:
                # 以日期为索引，收盘价为值
                df = df.set_index("trade_date")["close"].rename(symbol)
                prices[symbol] = df

        if not prices:
            raise ValueError("没有有效的价格数据")

        # 合并为宽格式
        price_df = pd.DataFrame(prices)
        price_df = price_df.ffill().dropna()
        logger.info(
            "回测数据: %s ~ %s, %d个交易日, %d只股票",
            price_df.index[0],
            price_df.index[-1],
            len(price_df),
            len(price_df.columns),
        )
        return price_df

    def run_backtest(
        self,
        price_data: pd.DataFrame,
        signal_algo: Callable,
        **kwargs
    ) -> bt.backtest.Result:
        """
        运行回测

        Args:
            price_data: 宽格式价格数据 (date x symbols)
            signal_algo: 信号生成函数，返回SelectAll/WeighTarget等Algo
            **kwargs: 传递给bt.Backtest的参数

        Returns:
            bt.backtest.Result
        """
        # 构建策略
        strategy = bt.Strategy(
            "ML_Signal_Strategy",
            [
                bt.algos.RunDaily(),
                bt.algos.SelectAll(),
                signal_algo(),
                bt.algos.Rebalance(),
            ],
        )

        backtest = bt.Backtest(
            strategy,
            price_data,
            initial_capital=kwargs.get("initial_capital", INITIAL_CAPITAL),
            commissions=lambda q, p: abs(q) * p * COMMISSION,
        )

        logger.info("开始回测...")
        self.results = bt.run(backtest)
        logger.info("回测完成")

        return self.results

    def get_metrics(self) -> Dict:
        """
        获取回测绩效指标
        对应 plan.md: "评估夏普比率、最大回撤、胜率"
        """
        if self.results is None:
            return {}

        r = self.results["ML_Signal_Strategy"]

        metrics = {
            "total_return": r.total_return,
            "cagr": r.cagr,
            "sharpe": r.sharpe,
            "max_drawdown": r.max_drawdown,
            "calmar": r.calmar,
            "volatility": r.volatility,
            "sortino": r.sortino,
            "daily_win_rate": (r.daily_returns > 0).mean(),
            "monthly_win_rate": (r.monthly_returns > 0).mean(),
            "avg_daily_return": r.daily_returns.mean(),
            "avg_daily_volatility": r.daily_returns.std(),
        }

        logger.info("回测绩效: 夏普=%.3f, 最大回撤=%.2f%%, 胜率=%.1f%%",
            metrics["sharpe"],
            metrics["max_drawdown"] * 100,
            metrics["daily_win_rate"] * 100,
        )
        return metrics

    def plot(self, output_path: str = None):
        """绘制回测图表"""
        if self.results is None:
            logger.warning("请先运行回测")
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = self.results["ML_Signal_Strategy"]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 1. 累计收益
        r.cumulative_returns.plot(ax=axes[0], title="Cumulative Returns")
        axes[0].axhline(0, color="black", linestyle="--", alpha=0.3)

        # 2. 回撤
        r.drawdown_series.plot(ax=axes[1], title="Drawdown", color="red")
        axes[1].fill_between(
            r.drawdown_series.index, r.drawdown_series, 0, alpha=0.3, color="red"
        )

        # 3. 日收益分布
        r.daily_returns.hist(ax=axes[2], bins=50, edgecolor="black")
        axes[2].set_title("Daily Returns Distribution")
        axes[2].axvline(0, color="red", linestyle="--")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("回测图表已保存: %s", output_path)
        else:
            plt.show()

        plt.close()

    def report(self, output_path: str = None) -> str:
        """生成回测报告文本"""
        metrics = self.get_metrics()
        if not metrics:
            return "无回测结果"

        report_lines = [
            "=" * 50,
            "回测绩效报告",
            "=" * 50,
            f"总收益率:    {metrics['total_return']:.2%}",
            f"年化收益率:  {metrics['cagr']:.2%}",
            f"夏普比率:    {metrics['sharpe']:.3f}",
            f"最大回撤:    {metrics['max_drawdown']:.2%}",
            f"Calmar比率:  {metrics['calmar']:.3f}",
            f"年化波动率:  {metrics['volatility']:.2%}",
            f"Sortino比率: {metrics['sortino']:.3f}",
            f"日胜率:      {metrics['daily_win_rate']:.1%}",
            f"月胜率:      {metrics['monthly_win_rate']:.1%}",
            "=" * 50,
        ]

        report_text = "\n".join(report_lines)
        logger.info("\n%s", report_text)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)

        return report_text


class MLSignalAlgo:
    """
    将ML模型预测包装为bt.Algo
    供BTBacktester使用
    """

    def __init__(self, ml_model, threshold: float = 0.55):
        """
        Args:
            ml_model: 已加载的MLTrainer实例
            threshold: 信号阈值
        """
        self.ml_model = ml_model
        self.threshold = threshold

    def __call__(self):
        """返回bt.Algo实例"""

        class _Algo(bt.algos.Algo):
            def __init__(inner_self, model, threshold):
                super().__init__()
                inner_self.model = model
                inner_self.threshold = threshold

            def __call__(inner_self, target):
                # 获取当前持仓的股票
                selected = target.temp["selected"]
                if not selected:
                    return True

                # 对每个选中股票进行ML预测
                weights = {}
                for symbol in selected:
                    try:
                        df_feat = inner_self.model.store.get_features(
                            symbol, n_days=30
                        )
                        if df_feat.empty:
                            weights[symbol] = 0
                            continue

                        pred = inner_self.model.predict(df_feat)
                        if pred["up_prob"] > inner_self.threshold:
                            weights[symbol] = pred["up_prob"]
                        else:
                            weights[symbol] = 0
                    except Exception:
                        weights[symbol] = 0

                # 归一化权重
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}

                target.temp["weights"] = weights
                return True

        return _Algo(self.ml_model, self.threshold)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 快速回测示例
    from models.ml_trainer import MLTrainer

    store = DataStore()
    symbols = store.get_all_symbols()[:10]

    if symbols:
        bt_engine = BTBacktester()
        prices = bt_engine.prepare_data(symbols, "20220101", "20241231")
        print(f"回测数据:\n{prices.head()}")
