"""
A股量化信号系统 - 内置策略示例

包含以下策略:
- MACrossStrategy: 均线金叉/死叉策略
- RSIStrategy: RSI超买超卖策略
- MACDStrategy: MACD金叉/死叉策略
- BreakoutStrategy: 突破策略（支撑位/压力位）
- MeanReversionStrategy: 均值回归策略
- MLHybridStrategy: ML+技术面混合策略（与LightGBM结合）

使用方式:
    from strategies.built_in import MACrossStrategy, RSIStrategy
    from strategies.registry import StrategyRegistry

    registry = StrategyRegistry()
    registry.register(MACrossStrategy(fast=5, slow=20))
    registry.register(RSIStrategy(period=14, overbought=70, oversold=30))

    results = registry.run_all(df_kline, symbol="000001.SZ")
    combined = registry.aggregate_voting(results)
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, SignalResult

logger = logging.getLogger(__name__)


# ============================================================
# 1. 均线交叉策略
# ============================================================

class MACrossStrategy(BaseStrategy):
    """
    双均线交叉策略
    快线从下向上穿越慢线 -> 买入信号（金叉）
    快线从上向下穿越慢线 -> 卖出信号（死叉）
    """

    name = "ma_cross"
    display_name = "均线交叉"
    description = "双均线金叉买入，死叉卖出"
    category = "technical"

    required_columns = ["close"]
    min_bars = 60

    def __init__(self, fast: int = 5, slow: int = 20):
        """
        Args:
            fast: 快线周期（默认5日）
            slow: 慢线周期（默认20日）
        """
        if fast >= slow:
            raise ValueError(f"快线周期({fast})必须小于慢线周期({slow})")
        super().__init__(fast=fast, slow=slow)
        self.fast = fast
        self.slow = slow

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        if not self.validate_data(df):
            return SignalResult(self.name, triggered=False)

        close = df["close"]
        ma_fast = close.rolling(self.fast).mean()
        ma_slow = close.rolling(self.slow).mean()

        # 需要至少 slow+1 条数据才能计算交叉
        if len(ma_fast.dropna()) < 2 or len(ma_slow.dropna()) < 2:
            return SignalResult(self.name, triggered=False)

        # 当前和前一日状态
        prev_fast, curr_fast = ma_fast.iloc[-2], ma_fast.iloc[-1]
        prev_slow, curr_slow = ma_slow.iloc[-2], ma_slow.iloc[-1]

        prev_diff = prev_fast - prev_slow
        curr_diff = curr_fast - curr_slow

        latest_close = close.iloc[-1]

        # 金叉: prev_diff < 0 且 curr_diff >= 0
        if prev_diff < 0 <= curr_diff:
            confidence = min(abs(curr_diff) / curr_slow * 100, 1.0)
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="buy",
                confidence=round(confidence, 4),
                price=round(latest_close, 2),
                target_price=round(latest_close * 1.05, 2),
                stop_loss=round(latest_close * 0.93, 2),
                position_pct=0.05,
                rationale=f"{self.fast}日均线上穿{self.slow}日均线形成金叉",
                metadata={
                    "ma_fast": round(curr_fast, 2),
                    "ma_slow": round(curr_slow, 2),
                    "diff": round(curr_diff, 4),
                },
            )

        # 死叉: prev_diff > 0 且 curr_diff <= 0
        elif prev_diff > 0 >= curr_diff:
            confidence = min(abs(curr_diff) / curr_slow * 100, 1.0)
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="sell",
                confidence=round(confidence, 4),
                price=round(latest_close, 2),
                rationale=f"{self.fast}日均线下穿{self.slow}日均线形成死叉",
                metadata={
                    "ma_fast": round(curr_fast, 2),
                    "ma_slow": round(curr_slow, 2),
                    "diff": round(curr_diff, 4),
                },
            )

        return SignalResult(
            strategy_name=self.name,
            triggered=False,
            rationale=f"均线交叉未触发 (fast={curr_fast:.2f}, slow={curr_slow:.2f})",
        )


# ============================================================
# 2. RSI策略
# ============================================================

class RSIStrategy(BaseStrategy):
    """
    RSI相对强弱指标策略
    RSI < oversold(超卖线) -> 买入信号（可能反弹）
    RSI > overbought(超买线) -> 卖出信号（可能回调）
    """

    name = "rsi"
    display_name = "RSI超买超卖"
    description = "RSI超卖区买入，超买区卖出"
    category = "technical"

    required_columns = ["close"]
    min_bars = 30

    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Args:
            period: RSI计算周期（默认14）
            overbought: 超买线（默认70）
            oversold: 超卖线（默认30）
        """
        super().__init__(period=period, overbought=overbought, oversold=oversold)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def _calculate_rsi(self, close: pd.Series) -> float:
        """计算RSI值"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        if not self.validate_data(df):
            return SignalResult(self.name, triggered=False)

        close = df["close"]
        rsi = self._calculate_rsi(close)
        latest_close = close.iloc[-1]

        # 超卖区买入
        if rsi < self.oversold:
            # 越接近0越有信心
            confidence = (self.oversold - rsi) / self.oversold
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="buy",
                confidence=round(confidence, 4),
                price=round(latest_close, 2),
                target_price=round(latest_close * 1.05, 2),
                stop_loss=round(latest_close * 0.95, 2),
                position_pct=0.03,
                rationale=f"RSI={rsi:.1f}进入超卖区(>{self.oversold})，预期反弹",
                risk_factors=["超卖可能持续", "下跌动能未完全释放"],
                metadata={"rsi": round(rsi, 2)},
            )

        # 超买区卖出
        elif rsi > self.overbought:
            confidence = (rsi - self.overbought) / (100 - self.overbought)
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="sell",
                confidence=round(confidence, 4),
                price=round(latest_close, 2),
                rationale=f"RSI={rsi:.1f}进入超买区(>{self.overbought})，预期回调",
                risk_factors=["超买可能持续", "强者恒强"],
                metadata={"rsi": round(rsi, 2)},
            )

        return SignalResult(
            strategy_name=self.name,
            triggered=False,
            rationale=f"RSI={rsi:.1f}，处于中性区间",
            metadata={"rsi": round(rsi, 2)},
        )


# ============================================================
# 3. MACD策略
# ============================================================

class MACDStrategy(BaseStrategy):
    """
    MACD指标策略
    DIF上穿DEA -> 买入（金叉）
    DIF下穿DEA -> 卖出（死叉）
    支持零轴判断：零轴上方金叉更强
    """

    name = "macd"
    display_name = "MACD金叉死叉"
    description = "MACD金叉买入，死叉卖出，结合零轴判断"
    category = "technical"

    required_columns = ["close"]
    min_bars = 40

    def __init__(
        self, fast: int = 12, slow: int = 26, signal: int = 9, use_zero_axis: bool = True
    ):
        """
        Args:
            fast: DIF快线周期
            slow: DIF慢线周期
            signal: DEA信号线周期
            use_zero_axis: 是否考虑零轴位置
        """
        super().__init__(fast=fast, slow=slow, signal=signal, use_zero_axis=use_zero_axis)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.use_zero_axis = use_zero_axis

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        if not self.validate_data(df):
            return SignalResult(self.name, triggered=False)

        close = df["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal_period, adjust=False).mean()
        hist = dif - dea

        if len(dif) < 2:
            return SignalResult(self.name, triggered=False)

        prev_dif, curr_dif = dif.iloc[-2], dif.iloc[-1]
        prev_dea, curr_dea = dea.iloc[-2], dea.iloc[-1]
        latest_close = close.iloc[-1]

        # 金叉: prev_dif < prev_dea 且 curr_dif >= curr_dea
        is_golden_cross = prev_dif < prev_dea <= curr_dif
        # 死叉: prev_dif > prev_dea 且 curr_dif <= curr_dea
        is_dead_cross = prev_dif > prev_dea >= curr_dif

        if is_golden_cross:
            # 零轴上方金叉 = 更强
            above_zero = curr_dea > 0 if self.use_zero_axis else False
            confidence = 0.6 + (0.2 if above_zero else 0)
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="buy",
                confidence=round(confidence, 4),
                price=round(latest_close, 2),
                target_price=round(latest_close * 1.06, 2),
                stop_loss=round(latest_close * 0.94, 2),
                position_pct=0.05 if above_zero else 0.03,
                rationale=f"MACD{'零轴上方' if above_zero else ''}金叉 (DIF={curr_dif:.3f}, DEA={curr_dea:.3f})",
                metadata={
                    "dif": round(curr_dif, 4),
                    "dea": round(curr_dea, 4),
                    "hist": round(hist.iloc[-1], 4),
                    "above_zero": above_zero,
                },
            )

        elif is_dead_cross:
            below_zero = curr_dea < 0 if self.use_zero_axis else False
            confidence = 0.6 + (0.2 if below_zero else 0)
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="sell",
                confidence=round(confidence, 4),
                price=round(latest_close, 2),
                rationale=f"MACD{'零轴下方' if below_zero else ''}死叉 (DIF={curr_dif:.3f}, DEA={curr_dea:.3f})",
                metadata={
                    "dif": round(curr_dif, 4),
                    "dea": round(curr_dea, 4),
                    "hist": round(hist.iloc[-1], 4),
                },
            )

        return SignalResult(
            strategy_name=self.name,
            triggered=False,
            rationale=f"MACD未交叉 (DIF={curr_dif:.3f}, DEA={curr_dea:.3f})",
            metadata={
                "dif": round(curr_dif, 4),
                "dea": round(curr_dea, 4),
            },
        )


# ============================================================
# 4. 突破策略
# ============================================================

class BreakoutStrategy(BaseStrategy):
    """
    突破策略
    突破N日最高价 -> 买入（可能开始上涨趋势）
    跌破N日最低价 -> 卖出（可能开始下跌趋势）
    """

    name = "breakout"
    display_name = "突破策略"
    description = "突破N日最高价买入，跌破N日最低价卖出"
    category = "technical"

    required_columns = ["close", "high", "low"]
    min_bars = 40

    def __init__(self, period: int = 20, volume_confirm: bool = True):
        """
        Args:
            period: 突破周期（默认20日）
            volume_confirm: 是否要求放量确认
        """
        super().__init__(period=period, volume_confirm=volume_confirm)
        self.period = period
        self.volume_confirm = volume_confirm

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        if not self.validate_data(df):
            return SignalResult(self.name, triggered=False)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        latest_close = close.iloc[-1]

        # N日最高/最低
        highest = high.rolling(self.period).max().iloc[-1]
        lowest = low.rolling(self.period).min().iloc[-1]
        prev_close = close.iloc[-2]

        # 突破前高
        if prev_close < highest <= latest_close:
            # 放量确认
            vol_confirmed = True
            if self.volume_confirm and "volume" in df.columns:
                avg_vol = df["volume"].rolling(5).mean().iloc[-1]
                vol_confirmed = df["volume"].iloc[-1] > avg_vol * 1.2

            if vol_confirmed:
                return SignalResult(
                    strategy_name=self.name,
                    triggered=True,
                    action="buy",
                    confidence=0.65,
                    price=round(latest_close, 2),
                    target_price=round(highest * 1.08, 2),
                    stop_loss=round(lowest * 0.97, 2),
                    position_pct=0.05,
                    rationale=f"突破{self.period}日最高价{highest:.2f}" + ("且放量确认" if self.volume_confirm else ""),
                    risk_factors=["假突破风险", "追高被套"],
                    metadata={
                        "highest": round(highest, 2),
                        "lowest": round(lowest, 2),
                        "volume_confirmed": vol_confirmed,
                    },
                )

        # 跌破前低
        elif prev_close > lowest >= latest_close:
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="sell",
                confidence=0.6,
                price=round(latest_close, 2),
                rationale=f"跌破{self.period}日最低价{lowest:.2f}",
                risk_factors=["超跌反弹", "恐慌抛售"],
                metadata={
                    "highest": round(highest, 2),
                    "lowest": round(lowest, 2),
                },
            )

        return SignalResult(
            strategy_name=self.name,
            triggered=False,
            rationale=f"价格{latest_close:.2f}在区间[{lowest:.2f}, {highest:.2f}]内",
            metadata={"highest": round(highest, 2), "lowest": round(lowest, 2)},
        )


# ============================================================
# 5. 均值回归策略
# ============================================================

class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略（布林带简化版）
    价格低于下轨（均线-N倍std）-> 买入
    价格高于上轨（均线+N倍std）-> 卖出
    """

    name = "mean_reversion"
    display_name = "均值回归"
    description = "布林带均值回归策略，价格偏离均线过多时反向操作"
    category = "technical"

    required_columns = ["close"]
    min_bars = 40

    def __init__(self, period: int = 20, dev: float = 2.0):
        """
        Args:
            period: 均线周期
            dev: 标准差倍数（默认2倍）
        """
        super().__init__(period=period, dev=dev)
        self.period = period
        self.dev = dev

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        if not self.validate_data(df):
            return SignalResult(self.name, triggered=False)

        close = df["close"]
        ma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()

        upper = (ma + self.dev * std).iloc[-1]
        lower = (ma - self.dev * std).iloc[-1]
        latest = close.iloc[-1]
        curr_ma = ma.iloc[-1]

        if latest < lower:
            # 跌破下轨 -> 买入（预期回归均值）
            deviation = (curr_ma - latest) / (curr_ma + 1e-10)
            confidence = min(deviation * 5, 1.0)
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="buy",
                confidence=round(confidence, 4),
                price=round(latest, 2),
                target_price=round(curr_ma, 2),
                stop_loss=round(lower * 0.98, 2),
                position_pct=0.03,
                rationale=f"价格{latest:.2f}跌破{self.dev}倍标准差下轨{lower:.2f}，预期回归均线{curr_ma:.2f}",
                risk_factors=["趋势延续风险", "下跌未尽"],
                metadata={"ma": round(curr_ma, 2), "upper": round(upper, 2), "lower": round(lower, 2)},
            )

        elif latest > upper:
            # 突破上轨 -> 卖出（预期回归均值）
            deviation = (latest - curr_ma) / (curr_ma + 1e-10)
            confidence = min(deviation * 5, 1.0)
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="sell",
                confidence=round(confidence, 4),
                price=round(latest, 2),
                target_price=round(curr_ma, 2),
                stop_loss=round(upper * 1.02, 2),
                position_pct=0.05,
                rationale=f"价格{latest:.2f}突破{self.dev}倍标准差上轨{upper:.2f}，预期回归均线{curr_ma:.2f}",
                risk_factors=[["强者恒强", "趋势延续"]],
                metadata={"ma": round(curr_ma, 2), "upper": round(upper, 2), "lower": round(lower, 2)},
            )

        return SignalResult(
            strategy_name=self.name,
            triggered=False,
            rationale=f"价格在布林带区间内 [{lower:.2f}, {upper:.2f}]",
            metadata={"ma": round(curr_ma, 2), "upper": round(upper, 2), "lower": round(lower, 2)},
        )


# ============================================================
# 6. ML+技术面混合策略
# ============================================================

class MLHybridStrategy(BaseStrategy):
    """
    ML+技术面混合策略
    结合LightGBM预测和技术指标，双重确认后发出信号
    需要传入 MLTrainer 实例
    """

    name = "ml_hybrid"
    display_name = "ML混合策略"
    description = "LightGBM预测+技术指标双重确认"
    category = "ml"

    required_columns = ["close"]
    min_bars = 60

    def __init__(
        self,
        ml_trainer,
        min_ml_prob: float = 0.55,
        require_macd_align: bool = True,
    ):
        """
        Args:
            ml_trainer: MLTrainer 实例（已加载模型）
            min_ml_prob: ML最低概率阈值
            require_macd_align: 是否要求MACD方向一致
        """
        super().__init__(
            min_ml_prob=min_ml_prob,
            require_macd_align=require_macd_align,
        )
        self.ml_trainer = ml_trainer
        self.min_ml_prob = min_ml_prob
        self.require_macd_align = require_macd_align

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        if not self.validate_data(df):
            return SignalResult(self.name, triggered=False)

        if self.ml_trainer is None or self.ml_trainer.model is None:
            return SignalResult(
                self.name, triggered=False, rationale="ML模型未加载"
            )

        latest_close = df["close"].iloc[-1]

        # Step 1: ML预测（从数据库获取特征）
        try:
            from data.data_store import DataStore

            store = DataStore()
            symbol = df["symbol"].iloc[-1] if "symbol" in df.columns else ""
            df_feat = store.get_features(symbol)
            if df_feat.empty:
                return SignalResult(self.name, triggered=False, rationale="无特征数据")

            ml_pred = self.ml_trainer.predict(df_feat)
            ml_prob = ml_pred["up_prob"]
        except Exception as e:
            return SignalResult(
                self.name, triggered=False, rationale=f"ML预测失败: {e}"
            )

        # Step 2: 检查ML概率
        if ml_prob < self.min_ml_prob:
            return SignalResult(
                self.name,
                triggered=False,
                rationale=f"ML概率{ml_prob:.3f}低于阈值{self.min_ml_prob}",
                metadata={"ml_prob": ml_prob},
            )

        # Step 3: MACD方向确认（可选）
        macd_aligned = True
        if self.require_macd_align:
            close = df["close"]
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            curr_dif = dif.iloc[-1]

            if ml_prob > 0.55 and curr_dif < 0:
                macd_aligned = False
            elif ml_prob < 0.45 and curr_dif > 0:
                macd_aligned = False

        if not macd_aligned:
            return SignalResult(
                self.name,
                triggered=False,
                rationale=f"ML概率{ml_prob:.3f}与MACD方向不一致(DIF={curr_dif:.3f})",
                metadata={"ml_prob": ml_prob, "dif": round(curr_dif, 4)},
            )

        # Step 4: 双重确认通过 -> 发出信号
        action = "buy" if ml_prob > 0.55 else "sell"
        confidence = abs(ml_prob - 0.5) * 2  # 归一化到0-1

        return SignalResult(
            strategy_name=self.name,
            triggered=True,
            action=action,
            confidence=round(confidence, 4),
            price=round(latest_close, 2),
            target_price=round(latest_close * (1.05 if action == "buy" else 0.97), 2),
            stop_loss=round(latest_close * (0.93 if action == "buy" else 1.05), 2),
            position_pct=0.05 if confidence > 0.6 else 0.03,
            rationale=f"ML概率{ml_prob:.3f}+{'' if macd_aligned else '不'}要求MACD确认，双重验证通过",
            metadata={"ml_prob": ml_prob, "dif": round(curr_dif, 4) if self.require_macd_align else None},
        )


# ============================================================
# 便捷函数：一键注册所有内置策略
# ============================================================

def register_all_built_in(registry, **kwargs) -> None:
    """
    一键注册所有内置策略

    Args:
        registry: StrategyRegistry 实例
        **kwargs: 可覆盖策略参数，如 ma_fast=10
    """
    strategies = [
        MACrossStrategy(fast=kwargs.get("ma_fast", 5), slow=kwargs.get("ma_slow", 20)),
        RSIStrategy(
            period=kwargs.get("rsi_period", 14),
            overbought=kwargs.get("rsi_overbought", 70),
            oversold=kwargs.get("rsi_oversold", 30),
        ),
        MACDStrategy(
            fast=kwargs.get("macd_fast", 12),
            slow=kwargs.get("macd_slow", 26),
            signal=kwargs.get("macd_signal", 9),
        ),
        BreakoutStrategy(period=kwargs.get("breakout_period", 20)),
        MeanReversionStrategy(period=kwargs.get("mr_period", 20)),
    ]

    for s in strategies:
        registry.register(s)

    logger.info("已注册 %d 个内置策略", len(strategies))
