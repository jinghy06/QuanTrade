"""
A股量化信号系统 - 策略基类与信号结果

所有自定义策略必须继承 BaseStrategy，实现 generate_signal 方法。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """
    策略信号结果
    每个策略 generate_signal 方法必须返回此对象
    """

    strategy_name: str          # 策略名称
    triggered: bool             # 是否触发信号
    action: str = "hold"        # 动作: buy/sell/hold/light_buy/light_sell
    confidence: float = 0.0     # 置信度 0-1
    price: float = 0.0          # 当前价格
    target_price: float = 0.0   # 目标价
    stop_loss: float = 0.0      # 止损价
    position_pct: float = 0.0   # 建议仓位比例
    rationale: str = ""         # 决策理由
    risk_factors: List[str] = field(default_factory=list)  # 风险因素
    metadata: Dict[str, Any] = field(default_factory=dict)  # 策略自定义数据

    def to_dict(self) -> Dict:
        """转为字典格式（用于序列化到数据库/飞书推送）"""
        return {
            "strategy_name": self.strategy_name,
            "triggered": self.triggered,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "price": round(self.price, 2) if self.price else None,
            "target_price": round(self.target_price, 2) if self.target_price else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "position_pct": round(self.position_pct, 4) if self.position_pct else None,
            "rationale": self.rationale,
            "risk_factors": self.risk_factors,
            **{f"meta_{k}": v for k, v in self.metadata.items()},
        }

    def __bool__(self):
        """便捷判断：signal.triggered 可简写为 if signal:"""
        return self.triggered


class BaseStrategy(ABC):
    """
    策略基类
    所有自定义策略必须继承此类并实现 generate_signal 方法
    """

    # 策略元信息（子类必须覆盖）
    name: str = "base"                    # 策略标识名（英文，用于代码）
    display_name: str = "基础策略"         # 策略显示名（中文，用于展示）
    description: str = ""                  # 策略描述
    category: str = "technical"            # 分类: technical/fundamental/ml/combined
    author: str = ""                       # 策略作者

    # 数据需求声明（子类可选覆盖，用于运行前校验）
    required_columns: List[str] = ["close"]  # 最低要求的数据列
    min_bars: int = 60                       # 最少需要的K线数量

    def __init__(self, **kwargs):
        """
        初始化策略参数
        子类可通过 kwargs 接收自定义参数
        """
        self.params = kwargs
        self._validate_params()
        logger.info(
            "策略初始化: %s | 参数: %s",
            self.name,
            {k: v for k, v in kwargs.items() if not k.startswith("_")},
        )

    def _validate_params(self):
        """参数校验，子类可覆盖做更严格的检查"""
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        校验输入数据是否满足策略运行条件
        子类可覆盖增加额外检查
        """
        if df is None or df.empty:
            logger.warning("[%s] 输入数据为空", self.name)
            return False

        if len(df) < self.min_bars:
            logger.warning(
                "[%s] K线数量不足: %d < %d",
                self.name,
                len(df),
                self.min_bars,
            )
            return False

        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            logger.warning(
                "[%s] 缺少必要列: %s",
                self.name,
                missing,
            )
            return False

        return True

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        """
        生成交易信号（核心方法，子类必须实现）

        Args:
            df: DataFrame，至少包含 OHLCV 列
                - close: 收盘价（必需）
                - open/high/low: 可选
                - volume: 可选
                - 其他特征列: 视策略需求

        Returns:
            SignalResult 对象

        示例实现:
            def generate_signal(self, df):
                if not self.validate_data(df):
                    return SignalResult(self.name, triggered=False)

                latest = df.iloc[-1]
                # ... 你的策略逻辑 ...

                return SignalResult(
                    strategy_name=self.name,
                    triggered=True,
                    action="buy",
                    confidence=0.7,
                    price=latest['close'],
                    rationale="金叉信号触发",
                )
        """
        pass

    def get_info(self) -> Dict:
        """获取策略信息摘要"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "author": self.author,
            "params": self.params,
            "required_columns": self.required_columns,
            "min_bars": self.min_bars,
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} params={self.params}>"
