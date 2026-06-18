"""
A股量化信号系统 - 策略工具包 (Strategy SDK)

使用方式:
    from strategies import StrategyRegistry, SignalResult
    from strategies.built_in import MACrossStrategy

    # 注册内置策略
    registry = StrategyRegistry()
    registry.register(MACrossStrategy(fast=5, slow=20))

    # 运行策略
    for name, strategy in registry.items():
        signal = strategy.generate_signal(df_kline)
        if signal.triggered:
            print(f"{name}: {signal.action} @ {signal.price}")
"""

from strategies.base import BaseStrategy, SignalResult
from strategies.registry import StrategyRegistry

__all__ = ["BaseStrategy", "SignalResult", "StrategyRegistry"]
