"""
自定义策略模板
复制此文件并重命名，然后实现你的策略逻辑

命名规范: xxx_strategy.py
存放位置: strategies/examples/ 或 strategies/custom/
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# 必须从策略基类继承
from strategies.base import BaseStrategy, SignalResult

logger = logging.getLogger(__name__)


class MyStrategy(BaseStrategy):
    """
    [策略名称]: [一句话描述策略核心逻辑]

    触发条件:
    - 买入: [描述买入触发条件]
    - 卖出: [描述卖出触发条件]

    参数说明:
    - param1: [参数1说明]
    - param2: [参数2说明]

    作者: [你的名字]
    日期: [创建日期]
    版本: 1.0
    """

    # ========== 策略元信息（必须填写） ==========
    name = "my_strategy"              # 英文标识（用于代码，必须唯一）
    display_name = "我的策略"          # 中文显示名
    description = "这是一个自定义策略模板"  # 策略描述
    category = "technical"             # 分类: technical/fundamental/ml/combined
    author = "你的名字"                 # 作者

    # ========== 数据需求（按需修改） ==========
    required_columns = ["close", "volume"]  # 策略需要的最低数据列
    min_bars = 30                            # 最少需要的K线数量

    def __init__(self, param1: int = 10, param2: float = 0.5):
        """
        初始化策略参数

        Args:
            param1: 参数1说明
            param2: 参数2说明
        """
        # 所有参数通过 super().__init__ 自动保存到 self.params
        super().__init__(param1=param1, param2=param2)
        self.param1 = param1
        self.param2 = param2

    def _validate_params(self):
        """
        参数校验（可选）
        在这里做更严格的参数检查
        """
        if self.param1 <= 0:
            raise ValueError("param1 必须大于0")
        if not (0 <= self.param2 <= 1):
            raise ValueError("param2 必须在0-1之间")

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        """
        生成交易信号（核心方法，必须实现）

        Args:
            df: DataFrame，包含至少 required_columns 中声明的列
                标准列名: open, high, low, close, volume, amount, turnover

        Returns:
            SignalResult 对象:
                - triggered: True/False 是否触发信号
                - action: "buy"/"sell"/"hold"/"light_buy"/"light_sell"
                - confidence: 0-1 置信度
                - price: 当前价格
                - target_price: 目标价（可选）
                - stop_loss: 止损价（可选）
                - position_pct: 建议仓位 0-1（可选）
                - rationale: 决策理由文字
                - risk_factors: 风险因素列表（可选）
                - metadata: 策略自定义数据字典（可选）
        """

        # Step 1: 数据校验（建议保留）
        if not self.validate_data(df):
            return SignalResult(self.name, triggered=False)

        # Step 2: 获取最新数据
        close = df["close"]
        latest_close = close.iloc[-1]
        prev_close = close.iloc[-2] if len(close) > 1 else latest_close

        # Step 3: 计算你的指标（在这里写策略逻辑）
        # 示例: 计算N日移动平均线
        ma = close.rolling(self.param1).mean()
        latest_ma = ma.iloc[-1]
        prev_ma = ma.iloc[-2] if len(ma.dropna()) > 1 else latest_ma

        # 示例指标计算（替换成你的逻辑）
        some_indicator = latest_close / latest_ma if latest_ma > 0 else 1.0

        # Step 4: 判断信号（在这里写触发条件）
        # 示例: 价格上穿均线 -> 买入
        if prev_close <= prev_ma < latest_close:
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="buy",
                confidence=round(min(some_indicator - 1, 1.0), 4),
                price=round(latest_close, 2),
                target_price=round(latest_close * 1.05, 2),
                stop_loss=round(latest_close * 0.93, 2),
                position_pct=0.05,
                rationale=f"价格上穿{self.param1}日均线，指标值={some_indicator:.3f}",
                risk_factors=["假突破风险"],
                metadata={
                    "indicator": round(some_indicator, 4),
                    "ma": round(latest_ma, 2),
                },
            )

        # 示例: 价格下穿均线 -> 卖出
        if prev_close >= prev_ma > latest_close:
            return SignalResult(
                strategy_name=self.name,
                triggered=True,
                action="sell",
                confidence=round(min(1 - some_indicator, 1.0), 4),
                price=round(latest_close, 2),
                rationale=f"价格下穿{self.param1}日均线，指标值={some_indicator:.3f}",
                metadata={
                    "indicator": round(some_indicator, 4),
                    "ma": round(latest_ma, 2),
                },
            )

        # Step 5: 未触发信号
        return SignalResult(
            strategy_name=self.name,
            triggered=False,
            rationale=f"指标值={some_indicator:.3f}，未达触发条件",
            metadata={
                "indicator": round(some_indicator, 4),
                "ma": round(latest_ma, 2),
            },
        )


# ========== 测试代码（开发时运行） ==========
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    # 创建模拟数据
    np.random.seed(42)
    n = 100
    prices = 10 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
    df = pd.DataFrame({
        "close": prices,
        "volume": np.random.randint(5000, 20000, n),
    })

    # 运行策略
    strategy = MyStrategy(param1=10, param2=0.5)
    signal = strategy.generate_signal(df)

    print(f"\n策略信息:")
    print(f"  名称: {strategy.display_name}")
    print(f"  参数: {strategy.params}")
    print(f"\n信号结果:")
    for k, v in signal.to_dict().items():
        print(f"  {k}: {v}")
