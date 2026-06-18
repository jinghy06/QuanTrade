"""
预测K线生成器
基于多锚点回归预测，生成未来N天的预测OHLC序列
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


class TrendForecaster:
    """
    基于ML锚点预测生成未来K线序列
    
    输入：
        - current_price: 当前收盘价
        - anchors: {1: price_1d, 3: price_3d, 5: price_5d, 10: price_10d}
        - predicted_volatility: 预测波动率（用于生成上下影线）
        - history_df: 历史K线DataFrame（用于画图）
    
    输出：
        - future_df: 未来N天的预测OHLC DataFrame
        - 可视化图表
    """
    
    HORIZONS = [1, 3, 5, 10]
    
    def __init__(self):
        pass
    
    def generate_future_kline(
        self,
        current_price: float,
        anchors: Dict[int, float],
        predicted_volatility: float,
        n_days: int = 10,
    ) -> pd.DataFrame:
        """
        生成未来N天预测K线
        
        方法：
        1. 用锚点做三次样条插值生成平滑收盘价序列
        2. 基于预测波动率生成High/Low区间（±1.5 * ATR估计）
        3. Open = 前一日Close（首日Open=current_price）
        """
        # 构建锚点：今天(0) + 预测锚点
        anchor_days = [0] + sorted(anchors.keys())
        anchor_prices = [current_price] + [anchors[d] for d in sorted(anchors.keys())]
        
        # 如果n_days > 最大锚点，外推最后一段趋势
        max_anchor_day = max(anchor_days)
        if n_days > max_anchor_day:
            # 计算最后两个锚点的斜率，线性外推
            last_slope = (anchor_prices[-1] - anchor_prices[-2]) / (anchor_days[-1] - anchor_days[-2])
            for d in range(max_anchor_day + 1, n_days + 1):
                anchor_days.append(d)
                anchor_prices.append(anchor_prices[-1] + last_slope)
        
        # 三次样条插值（需要至少4个点，否则用线性）
        future_days = np.arange(1, n_days + 1)
        if len(anchor_days) >= 4:
            cs = CubicSpline(anchor_days, anchor_prices)
            future_close = cs(future_days)
        else:
            future_close = np.interp(future_days, anchor_days, anchor_prices)
        
        # 确保价格单调性不突变（简单平滑）
        future_close = pd.Series(future_close).ewm(span=3).mean().values
        
        # 生成OHLC
        future_records = []
        prev_close = current_price
        
        # ATR估计：基于预测波动率
        atr_estimate = current_price * predicted_volatility * 1.5  # 1.5倍系数让影线更真实
        
        for i, day in enumerate(future_days):
            close = float(future_close[i])
            # Open ≈ 前一日Close，加小幅随机偏移
            open_price = prev_close * (1 + np.random.normal(0, predicted_volatility * 0.3))
            
            # High/Low = Close ± ATR估计，确保High >= max(Open, Close), Low <= min(Open, Close)
            high = max(open_price, close) + np.random.uniform(0, atr_estimate * 0.5)
            low = min(open_price, close) - np.random.uniform(0, atr_estimate * 0.5)
            
            # 确保合理性
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            future_records.append({
                "day_offset": int(day),
                "open": round(float(open_price), 2),
                "high": round(float(high), 2),
                "low": round(float(low), 2),
                "close": round(float(close), 2),
                "is_predicted": True,
            })
            prev_close = close
        
        return pd.DataFrame(future_records)
    
    def classify_trend(self, anchors: Dict[int, float], current_price: float) -> Dict:
        """
        基于多锚点判断走势类型
        """
        returns = {d: (p - current_price) / current_price for d, p in anchors.items()}
        
        r1 = returns.get(1, 0)
        r3 = returns.get(3, 0)
        r5 = returns.get(5, 0)
        r10 = returns.get(10, 0)
        
        # 走势分类逻辑
        if r1 > 0 and r3 > r1 and r5 > r3:
            trend = "强势上涨"
            desc = "短期加速上涨，趋势确立"
        elif r1 < 0 and r3 > r1 and r5 > 0:
            trend = "先抑后扬"
            desc = "短期回调后反弹，中期向好"
        elif r1 > 0 and r3 < r1 and r5 < 0:
            trend = "冲高回落"
            desc = "短期冲高后回落，注意止盈"
        elif abs(r5) < 0.02 and abs(r10) < 0.03:
            trend = "震荡整理"
            desc = "上下空间有限，观望为主"
        elif r1 < 0 and r3 < r1 and r5 < r3:
            trend = "持续下跌"
            desc = "下跌趋势明确，回避为主"
        elif r5 > 0 and r10 < r5:
            trend = "反弹遇阻"
            desc = "中期反弹但长期承压"
        else:
            trend = "趋势不明"
            desc = "信号混杂，等待明朗"
        
        return {
            "trend_type": trend,
            "description": desc,
            "returns": {f"{k}d": round(v * 100, 2) for k, v in returns.items()},
            "overall_direction": "up" if r5 > 0 else "down" if r5 < 0 else "neutral",
        }
