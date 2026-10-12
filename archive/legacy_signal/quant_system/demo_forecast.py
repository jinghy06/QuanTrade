"""
预测K线演示脚本
用模拟数据展示：历史K线 + ML多视野预测 + 走势分类 + 可视化
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import numpy as np
import pandas as pd

from trend_forecaster import TrendForecaster
from kline_plotter import KlinePlotter

logging.basicConfig(level=logging.INFO)


def generate_mock_history(n_days: int = 80) -> pd.DataFrame:
    """生成模拟历史K线数据"""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
    
    # 随机游走 + 轻微趋势
    returns = np.random.normal(loc=0.001, scale=0.02, size=n_days)
    close = 100.0 * np.exp(np.cumsum(returns))
    
    open_ = close * (1 + np.random.normal(0, 0.005, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.012, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.012, n_days)))
    
    df = pd.DataFrame({
        "trade_date": dates.strftime("%Y%m%d"),
        "symbol": "000001.SZ",
        "open": open_.round(2),
        "high": high.round(2),
        "low": low.round(2),
        "close": close.round(2),
        "volume": np.random.randint(1_000_000, 10_000_000, size=n_days),
        "amount": (np.random.randint(1_000_000, 10_000_000, size=n_days) * close).round(2),
        "amplitude": ((high - low) / low * 100).round(2),
        "pct_change": pd.Series(close / np.roll(close, 1) - 1).fillna(0).values * 100,
        "change": pd.Series(close - np.roll(close, 1)).fillna(0).round(2).values,
        "turnover": np.random.uniform(1.0, 5.0, size=n_days).round(2),
    })
    return df


def demo_forecast_kline():
    """演示：生成预测K线图"""
    print("=" * 60)
    print("预测K线演示")
    print("=" * 60)
    
    # 1. 模拟历史数据
    history = generate_mock_history(80)
    current_price = history["close"].iloc[-1]
    print(f"\n当前价格: {current_price:.2f} 元")
    
    # 2. 模拟ML锚点预测（实际应由MLTrainer.predict_trend输出）
    # 这里模拟一个"先抑后扬"的走势
    anchors = {
        1: current_price * 0.985,   # 第1天回调 -1.5%
        3: current_price * 0.995,   # 第3天回到-0.5%
        5: current_price * 1.025,     # 第5天反弹 +2.5%
        10: current_price * 1.055,  # 第10天加速 +5.5%
    }
    predicted_volatility = 0.018  # 1.8%日波动率
    
    print("\nML锚点预测:")
    for d, p in anchors.items():
        ret = (p - current_price) / current_price * 100
        print(f"  +{d}d: {p:.2f} 元 ({ret:+.2f}%)")
    
    # 3. 走势分类
    forecaster = TrendForecaster()
    trend_info = forecaster.classify_trend(anchors, current_price)
    print(f"\n走势判断: {trend_info['trend_type']}")
    print(f"描述: {trend_info['description']}")
    print(f"整体方向: {trend_info['overall_direction']}")
    
    # 4. 生成预测K线序列
    forecast_df = forecaster.generate_future_kline(
        current_price=current_price,
        anchors=anchors,
        predicted_volatility=predicted_volatility,
        n_days=10,
    )
    print(f"\n预测K线序列（未来10天）:")
    print(forecast_df[["day_offset", "open", "high", "low", "close"]].to_string(index=False))
    
    # 5. 画图
    plotter = KlinePlotter(save_dir="plots")
    save_path = plotter.plot_with_forecast(
        history_df=history,
        forecast_df=forecast_df,
        symbol="000001.SZ",
        trend_info=trend_info,
    )
    print(f"\n[OK] 预测K线图已保存: {save_path}")
    
    # 6. 模拟投资建议
    print("\n" + "=" * 60)
    print("投资建议（基于走势预测）")
    print("=" * 60)
    
    if trend_info["overall_direction"] == "up":
        entry = min(anchors[1], current_price * 0.99)
        target = anchors[10] * 1.02
        stop = current_price * 0.95
        print(f"  操作: 回调建仓")
        print(f"  入场区间: {entry:.2f} - {current_price:.2f} 元")
        print(f"  目标价: {target:.2f} 元")
        print(f"  止损价: {stop:.2f} 元")
        print(f"  建议持仓: 5-7天")
    elif trend_info["overall_direction"] == "down":
        print(f"  操作: 回避/减仓")
        print(f"  理由: 下跌趋势明确")
    else:
        print(f"  操作: 观望")
        print(f"  理由: 趋势不明，等待信号")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_forecast_kline()
