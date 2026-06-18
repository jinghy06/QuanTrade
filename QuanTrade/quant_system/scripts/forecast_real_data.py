"""
用真实历史数据生成预测K线图
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np

from data.data_store import DataStore
from trend_forecaster import TrendForecaster
from kline_plotter import KlinePlotter

logging.basicConfig(level=logging.INFO)


def generate_forecast_with_real_data(symbol: str = "000002.SZ"):
    """用真实历史数据生成预测K线"""
    store = DataStore()
    forecaster = TrendForecaster()
    plotter = KlinePlotter(save_dir="plots")

    # 1. 获取真实历史K线
    history = store.get_kline(symbol, n_days=80)
    if len(history) < 60:
        print(f"{symbol} 历史数据不足: {len(history)}条")
        return

    current_price = history["close"].iloc[-1]
    print(f"\n{'='*60}")
    print(f"股票: {symbol}")
    print(f"当前价格: {current_price:.2f}")
    print(f"历史数据: {len(history)}条 ({history['trade_date'].iloc[0]} ~ {history['trade_date'].iloc[-1]})")

    # 2. 模拟ML锚点预测（实际应由MLTrainer.predict_trend输出）
    # 基于近期趋势做简单外推模拟
    returns_5d = (history["close"].iloc[-1] - history["close"].iloc[-6]) / history["close"].iloc[-6]
    returns_20d = (history["close"].iloc[-1] - history["close"].iloc[-21]) / history["close"].iloc[-21]

    # 模拟预测：延续近期趋势，但逐步衰减
    anchors = {
        1: current_price * (1 + returns_5d * 0.3),      # 短期延续
        3: current_price * (1 + returns_5d * 0.5),      # 中期
        5: current_price * (1 + returns_20d * 0.3),     # 参考20日趋势
        10: current_price * (1 + returns_20d * 0.2),    # 长期衰减
    }

    # 确保价格合理（不超过±15%）
    for d in anchors:
        anchors[d] = max(current_price * 0.85, min(current_price * 1.15, anchors[d]))

    print(f"\n模拟锚点预测:")
    for d, p in sorted(anchors.items()):
        ret = (p - current_price) / current_price * 100
        print(f"  +{d}d: {p:.2f} ({ret:+.2f}%)")

    # 3. 走势分类
    trend_info = forecaster.classify_trend(anchors, current_price)
    print(f"\n走势判断: {trend_info['trend_type']}")
    print(f"描述: {trend_info['description']}")

    # 4. 生成预测K线
    # 用历史波动率估计
    hist_returns = history["close"].pct_change().dropna()
    predicted_vol = hist_returns.std() if len(hist_returns) > 0 else 0.02

    forecast_df = forecaster.generate_future_kline(
        current_price=current_price,
        anchors=anchors,
        predicted_volatility=predicted_vol,
        n_days=10,
    )

    print(f"\n预测K线序列:")
    print(forecast_df[["day_offset", "open", "high", "low", "close"]].to_string(index=False))

    # 5. 画图
    save_path = plotter.plot_with_forecast(
        history_df=history,
        forecast_df=forecast_df,
        symbol=symbol,
        trend_info=trend_info,
    )
    print(f"\n[OK] 预测K线图已保存: {save_path}")

    # 6. 投资建议
    print(f"\n{'='*60}")
    print("投资建议")
    print(f"{'='*60}")

    if trend_info["overall_direction"] == "up":
        entry = min(anchors[1], current_price * 0.99)
        target = max(anchors.values()) * 1.02
        stop = current_price * 0.95
        print(f"  操作: 回调建仓/持有")
        print(f"  入场区间: {entry:.2f} - {current_price:.2f}")
        print(f"  目标价: {target:.2f}")
        print(f"  止损价: {stop:.2f}")
        print(f"  建议持仓: 5-7天")
    elif trend_info["overall_direction"] == "down":
        print(f"  操作: 回避/减仓")
        print(f"  理由: {trend_info['description']}")
    else:
        print(f"  操作: 观望")
        print(f"  理由: {trend_info['description']}")

    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="000002.SZ")
    args = parser.parse_args()

    generate_forecast_with_real_data(args.symbol)
