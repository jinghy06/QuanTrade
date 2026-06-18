"""
K线可视化：历史K线 + 预测K线
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class KlinePlotter:
    """画出历史K线 + 预测K线 + 置信区间"""
    
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_with_forecast(
        self,
        history_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        symbol: str,
        trend_info: Dict,
        save_path: Optional[str] = None,
    ) -> str:
        """
        画出历史K线 + 预测K线
        
        Args:
            history_df: 历史K线，columns=[trade_date, open, high, low, close]
            forecast_df: 预测K线，columns=[day_offset, open, high, low, close, is_predicted]
            symbol: 股票代码
            trend_info: TrendForecaster.classify_trend 的输出
        
        Returns:
            保存的图片路径
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # --- 准备历史数据 ---
        hist = history_df.tail(60).copy()  # 最近60天
        hist["trade_date"] = pd.to_datetime(hist["trade_date"])
        
        # --- 准备预测数据日期 ---
        last_date = hist["trade_date"].iloc[-1]
        forecast_df = forecast_df.copy()
        forecast_df["date"] = [last_date + timedelta(days=int(d)) for d in forecast_df["day_offset"]]
        
        # --- 画历史K线（蜡烛图简化版：用线段） ---
        for _, row in hist.iterrows():
            color = "red" if row["close"] >= row["open"] else "green"
            # 实体
            ax.plot([row["trade_date"], row["trade_date"]], 
                   [row["low"], row["high"]], color=color, linewidth=1, alpha=0.7)
            ax.plot([row["trade_date"], row["trade_date"]], 
                   [row["open"], row["close"]], color=color, linewidth=3, solid_capstyle="butt")
        
        # --- 画预测K线（蓝色虚线框） ---
        for _, row in forecast_df.iterrows():
            color = "#3498db"  # 蓝色
            # 影线
            ax.plot([row["date"], row["date"]], 
                   [row["low"], row["high"]], color=color, linewidth=1, alpha=0.6, linestyle="--")
            # 实体（空心框）
            ax.plot([row["date"], row["date"]], 
                   [row["open"], row["close"]], color=color, linewidth=3, 
                   linestyle="--", solid_capstyle="butt")
            # 画小横线标记open/close
            ax.plot([row["date"] - timedelta(hours=8), row["date"]], 
                   [row["open"], row["open"]], color=color, linewidth=1, linestyle="--")
            ax.plot([row["date"], row["date"] + timedelta(hours=8)], 
                   [row["close"], row["close"]], color=color, linewidth=1, linestyle="--")
        
        # --- 连接线（历史最后一日 → 预测第一日） ---
        last_hist = hist.iloc[-1]
        first_fc = forecast_df.iloc[0]
        ax.plot([last_hist["trade_date"], first_fc["date"]], 
               [last_hist["close"], first_fc["open"]], 
               color="gray", linewidth=1, linestyle=":", alpha=0.5)
        
        # --- 置信区间阴影（High/Low包络线） ---
        fc_dates = forecast_df["date"].values
        ax.fill_between(fc_dates, forecast_df["low"], forecast_df["high"], 
                       alpha=0.1, color="blue", label="预测区间")
        
        # --- 锚点标记 ---
        for _, row in forecast_df.iterrows():
            if int(row["day_offset"]) in [1, 3, 5, 10]:
                ax.scatter(row["date"], row["close"], color="blue", s=50, zorder=5)
                ax.annotate(f"{int(row['day_offset'])}d", 
                           (row["date"], row["close"]),
                           textcoords="offset points", xytext=(0, 10),
                           ha="center", fontsize=8, color="blue")
        
        # --- 标题和标签 ---
        trend_type = trend_info.get("trend_type", "未知")
        desc = trend_info.get("description", "")
        returns = trend_info.get("returns", {})
        ret_str = " | ".join([f"{k}: {v:+.1f}%" for k, v in returns.items()])
        
        ax.set_title(f"{symbol} - {trend_type}\n{desc}\n{ret_str}", fontsize=12)
        ax.set_xlabel("日期")
        ax.set_ylabel("价格")
        
        # 添加图例说明
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="red", linewidth=3, label="历史上涨"),
            Line2D([0], [0], color="green", linewidth=3, label="历史下跌"),
            Line2D([0], [0], color="#3498db", linewidth=3, linestyle="--", label="预测K线"),
            Patch(facecolor="blue", alpha=0.1, label="预测区间"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")
        
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        
        # --- 保存 ---
        if save_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.save_dir / f"{symbol}_{ts}_forecast.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info("预测K线图已保存: %s", save_path)
        return str(save_path)
