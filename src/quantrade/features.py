"""Causal, per-instrument features; unknown future targets remain NaN."""

import numpy as np
import pandas as pd

FEATURE_VERSION = "etf-v3.0"
FEATURES = [
    "return_5",
    "return_20",
    "return_60",
    "volatility_20",
    "ma_distance",
    "volume_ratio",
    "intraday_range",
]


def compute_features(prices, horizons=(1, 5, 10, 20)):
    chunks = []
    for symbol, bars in prices.groupby("symbol", sort=True):
        g = bars.sort_values("date").copy()
        close = g.close
        for days in (5, 20, 60):
            g[f"return_{days}"] = close.pct_change(days, fill_method=None)
        g["volatility_20"] = close.pct_change(fill_method=None).rolling(20).std()
        g["ma_distance"] = close / close.rolling(20).mean() - 1
        g["volume_ratio"] = g.volume / g.volume.rolling(20).mean().replace(0, np.nan)
        g["intraday_range"] = (g.high - g.low) / close
        for horizon in horizons:
            g[f"target_{horizon}"] = close.shift(-horizon) / close - 1
            g[f"label_end_{horizon}"] = g.date.shift(-horizon)
        chunks.append(g)
    if not chunks:
        return pd.DataFrame(columns=["symbol", "date"] + FEATURES)
    return (
        pd.concat(chunks, ignore_index=True)
        .replace([np.inf, -np.inf], np.nan)
        .sort_values(["date", "symbol"])
    )
