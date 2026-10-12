"""Model anchors and illustrative interpolated candles; never execution prices."""

import numpy as np
import pandas as pd


def forecast_path(close, anchors, residuals, start_date, seed=42, interval_widths=None):
    if close <= 0 or not anchors:
        raise ValueError("positive close and prediction anchors required")
    days = sorted(int(h) for h in anchors)
    if min(days) < 1:
        raise ValueError("positive prediction horizons required")
    prices = [max(close * (1 + anchors[h]), close * 0.01) for h in days]
    grid = np.arange(1, max(days) + 1)
    path = np.interp(grid, [0] + days, [close] + prices)
    sigma = np.interp(grid, [0] + days, [0.0] + [residuals[h] for h in days])
    widths = np.interp(
        grid,
        [0] + days,
        [0.0] + [interval_widths[h] if interval_widths else 1.645 * residuals[h] for h in days],
    )
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(pd.Timestamp(start_date) + pd.offsets.BDay(1), periods=len(grid))
    previous = close
    rows = []
    for day, date, value, spread, width in zip(grid, dates, path, sigma, widths):
        open_price = previous
        simulated_wick = float(rng.uniform(0, max(spread * close, 0.0)))
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "horizon": int(day),
                "open": open_price,
                "close": float(value),
                "high": max(open_price, value) + simulated_wick,
                "low": max(0.0, min(open_price, value) - simulated_wick),
                "lower": max(0.0, float(value - width * close)),
                "upper": float(value + width * close),
                "kind": "model_anchor" if day in days else "interpolated_scenario",
            }
        )
        previous = float(value)
    return {
        "candles": rows,
        "seed": seed,
        "note": "OHLC wicks are illustrative simulations, not trained OHLC predictions. "
        + (
            "Anchor bounds use the validation 90% absolute-error quantile; future coverage is not guaranteed. "
            if interval_widths
            else "Bounds use validation RMSE, not calibrated coverage. "
        )
        + "Dates are weekdays, not an exchange holiday calendar.",
    }
