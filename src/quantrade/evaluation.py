"""Explicit metric definitions from NAV and actual traded notional."""

import numpy as np
import pandas as pd


def audit_ledger(nav, fills, cash_events, prices, initial_cash):
    """Independent cash/share reconstruction, deliberately not using Portfolio."""
    cash = initial_cash
    quantities, last_prices = {}, {}
    max_error = 0.0
    for row in nav.itertuples():
        for event in cash_events:
            if event["date"] != row.date:
                continue
            symbol = event["symbol"]
            dividend = quantities.get(symbol, 0) * event["dividend_per_share"]
            if abs(dividend - event["amount"]) > 1e-6:
                raise AssertionError("cash event attribution mismatch")
            cash += dividend
            quantities[symbol] = quantities.get(symbol, 0) * event["split"]
            if symbol in last_prices:
                last_prices[symbol] /= event["split"]
        for fill in fills:
            if fill["date"] != row.date:
                continue
            sign = 1 if fill["side"] == "buy" else -1
            cash -= sign * fill["quantity"] * fill["price"] + fill["fee"]
            quantities[fill["symbol"]] = quantities.get(fill["symbol"], 0) + sign * fill["quantity"]
            last_prices[fill["symbol"]] = fill["price"]
        bars = prices[prices.date == row.date]
        last_prices.update(dict(zip(bars.symbol, bars.close)))
        reconstructed = cash + sum(q * last_prices.get(s, 0) for s, q in quantities.items())
        error = max(abs(cash - row.cash), abs(reconstructed - row.nav))
        max_error = max(max_error, error)
        if error > 1e-5 or cash < -1e-7 or any(q < -1e-7 for q in quantities.values()):
            raise AssertionError(f"independent ledger audit failed on {row.date}")
    return {
        "passed": True,
        "days": len(nav),
        "fills": len(fills),
        "max_absolute_error": max_error,
        "method": "cash and quantities reconstructed directly from fills and actions",
    }


def evaluate(nav, fills, initial_cash, benchmark=None):
    if nav.empty:
        return {
            "validity": "insufficient",
            "total_return": 0.0,
            "n_days": 0,
            "limitations": ["empty period"],
        }
    if not np.isfinite(nav.nav).all() or (nav.nav <= 0).any():
        return {"validity": "invalid", "limitations": ["non-finite or nonpositive NAV"]}
    # Prepend starting capital so first-day fees are included in return/drawdown.
    values = np.r_[initial_cash, nav.nav.to_numpy()]
    returns = pd.Series(values[1:] / values[:-1] - 1, index=nav.date)
    dd = values / np.maximum.accumulate(values) - 1
    below = dd < -1e-12
    duration = longest = 0
    for value in below:
        duration = duration + 1 if value else 0
        longest = max(longest, duration)
    n = len(nav)
    total = values[-1] / initial_cash - 1
    std = float(returns.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else None
    notional = sum(f["quantity"] * f["price"] for f in fills)
    result = {
        "validity": "measured" if n >= 30 else "insufficient",
        "n_days": n,
        "total_return": float(total),
        "annual_return": float((1 + total) ** (252 / n) - 1),
        "max_drawdown": float(dd.min()),
        "max_drawdown_days": longest,
        "volatility": std * np.sqrt(252),
        "sharpe": sharpe,
        "fees": float(sum(f["fee"] for f in fills)),
        "fills": len(fills),
        "gross_notional": float(notional),
        "turnover_gross": float(notional / nav.nav.mean()),
        "average_cash_weight": float((nav.cash / nav.nav).mean()),
        "annualization": 252,
        "risk_free_rate": 0.0,
        "turnover_definition": "sum of buy and sell notional / mean NAV",
        "limitations": [
            "Metrics do not establish statistical significance or future profitability."
        ],
    }
    if benchmark is not None and not benchmark.empty:
        b = benchmark.set_index("date").nav
        bvalues = np.r_[initial_cash, b.to_numpy()]
        br = pd.Series(bvalues[1:] / bvalues[:-1] - 1, index=b.index)
        aligned = pd.concat([returns.rename("s"), br.rename("b")], axis=1).dropna()
        tracking = float((aligned.s - aligned.b).std()) if len(aligned) > 1 else 0.0
        result["benchmark_return"] = float(b.iloc[-1] / initial_cash - 1)
        result["excess_return"] = result["total_return"] - result["benchmark_return"]
        result["information_ratio"] = (
            float((aligned.s - aligned.b).mean() / tracking * np.sqrt(252))
            if tracking > 0
            else None
        )
        result["aligned_dates"] = len(aligned)
    year_results = {}
    for year, r in returns.groupby(returns.index.str[:4]):
        year_results[year] = float((1 + r).prod() - 1)
    result["year_returns"] = year_results
    return result
