"""Persisted accounts, advice, and human-readable run reports."""

import json
from pathlib import Path

import pandas as pd

from .domain import Fill
from .execution import Executor
from .features import compute_features
from .forecast import forecast_path
from .models import published_bundle
from .portfolio import Portfolio
from .strategy import ETFStrategy


def reconstruct_account(initial_cash, fills, instruments, actions, through_date):
    """Actual account valuations replay raw fills and actions, never double adjust."""
    portfolio = Portfolio(initial_cash, instruments)
    included = [f for f in fills if f["date"] <= through_date]
    eligible_actions = actions[actions.date <= through_date] if not actions.empty else actions
    dates = sorted(
        {f["date"] for f in included} | set(eligible_actions.date if len(eligible_actions) else [])
    )
    for date in dates:
        if not eligible_actions.empty:
            for row in eligible_actions[eligible_actions.date == date].itertuples():
                portfolio.corporate_action(row.event_id, date, row.symbol, row.split, row.dividend)
        for record in included:
            if record["date"] == date:
                portfolio.apply_fill(Fill(**record))
    return portfolio


def import_fills(store, path, account_id, initial_cash, instruments):
    data = store.get("account", account_id)
    if data is None:
        raise ValueError("initialize the account with starting cash first")
    if data["mode"] != "actual":
        raise ValueError("actual fills require an actual account")
    portfolio = Portfolio.from_dict(data["portfolio"], instruments)
    fills = pd.read_csv(path, dtype={"symbol": str, "fill_id": str, "order_id": str})
    needed = {"fill_id", "order_id", "date", "symbol", "side", "quantity", "price", "fee"}
    if not needed.issubset(fills):
        raise ValueError(f"missing fill columns {sorted(needed - set(fills))}")
    count = 0
    last_date = max((f["date"] for f in portfolio.fills), default="")
    # Process file order within each date. Sell-before-buy must be intentional.
    for row in fills.sort_values("date", kind="stable").to_dict("records"):
        row = {key: row[key] for key in needed}
        row["date"] = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        if row["symbol"] not in instruments:
            raise ValueError("unknown instrument")
        old = next((f for f in portfolio.fills if f["fill_id"] == row["fill_id"]), None)
        if old:
            if any(old[k] != row[k] for k in needed):
                raise ValueError("conflicting fill id; use a reviewed correction")
            continue
        if row["date"] < last_date:
            raise ValueError("backdated fills require a full ordered replay")
        # Rebuild the source ledger through this execution day so splits and
        # paid dividends precede its fills. Nothing is persisted mid-batch.
        portfolio = reconstruct_account(
            portfolio.initial_cash,
            portfolio.fills,
            instruments,
            store.frame("corporate_actions"),
            row["date"],
        )
        count += portfolio.apply_fill(Fill(**row, source="actual"))
        last_date = row["date"]
    # Only persist after the entire import succeeds (atomic batch).
    data["portfolio"] = portfolio.to_dict()
    store.put("account", account_id, data)
    return {"account": account_id, "imported": count}


def generate_forecast(prices, date, bundle, symbol=None):
    if bundle is None:
        return {"status": "unavailable", "reason": "no published model"}
    if not bundle.can_predict(date):
        return {
            "status": "unavailable",
            "reason": "model_unreleased_stale_or_not_valid_at_requested_date",
        }
    history = prices[prices.date <= date]
    features = compute_features(history)
    latest = features[features.date == date]
    if symbol:
        latest = latest[latest.symbol == symbol]
    if latest.empty:
        return {"status": "unavailable", "reason": "no current price"}
    prediction = bundle.predict(latest, date)
    forecasts = {}
    for row in latest.itertuples():
        anchors = {h: values[row.symbol] for h, values in prediction.items()}
        errors = {h: bundle.metadata["horizons"][str(h)]["validation"]["rmse"] for h in anchors}
        widths = {h: bundle.metadata["horizons"][str(h)]["interval_width_90"] for h in anchors}
        forecasts[row.symbol] = forecast_path(
            row.close, anchors, errors, date, bundle.metadata["seed"], widths
        )
    return {
        "status": "generated",
        "model_version": bundle.metadata["version"],
        "as_of": date,
        "forecasts": forecasts,
    }


def account_report(store, config, account_id, date):
    account = store.get("account", account_id)
    if account is None:
        raise ValueError("account missing; initialize actual or simulation account explicitly")
    instruments = store.instruments()
    if any(f["date"] > date for f in account["portfolio"]["fills"]):
        raise ValueError("account contains future fills for requested date")
    portfolio = reconstruct_account(
        account["portfolio"]["initial_cash"],
        account["portfolio"]["fills"],
        instruments,
        store.frame("corporate_actions"),
        date,
    )
    if any(f["date"] > date for f in portfolio.fills):
        raise ValueError("account contains future fills for requested date")
    prices = store.prices()
    history = prices[prices.date <= date]
    bars = history[history.date == date]
    if bars.empty:
        raise ValueError("no quotes on requested date")
    for row in history.sort_values(["symbol", "date"]).groupby("symbol").tail(1).itertuples():
        price = row.close
        actions = store.frame("corporate_actions")
        if len(actions):
            subsequent = actions[
                (actions.symbol == row.symbol) & (actions.date > row.date) & (actions.date <= date)
            ]
            for factor in subsequent.split:
                price /= factor
        portfolio.mark(row.date, {row.symbol: price})
    bundle = published_bundle(config["paths"]["models"])
    # Reports are read-only. Persisted decision state ensures repeated reports do
    # not invent a new initial rebalance on each day.
    strategy = ETFStrategy(config, instruments)
    state = account.get("strategy_state", {})
    with store.connect() as c:
        saved = c.execute(
            "SELECT payload FROM artifacts WHERE kind='decision_state' AND artifact_id=?",
            (account_id,),
        ).fetchone()
    if saved:
        checkpoint = json.loads(saved[0])
        if checkpoint["date"] > date:
            raise ValueError("historical reports require a historical decision/account snapshot")
        state = (
            checkpoint.get("before_state", {})
            if checkpoint["date"] == date
            else checkpoint["state"]
        )
    before_state = dict(state)
    for key in ("initialized", "risk_active", "last_risk_date"):
        if key in state:
            setattr(strategy, key, state[key])
    future_sessions = sorted(prices[prices.date > date].date.unique())
    next_date = (
        future_sessions[0]
        if future_sessions
        else (pd.Timestamp(date) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    )
    target = strategy.decide(history, date, next_date, portfolio, store.frame("events"), bundle)
    report = {
        "account": account_id,
        "mode": account["mode"],
        "as_of": date,
        "status": "ready" if account.get("complete", False) else "incomplete_account",
        "valuation": portfolio.snapshot(date),
        "target": target.to_dict() if target else None,
        "orders": [],
        "forecast": generate_forecast(prices, date, bundle),
        "notes": [
            "Advice is not a fill. Next execution session must be confirmed against the exchange calendar."
        ],
    }
    report["decision_state"] = {
        key: getattr(strategy, key) for key in ("initialized", "risk_active", "last_risk_date")
    }
    report["before_decision_state"] = before_state
    if target and account.get("complete", False):
        estimate = Portfolio.from_dict(portfolio.to_dict(), instruments)
        # Next open is unknown: today's close is only an order cost estimate.
        report["orders"] = Executor(config, instruments).execute(
            estimate, target, bars, next_date, account="advice", price_column="close"
        )
        for order in report["orders"]:
            order["estimated_status"] = order.pop("status")
            order["status"] = "suggested"
        report["notes"].append(
            "Quantities and fees use today's close as an estimate; confirm at execution."
        )
    return report


def markdown_run(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    lines = [
        "# QuanTrade v3.0 回测报告",
        "",
        f"实验：`{manifest['run_id']}`",
        "",
        f"数据状态：**{manifest['performance_status']}**",
        "",
        "| 策略 | 总收益 | 最大回撤 | 费用 | 成交数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, metrics in manifest["metrics"].items():
        lines.append(
            f"| {mode} | {metrics.get('total_return', 0):.2%} | {metrics.get('max_drawdown', 0):.2%} | {metrics.get('fees', 0):.2f} | {metrics.get('fills', 0)} |"
        )
    lines += [
        "",
        "收益包含建仓费用。旧数据口径、交易规则及未解释异常未核验时仅用于探索。",
        "预测 K 线中的插值与模拟影线不能用于认定实际成交。程序验证与策略实盘准入分别判断。",
        "",
    ]
    path = folder / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
