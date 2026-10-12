"""Close decision -> next session open -> close valuation, one share ledger."""

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from .evaluation import audit_ledger, evaluate
from .execution import Executor
from .models import ModelBundle, fit_bundle
from .portfolio import Portfolio
from .storage import write_json
from .strategy import ETFStrategy


def replay(
    prices,
    instruments,
    config,
    mode="full",
    start=None,
    end=None,
    events=None,
    actions=None,
    bundle=None,
    walk_forward=False,
    snapshot_id="unspecified",
):
    if prices.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate market dates")
    prices = prices.sort_values(["date", "symbol"]).copy()
    if end:
        prices = prices[prices.date <= end]
    dates = sorted(prices.date.unique())
    if start:
        dates = [d for d in dates if d >= start]
    portfolio = Portfolio(config["execution"]["initial_cash"], instruments)
    executor = Executor(config, instruments)
    strategy = ETFStrategy(config, instruments, mode)
    pending = None
    rows, positions, orders, targets, predictions, windows = [], [], [], [], [], []
    actions = actions if actions is not None else pd.DataFrame()
    active_bundle = bundle
    for index, date in enumerate(dates):
        bars = prices[prices.date == date]
        # Action dividend field represents cash actually paid on this date.
        if not actions.empty:
            for row in actions[actions.date == date].itertuples():
                portfolio.corporate_action(row.event_id, date, row.symbol, row.split, row.dividend)
        if pending is not None:
            orders.extend(executor.execute(portfolio, pending, bars, date))
            pending = None
        portfolio.mark(date, dict(zip(bars.symbol, bars.close)))
        snap = portfolio.snapshot(date)
        rows.append({k: v for k, v in snap.items() if k != "positions"})
        positions.extend([dict(date=date, symbol=s, **v) for s, v in snap["positions"].items()])
        if index == len(dates) - 1:
            continue
        history = prices[prices.date <= date]
        if (
            walk_forward
            and mode in ("model", "full")
            and strategy.scheduled(date, dates[index + 1])
        ):
            past = sorted(history.date.unique())
            if len(past) > 1:
                try:
                    manifest = fit_bundle(
                        history, config, past[-2], config["paths"]["models"], snapshot_id
                    )
                    active_bundle = ModelBundle(
                        Path(config["paths"]["models"]) / manifest["version"]
                    )
                    windows.append(
                        dict(
                            prediction_date=date,
                            **{
                                k: manifest[k]
                                for k in (
                                    "version",
                                    "trained_through",
                                    "train_end",
                                    "validation_end",
                                    "test_end",
                                )
                            },
                        )
                    )
                except ValueError as exc:
                    windows.append(
                        {
                            "prediction_date": date,
                            "status": "insufficient_training_data",
                            "reason": str(exc),
                        }
                    )
        target = strategy.decide(history, date, dates[index + 1], portfolio, events, active_bundle)
        if target:
            targets.append(target.to_dict())
            pending = target
            if target.model_version:
                from .features import compute_features

                latest = compute_features(history)
                latest = latest[latest.date == date]
                result = active_bundle.predict(latest, date)
                predictions.extend(
                    {
                        "date": date,
                        "symbol": s,
                        "horizon": h,
                        "predicted_return": v,
                        "model_version": target.model_version,
                    }
                    for h, values in result.items()
                    for s, v in values.items()
                )
    nav = pd.DataFrame(
        rows, columns=["date", "cash", "nav", "fees", "realized_pnl", "unrealized_pnl", "dividends"]
    )
    # Reconcile NAV attribution including capitalized buy and realized sell costs.
    if len(nav):
        reconstructed = (
            portfolio.initial_cash + nav.realized_pnl + nav.unrealized_pnl + nav.dividends
        )
        if (reconstructed - nav.nav).abs().max() > 1e-5:
            raise AssertionError("NAV / P&L reconciliation failed")
    audit = audit_ledger(
        nav, portfolio.fills, portfolio.cash_events, prices, portfolio.initial_cash
    )
    return {
        "mode": mode,
        "nav": nav,
        "positions": positions,
        "orders": orders,
        "targets": targets,
        "predictions": predictions,
        "windows": windows,
        "fills": portfolio.fills,
        "cash_events": portfolio.cash_events,
        "account": portfolio.to_dict(),
        "audit": audit,
        "metrics": evaluate(nav, portfolio.fills, portfolio.initial_cash),
    }


def compare(
    prices,
    instruments,
    config,
    start=None,
    end=None,
    events=None,
    actions=None,
    bundle=None,
    walk_forward=False,
    snapshot_id="unspecified",
):
    results = {}
    modes = ("buy_hold", "fixed", "momentum", "model", "full")
    for mode in modes:
        mode_config = config
        if mode == "buy_hold":
            from copy import deepcopy

            mode_config = deepcopy(config)
            # Preserve a fully invested benchmark alongside constrained strategies.
            mode_config["strategy"].update(cash_reserve=0.0, max_weight=1.0, max_theme_weight=1.0)
        results[mode] = replay(
            prices,
            instruments,
            mode_config,
            mode,
            start,
            end,
            events,
            actions,
            bundle,
            walk_forward,
            snapshot_id,
        )
    benchmark = results["buy_hold"]["nav"]
    for result in results.values():
        result["metrics"] = evaluate(
            result["nav"], result["fills"], config["execution"]["initial_cash"], benchmark
        )
    return results


def persist_run(results, config, snapshot_id, validation, directory, provenance=None):
    identity = {
        "config": config,
        "snapshot_id": snapshot_id,
        "provenance": provenance or {},
        "modes": list(results),
        "metrics": {m: r["metrics"] for m, r in results.items()},
    }
    run_id = (
        "v3-"
        + sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    )
    folder = Path(directory) / run_id
    if (folder / "manifest.json").exists():
        return run_id, folder
    folder.mkdir(parents=True, exist_ok=True)
    for mode, result in results.items():
        output = folder / mode
        output.mkdir(exist_ok=True)
        result["nav"].to_csv(output / "nav.csv", index=False)
        for key in (
            "positions",
            "orders",
            "targets",
            "predictions",
            "windows",
            "fills",
            "cash_events",
            "account",
            "metrics",
            "audit",
        ):
            write_json(output / f"{key}.json", result[key])
    write_json(
        folder / "manifest.json",
        dict(
            run_id=run_id,
            **identity,
            validation=validation,
            performance_status="validated_data" if validation["valid"] else "exploratory",
            note="Engineering/data checks and profitable strategy admission are separate.",
        ),
    )
    return run_id, folder
