"""Single supported command entry; importing this module performs no work."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .backtest import compare, persist_run, replay
from .config import ROOT, load_config
from .data import (
    import_instruments,
    inspect_prices,
    migrate_legacy,
    normalize_prices,
    record_issues,
    save_prices,
    snapshot,
    update_etfs,
    validate_store,
)
from .factors import import_events
from .models import ModelBundle, fit_bundle, publish_model, published_bundle, validate_model
from .portfolio import Portfolio
from .reporting import account_report, generate_forecast, import_fills, markdown_run
from .storage import Store, json_default, write_json


def provenance():
    def git(*args):
        r = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else "unavailable"

    diff = git("diff", "--binary", "HEAD")
    source = hashlib.sha256()
    for path in sorted((ROOT / "src/quantrade").glob("*.py")):
        source.update(path.name.encode() + path.read_bytes())
    dependencies = {}
    for name in ("pandas", "numpy", "lightgbm", "scikit-learn"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "unavailable"
    return {
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain"),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "package_sha256": source.hexdigest(),
        "python": sys.version,
        "dependencies": dependencies,
    }


def parser():
    p = argparse.ArgumentParser(description="QuanTrade 3.0 ETF 中低频交易系统")
    p.add_argument("--config", help="Configuration TOML path")
    groups = p.add_subparsers(dest="command", required=True)
    data = groups.add_parser("data").add_subparsers(dest="action", required=True)
    migrate = data.add_parser("migrate")
    migrate.add_argument("--source")
    data.add_parser("validate")
    data.add_parser("snapshot")
    instruments = data.add_parser("import-instruments")
    instruments.add_argument("--file", required=True)
    prices = data.add_parser("import-prices")
    prices.add_argument("--file", required=True)
    prices.add_argument("--basis", choices=["raw", "research", "unknown"], default="unknown")
    events = data.add_parser("import-events")
    events.add_argument("--file", required=True)
    actions = data.add_parser("import-actions")
    actions.add_argument("--file", required=True)
    resolve = data.add_parser("resolve-issue")
    resolve.add_argument("--id", required=True)
    resolve.add_argument("--explanation", required=True)
    update = data.add_parser("update")
    update.add_argument("--symbols", nargs="+")
    update.add_argument("--start", required=True)
    update.add_argument("--end", required=True)
    models = groups.add_parser("model").add_subparsers(dest="action", required=True)
    train = models.add_parser("train")
    train.add_argument("--cutoff", required=True)
    for action in ("validate", "publish"):
        sub = models.add_parser(action)
        sub.add_argument("--version", required=True)
    bt = groups.add_parser("backtest")
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--walk-forward", action="store_true")
    bt.add_argument("--allow-unverified", action="store_true")
    bt.add_argument(
        "--cost-levels",
        nargs="+",
        type=float,
        help="Rerun full strategy with these one-way commission rates",
    )
    forecast = groups.add_parser("forecast")
    forecast.add_argument("--date", required=True)
    forecast.add_argument("--symbol")
    forecast.add_argument("--model-version", help="Explicit historical research model")
    forecast.add_argument(
        "--research",
        action="store_true",
        help="Use an unpublished, strictly earlier historical model",
    )
    accounts = groups.add_parser("account").add_subparsers(dest="action", required=True)
    initialize = accounts.add_parser("init")
    initialize.add_argument("--id", default="default")
    initialize.add_argument("--cash", type=float, required=True)
    initialize.add_argument("--mode", choices=["actual", "simulation"], required=True)
    initialize.add_argument(
        "--complete",
        action="store_true",
        help="Confirm all-cash starting state or complete replay basis",
    )
    fills = accounts.add_parser("import-fills")
    fills.add_argument("--file", required=True)
    fills.add_argument("--id", default="default")
    report = groups.add_parser("report")
    report.add_argument("--date", required=True)
    report.add_argument("--account", default="default")
    groups.add_parser("dashboard")
    return p


def run(args, config):
    store = Store(config["paths"]["database"])
    if args.command == "data":
        if args.action == "migrate":
            return migrate_legacy(
                store,
                args.source or config["paths"]["legacy_database"],
                ROOT / "data/migration",
                config["data"]["jump_threshold"],
            )
        if args.action == "validate":
            return validate_store(store, config["data"]["jump_threshold"], config["data"])
        if args.action == "snapshot":
            validation = validate_store(store, config["data"]["jump_threshold"], config["data"])
            digest, path = snapshot(store, ROOT / "data/snapshots", validation)
            return dict(snapshot_id=digest, folder=str(path), **validation)
        if args.action == "import-instruments":
            import_instruments(store, args.file)
            return {"instruments": len(store.instruments())}
        if args.action == "import-prices":
            raw = pd.read_csv(args.file, dtype={"symbol": str})
            df = normalize_prices(raw, source="csv_import", basis=args.basis)
            issues = inspect_prices(df, config["data"]["jump_threshold"])
            record_issues(store, issues)
            if any(i["kind"] not in ("duplicate_identical", "unexplained_jump") for i in issues):
                raise ValueError("malformed or conflicting prices rejected")
            save_prices(store, df)
            return {"rows": len(df), "issues": issues}
        if args.action == "import-events":
            return import_events(store, args.file)
        if args.action == "import-actions":
            actions = pd.read_csv(args.file, dtype={"symbol": str, "event_id": str})
            if not {"event_id", "date", "symbol", "split", "dividend"}.issubset(actions):
                raise ValueError("invalid action fields")
            with store.connect() as c:
                for row in actions[["event_id", "date", "symbol", "split", "dividend"]].itertuples(
                    index=False, name=None
                ):
                    event_id, date, symbol, split, dividend = row
                    if (
                        pd.isna(event_id)
                        or not event_id
                        or not math.isfinite(split)
                        or not math.isfinite(dividend)
                        or split <= 0
                        or dividend < 0
                    ):
                        raise ValueError("invalid corporate action")
                    date = pd.Timestamp(date).strftime("%Y-%m-%d")
                    if symbol not in store.instruments():
                        raise ValueError("corporate action for unknown instrument")
                    row = (event_id, date, symbol, split, dividend)
                    old = c.execute(
                        "SELECT * FROM corporate_actions WHERE event_id=?", (event_id,)
                    ).fetchone()
                    if old and old != row:
                        raise ValueError("conflicting corporate action id")
                    c.execute("INSERT OR IGNORE INTO corporate_actions VALUES(?,?,?,?,?)", row)
            return {"actions": len(actions)}
        if args.action == "resolve-issue":
            if not args.explanation.strip():
                raise ValueError("resolution requires evidence/explanation")
            with store.connect() as c:
                if not c.execute(
                    "SELECT 1 FROM data_issues WHERE issue_id=?", (args.id,)
                ).fetchone():
                    raise ValueError("unknown issue id")
                c.execute("UPDATE data_issues SET resolved=1 WHERE issue_id=?", (args.id,))
            store.put(
                "issue_resolution",
                args.id,
                {"explanation": args.explanation, "resolved_at": datetime.now(UTC).isoformat()},
            )
            return {"resolved": args.id}
        if args.action == "update":
            symbols = args.symbols or list(store.instruments())
            if not symbols:
                raise ValueError("import instruments or supply --symbols first")
            return update_etfs(store, symbols, args.start, args.end)
    if args.command == "model":
        if args.action == "train":
            validation = validate_store(store, config["data"]["jump_threshold"], config["data"])
            digest, _ = snapshot(store, ROOT / "data/snapshots", validation)
            manifest = fit_bundle(
                store.prices(), config, args.cutoff, config["paths"]["models"], digest
            )
            manifest["data_validity"] = validation["validity"]
            write_json(
                Path(config["paths"]["models"]) / manifest["version"] / "manifest.json", manifest
            )
            return manifest
        if Path(args.version).name != args.version:
            raise ValueError("version must be a model id, not a path")
        if args.action == "validate":
            return validate_model(Path(config["paths"]["models"]) / args.version)
        if args.action == "publish":
            return publish_model(config["paths"]["models"], args.version)
    if args.command == "backtest":
        validation = validate_store(store, config["data"]["jump_threshold"], config["data"])
        if not validation["valid"] and not args.allow_unverified:
            raise ValueError(
                "data not admitted; review data validate, or explicitly use --allow-unverified for exploratory research"
            )
        digest, _ = snapshot(store, ROOT / "data/snapshots", validation)
        prices, instruments = store.prices(), store.instruments()
        events, actions = store.frame("events"), store.frame("corporate_actions")
        results = compare(
            prices,
            instruments,
            config,
            args.start,
            args.end,
            events,
            actions,
            published_bundle(config["paths"]["models"]),
            args.walk_forward,
            digest,
        )
        if args.cost_levels:
            sensitivity = {}
            for cost in args.cost_levels:
                if cost < 0:
                    raise ValueError("negative cost")
                alternative = deepcopy(config)
                alternative["execution"]["commission"] = cost
                result = replay(
                    prices,
                    instruments,
                    alternative,
                    "full",
                    args.start,
                    args.end,
                    events,
                    actions,
                    published_bundle(config["paths"]["models"]),
                    args.walk_forward,
                    digest,
                )
                sensitivity[str(cost)] = result["metrics"]
            results["full"]["metrics"]["cost_sensitivity"] = sensitivity
        provenance_info = provenance()
        provenance_info["arguments"] = vars(args)
        run_id, folder = persist_run(
            results, config, digest, validation, config["paths"]["runs"], provenance_info
        )
        report = markdown_run(folder)
        store.put("run", run_id, {"folder": str(folder), "validity": validation["validity"]})
        return {
            "run_id": run_id,
            "folder": str(folder),
            "report": str(report),
            "performance_status": validation["validity"],
            "metrics": {m: r["metrics"] for m, r in results.items()},
        }
    if args.command == "forecast":
        bundle = published_bundle(config["paths"]["models"])
        if args.model_version:
            if not args.research or Path(args.model_version).name != args.model_version:
                raise ValueError("historical model requires --research and a valid version id")
            bundle = ModelBundle(Path(config["paths"]["models"]) / args.model_version)
        result = generate_forecast(store.prices(), args.date, bundle, args.symbol)
        result["mode"] = "historical_research" if args.model_version else "published"
        store.put("forecast", args.date, result)
        write_json(Path(config["paths"]["runs"]) / "forecasts" / f"{args.date}.json", result)
        return result
    if args.command == "account":
        if args.action == "init":
            if store.get("account", args.id):
                raise ValueError("account exists; refusing to reset ledger")
            portfolio = Portfolio(args.cash, store.instruments())
            store.put(
                "account",
                args.id,
                {
                    "mode": args.mode,
                    "complete": args.complete,
                    "portfolio": portfolio.to_dict(),
                    "strategy_state": {},
                },
            )
            return {"account": args.id, "mode": args.mode, "complete": args.complete}
        if args.action == "import-fills":
            return import_fills(
                store, args.file, args.id, config["execution"]["initial_cash"], store.instruments()
            )
    if args.command == "report":
        # Reports freeze a decision for the day. Same-input reruns reuse it;
        # they never manufacture a second initial/risk rebalance.
        account = store.get("account", args.account)
        signature = hashlib.sha256(
            json.dumps(
                {
                    "account": account,
                    "prices": store.prices().query("date <= @args.date").to_csv(index=False),
                    "config": config,
                    "events": store.frame("events").to_json(),
                    "actions": store.frame("corporate_actions").to_json(),
                    "instruments": {s: vars(i) for s, i in store.instruments().items()},
                    "published": (Path(config["paths"]["models"]) / "published.json").read_text(
                        encoding="utf-8"
                    )
                    if (Path(config["paths"]["models"]) / "published.json").exists()
                    else None,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        previous = store.get("report", f"{args.account}:{args.date}")
        if previous and previous.get("input_signature") == signature:
            return previous
        result = account_report(store, config, args.account, args.date)
        validation = validate_store(store, config["data"]["jump_threshold"], config["data"])
        result["data_validation"] = validation
        if not validation["valid"] and result["status"] == "ready":
            result["status"] = "research_advice_unverified_data"
            for order in result["orders"]:
                order["status"] = "research_suggestion"
        result["input_signature"] = signature
        store.put("report", f"{args.account}:{args.date}", result)
        store.put(
            "decision_state",
            args.account,
            {
                "date": args.date,
                "state": result["decision_state"],
                "before_state": result["before_decision_state"],
            },
        )
        output = Path(config["paths"]["runs"]) / "reports" / args.account / f"{args.date}.json"
        write_json(output, result)
        return dict(file=str(output), **result)
    if args.command == "dashboard":
        return_code = subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "ui/app.py"),
                "--server.headless=true",
                "--server.address=127.0.0.1",
                "--",
                "--config",
                config["config_file"],
            ],
            cwd=ROOT,
        )
        if return_code:
            raise ValueError("dashboard failed")
        return {"status": "closed"}
    raise ValueError("unsupported command")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        result = run(args, load_config(args.config))
        print(
            json.dumps(result, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        p.exit(2, f"QuanTrade: {exc}\n")


if __name__ == "__main__":
    main()
