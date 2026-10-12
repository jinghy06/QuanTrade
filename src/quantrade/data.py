"""Data contracts, conservative legacy migration, snapshots and ETF updates."""

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .domain import Instrument
from .storage import write_json

PRICE_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "fetched_at",
    "price_basis",
    "verified",
]


def normalize_prices(frame, source="unknown", basis="unknown", verified=False):
    df = frame.copy().rename(columns={"trade_date": "date", "code": "symbol"})
    required = {"symbol", "date", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing price columns: {sorted(required - set(df.columns))}")
    raw_date = df["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    # Mixed compact and ISO date formats occur in the legacy database.
    df["date"] = pd.to_datetime(raw_date, format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    df["symbol"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    defaults = {
        "volume": np.nan,
        "amount": np.nan,
        "source": source,
        "price_basis": basis,
        "verified": int(verified),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    for key, value in defaults.items():
        if key not in df:
            df[key] = value
    for key in ("open", "high", "low", "close", "volume", "amount"):
        df[key] = pd.to_numeric(df[key], errors="coerce")
    return df[PRICE_COLUMNS].sort_values(["symbol", "date"]).reset_index(drop=True)


def inspect_prices(df, jump_threshold=0.35):
    issues = []
    for row in df.itertuples(index=False):
        problems = []
        if pd.isna(row.date):
            problems.append("invalid_date")
        if not row.symbol.isdigit() or len(row.symbol) != 6:
            problems.append("invalid_symbol")
        p = np.array([row.open, row.high, row.low, row.close], dtype=float)
        if not np.isfinite(p).all() or (p <= 0).any():
            problems.append("invalid_price")
        elif (
            row.high < max(row.open, row.close)
            or row.low > min(row.open, row.close)
            or row.high < row.low
        ):
            problems.append("invalid_ohlc")
        if not np.isfinite(row.volume) or row.volume < 0:
            problems.append("invalid_volume")
        for problem in problems:
            issues.append(
                {
                    "symbol": row.symbol,
                    "date": None if pd.isna(row.date) else row.date,
                    "kind": problem,
                }
            )
    duplicated = df[df.duplicated(["symbol", "date"], keep=False)]
    for (symbol, date), group in duplicated.groupby(["symbol", "date"]):
        conflict = (
            len(group[["open", "high", "low", "close", "volume", "price_basis"]].drop_duplicates())
            > 1
        )
        issues.append(
            {
                "symbol": symbol,
                "date": date,
                "kind": "duplicate_conflict" if conflict else "duplicate_identical",
            }
        )
    unique = df.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    change = unique.groupby("symbol")["close"].pct_change(fill_method=None)
    for _, row in unique[change.abs() > jump_threshold].iterrows():
        issues.append(
            {
                "symbol": row.symbol,
                "date": row.date,
                "kind": "unexplained_jump",
                "return_value": float(change.loc[row.name]),
            }
        )
    return issues


def save_prices(store, df):
    # All validation happens before starting writes. Conflicts must never be
    # resolved by an arbitrary INSERT OR REPLACE.
    duplicates = [i for i in inspect_prices(df) if i["kind"] == "duplicate_conflict"]
    if duplicates:
        raise ValueError("conflicting duplicate prices")
    df = df.drop_duplicates(["symbol", "date"])
    with store.connect() as c:
        for row in df.itertuples(index=False, name=None):
            clean = tuple(None if pd.isna(x) else x for x in row)
            old = c.execute(
                "SELECT open,high,low,close,volume,price_basis FROM prices WHERE symbol=? AND date=?",
                clean[:2],
            ).fetchone()
            expected = clean[2:7] + (clean[10],)
            if old and old != expected:
                raise ValueError(f"existing price conflict: {clean[:2]}")
            c.execute("INSERT OR IGNORE INTO prices VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", clean)


def record_issues(store, issues):
    with store.connect() as c:
        for issue in issues:
            payload = json.dumps(issue, ensure_ascii=False, sort_keys=True, allow_nan=False)
            issue_id = sha256(payload.encode()).hexdigest()
            c.execute(
                "INSERT OR IGNORE INTO data_issues(issue_id,payload) VALUES(?,?)",
                (issue_id, payload),
            )


def migrate_legacy(store, legacy_path, output_dir, jump_threshold=0.35):
    legacy = Path(legacy_path).resolve()
    if legacy == store.path.resolve():
        raise ValueError("source and destination must differ")
    with sqlite3.connect(legacy.as_uri() + "?mode=ro", uri=True) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        chunks = []
        for table in ("etf_daily_prices", "gold_daily_prices"):
            if table in tables:
                chunks.append(
                    normalize_prices(
                        pd.read_sql_query(f"SELECT * FROM {table}", c), source=f"legacy:{table}"
                    )
                )
    if not chunks:
        raise ValueError("no legacy ETF prices")
    raw = pd.concat(chunks, ignore_index=True).sort_values(["symbol", "date"])
    issues = inspect_prices(raw, jump_threshold)
    # Quarantine keys with inconsistent duplicate observations or malformed bars.
    invalid = {
        (i["symbol"], i["date"])
        for i in issues
        if i["kind"] not in ("duplicate_identical", "unexplained_jump")
    }
    clean = raw[
        [(r.symbol, None if pd.isna(r.date) else r.date) not in invalid for r in raw.itertuples()]
    ].drop_duplicates(["symbol", "date"])
    clean = clean.sort_values(["date", "symbol"]).reset_index(drop=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "legacy_raw.csv", index=False)
    record_issues(store, issues)
    save_prices(store, clean)
    existing = store.instruments()
    store.save_instruments([Instrument(s) for s in clean.symbol.unique() if s not in existing])
    invalid_count = sum(
        (r.symbol, None if pd.isna(r.date) else r.date) in invalid for r in raw.itertuples()
    )
    report = {
        "source": str(legacy),
        "source_sha256": sha256(legacy.read_bytes()).hexdigest(),
        "source_rows": len(raw),
        "migrated_rows": len(clean),
        "quarantined_rows": invalid_count,
        "deduplicated_rows": len(raw) - len(clean) - invalid_count,
        "symbols": int(clean.symbol.nunique()),
        "issues": issues,
        "validity": "exploratory",
        "limitation": "Legacy price basis and instrument metadata are unverified.",
    }
    write_json(output / "migration.json", report)
    return report


def validate_store(store, jump_threshold=0.35, assurance=None):
    prices = store.prices()
    issues = inspect_prices(prices, jump_threshold) if len(prices) else [{"kind": "empty_prices"}]
    instruments = store.instruments()
    for symbol in prices.symbol.unique():
        inst = instruments.get(symbol)
        if inst is None or not inst.verified or not inst.listing_date or inst.theme == "unknown":
            issues.append({"symbol": symbol, "kind": "unverified_instrument"})
        if (
            inst
            and inst.listing_date
            and (prices.loc[prices.symbol == symbol, "date"] < inst.listing_date).any()
        ):
            issues.append({"symbol": symbol, "kind": "pre_listing_prices"})
    if len(prices) and ((prices.price_basis != "raw") | (prices.verified != 1)).any():
        issues.append({"kind": "unverified_execution_prices"})
    assurance = assurance or {}
    for key in ("corporate_actions_verified", "execution_rules_verified", "calendar_verified"):
        if not assurance.get(key, False):
            issues.append({"kind": key.replace("_verified", "_unverified")})
    unresolved = store.frame("data_issues")
    if len(unresolved):
        resolved_jumps = {
            (json.loads(r.payload).get("symbol"), json.loads(r.payload).get("date"))
            for r in unresolved[unresolved.resolved == 1].itertuples()
            if json.loads(r.payload)["kind"] == "unexplained_jump"
        }
        issues = [
            i
            for i in issues
            if not (i["kind"] == "unexplained_jump" and (i["symbol"], i["date"]) in resolved_jumps)
        ]
    if len(unresolved):
        for row in unresolved[unresolved.resolved == 0].itertuples():
            issue = json.loads(row.payload)
            if issue["kind"] != "duplicate_identical" and issue not in issues:
                issues.append(issue)
    return {
        "rows": len(prices),
        "symbols": int(prices.symbol.nunique()),
        "issues": issues,
        "valid": not issues,
        "validity": "validated" if not issues else "exploratory",
    }


def snapshot(store, directory, validation):
    df = store.prices()
    # Fetched timestamps are preserved but deterministic across fixed inputs.
    content = df.to_csv(index=False, float_format="%.12g").encode("utf-8")
    metadata = {
        "instruments": {s: asdict(i) for s, i in store.instruments().items()},
        "corporate_actions": store.frame("corporate_actions")
        .sort_values("event_id")
        .to_dict("records"),
        "events": store.frame("events").sort_values("event_id").to_dict("records"),
        "validation": validation,
    }
    digest = sha256(
        content + json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    folder = Path(directory) / digest[:16]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prices.csv").write_bytes(content)
    write_json(folder / "manifest.json", dict(snapshot_id=digest, **metadata))
    return digest, folder


def update_etfs(store, symbols, start, end):
    import akshare as ak

    errors = []
    for symbol in symbols:
        try:
            raw = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="",
            )
            df = raw.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume",
                    "成交额": "amount",
                }
            )
            df["symbol"] = symbol
            # Price basis is known; official verification still requires actions and rules.
            df = normalize_prices(df, source="akshare:fund_etf_hist_em", basis="raw")
            issues = inspect_prices(df)
            record_issues(store, issues)
            if any(i["kind"] not in ("duplicate_identical", "unexplained_jump") for i in issues):
                raise ValueError("invalid fetched bars; see data issues")
            save_prices(store, df)
            if symbol not in store.instruments():
                store.save_instruments([Instrument(symbol)])
        except Exception as exc:  # noqa: BLE001 -- isolate heterogeneous provider failures per ETF
            errors.append({"symbol": symbol, "error": str(exc)})
    return {"errors": errors, "validity": "exploratory"}


def import_instruments(store, path):
    df = pd.read_csv(path, dtype={"symbol": str, "listing_date": str}).fillna("")
    items = []
    for row in df.to_dict("records"):
        verified = str(row.get("verified", "false")).lower() in ("true", "1")
        inst = Instrument(
            symbol=row["symbol"],
            name=row.get("name", ""),
            theme=row.get("theme", "unknown"),
            listing_date=row.get("listing_date", ""),
            lot_size=int(row.get("lot_size", 100)),
            t_plus=int(row.get("t_plus", 1)),
            verified=verified,
        )
        if (
            inst.lot_size < 1
            or inst.t_plus not in (0, 1)
            or not inst.symbol.isdigit()
            or len(inst.symbol) != 6
        ):
            raise ValueError("invalid instrument rules")
        if inst.listing_date:
            pd.Timestamp(inst.listing_date)
        if verified and (not inst.name or not inst.listing_date or inst.theme == "unknown"):
            raise ValueError("verified instruments require name, listing date and theme")
        items.append(inst)
    store.save_instruments(items)
