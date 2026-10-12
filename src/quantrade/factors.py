"""Point-in-time factor signals. Price proxies are explicitly labelled."""

import math

import numpy as np
import pandas as pd

CATEGORIES = {"sentiment", "policy", "geopolitical"}

LEXICONS = {
    "sentiment": (
        ["乐观", "看涨", "追涨", "热潮", "狂热", "资金流入"],
        ["恐慌", "悲观", "看跌", "抛售", "资金流出"],
    ),
    "policy": (
        ["降准", "降息", "宽松", "减税", "财政补贴", "产业扶持", "技术创新"],
        ["加息", "收紧", "紧缩", "禁止", "处罚", "金融风险", "债务风险"],
    ),
    "geopolitical": (
        ["贸易战", "加征关税", "出口管制", "芯片禁令", "军事冲突", "战争", "制裁"],
        ["贸易谈判", "达成协议", "取消关税", "缓和", "和平", "停火"],
    ),
}


def score_text(category, text):
    if category not in LEXICONS:
        raise ValueError("unknown event category")
    positive, negative = LEXICONS[category]
    a = sum(word in text for word in positive)
    b = sum(word in text for word in negative)
    return {
        "score": (a - b) / (a + b) if a + b else 0.0,
        "confidence": 0.6 if a + b else 0.0,
        "analysis_version": "keyword-v3.0",
        "limitation": "Keyword proxy, not a calibrated causal interpretation; inspect negation and context.",
    }


def normalize_events(frame):
    required = {
        "event_id",
        "category",
        "published_at",
        "available_at",
        "valid_until",
        "score",
        "confidence",
        "source",
        "content",
    }
    if not required.issubset(frame):
        raise ValueError(f"missing event fields {sorted(required - set(frame))}")
    df = frame.copy()
    if df.event_id.duplicated().any() or df.event_id.isna().any():
        raise ValueError("events require unique nonempty ids")
    for col in ("published_at", "available_at", "valid_until"):
        # Require an explicit timezone, preventing local/UTC lookahead mistakes.
        times = [pd.Timestamp(v) for v in df[col]]
        if any(t.tzinfo is None for t in times):
            raise ValueError("event timestamps require timezone")
        df[col] = [t.tz_convert("UTC").isoformat() for t in times]
    for row in df.itertuples():
        if row.category not in CATEGORIES or not row.source or pd.isna(row.source):
            raise ValueError("invalid event category/source")
        if not math.isfinite(row.score) or not -1 <= row.score <= 1:
            raise ValueError("score must be in [-1,1]")
        if not math.isfinite(row.confidence) or not 0 <= row.confidence <= 1:
            raise ValueError("invalid confidence")
        if pd.Timestamp(row.available_at) < pd.Timestamp(row.published_at) or pd.Timestamp(
            row.valid_until
        ) < pd.Timestamp(row.available_at):
            raise ValueError("invalid event timing")
    return df[list(required)].sort_values("event_id")


def import_events(store, path):
    raw = pd.read_csv(path, dtype={"event_id": str}).fillna("")
    analyses = {}
    if "category" not in raw or "content" not in raw:
        raise ValueError("events require category and content")
    for index, row in raw.iterrows():
        if "score" not in raw or row.get("score", "") == "":
            analysis = score_text(row.category, row.content)
            raw.loc[index, "score"] = analysis["score"]
            if "confidence" not in raw or row.get("confidence", "") == "":
                raw.loc[index, "confidence"] = analysis["confidence"]
            analyses[row.event_id] = analysis
        else:
            analyses[row.event_id] = {
                "analysis_version": "provided-structured-v1",
                "limitation": "Input interpretation must be validated against its frozen source.",
            }
    raw["score"] = pd.to_numeric(raw["score"], errors="raise")
    raw["confidence"] = pd.to_numeric(raw["confidence"], errors="raise")
    df = normalize_events(raw)
    columns = [
        "event_id",
        "category",
        "published_at",
        "available_at",
        "valid_until",
        "score",
        "confidence",
        "source",
        "content",
    ]
    with store.connect() as c:
        for row in df[columns].itertuples(index=False, name=None):
            old = c.execute("SELECT * FROM events WHERE event_id=?", (row[0],)).fetchone()
            if old and tuple(old) != row:
                raise ValueError("event revision requires a new id, never overwrite history")
            c.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,?,?)", row)
            import json

            c.execute(
                "INSERT OR IGNORE INTO artifacts VALUES(?,?,?)",
                ("event_analysis", row[0], json.dumps(analyses[row[0]], ensure_ascii=False)),
            )
    return {"imported": len(df)}


def calculate_factors(history, events, date, min_confidence=0.6):
    as_of = pd.Timestamp(f"{date}T15:30:00+08:00").tz_convert("UTC")
    result = {
        kind: {"score": 0.0, "mode": "unavailable", "sources": [], "available": False}
        for kind in CATEGORIES
    }
    current = history[history.date <= date].sort_values(["symbol", "date"])
    scores, policy = [], []
    for symbol, bars in current.groupby("symbol"):
        if len(bars) < 61 or bars.date.iloc[-1] != date:
            continue
        mom = bars.close.iloc[-1] / bars.close.iloc[-21] - 1
        long = bars.close.iloc[-1] / bars.close.iloc[-61] - 1
        scores.append(float(np.tanh(mom * 8)))
        policy.append(float(np.tanh(long * 3)))
    if scores:
        result["sentiment"] = {
            "score": float(np.mean(scores)),
            "mode": "price_proxy",
            "sources": ["ETF 20-session momentum"],
            "available": True,
        }
        result["policy"] = {
            "score": float(np.mean(policy)),
            "mode": "price_proxy",
            "sources": ["ETF 60-session trend; not policy text"],
            "available": True,
        }
    gold = current[(current.symbol == "518880") & (current.date <= date)]
    if len(gold) > 20 and gold.date.iloc[-1] == date:
        risk = float(np.tanh((gold.close.iloc[-1] / gold.close.iloc[-21] - 1) * 10))
        result["geopolitical"] = {
            "score": risk,
            "mode": "price_proxy",
            "sources": ["gold momentum; not geopolitical news"],
            "available": True,
        }
    if events is not None and not events.empty:
        eligible = events[
            (pd.to_datetime(events.published_at, utc=True) <= as_of)
            & (pd.to_datetime(events.available_at, utc=True) <= as_of)
            & (pd.to_datetime(events.valid_until, utc=True) >= as_of)
            & (events.confidence >= min_confidence)
        ]
        for kind, group in eligible.groupby("category"):
            if kind not in CATEGORIES:
                continue
            # Multiple frozen events can coexist; conservative geopolitical risk
            # uses the largest credible risk score, other categories average.
            score = (
                float(group.score.max())
                if kind == "geopolitical"
                else float(np.average(group.score, weights=group.confidence.clip(lower=1e-9)))
            )
            result[kind] = {
                "score": score,
                "mode": "event",
                "sources": group.source.tolist(),
                "event_ids": group.event_id.tolist(),
                "available": True,
            }
    return result
