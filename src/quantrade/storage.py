"""Separate v3 SQLite; never mutate the legacy database."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .domain import Instrument


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS instruments(
                symbol TEXT PRIMARY KEY, name TEXT, theme TEXT, listing_date TEXT,
                lot_size INTEGER, t_plus INTEGER, verified INTEGER);
            CREATE TABLE IF NOT EXISTS prices(
                symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
                volume REAL, amount REAL, source TEXT, fetched_at TEXT,
                price_basis TEXT, verified INTEGER,
                PRIMARY KEY(symbol, date));
            CREATE TABLE IF NOT EXISTS data_issues(
                issue_id TEXT PRIMARY KEY, payload TEXT NOT NULL, resolved INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS corporate_actions(
                event_id TEXT PRIMARY KEY, date TEXT, symbol TEXT,
                split REAL DEFAULT 1, dividend REAL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events(
                event_id TEXT PRIMARY KEY, category TEXT, published_at TEXT,
                available_at TEXT, valid_until TEXT, score REAL, confidence REAL,
                source TEXT, content TEXT);
            CREATE TABLE IF NOT EXISTS artifacts(
                kind TEXT, artifact_id TEXT, payload TEXT,
                PRIMARY KEY(kind, artifact_id));
            """)

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.path)
        try:
            yield c
            c.commit()
        except BaseException:
            c.rollback()
            raise
        finally:
            c.close()

    def instruments(self):
        with self.connect() as c:
            rows = c.execute("SELECT * FROM instruments ORDER BY symbol").fetchall()
        return {r[0]: Instrument(*r[:-1], verified=bool(r[-1])) for r in rows}

    def save_instruments(self, instruments):
        with self.connect() as c:
            c.executemany(
                "INSERT OR REPLACE INTO instruments VALUES(?,?,?,?,?,?,?)",
                [tuple(asdict(i).values()) for i in instruments],
            )

    def prices(self):
        with self.connect() as c:
            return pd.read_sql_query("SELECT * FROM prices ORDER BY date, symbol", c)

    def put(self, kind, artifact_id, payload):
        encoded = json.dumps(payload, ensure_ascii=False, default=json_default, allow_nan=False)
        with self.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO artifacts VALUES(?,?,?)", (kind, artifact_id, encoded)
            )

    def get(self, kind, artifact_id):
        with self.connect() as c:
            row = c.execute(
                "SELECT payload FROM artifacts WHERE kind=? AND artifact_id=?", (kind, artifact_id)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def frame(self, table):
        if table not in ("corporate_actions", "events", "data_issues"):
            raise ValueError("unsupported table")
        with self.connect() as c:
            return pd.read_sql_query(f"SELECT * FROM {table}", c)
