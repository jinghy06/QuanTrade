import pandas as pd
import pytest
from test_v3_pipeline import config, market

from quantrade.cli import parser, run
from quantrade.config import ROOT
from quantrade.data import normalize_prices, save_prices
from quantrade.domain import Instrument
from quantrade.factors import import_events
from quantrade.forecast import forecast_path
from quantrade.reporting import reconstruct_account
from quantrade.storage import Store


def setup_store(tmp_path):
    c = config()
    c["paths"].update(
        database=str(tmp_path / "db.sqlite"),
        runs=str(tmp_path / "runs"),
        models=str(tmp_path / "models"),
    )
    store = Store(c["paths"]["database"])
    store.save_instruments([Instrument("510300", theme="index")])
    save_prices(store, normalize_prices(market(120, ("510300",))))
    return c, store


def test_daily_reports_are_idempotent_and_only_initial_or_scheduled(tmp_path):
    c, store = setup_store(tmp_path)
    parse = parser().parse_args
    run(
        parse(
            [
                "account",
                "init",
                "--id",
                "demo",
                "--mode",
                "simulation",
                "--cash",
                "50000",
                "--complete",
            ]
        ),
        c,
    )
    args = parse(["report", "--account", "demo", "--date", "2023-05-10"])
    first = run(args, c)
    assert first["target"]
    assert first["target"]["signal_time"].startswith("2023-05-10")
    second = run(args, c)
    assert first["target"] == second["target"]
    next_day = run(parse(["report", "--account", "demo", "--date", "2023-05-11"]), c)
    assert next_day["target"] is None
    assert next_day["orders"] == []
    assert store.get("account", "demo")["portfolio"]["fills"] == []


def test_account_reconstruction_handles_dividends_before_later_sales():
    fills = [
        {
            "fill_id": "b",
            "order_id": "b",
            "date": "2024-01-02",
            "symbol": "510300",
            "side": "buy",
            "quantity": 100,
            "price": 10,
            "fee": 0,
            "source": "actual",
        },
        {
            "fill_id": "s",
            "order_id": "s",
            "date": "2024-01-05",
            "symbol": "510300",
            "side": "sell",
            "quantity": 200,
            "price": 4.9,
            "fee": 0,
            "source": "actual",
        },
    ]
    actions = pd.DataFrame(
        [
            {
                "event_id": "split",
                "date": "2024-01-03",
                "symbol": "510300",
                "split": 2,
                "dividend": 0,
            },
            {
                "event_id": "div",
                "date": "2024-01-04",
                "symbol": "510300",
                "split": 1,
                "dividend": 0.1,
            },
        ]
    )
    portfolio = reconstruct_account(
        1000, fills, {"510300": Instrument("510300")}, actions, "2024-01-05"
    )
    assert portfolio.cash == pytest.approx(1000)
    assert portfolio.positions["510300"].quantity == 0


def test_policy_and_geopolitical_text_are_frozen_and_versioned(tmp_path):
    _c, store = setup_store(tmp_path)
    path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "event_id": "test-policy",
                "category": "policy",
                "content": "降准，产业扶持",
                "published_at": "2023-05-01T09:00:00+08:00",
                "available_at": "2023-05-01T10:00:00+08:00",
                "valid_until": "2023-05-31T10:00:00+08:00",
                "source": "fixture",
            }
        ]
    ).to_csv(path, index=False)
    assert import_events(store, path)["imported"] == 1
    assert store.frame("events").iloc[0].score == 1
    assert store.get("event_analysis", "test-policy")["analysis_version"] == "keyword-v3.0"


def test_dashboard_with_saved_report_and_forecast(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    c, store = setup_store(tmp_path)
    parse = parser().parse_args
    run(
        parse(
            [
                "account",
                "init",
                "--id",
                "demo",
                "--mode",
                "simulation",
                "--cash",
                "50000",
                "--complete",
            ]
        ),
        c,
    )
    report = run(parse(["report", "--account", "demo", "--date", "2023-05-10"]), c)
    report["forecast"] = {
        "status": "fixture",
        "forecasts": {
            "510300": forecast_path(10, {1: 0.01, 5: 0.03}, {1: 0.01, 5: 0.02}, "2023-05-10")
        },
    }
    store.put("report", "demo:2023-05-10", report)
    monkeypatch.setattr("quantrade.config.load_config", lambda path=None: c)
    app = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=30).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["账户", "调仓", "回测", "数据", "预测与决策"]
    app.selectbox[0].set_value("demo").run()
    assert not app.exception
