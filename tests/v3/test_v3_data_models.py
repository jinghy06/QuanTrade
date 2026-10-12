import sqlite3

import pandas as pd
import pytest
from test_v3_pipeline import config, market

from quantrade.data import migrate_legacy, normalize_prices, save_prices, validate_store
from quantrade.domain import Instrument
from quantrade.features import compute_features
from quantrade.models import ModelBundle, fit_bundle, publish_model, validate_model
from quantrade.portfolio import Portfolio
from quantrade.reporting import import_fills
from quantrade.storage import Store


def test_migration_is_readonly_and_quarantines_conflicting_dates(tmp_path):
    source = tmp_path / "legacy.db"
    frame = market(5, ("510300",)).rename(columns={"date": "trade_date"})
    frame = pd.concat([frame, frame.iloc[[0]], frame.iloc[[1]].assign(close=20)], ignore_index=True)
    with sqlite3.connect(source) as c:
        frame.to_sql("etf_daily_prices", c, index=False)
    before = source.read_bytes()
    store = Store(tmp_path / "v3.db")
    report = migrate_legacy(store, source, tmp_path / "migration")
    assert source.read_bytes() == before
    assert report["quarantined_rows"] == 2
    assert report["deduplicated_rows"] == 1
    assert len(store.prices()) == 4
    assert not validate_store(store)["valid"]
    migrate_legacy(store, source, tmp_path / "migration")
    assert len(store.prices()) == 4


def test_prices_refuse_conflicting_overwrite(tmp_path):
    store = Store(tmp_path / "v3.db")
    df = normalize_prices(market(5))
    save_prices(store, df)
    wrong = df.copy()
    wrong["close"] *= 2
    with pytest.raises(ValueError, match="conflict"):
        save_prices(store, wrong)
    assert store.prices().close.max() < 11


def test_model_uses_past_dates_and_saved_preprocessing(tmp_path):
    c = config()
    c["model"].update(
        min_train_dates=80, validation_dates=40, test_dates=40, horizons=[1, 20], n_estimators=5
    )
    prices = market(210)
    dates = sorted(prices.date.unique())
    manifest = fit_bundle(prices, c, dates[199], tmp_path)
    bundle = ModelBundle(tmp_path / manifest["version"])
    latest = compute_features(prices)
    latest = latest[latest.date == dates[200]]
    expected = bundle.predict(latest, dates[200])
    assert expected == ModelBundle(tmp_path / manifest["version"]).predict(latest, dates[200])
    with pytest.raises(ValueError, match="future"):
        bundle.predict(latest, dates[199])
    with pytest.raises(ValueError, match="validate"):
        publish_model(tmp_path, manifest["version"])
    validate_model(tmp_path / manifest["version"])
    assert publish_model(tmp_path, manifest["version"])["status"] == "published"


def test_actual_import_atomic_idempotent_and_cannot_replace_fill(tmp_path):
    store = Store(tmp_path / "v3.db")
    inst = {"510300": Instrument("510300")}
    store.save_instruments(inst.values())
    initial = {"mode": "actual", "complete": True, "portfolio": Portfolio(10000, inst).to_dict()}
    store.put("account", "actual", initial)
    fill = {
        "fill_id": "b",
        "order_id": "b",
        "date": "2024-01-02",
        "symbol": "510300",
        "side": "buy",
        "quantity": 100,
        "price": 10,
        "fee": 1,
    }
    path = tmp_path / "fills.csv"
    pd.DataFrame([fill]).to_csv(path, index=False)
    assert import_fills(store, path, "actual", 10000, inst)["imported"] == 1
    assert import_fills(store, path, "actual", 10000, inst)["imported"] == 0
    before = store.get("account", "actual")
    pd.DataFrame([{**fill, "quantity": 200}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting"):
        import_fills(store, path, "actual", 10000, inst)
    assert store.get("account", "actual") == before
