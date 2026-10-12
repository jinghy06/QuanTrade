import pandas as pd
import pytest

from quantrade.backtest import replay
from quantrade.config import load_config
from quantrade.domain import Instrument, Target
from quantrade.execution import Executor
from quantrade.factors import calculate_factors, normalize_events
from quantrade.features import compute_features
from quantrade.forecast import forecast_path
from quantrade.models import purged_partition
from quantrade.portfolio import Portfolio
from quantrade.strategy import ETFStrategy


def market(n=160, symbols=("510300", "510500")):
    dates = pd.bdate_range("2023-01-02", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame(
        [
            {
                "symbol": s,
                "date": d,
                "open": 10 * (1.001**i),
                "high": 10.1 * (1.001**i),
                "low": 9.9 * (1.001**i),
                "close": 10 * (1.001**i),
                "volume": 100000,
                "amount": 1000000,
            }
            for s in symbols
            for i, d in enumerate(dates)
        ]
    )


def instruments():
    return {s: Instrument(s, theme=s) for s in ("510300", "510500")}


def config():
    c = load_config()
    c["strategy"].update(
        lookback=20, max_weight=0.8, max_theme_weight=0.8, cash_reserve=0.0, min_weight_change=0.0
    )
    c["execution"].update(commission=0.0, slippage=0.0)
    return c


def test_future_changes_cannot_change_past_signal_nav():
    prices = market()
    c = config()
    first = replay(prices, instruments(), c, "momentum")
    altered = prices.copy()
    boundary = "2023-05-01"
    altered.loc[altered.date > boundary, ["open", "high", "low", "close"]] *= 2
    second = replay(altered, instruments(), c, "momentum")
    assert [t for t in first["targets"] if t["signal_time"][:10] <= boundary] == [
        t for t in second["targets"] if t["signal_time"][:10] <= boundary
    ]
    pd.testing.assert_frame_equal(
        first["nav"][first["nav"].date <= boundary], second["nav"][second["nav"].date <= boundary]
    )
    assert all(
        f["date"]
        > next(t["signal_time"][:10] for t in first["targets"] if t["signal_time"][:10] < f["date"])
        for f in first["fills"]
    )


def test_signal_executes_next_open_and_does_not_earn_previous_gap():
    prices = market(70, ("510300",))
    c = config()
    c["strategy"].update(lookback=2, max_weight=1.0, max_theme_weight=1.0)
    start = sorted(prices.date.unique())[30]
    dates = sorted(prices.date.unique())
    prices.loc[prices.date == dates[31], ["open", "high", "low", "close"]] *= 2
    result = replay(prices, {"510300": Instrument("510300", theme="index")}, c, "fixed", start)
    assert result["fills"][0]["date"] == dates[31]
    assert result["nav"].iloc[1].nav == pytest.approx(c["execution"]["initial_cash"])


def test_executor_budget_and_repeated_target():
    c = config()
    c["execution"].update(commission=0.01, min_fee=5)
    p = Portfolio(1000, instruments())
    target = Target("2024-01-01T15:30:00+08:00", {"510300": 1}, [], "2024-01-02T15:00:00+08:00")
    bars = market(1, ("510300",))
    bars["open"] = 1
    executor = Executor(c, instruments())
    executor.execute(p, target, bars, "2024-01-02")
    assert p.cash >= 0 and p.positions["510300"].quantity == 900
    previous = p.to_dict()
    executor.execute(p, target, bars, "2024-01-02")
    assert p.to_dict() == previous


def test_targets_obey_theme_cap_and_unallocated_cash():
    c = config()
    c["strategy"]["max_theme_weight"] = 0.3
    inst = {s: Instrument(s, theme="same") for s in instruments()}
    strategy = ETFStrategy(c, inst, "momentum")
    prices = market()
    date = prices.date.max()
    target = strategy.decide(prices, date, "2024-01-02", Portfolio(), None)
    assert sum(target.weights.values()) <= 0.3 + 1e-8


def test_labels_remain_unknown_and_split_is_by_dates_with_purge():
    f = compute_features(market(), (20,))
    assert f.groupby("symbol").tail(20).target_20.isna().all()
    dates = sorted(f.date.unique())
    train, val, test = purged_partition(f, dates[79], dates[119], dates[-1], 20)
    assert train.label_end_20.max() <= dates[79]
    assert val.label_end_20.max() <= dates[119]
    assert set(train.date).isdisjoint(val.date)
    assert set(val.date).isdisjoint(test.date)


def test_event_first_available_time_and_expiry():
    frame = pd.DataFrame(
        [
            {
                "event_id": "policy1",
                "category": "policy",
                "score": -1.0,
                "confidence": 1.0,
                "published_at": "2023-04-01T12:00:00+08:00",
                "available_at": "2023-05-01T16:00:00+08:00",
                "valid_until": "2023-05-03T16:00:00+08:00",
                "source": "fixture",
                "content": "test",
            }
        ]
    )
    events = normalize_events(frame)
    early = calculate_factors(market(), events, "2023-05-01")
    later = calculate_factors(market(), events, "2023-05-02")
    expired = calculate_factors(market(), events, "2023-05-04")
    assert early["policy"]["mode"] != "event"
    assert later["policy"]["score"] == -1
    assert expired["policy"]["mode"] != "event"


def test_prediction_scenarios_are_reproducible_and_valid_ohlc():
    a = forecast_path(10, {1: 0.01, 5: -0.03}, {1: 0.01, 5: 0.03}, "2024-01-01")
    assert a == forecast_path(10, {1: 0.01, 5: -0.03}, {1: 0.01, 5: 0.03}, "2024-01-01")
    for row in a["candles"]:
        assert row["high"] >= max(row["open"], row["close"])
        assert row["low"] <= min(row["open"], row["close"])
    assert a["candles"][0]["kind"] == "model_anchor"
    assert a["candles"][1]["kind"] == "interpolated_scenario"
