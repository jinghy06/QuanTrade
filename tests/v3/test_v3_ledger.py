import pytest

from quantrade.domain import Fill
from quantrade.portfolio import Portfolio


def fill(fid, side, q, p, fee=0, date="2024-01-02"):
    return Fill(fid, fid, date, "510300", side, q, p, fee)


def test_cumulative_valuation_and_fee_reconciliation():
    p = Portfolio(1000)
    p.apply_fill(fill("buy", "buy", 90, 10, 1))
    p.mark("2024-01-03", {"510300": 11})
    assert p.nav == pytest.approx(1089)
    p.mark("2024-01-04", {"510300": 12.1})
    assert p.nav == pytest.approx(1188)
    p.apply_fill(fill("sell", "sell", 40, 12.1, 2, "2024-01-04"))
    snap = p.snapshot("2024-01-04")
    assert snap["nav"] == pytest.approx(1186)
    assert 1000 + snap["realized_pnl"] + snap["unrealized_pnl"] == pytest.approx(snap["nav"])


def test_missing_price_never_erases_position_and_downtrend_compounds():
    p = Portfolio(1000)
    p.apply_fill(fill("b", "buy", 100, 10))
    p.mark("2024-01-03", {"510300": 9})
    p.mark("2024-01-04", {"510300": 8.1})
    p.mark("2024-01-05", {})
    assert p.nav == 810
    assert p.snapshot("2024-01-05")["positions"]["510300"]["stale"]


def test_buy_cost_budget_and_idempotence():
    p = Portfolio(1000)
    with pytest.raises(ValueError, match="cash"):
        p.apply_fill(fill("bad", "buy", 100, 10, 1))
    assert p.cash == 1000 and not p.positions
    assert p.apply_fill(fill("good", "buy", 99, 10, 1))
    assert not p.apply_fill(fill("good", "buy", 99, 10, 1))
    assert p.cash == 9


def test_t_plus_one_and_split_dividend():
    p = Portfolio(1000)
    p.apply_fill(fill("b", "buy", 100, 10))
    with pytest.raises(ValueError, match="sellable"):
        p.apply_fill(fill("s", "sell", 100, 10))
    p.corporate_action("split", "2024-01-03", "510300", split=2)
    p.mark("2024-01-03", {"510300": 5})
    assert p.nav == 1000
    p.corporate_action("div", "2024-01-04", "510300", dividend=0.1)
    p.mark("2024-01-04", {"510300": 4.9})
    assert p.nav == pytest.approx(1000)
    assert not p.corporate_action("div", "2024-01-04", "510300", dividend=0.1)
    p.apply_fill(fill("s", "sell", 200, 4.9, date="2024-01-04"))
    assert p.cash == pytest.approx(1000)
