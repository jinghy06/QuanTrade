"""Share ledger. Trade cost is capitalized on buys and realized on sells."""

import math
from dataclasses import asdict

from .domain import Fill, Instrument, Position


class Portfolio:
    def __init__(self, cash=50000.0, instruments=None):
        if not math.isfinite(cash) or cash < 0:
            raise ValueError("invalid starting cash")
        self.initial_cash = self.cash = float(cash)
        self.instruments = instruments or {}
        self.positions = {}
        self.fills = []
        self.cash_events = []
        self.applied_ids = set()
        self.fees = self.realized_pnl = self.dividends = 0.0

    def sellable(self, symbol, date):
        pos = self.positions.get(symbol, Position())
        inst = self.instruments.get(symbol, Instrument(symbol))
        if inst.t_plus == 0:
            return pos.quantity
        # ETF T+1 defaults are conservative. Longer rules require a calendar.
        if inst.t_plus != 1:
            raise ValueError("Only configured T+0 and T+1 instruments supported")
        locked = sum(q for bought, q in pos.locked if bought >= date)
        return max(0.0, pos.quantity - locked)

    def apply_fill(self, fill: Fill):
        if fill.fill_id in self.applied_ids:
            previous = next((f for f in self.fills if f["fill_id"] == fill.fill_id), None)
            if previous != asdict(fill):
                raise ValueError("conflicting or reused ledger event id")
            return False
        if fill.side not in ("buy", "sell"):
            raise ValueError("side must be buy or sell")
        if not all(math.isfinite(x) for x in (fill.quantity, fill.price, fill.fee)):
            raise ValueError("non-finite fill")
        if fill.quantity <= 0 or fill.price <= 0 or fill.fee < 0:
            raise ValueError("invalid fill")
        pos = self.positions.get(fill.symbol, Position())
        amount = fill.quantity * fill.price
        before = self.cash + pos.quantity * fill.price
        if fill.side == "buy":
            if amount + fill.fee > self.cash + 1e-8:
                raise ValueError("insufficient cash including fees")
            pos.average_cost = (pos.quantity * pos.average_cost + amount + fill.fee) / (
                pos.quantity + fill.quantity
            )
            pos.quantity += fill.quantity
            pos.locked.append((fill.date, fill.quantity))
            self.cash -= amount + fill.fee
        else:
            if fill.quantity > self.sellable(fill.symbol, fill.date) + 1e-8:
                raise ValueError("insufficient sellable shares")
            if self.cash + amount < fill.fee:
                raise ValueError("sale fee exceeds available proceeds and cash")
            self.realized_pnl += amount - fill.fee - fill.quantity * pos.average_cost
            pos.quantity -= fill.quantity
            self.cash += amount - fill.fee
            pos.locked = [(d, q) for d, q in pos.locked if d >= fill.date]
            if pos.quantity < 1e-8:
                pos.quantity = pos.average_cost = 0.0
                pos.locked = []
        pos.last_price, pos.price_date = fill.price, fill.date
        self.positions[fill.symbol] = pos
        self.cash = max(0.0, self.cash)
        self.fees += fill.fee
        self.fills.append(asdict(fill))
        self.applied_ids.add(fill.fill_id)
        assert abs(before - (self.cash + pos.quantity * fill.price) - fill.fee) < 1e-6
        return True

    def corporate_action(self, event_id, date, symbol, split=1.0, dividend=0.0):
        if event_id in self.applied_ids:
            return False
        if not math.isfinite(split) or split <= 0 or not math.isfinite(dividend) or dividend < 0:
            raise ValueError("invalid corporate action")
        pos = self.positions.get(symbol)
        income = 0.0
        if pos:
            income = pos.quantity * dividend
            self.cash += income
            self.dividends += income
            pos.quantity *= split
            pos.average_cost /= split
            pos.last_price /= split
            pos.locked = [(d, q * split) for d, q in pos.locked]
        self.cash_events.append(
            {
                "event_id": event_id,
                "date": date,
                "symbol": symbol,
                "kind": "corporate_action",
                "amount": income,
                "split": split,
                "dividend_per_share": dividend,
            }
        )
        self.applied_ids.add(event_id)
        return True

    def mark(self, date, prices):
        for symbol, pos in self.positions.items():
            price = prices.get(symbol)
            if price is not None and math.isfinite(price) and price > 0:
                pos.last_price, pos.price_date = float(price), date

    @property
    def nav(self):
        return self.cash + sum(p.quantity * p.last_price for p in self.positions.values())

    def snapshot(self, date):
        positions = {
            s: {
                **asdict(p),
                "market_value": p.quantity * p.last_price,
                "stale": p.price_date != date,
            }
            for s, p in self.positions.items()
            if p.quantity > 0
        }
        unrealized = sum(
            p.quantity * (p.last_price - p.average_cost) for p in self.positions.values()
        )
        return {
            "date": date,
            "cash": self.cash,
            "nav": self.nav,
            "fees": self.fees,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "dividends": self.dividends,
            "positions": positions,
        }

    def to_dict(self):
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "fills": self.fills,
            "cash_events": self.cash_events,
            "applied_ids": sorted(self.applied_ids),
            "fees": self.fees,
            "realized_pnl": self.realized_pnl,
            "dividends": self.dividends,
        }

    @classmethod
    def from_dict(cls, data, instruments=None):
        p = cls(data["initial_cash"], instruments)
        p.cash = data["cash"]
        p.positions = {s: Position(**v) for s, v in data["positions"].items()}
        for key in ("fills", "cash_events", "fees", "realized_pnl", "dividends"):
            setattr(p, key, data[key])
        p.applied_ids = set(data["applied_ids"])
        return p
