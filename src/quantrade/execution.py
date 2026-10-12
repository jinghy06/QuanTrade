"""Common target-to-order execution for simulation and estimated advice."""

import math
from hashlib import sha256

import pandas as pd

from .domain import Fill


class Executor:
    def __init__(self, config, instruments):
        self.cfg = config["execution"]
        self.threshold = config["strategy"]["min_weight_change"]
        self.instruments = instruments

    def fee(self, amount):
        return max(self.cfg["min_fee"], amount * self.cfg["commission"])

    def execute(self, portfolio, target, bars, date, account="simulation", price_column="open"):
        if date <= target.signal_time[:10]:
            raise ValueError("cannot execute a closing signal on the signal date")
        if pd.Timestamp(date + "T09:30:00+08:00") > pd.Timestamp(target.valid_until):
            return [{"date": date, "status": "expired", "reason": "target_expired"}]
        weights = target.weights
        if (
            any(not math.isfinite(w) or w < 0 for w in weights.values())
            or sum(weights.values()) > 1 + 1e-8
        ):
            raise ValueError("invalid target weights")
        quotes = {r.symbol: r._asdict() for r in bars.itertuples(index=False)}
        portfolio.mark(date, {s: q[price_column] for s, q in quotes.items()})
        nav = portfolio.nav
        orders = []
        eligible = {}
        for symbol in sorted(set(weights) | set(portfolio.positions)):
            inst = self.instruments.get(symbol)
            quote = quotes.get(symbol)
            reason = None
            if inst is None:
                reason = "missing_instrument_rules"
            elif quote is None:
                reason = "missing_quote"
            elif not math.isfinite(quote[price_column]) or quote[price_column] <= 0:
                reason = "invalid_execution_price"
            elif not math.isfinite(quote.get("volume", 0)) or quote.get("volume", 0) <= 0:
                reason = "not_tradable"
            if reason:
                orders.append(
                    {"date": date, "symbol": symbol, "status": "rejected", "reason": reason}
                )
                continue
            price = quote[price_column]
            pos = portfolio.positions.get(symbol)
            current = pos.quantity if pos else 0
            desired = (
                math.floor(weights.get(symbol, 0) * nav / price / inst.lot_size) * inst.lot_size
            )
            delta = desired - current
            if (
                nav
                and abs(delta * price) / nav < self.threshold
                and not (desired == 0 and current > 0)
            ):
                continue
            eligible[symbol] = (delta, price, quote)
        # Sells fund buys; deterministic ordering and pro-rata buy budgets.
        buy_need = sum(
            delta * p * (1 + self.cfg["slippage"]) * (1 + self.cfg["commission"])
            + self.cfg["min_fee"]
            for delta, p, q in eligible.values()
            if delta > 0
        )
        for side in ("sell", "buy"):
            scale = min(1.0, portfolio.cash / buy_need) if side == "buy" and buy_need else 1.0
            for symbol, (delta, mid, quote) in sorted(eligible.items()):
                if (side == "buy" and delta <= 0) or (side == "sell" and delta >= 0):
                    continue
                inst = self.instruments[symbol]
                price = mid * (
                    1 + self.cfg["slippage"] if side == "buy" else 1 - self.cfg["slippage"]
                )
                order_id = sha256(
                    f"{account}|{target.signal_time}|{symbol}|{side}".encode()
                ).hexdigest()[:24]
                order = {
                    "order_id": order_id,
                    "date": date,
                    "symbol": symbol,
                    "side": side,
                    "requested_quantity": abs(delta),
                    "price": price,
                    "status": "rejected",
                }
                # Known limit quotes must not be crossed. When unknown, official
                # run admission is separately blocked by unverified data/rules.
                limit = quote.get("limit_up" if side == "buy" else "limit_down")
                if (
                    limit is not None
                    and math.isfinite(limit)
                    and (mid >= limit if side == "buy" else mid <= limit)
                ):
                    order["reason"] = "price_limit"
                    orders.append(order)
                    continue
                if side == "sell":
                    quantity = min(abs(delta), portfolio.sellable(symbol, date))
                else:
                    quantity = math.floor(delta * scale / inst.lot_size) * inst.lot_size
                    # Avoid implicit borrowing even with rounding/minimum fees.
                    affordable = (
                        math.floor(
                            max(0.0, portfolio.cash - self.cfg["min_fee"])
                            / (price * (1 + self.cfg["commission"]))
                            / inst.lot_size
                        )
                        * inst.lot_size
                    )
                    quantity = min(quantity, affordable)
                if quantity <= 0:
                    order["reason"] = "cash_or_sellable_quantity_insufficient"
                else:
                    fee = self.fee(quantity * price)
                    fill = Fill(
                        fill_id=order_id,
                        order_id=order_id,
                        date=date,
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        fee=fee,
                        source=account,
                    )
                    applied = portfolio.apply_fill(fill)
                    order.update(
                        quantity=quantity,
                        fee=fee,
                        status=("filled" if abs(quantity - abs(delta)) < 1e-8 else "partial")
                        if applied
                        else "duplicate",
                        reason="budget_or_trading_unit"
                        if abs(quantity - abs(delta)) > 1e-8
                        else "target",
                    )
                orders.append(order)
        return orders
