"""One ETF target interface; risk constraints always follow scoring."""

from datetime import date as Date

from .domain import Target
from .factors import calculate_factors
from .features import compute_features


class ETFStrategy:
    def __init__(self, config, instruments, mode="full"):
        if mode not in ("buy_hold", "fixed", "momentum", "model", "full"):
            raise ValueError("invalid strategy mode")
        self.config = config
        self.instruments = instruments
        self.mode = mode
        self.risk_active = False
        self.last_risk_date = None
        self.initialized = False

    def scheduled(self, date, next_date):
        if not self.initialized:
            return True
        if self.mode == "buy_hold":
            return False
        a, b = Date.fromisoformat(date), Date.fromisoformat(next_date)
        changed = (a.year, a.month) != (b.year, b.month)
        return changed and (
            self.config["strategy"]["frequency"] == "monthly" or b.month in (1, 4, 7, 10)
        )

    def decide(self, history, date, next_date, portfolio, events=None, bundle=None):
        cfg = self.config["strategy"]
        history = history[history.date <= date]
        factors = calculate_factors(
            history, events, date, self.config["data"]["event_min_confidence"]
        )
        regular = self.scheduled(date, next_date)
        risk = False
        risk_reasons = []
        if self.mode == "full" and cfg["factors_enabled"]:
            for kind, threshold, lower in (
                ("sentiment", cfg["sentiment_threshold"], False),
                ("policy", cfg["policy_threshold"], True),
                ("geopolitical", cfg["geopolitical_threshold"], False),
            ):
                value = factors[kind]
                if value["available"] and (
                    value["score"] < threshold if lower else value["score"] > threshold
                ):
                    risk = True
                    risk_reasons.append(f"risk:{kind}:{value['mode']}")
        cooled = (
            self.last_risk_date is None
            or (Date.fromisoformat(date) - Date.fromisoformat(self.last_risk_date)).days
            >= cfg["cooldown_days"]
        )
        # Latch a persistent event: no fresh daily orders during a risk regime.
        event_trigger = risk and not self.risk_active and cooled
        if not regular and not event_trigger:
            return None
        recovering = regular and self.risk_active and not risk and cooled
        if self.risk_active and not recovering:
            risk = True
        if event_trigger:
            self.last_risk_date = date
        self.risk_active = risk
        reasons = ["scheduled" if regular else "risk_event"] + risk_reasons
        if recovering:
            reasons.append("risk_recovery_at_scheduled_rebalance")
        lookback = cfg["lookback"]
        candidates = {}
        for symbol, bars in history.groupby("symbol"):
            inst = self.instruments.get(symbol)
            bars = bars.sort_values("date")
            if inst is None or (inst.listing_date and date < inst.listing_date):
                continue
            if bars.date.iloc[-1] != date or len(bars) <= lookback or bars.volume.iloc[-1] <= 0:
                continue
            candidates[symbol] = float(bars.close.iloc[-1] / bars.close.iloc[-lookback - 1] - 1)
        predictions = {}
        version = None
        if self.mode in ("model", "full"):
            if bundle is not None and bundle.can_predict(date):
                features = compute_features(history)
                latest = features[features.date == date]
                horizon = max(bundle.models)
                predictions = bundle.predict(latest, date)[horizon]
                version = bundle.metadata["version"]
            else:
                reasons.append("fallback:momentum:model_unavailable_or_not_yet_valid")
        scored = {}
        for symbol, momentum in candidates.items():
            if self.mode in ("buy_hold", "fixed"):
                scored[symbol] = 0.0
            elif momentum > 0:
                w = cfg["model_weight"] if symbol in predictions else 0.0
                scored[symbol] = (1 - w) * momentum + w * predictions.get(symbol, 0.0)
        ranked = sorted(scored, key=lambda s: (-scored[s], s))
        if self.mode not in ("buy_hold", "fixed"):
            ranked = ranked[: cfg["max_positions"]]
        exposure = 1.0 - cfg["cash_reserve"]
        if risk:
            exposure *= cfg["risk_reduction"]
        weights, themes = {}, {}
        for symbol in ranked:
            # Unknown classifications share a conservative single bucket.
            theme = self.instruments[symbol].theme or "unknown"
            equal = exposure / len(ranked)
            weight = min(
                equal, cfg["max_weight"], max(0.0, cfg["max_theme_weight"] - themes.get(theme, 0.0))
            )
            if weight > 0:
                weights[symbol] = weight
                themes[theme] = themes.get(theme, 0.0) + weight
        if event_trigger and not regular and portfolio.nav > 0:
            # Emergency targets only reduce existing risk. No new names or
            # accidental increase while cutting risk outside the schedule.
            weights, themes = {}, {}
            for s, p in sorted(portfolio.positions.items()):
                if p.quantity <= 0:
                    continue
                theme = self.instruments[s].theme or "unknown"
                weight = min(
                    p.quantity * p.last_price / portfolio.nav * cfg["risk_reduction"],
                    cfg["max_weight"],
                    max(0, cfg["max_theme_weight"] - themes.get(theme, 0)),
                )
                if weight > 0:
                    weights[s] = weight
                    themes[theme] = themes.get(theme, 0) + weight
        self.initialized = True
        if not weights:
            reasons.append("cash:no_eligible_candidates")
        reasons.append("unallocated_weight_remains_cash")
        return Target(
            signal_time=f"{date}T15:30:00+08:00",
            weights=weights,
            reasons=reasons,
            valid_until=f"{next_date}T15:00:00+08:00",
            model_version=version,
            factors=factors,
        )
