"""Tests for the strategy system."""

import pandas as pd
import pytest

from strategies.base import BaseStrategy, SignalResult
from strategies.registry import StrategyRegistry


class TestSignalResult:
    def test_to_dict_serializes_correctly(self):
        sr = SignalResult(
            strategy_name="test",
            triggered=True,
            action="buy",
            confidence=0.75,
            price=10.5,
            rationale="test rationale",
        )
        d = sr.to_dict()
        assert d["strategy_name"] == "test"
        assert d["triggered"] is True
        assert d["action"] == "buy"
        assert d["confidence"] == 0.75

    def test_bool_returns_triggered(self):
        assert bool(SignalResult("s1", triggered=True))
        assert not bool(SignalResult("s1", triggered=False))


class MockBuyStrategy(BaseStrategy):
    name = "mock_buy"
    display_name = "Mock Buy"

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        if not self.validate_data(df):
            return SignalResult(self.name, triggered=False)
        return SignalResult(
            strategy_name=self.name,
            triggered=True,
            action="buy",
            confidence=0.8,
            price=df["close"].iloc[-1],
            rationale="mock buy signal",
        )


class MockSellStrategy(BaseStrategy):
    name = "mock_sell"
    display_name = "Mock Sell"

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        if not self.validate_data(df):
            return SignalResult(self.name, triggered=False)
        return SignalResult(
            strategy_name=self.name,
            triggered=True,
            action="sell",
            confidence=0.7,
            price=df["close"].iloc[-1],
            rationale="mock sell signal",
        )


class TestStrategyRegistry:
    def test_register_and_unregister(self):
        reg = StrategyRegistry()
        strategy = MockBuyStrategy()
        reg.register(strategy)
        assert "mock_buy" in reg
        assert reg.unregister("mock_buy")
        assert "mock_buy" not in reg

    def test_register_non_strategy_raises(self):
        reg = StrategyRegistry()
        with pytest.raises(TypeError):
            reg.register("not a strategy")

    def test_run_all_executes_strategies(self, sample_kline_df):
        reg = StrategyRegistry()
        reg.register(MockBuyStrategy())
        results = reg.run_all(sample_kline_df, symbol="000001.SZ")
        assert "mock_buy" in results
        assert results["mock_buy"].triggered

    def test_aggregate_voting_majority_tie(self, sample_kline_df):
        """1 buy vs 1 sell -> majority tie -> no signal."""
        reg = StrategyRegistry()
        results = {
            "mock_buy": SignalResult(
                strategy_name="mock_buy", triggered=True, action="buy", confidence=0.8
            ),
            "mock_sell": SignalResult(
                strategy_name="mock_sell", triggered=True, action="sell", confidence=0.7
            ),
        }
        agg = reg.aggregate_voting(results, method="majority")
        assert not agg.triggered
        assert "平局" in agg.rationale

    def test_aggregate_voting_confidence_weighted_buy(self, sample_kline_df):
        reg = StrategyRegistry()
        results = {
            "s1": SignalResult("s1", triggered=True, action="buy", confidence=0.9),
            "s2": SignalResult("s2", triggered=True, action="buy", confidence=0.8),
            "s3": SignalResult("s3", triggered=True, action="sell", confidence=0.3),
        }
        agg = reg.aggregate_voting(results, method="confidence_weighted")
        assert agg.triggered
        assert agg.action == "buy"

    def test_aggregate_voting_unanimous(self, sample_kline_df):
        reg = StrategyRegistry()
        results = {
            "s1": SignalResult("s1", triggered=True, action="buy", confidence=0.9),
            "s2": SignalResult("s2", triggered=True, action="buy", confidence=0.8),
        }
        agg = reg.aggregate_voting(results, method="unanimous")
        assert agg.triggered
        assert agg.action == "buy"

        # Mixed directions should fail unanimous
        results["s3"] = SignalResult("s3", triggered=True, action="sell", confidence=0.9)
        agg2 = reg.aggregate_voting(results, method="unanimous")
        assert not agg2.triggered


class TestBaseStrategy:
    def test_validate_data_empty(self):
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def generate_signal(self, df):
                return SignalResult(self.name, triggered=False)

        s = DummyStrategy()
        assert not s.validate_data(pd.DataFrame())
        assert not s.validate_data(None)

    def test_validate_data_missing_columns(self):
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            required_columns = ["close", "volume"]
            def generate_signal(self, df):
                return SignalResult(self.name, triggered=False)

        s = DummyStrategy()
        df = pd.DataFrame({"open": [1, 2, 3]})
        assert not s.validate_data(df)

    def test_validate_data_insufficient_bars(self):
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            min_bars = 10
            def generate_signal(self, df):
                return SignalResult(self.name, triggered=False)

        s = DummyStrategy()
        df = pd.DataFrame({"close": range(5)})
        assert not s.validate_data(df)

    def test_validate_data_passes(self):
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def generate_signal(self, df):
                return SignalResult(self.name, triggered=False)

        s = DummyStrategy()
        df = pd.DataFrame({"close": range(100)})
        assert s.validate_data(df)
