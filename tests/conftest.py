"""pytest fixtures for the quant system test suite."""

import sys
from pathlib import Path

# Ensure the quant_system package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "QuanTrade" / "quant_system"))

import numpy as np
import pandas as pd
import pytest

from data.data_store import DataStore
from features.feature_engine import FeatureEngine


@pytest.fixture
def sample_kline_df() -> pd.DataFrame:
    """Return a DataFrame with 100 days of synthetic OHLCV data."""
    np.random.seed(42)
    n = 100
    base_price = 100.0
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")

    # Generate a random walk for close prices
    returns = np.random.normal(loc=0.0005, scale=0.02, size=n)
    close = base_price * np.exp(np.cumsum(returns))

    # Derive open, high, low from close with small noise
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.01, n)))
    volume = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)
    amount = volume * close * np.random.uniform(0.9, 1.1, n)
    amplitude = ((high - low) / low * 100).round(2)
    pct_change = pd.Series(close / np.roll(close, 1) - 1).fillna(0) * 100
    change = pd.Series(close - np.roll(close, 1)).fillna(0)
    turnover = np.random.uniform(0.5, 5.0, size=n).round(2)

    df = pd.DataFrame({
        "trade_date": dates.strftime("%Y%m%d"),
        "symbol": "000001.SZ",
        "open": open_.round(2),
        "high": high.round(2),
        "low": low.round(2),
        "close": close.round(2),
        "volume": volume.astype(int),
        "amount": amount.round(2),
        "amplitude": amplitude,
        "pct_change": pct_change.round(2),
        "change": change.round(2),
        "turnover": turnover,
    })
    return df


@pytest.fixture
def temp_db(tmp_path) -> DataStore:
    """Return a DataStore backed by a temporary SQLite database."""
    db_path = tmp_path / "test_quant.db"
    store = DataStore(db_path=str(db_path))
    yield store
    # Cleanup is handled by tmp_path pytest fixture


@pytest.fixture
def mock_feature_engine() -> FeatureEngine:
    """Return a FeatureEngine instance."""
    return FeatureEngine()
