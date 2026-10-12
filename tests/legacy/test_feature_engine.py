"""Tests for the feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from features.feature_engine import FeatureEngine


class TestComputeFeatures:
    def test_returns_expected_columns(self, sample_kline_df, mock_feature_engine):
        df = mock_feature_engine.compute_features(sample_kline_df)
        expected_cols = [
            "trade_date", "symbol",
            "return_5d", "return_10d", "return_20d",
            "rsi_14", "macd_dif", "macd_dea", "macd_hist",
            "std_5d", "std_20d", "atr_14",
            "volume_ma5", "volume_ma20", "obv", "turnover_ma5",
            "bond_yield_10y",
            "rd_factor_1", "rd_factor_2", "rd_factor_3",
            "target_next_day_return", "target_direction",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_empty_input_returns_empty(self, mock_feature_engine):
        df = mock_feature_engine.compute_features(pd.DataFrame())
        assert df.empty

    def test_rsi_range(self, sample_kline_df, mock_feature_engine):
        df = mock_feature_engine.compute_features(sample_kline_df)
        rsi = df["rsi_14"].dropna()
        assert ((rsi >= 0) & (rsi <= 100)).all()

    def test_macd_columns_not_all_nan(self, sample_kline_df, mock_feature_engine):
        df = mock_feature_engine.compute_features(sample_kline_df)
        assert df["macd_dif"].notna().sum() > 0
        assert df["macd_dea"].notna().sum() > 0
        assert df["macd_hist"].notna().sum() > 0


class TestAddTarget:
    def test_target_direction_is_binary(self, sample_kline_df, mock_feature_engine):
        df = mock_feature_engine.compute_features(sample_kline_df)
        target = df["target_direction"].dropna()
        assert set(target.unique()).issubset({0, 1})

    def test_target_next_day_return_matches_direction(self, sample_kline_df, mock_feature_engine):
        df = mock_feature_engine.compute_features(sample_kline_df)
        valid = df.dropna(subset=["target_next_day_return", "target_direction"])
        expected = (valid["target_next_day_return"] > 0).astype(int)
        expected.name = "target_direction"
        actual = valid["target_direction"].reset_index(drop=True)
        expected = expected.reset_index(drop=True)
        pd.testing.assert_series_equal(actual, expected)


class TestMomentumFeatures:
    def test_return_windows(self, sample_kline_df, mock_feature_engine):
        df = mock_feature_engine._add_momentum_features(sample_kline_df.copy())
        assert "return_5d" in df.columns
        assert "return_10d" in df.columns
        assert "return_20d" in df.columns

    def test_rsi_computed(self, sample_kline_df, mock_feature_engine):
        df = mock_feature_engine._add_momentum_features(sample_kline_df.copy())
        assert "rsi_14" in df.columns
        assert df["rsi_14"].notna().sum() > 0


class TestVolumeFeatures:
    def test_obv_monotonic_when_always_up(self, mock_feature_engine):
        """If price only goes up, OBV should equal cumulative volume."""
        n = 20
        df = pd.DataFrame({
            "close": np.arange(100, 100 + n),
            "volume": np.ones(n) * 1000,
        })
        df = mock_feature_engine._add_volume_features(df)
        # OBV: first row has no prior close, so diff is NaN -> sign=0 -> OBV starts at 0
        expected_obv = np.concatenate([[0.0], np.cumsum(np.ones(n - 1) * 1000)])
        np.testing.assert_array_equal(df["obv"].values, expected_obv)
