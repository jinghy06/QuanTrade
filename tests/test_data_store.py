"""Tests for the data store module."""

import pandas as pd
import pytest

from data.data_store import DataStore


class TestKlineOperations:
    def test_save_and_get_kline_roundtrip(self, temp_db, sample_kline_df):
        temp_db.save_klines(sample_kline_df)
        df = temp_db.get_kline("000001.SZ", n_days=50)
        assert not df.empty
        assert "close" in df.columns
        assert len(df) <= 50

    def test_get_kline_date_range(self, temp_db, sample_kline_df):
        temp_db.save_klines(sample_kline_df)
        min_date, max_date = temp_db.get_date_range("000001.SZ")
        assert min_date is not None
        assert max_date is not None
        assert min_date <= max_date

    def test_get_all_symbols(self, temp_db, sample_kline_df):
        temp_db.save_klines(sample_kline_df)
        symbols = temp_db.get_all_symbols()
        assert "000001.SZ" in symbols


class TestSignalOperations:
    def test_has_today_signal_false_when_empty(self, temp_db):
        assert not temp_db.has_today_signal("000001.SZ")

    def test_save_signal_and_has_today(self, temp_db):
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        signal = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "trade_date": today,
            "symbol": "000001.SZ",
            "name": "平安银行",
            "signal": "轻仓试探",
            "confidence": 0.62,
            "ml_prediction": {"up_prob": 0.58, "volatility": "中"},
            "technical": {"trend": "短期反弹", "rsi": 45, "macd": "金叉初期"},
            "macro": {"rate_env": "降息周期", "market_sentiment": "谨慎"},
            "suggestion": {
                "action": "MACD金叉，ML显示58%上涨概率，轻仓试探",
                "target_price": 12.50,
                "stop_loss": 11.80,
                "position_pct": 0.05,
                "rationale": "ML模型显示58%上涨概率，MACD刚形成金叉。",
                "risk_factors": ["大盘情绪谨慎"],
            },
        }
        sid = temp_db.save_signal(signal)
        assert sid > 0
        assert temp_db.has_today_signal("000001.SZ")

    def test_get_recent_signals(self, temp_db):
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        signal = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "trade_date": today,
            "symbol": "000001.SZ",
            "name": "平安银行",
            "signal": "轻仓试探",
            "confidence": 0.62,
            "ml_prediction": {"up_prob": 0.58, "volatility": "中"},
            "technical": {"trend": "短期反弹", "rsi": 45, "macd": "金叉初期"},
            "macro": {"rate_env": "降息周期", "market_sentiment": "谨慎"},
            "suggestion": {
                "action": "试探",
                "target_price": 12.50,
                "stop_loss": 11.80,
                "position_pct": 0.05,
                "rationale": "测试",
                "risk_factors": [],
            },
        }
        temp_db.save_signal(signal)
        df = temp_db.get_recent_signals(symbol="000001.SZ", n=5)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "000001.SZ"


class TestFeatureOperations:
    def test_save_and_get_features(self, temp_db, sample_kline_df):
        from features.feature_engine import FeatureEngine
        engine = FeatureEngine()
        df_feat = engine.compute_features(sample_kline_df)
        if not df_feat.empty:
            temp_db.save_features(df_feat)
            df = temp_db.get_features("000001.SZ", n_days=10)
            assert not df.empty
            assert "rsi_14" in df.columns
