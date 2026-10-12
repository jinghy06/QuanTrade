"""
A股量化信号系统 - 特征工程
Alpha158简化版 + RD-Agent新因子插槽
对应计划书: "特征工程(Alpha158简化版 + RD-Agent新因子)"
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    ATR_PERIOD,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MOMENTUM_WINDOWS,
    RSI_PERIOD,
    VOLATILITY_WINDOWS,
    VOLUME_MA_WINDOWS,
)
from data.data_store import DataStore

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    特征工程引擎
    输入: 原始日K数据 (daily_prices表)
    输出: ML训练特征 (features表)
    """

    def __init__(self):
        self.store = DataStore()

    # ============================================================
    # 主入口
    # ============================================================

    def compute_features(
        self, df_price: pd.DataFrame, df_macro: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        从日K数据计算全部特征

        Args:
            df_price: 单只股票的日K数据，需包含 OHLCV
            df_macro: 宏观数据（如10年期国债收益率），可选

        Returns:
            DataFrame 包含所有特征列
        """
        if df_price.empty:
            return pd.DataFrame()

        df = df_price.copy().sort_values("trade_date")
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        logger.info("计算特征: %s, %d条K线", df["symbol"].iloc[0], len(df))

        # ---------- 1. 动量特征 ----------
        df = self._add_momentum_features(df)

        # ---------- 2. 波动率特征 ----------
        df = self._add_volatility_features(df)

        # ---------- 3. 量价特征 ----------
        df = self._add_volume_features(df)

        # ---------- 4. 趋势特征 ----------
        df = self._add_trend_features(df)

        # ---------- 5. 宏观代理变量 ----------
        if df_macro is not None and not df_macro.empty:
            df = self._add_macro_features(df, df_macro)
        else:
            df["bond_yield_10y"] = np.nan

        # ---------- 6. RD-Agent因子插槽（预留） ----------
        df = self._add_rd_agent_factors(df)

        # ---------- 7. 目标变量 ----------
        df = self._add_targets(df)

        # 选择输出列
        feature_cols = [
            "trade_date",
            "symbol",
            "return_5d",
            "return_10d",
            "return_20d",
            "rsi_14",
            "macd_dif",
            "macd_dea",
            "macd_hist",
            "std_5d",
            "std_20d",
            "atr_14",
            "volume_ma5",
            "volume_ma20",
            "obv",
            "turnover_ma5",
            "ma_alignment",
            "price_position",
            "trend_slope",
            "vol_percentile",
            "dist_to_support",
            "dist_to_resistance",
            "divergence_bear",
            "bond_yield_10y",
            "rd_factor_1",
            "rd_factor_2",
            "rd_factor_3",
            "target_close_1d",
            "target_close_3d",
            "target_close_5d",
            "target_close_10d",
            "target_return_1d",
            "target_return_3d",
            "target_return_5d",
            "target_return_10d",
            "target_volatility_5d",
            "target_next_day_return",
            "target_direction",
        ]

        return df[[c for c in feature_cols if c in df.columns]].copy()

    def compute_and_save(
        self,
        symbol: str,
        df_price: pd.DataFrame = None,
        df_macro: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        计算特征并保存到数据库
        如果df_price为None，自动从数据库读取
        """
        if df_price is None:
            df_price = self.store.get_kline(symbol)

        df_features = self.compute_features(df_price, df_macro)

        if not df_features.empty:
            self.store.save_features(df_features)

        return df_features

    # ============================================================
    # 特征计算细节
    # ============================================================

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        动量特征
        - N日收益率
        - RSI
        - MACD
        """
        close = df["close"]

        # N日收益率
        for w in MOMENTUM_WINDOWS:
            df[f"return_{w}d"] = close.pct_change(w)

        # RSI (Relative Strength Index)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
        rs = gain / (loss + 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        df["macd_dif"] = ema_fast - ema_slow
        df["macd_dea"] = df["macd_dif"].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df["macd_hist"] = df["macd_dif"] - df["macd_dea"]

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        波动率特征
        - N日标准差
        - ATR (Average True Range)
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # N日标准差
        for w in VOLATILITY_WINDOWS:
            df[f"std_{w}d"] = close.pct_change().rolling(w).std()

        # ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(ATR_PERIOD).mean()

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        量价特征
        - 成交量均线
        - OBV (On Balance Volume)
        - 换手率均线
        """
        volume = df["volume"]

        # 成交量均线
        for w in VOLUME_MA_WINDOWS:
            df[f"volume_ma{w}"] = volume.rolling(w).mean()

        # OBV: 涨时累加成交量，跌时累减（向量化）
        sign = np.sign(df["close"].diff())
        df["obv"] = (sign * volume).fillna(0).cumsum()

        # 换手率均线
        if "turnover" in df.columns:
            df["turnover_ma5"] = df["turnover"].rolling(5).mean()

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """趋势与形态特征"""
        close = df["close"]

        # 1. 均线排列状态（多头排列=1, 空头=-1, 混乱=0）
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        df["ma_alignment"] = ((ma5 > ma10) & (ma10 > ma20)).astype(int) - ((ma5 < ma10) & (ma10 < ma20)).astype(int)

        # 2. 价格相对20日区间的位置（0-1）
        df["price_position"] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 1e-10)

        # 3. 趋势斜率（20日线性回归斜率）
        x = np.arange(20)
        slope = close.rolling(20).apply(lambda y: np.polyfit(x[-len(y):], y, 1)[0] if len(y) >= 10 else np.nan, raw=False)
        df["trend_slope"] = slope

        # 4. 波动率状态（当前ATR处于20日ATR的什么分位）
        atr = df["atr_14"]
        df["vol_percentile"] = atr.rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan, raw=False)

        # 5. 支撑/压力距离
        support = close.rolling(20).min()
        resistance = close.rolling(20).max()
        df["dist_to_support"] = (close - support) / close
        df["dist_to_resistance"] = (resistance - close) / close

        # 6. 量价背离（价格新高但OBV未新高）
        price_high = close == close.rolling(10).max()
        obv_high = df["obv"] == df["obv"].rolling(10).max()
        df["divergence_bear"] = (price_high & ~obv_high).astype(int)

        return df

    def _add_macro_features(
        self, df: pd.DataFrame, df_macro: pd.DataFrame
    ) -> pd.DataFrame:
        """
        宏观代理变量
        - 10年期国债收益率
        """
        if "date" not in df_macro.columns:
            return df

        df_macro = df_macro.copy()
        df_macro["date"] = pd.to_datetime(df_macro["date"])

        # 按最近的有效值前向填充合并
        df = df.merge(
            df_macro[["date", "yield_10y"]],
            left_on="trade_date",
            right_on="date",
            how="left",
        )
        df["bond_yield_10y"] = df["yield_10y"].ffill()
        df = df.drop(columns=["date", "yield_10y"], errors="ignore")

        return df

    def _add_rd_agent_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        RD-Agent生成因子的插槽
        对应 plan.md: "RD-Agent自动生成新因子，人工审核后入库"

        当前为占位符，每月RD-Agent运行后:
        1. 将有效因子表达式写入 features/rd_agent_new/
        2. 从文件动态加载计算
        3. 存入数据库 rd_factor_1/2/3 列
        """
        # TODO: 从 rd_agent_new/ 目录动态加载因子
        # 示例因子（量稳比）: ts_mean(volume, 20) / ts_std(close, 60)
        if len(df) >= 60:
            df["rd_factor_1"] = (
                df["volume"].rolling(20).mean()
                / (df["close"].rolling(60).std() + 1e-10)
            )
        else:
            df["rd_factor_1"] = np.nan

        df["rd_factor_2"] = np.nan  # 预留
        df["rd_factor_3"] = np.nan  # 预留

        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        多步回归目标变量（用于预测K线）
        - target_return_1d/3d/5d/10d: 未来N日收益率（回归目标）
        - target_close_1d/3d/5d/10d: 未来N日收盘价（辅助验证）
        - target_volatility_5d: 未来5日波动率（用于画上下影线）
        - target_direction: 下一日涨跌方向（兼容旧分类逻辑）
        """
        close = df["close"]

        # 未来N日收盘价和收益率
        for horizon in [1, 3, 5, 10]:
            df[f"target_close_{horizon}d"] = close.shift(-horizon)
            df[f"target_return_{horizon}d"] = close.pct_change(horizon).shift(-horizon)

        # 保留旧的方向目标（兼容分类模式）
        df["target_next_day_return"] = close.pct_change().shift(-1)
        df["target_direction"] = (df["target_next_day_return"] > 0).astype(int)

        # 未来波动率目标（用于预测K线影线）
        df["target_volatility_5d"] = close.pct_change().rolling(5).std().shift(-5)

        return df

    # ============================================================
    # 批量处理
    # ============================================================

    def batch_compute(
        self, symbols: List[str], df_macro: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        批量计算多只股票特征并保存
        对应 plan.md Phase 0: "计算第一批技术指标，训练第一个LightGBM"
        """
        all_features = []
        success, failed = 0, 0

        for symbol in symbols:
            try:
                df_price = self.store.get_kline(symbol)
                if len(df_price) < 60:
                    logger.warning(
                        "%s K线数据不足60天，跳过", symbol
                    )
                    continue

                df_feat = self.compute_features(df_price, df_macro)
                if not df_feat.empty:
                    all_features.append(df_feat)
                    success += 1

            except Exception as e:
                logger.error("特征计算失败 %s: %s", symbol, e)
                failed += 1

        if all_features:
            df_all = pd.concat(all_features, ignore_index=True)
            self.store.save_features(df_all)
            logger.info(
                "批量特征计算完成: 成功%d只, 失败%d只, 共%d条",
                success,
                failed,
                len(df_all),
            )
            return df_all
        else:
            return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = FeatureEngine()

    # 快速测试
    from data.data_store import DataStore

    store = DataStore()
    symbols = store.get_all_symbols()
    if symbols:
        df = store.get_kline(symbols[0], n_days=100)
        feats = engine.compute_features(df)
        print(f"\n特征列: {list(feats.columns)}")
        print(f"\n最新特征:\n{feats.iloc[-1]}")
