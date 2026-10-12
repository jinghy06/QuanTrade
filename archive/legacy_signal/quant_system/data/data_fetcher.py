"""
A股数据获取模块 - 双源容灾
主源: AkShare（免费，覆盖全面）
兜底: Tushare（稳定性高，需token）
"""

import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd

from config.settings import CACHE_DIR, DB_PATH, TUSHARE_TOKEN

logger = logging.getLogger(__name__)


class DataFetcher:
    """A股数据获取器，支持AkShare主源+Tushare兜底"""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.tushare_token = TUSHARE_TOKEN
        self._tushare_api = None
        logger.info("DataFetcher初始化完成，缓存目录: %s", self.cache_dir)

    def _get_tushare_api(self):
        """惰性初始化Tushare API"""
        if self._tushare_api is None and self.tushare_token:
            try:
                import tushare as ts

                self._tushare_api = ts.pro_api(self.tushare_token)
                logger.info("Tushare API初始化成功")
            except Exception as e:
                logger.warning("Tushare初始化失败: %s", e)
        return self._tushare_api

    def _cache_path(self, symbol: str, data_type: str) -> Path:
        """生成缓存文件路径"""
        return self.cache_dir / f"{symbol}_{data_type}.csv"

    def _save_cache(self, df: pd.DataFrame, symbol: str, data_type: str):
        """保存数据到本地缓存"""
        path = self._cache_path(symbol, data_type)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.debug("缓存已保存: %s", path)

    def _load_cache(
        self, symbol: str, data_type: str, max_age_days: int = 1
    ) -> Optional[pd.DataFrame]:
        """加载本地缓存，检查新鲜度"""
        path = self._cache_path(symbol, data_type)
        if not path.exists():
            return None

        # 检查缓存新鲜度
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=max_age_days):
            logger.debug("缓存过期: %s", path)
            return None

        try:
            df = pd.read_csv(path, parse_dates=["trade_date"])
            logger.debug("缓存命中: %s, %d条记录", path, len(df))
            return df
        except Exception as e:
            logger.warning("读取缓存失败 %s: %s", path, e)
            return None

    # ============================================================
    # 公开接口
    # ============================================================

    def get_daily_k(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取日K数据

        Args:
            symbol: A股代码，如"000001.SZ"
            start_date: 起始日期，格式"YYYYMMDD"，默认3年前
            end_date: 结束日期，默认昨天
            use_cache: 是否使用本地缓存

        Returns:
            DataFrame with columns: trade_date, open, high, low, close, volume, amount
        """
        # 默认值
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365 * 3)).strftime(
                "%Y%m%d"
            )

        # 尝试缓存
        if use_cache:
            cached = self._load_cache(symbol, "daily", max_age_days=1)
            if cached is not None:
                mask = (cached["trade_date"] >= start_date) & (
                    cached["trade_date"] <= end_date
                )
                filtered = cached.loc[mask].copy()
                if len(filtered) > 0:
                    return filtered

        # AkShare获取
        df = self._fetch_akshare_daily(symbol, start_date, end_date)

        # AkShare失败，尝试Tushare兜底
        if df is None or df.empty:
            logger.warning("AkShare获取失败，尝试Tushare兜底: %s", symbol)
            df = self._fetch_tushare_daily(symbol, start_date, end_date)

        if df is not None and not df.empty:
            self._save_cache(df, symbol, "daily")
            return df

        raise RuntimeError(f"无法获取 {symbol} 的日K数据")

    def get_hs300_components(self) -> list:
        """获取沪深300成分股列表"""
        try:
            df = ak.index_stock_cons_weight_csindex(symbol="000300")
            symbols = df["成分券代码"].tolist()
            # 格式化代码
            formatted = []
            for s in symbols:
                s = str(s).zfill(6)
                if s.startswith("6"):
                    formatted.append(f"{s}.SH")
                else:
                    formatted.append(f"{s}.SZ")
            logger.info("沪深300成分股获取: %d只", len(formatted))
            return formatted
        except Exception as e:
            logger.error("获取沪深300成分股失败: %s", e)
            return []

    def get_macro_bond_yield(self) -> pd.DataFrame:
        """
        获取中国10年期国债收益率（宏观代理变量）
        Returns DataFrame with columns: date, yield_10y
        """
        try:
            # AkShare获取国债收益率
            df = ak.bond_zh_us_rate()
            df = df[["日期", "中国国债收益率10年"]].copy()
            df.columns = ["date", "yield_10y"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna()
            logger.info("10年期国债收益率获取: %d条", len(df))
            return df
        except Exception as e:
            logger.error("获取国债收益率失败: %s", e)
            return pd.DataFrame(columns=["date", "yield_10y"])

    def get_stock_name(self, symbol: str) -> str:
        """通过代码获取股票名称"""
        try:
            code = symbol.split(".")[0]
            df = ak.stock_individual_info_em(symbol=code)
            if not df.empty and "股票简称" in df["item"].values:
                return df.loc[df["item"] == "股票简称", "value"].values[0]
        except Exception as e:
            logger.warning("获取股票名称失败 %s: %s", symbol, e)
        return symbol

    # ============================================================
    # 私有获取方法
    # ============================================================

    def _fetch_akshare_daily(
        self, symbol: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """通过AkShare获取日K"""
        try:
            code = symbol.split(".")[0]
            # 判断市场
            market = "sh" if symbol.endswith(".SH") else "sz"

            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",  # 前复权
            )

            if df is None or df.empty:
                return None

            df = df.rename(
                columns={
                    "日期": "trade_date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_change",
                    "涨跌额": "change",
                    "换手率": "turnover",
                    "股票代码": "symbol_code",  # 新版本akshare可能返回
                }
            )
            
            # 删除不在数据库表结构中的列
            expected_cols = [
                "trade_date", "open", "high", "low", "close", 
                "volume", "amount", "amplitude", "pct_change", "change", "turnover"
            ]
            df = df[[c for c in df.columns if c in expected_cols or c == "symbol"]]
            
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime(
                "%Y%m%d"
            )
            df["symbol"] = symbol

            # 按日期升序排列
            df = df.sort_values("trade_date").reset_index(drop=True)

            logger.info(
                "AkShare获取 %s 日K: %d条 (%s ~ %s)",
                symbol,
                len(df),
                df["trade_date"].iloc[0],
                df["trade_date"].iloc[-1],
            )
            return df

        except Exception as e:
            logger.error("AkShare获取 %s 失败: %s", symbol, e)
            return None

    def _fetch_tushare_daily(
        self, symbol: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """通过Tushare获取日K（兜底）"""
        api = self._get_tushare_api()
        if api is None:
            return None

        try:
            # 转换代码格式 000001.SZ -> 000001.SZ
            ts_code = symbol

            df = api.daily(ts_code=ts_code, start_date=start, end_date=end)

            if df is None or df.empty:
                return None

            df = df.rename(
                columns={
                    "trade_date": "trade_date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "vol": "volume",
                    "amount": "amount",
                }
            )
            df["symbol"] = symbol
            df = df.sort_values("trade_date").reset_index(drop=True)

            logger.info(
                "Tushare兜底获取 %s: %d条",
                symbol,
                len(df),
            )
            return df

        except Exception as e:
            logger.error("Tushare获取 %s 失败: %s", symbol, e)
            return None


def init_stock_data(
    symbols: list, db_path: str = None, years: int = 3
) -> None:
    """
    批量初始化股票数据，下载到SQLite数据库
    首次运行或新增股票时调用

    对应计划书 Phase 0: "AkShare下载沪深300成分股3年历史日K数据"
    """
    db = db_path or str(DB_PATH)
    fetcher = DataFetcher()
    end = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # 建表（daily_prices）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            trade_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            amplitude REAL,
            pct_change REAL,
            change REAL,
            turnover REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol)
        )
    """)

    # 创建索引加速查询
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_daily_symbol ON daily_prices(symbol)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_prices(trade_date)"
    )
    conn.commit()

    def _fetch_and_save(symbol):
        time.sleep(0.3)  # 速率限制
        df = fetcher.get_daily_k(symbol, start, end, use_cache=True)
        if df is not None and not df.empty:
            # 使用独立连接写入（线程安全）
            with sqlite3.connect(db) as write_conn:
                write_conn.execute("DELETE FROM daily_prices WHERE symbol = ?", (symbol,))
                df.to_sql("daily_prices", write_conn, if_exists="append", index=False)
            return symbol
        return None

    success, failed = 0, 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_and_save, s): s for s in symbols}
        for i, future in enumerate(as_completed(futures), 1):
            sym = futures[future]
            try:
                result = future.result()
                if result:
                    success += 1
                    logger.info("[%d/%d] 下载完成 %s", i, len(symbols), sym)
                else:
                    failed += 1
                    logger.warning("[%d/%d] 无数据 %s", i, len(symbols), sym)
            except Exception as e:
                failed += 1
                logger.error("下载 %s 失败: %s", sym, e)

    conn.close()

    logger.info(
        "数据初始化完成: 成功%d只, 失败%d只, 数据库: %s", success, failed, db
    )


if __name__ == "__main__":
    # 命令行快速测试
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        fetcher = DataFetcher()
        df = fetcher.get_daily_k(symbol)
        print(df.tail())
    else:
        print("用法: python data_fetcher.py 000001.SZ")
