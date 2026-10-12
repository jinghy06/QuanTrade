"""
A股数据获取适配器 - baostock版
作为akshare的替代/兜底方案，专门用于历史日K数据下载
"""

import sys
from pathlib import Path

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import baostock as bs
import pandas as pd

from config.settings import CACHE_DIR

logger = logging.getLogger(__name__)


class BaostockFetcher:
    """
    baostock数据获取器
    优势: 稳定、免费、支持前复权、支持批量
    劣势: 数据有延迟(当天收盘后次日才能获取)
    """

    # 字段映射: baostock字段 -> 项目标准字段
    FIELD_MAP = {
        "date": "trade_date",
        "code": "symbol_raw",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "preclose": "pre_close",
        "volume": "volume",
        "amount": "amount",
        "turn": "turnover",
        "pctChg": "pct_change",
    }

    def __init__(self):
        self._logged_in = False
        self.cache_dir = CACHE_DIR

    def _ensure_login(self):
        """确保已登录baostock"""
        if not self._logged_in:
            result = bs.login()
            if result.error_code != "0":
                raise RuntimeError(f"baostock登录失败: {result.error_msg}")
            self._logged_in = True
            logger.info("baostock登录成功")

    def _convert_symbol(self, symbol: str) -> str:
        """
        将项目标准格式转换为baostock格式
        000001.SZ -> sz.000001
        600519.SH -> sh.600519
        """
        code, market = symbol.split(".")
        market = market.lower()
        return f"{market}.{code}"

    def _to_standard_df(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """将baostock原始DataFrame转换为项目标准格式"""
        if df.empty:
            return pd.DataFrame()

        # 重命名列
        df = df.rename(columns=self.FIELD_MAP)

        # 转换数据类型
        numeric_cols = ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover", "pct_change"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 格式化日期
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")

        # 添加标准symbol
        df["symbol"] = symbol

        # 计算change（涨跌额）
        if "close" in df.columns and "pre_close" in df.columns:
            df["change"] = df["close"] - df["pre_close"]

        # 计算振幅（如果没有）
        if "amplitude" not in df.columns and all(c in df.columns for c in ["high", "low", "close"]):
            df["amplitude"] = ((df["high"] - df["low"]) / df["pre_close"] * 100).round(2)

        # 选择标准列
        standard_cols = [
            "trade_date", "symbol", "open", "high", "low", "close",
            "volume", "amount", "amplitude", "pct_change", "change", "turnover",
        ]
        available = [c for c in standard_cols if c in df.columns]
        df = df[available].copy()

        # 按日期升序
        df = df.sort_values("trade_date").reset_index(drop=True)

        return df

    def get_daily_k(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "3",  # "3"=前复权, "2"=后复权, "1"=不复权
    ) -> pd.DataFrame:
        """
        获取日K数据

        Args:
            symbol: 如 "000001.SZ"
            start_date: "YYYYMMDD" 或 "YYYY-MM-DD"
            end_date: "YYYYMMDD" 或 "YYYY-MM-DD"
            adjust: 复权类型
        """
        self._ensure_login()

        # 默认日期
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

        # 统一格式为 YYYY-MM-DD
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                start_date = datetime.strptime(start_date, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                pass
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                end_date = datetime.strptime(end_date, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                pass

        bs_symbol = self._convert_symbol(symbol)

        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"

        rs = bs.query_history_k_data_plus(
            bs_symbol,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjust,
        )

        if rs.error_code != "0":
            logger.error("baostock获取 %s 失败: %s", symbol, rs.error_msg)
            return pd.DataFrame()

        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            logger.warning("baostock无数据: %s", symbol)
            return pd.DataFrame()

        df_raw = pd.DataFrame(data_list, columns=rs.fields)
        df = self._to_standard_df(df_raw, symbol)

        logger.info(
            "baostock获取 %s 日K: %d条 (%s ~ %s)",
            symbol, len(df), df["trade_date"].iloc[0], df["trade_date"].iloc[-1],
        )
        return df

    def batch_download(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "3",
    ) -> pd.DataFrame:
        """批量下载多只股票"""
        all_data = []
        success = 0
        failed = 0

        for symbol in symbols:
            try:
                df = self.get_daily_k(symbol, start_date, end_date, adjust)
                if not df.empty:
                    all_data.append(df)
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error("下载 %s 失败: %s", symbol, e)
                failed += 1

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            logger.info(
                "批量下载完成: 成功%d, 失败%d, 共%d条",
                success, failed, len(df_all),
            )
            return df_all
        else:
            logger.warning("批量下载无数据")
            return pd.DataFrame()

    def get_hs300_components(self) -> List[str]:
        """获取沪深300成分股列表"""
        self._ensure_login()

        rs = bs.query_hs300_stocks()
        if rs.error_code != "0":
            logger.error("获取沪深300失败: %s", rs.error_msg)
            return []

        symbols = []
        while (rs.error_code == "0") & rs.next():
            row = rs.get_row_data()
            # row[0]=updateDate, row[1]=code, row[2]=codeName
            code = row[1]
            # 判断市场
            if code.startswith("6"):
                symbols.append(f"{code}.SH")
            else:
                symbols.append(f"{code}.SZ")

        logger.info("沪深300成分股: %d只", len(symbols))
        return symbols

    def logout(self):
        """登出"""
        if self._logged_in:
            bs.logout()
            self._logged_in = False
            logger.info("baostock已登出")

    def __del__(self):
        self.logout()


# ============================================================
# 便捷函数
# ============================================================

def download_to_database(
    symbols: List[str],
    db_path: str,
    years: int = 3,
    adjust: str = "3",
):
    """
    批量下载股票数据并写入SQLite

    Args:
        symbols: 股票代码列表
        db_path: SQLite数据库路径
        years: 下载年数
        adjust: 复权类型 "3"=前复权
    """
    from data.data_store import DataStore

    end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    fetcher = BaostockFetcher()
    store = DataStore(db_path)

    logger.info("批量下载 | 股票池:%d只 | %s ~ %s", len(symbols), start, end)

    for symbol in symbols:
        try:
            df = fetcher.get_daily_k(symbol, start_date=start, end_date=end, adjust=adjust)
            if not df.empty:
                store.save_klines(df)
                logger.info("已保存 %s: %d条", symbol, len(df))
        except Exception as e:
            logger.error("下载 %s 失败: %s", symbol, e)

    fetcher.logout()
    logger.info("批量下载完成")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 快速测试
    fetcher = BaostockFetcher()
    df = fetcher.get_daily_k("000001.SZ", start_date="20250601", end_date="20250605")
    print(df)
    fetcher.logout()
