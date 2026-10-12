"""
下载更多A股大盘股数据 - 多源容灾版
脚本位置: QuanTrade/quant_system/scripts/download_more_stocks.py

任务: 下载10只高市值、高流动性A股大盘股历史日K数据
数据源: AkShare (主源, 预期网络受限) -> 新浪财经 (兜底, 支持1000条)
入库: daily_prices 表 (INSERT OR REPLACE)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import time
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import requests

from data.data_store import DataStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("DownloadMoreStocks")

# ============================================================
# 配置
# ============================================================
DB_PATH = r"C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db"
START_DATE = "20240102"
END_DATE = "20260605"
SINA_DATALEN = 1000  # 新浪财经单次可获取约1000条，覆盖2024-01-02~2026-06-05

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

# 10只高市值、高流动性A股大盘股（与现有4只不同行业）
NEW_STOCKS = [
    "600519.SH",   # 贵州茅台 (白酒)
    "300750.SZ",   # 宁德时代 (新能源)
    "601318.SH",   # 中国平安 (金融)
    "600036.SH",   # 招商银行 (银行)
    "000858.SZ",   # 五粮液 (白酒)
    "002594.SZ",   # 比亚迪 (汽车)
    "600900.SH",   # 长江电力 (电力)
    "601012.SH",   # 隆基绿能 (光伏)
    "600276.SH",   # 恒瑞医药 (医药)
    "000725.SZ",   # 京东方A (电子)
]


# ============================================================
# AkShare 股票接口 (预期网络受限)
# ============================================================
def _fetch_akshare_stock(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """用akshare获取A股历史数据（带请求头伪装）"""
    import akshare as ak
    code = symbol.split(".")[0]

    original_get = requests.get

    def _patched_get(url, **kwargs):
        kwargs.setdefault("headers", {})
        kwargs["headers"].update(HEADERS)
        kwargs.setdefault("timeout", 20)
        return original_get(url, **kwargs)

    requests.get = _patched_get
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    finally:
        requests.get = original_get
    return df


# ============================================================
# 新浪财经 股票接口 (兜底)
# ============================================================
def _fetch_sina_stock(symbol: str, n_days: int = SINA_DATALEN) -> pd.DataFrame:
    """用新浪财经获取A股历史数据"""
    code, market = symbol.split(".")
    sina_symbol = f"{market.lower()}{code}"

    url = (
        f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={sina_symbol}"
        f"&scale=240&ma=5&datalen={n_days}"
    )

    r = requests.get(url, timeout=20, headers=HEADERS)
    data = r.json()
    if not data:
        return pd.DataFrame()

    records = []
    prev_close = None
    for item in data:
        close = float(item["close"])
        open_ = float(item["open"])
        high = float(item["high"])
        low = float(item["low"])
        volume = float(item["volume"])

        if prev_close is not None and prev_close > 0:
            pct_change = round((close - prev_close) / prev_close * 100, 4)
            change = round(close - prev_close, 4)
            amplitude = round((high - low) / prev_close * 100, 4)
        else:
            pct_change = 0.0
            change = 0.0
            amplitude = round((high - low) / open_ * 100, 4) if open_ > 0 else 0.0

        prev_close = close

        records.append({
            "trade_date": item["day"].replace("-", ""),
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": 0.0,
            "amplitude": amplitude,
            "pct_change": pct_change,
            "change": change,
            "turnover": 0.0,
        })

    df = pd.DataFrame(records)
    return df.sort_values("trade_date").reset_index(drop=True)


# ============================================================
# 多源下载主函数
# ============================================================
def download_stock_history(
    symbol: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    下载单只股票历史日K - 多源容灾
    优先AkShare，失败则fallback新浪财经
    """
    sources_tried = []
    df = pd.DataFrame()

    # ---------- 尝试1: AkShare (主源) ----------
    try:
        logger.info("[%s] 尝试AkShare获取...", symbol)
        df = _fetch_akshare_stock(symbol, start_date, end_date, adjust="qfq")
        if df is not None and not df.empty:
            logger.info("[%s] AkShare成功: %d条", symbol, len(df))
        else:
            logger.warning("[%s] AkShare返回空数据", symbol)
            df = pd.DataFrame()
    except Exception as e:
        logger.warning("[%s] AkShare失败: %s", symbol, e)
        sources_tried.append(f"akshare:{e}")
        df = pd.DataFrame()

    # ---------- 尝试2: 新浪财经 (兜底) ----------
    if df.empty:
        try:
            logger.info("[%s] 尝试新浪财经获取...", symbol)
            df = _fetch_sina_stock(symbol, n_days=SINA_DATALEN)
            if not df.empty:
                # 过滤日期范围
                df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
                logger.info("[%s] 新浪财经成功: %d条 (过滤后)", symbol, len(df))
            else:
                logger.warning("[%s] 新浪财经返回空数据", symbol)
        except Exception as e:
            logger.warning("[%s] 新浪财经失败: %s", symbol, e)
            sources_tried.append(f"sina:{e}")
            df = pd.DataFrame()

    if df.empty:
        logger.error("[%s] 所有数据源均失败: %s", symbol, sources_tried)
        return pd.DataFrame()

    # ---------- 标准化处理 ----------
    # 列名映射（中文 -> 英文）- 仅akshare返回中文列名时需要
    rename_map = {
        "日期": "trade_date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    df = df.rename(columns=rename_map)

    # 确保symbol列存在
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    # 格式化日期
    if pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
        df["trade_date"] = df["trade_date"].dt.strftime("%Y%m%d")
    else:
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")

    # 数值转换
    numeric_cols = ["open", "high", "low", "close", "volume", "amount",
                    "amplitude", "pct_change", "change", "turnover"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 确保标准列顺序
    standard_cols = [
        "trade_date", "symbol", "open", "high", "low", "close",
        "volume", "amount", "amplitude", "pct_change", "change", "turnover",
    ]
    available = [c for c in standard_cols if c in df.columns]
    df = df[available].copy()

    df = df.sort_values("trade_date").reset_index(drop=True)

    if not df.empty:
        logger.info(
            "[%s] 最终获取: %d条 (%s ~ %s)",
            symbol, len(df), df["trade_date"].iloc[0], df["trade_date"].iloc[-1],
        )
    return df


# ============================================================
# 批量下载并入库
# ============================================================
def batch_download_stocks(
    symbols: List[str],
    db_path: str = DB_PATH,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
):
    """批量下载股票并入库"""
    store = DataStore(db_path)

    logger.info("=" * 60)
    logger.info("批量下载A股大盘股 | 数量: %d | %s ~ %s", len(symbols), start_date, end_date)
    logger.info("=" * 60)

    success = 0
    failed = 0
    stats = {}

    for symbol in symbols:
        try:
            df = download_stock_history(symbol, start_date=start_date, end_date=end_date)
            if not df.empty:
                store.save_klines(df)
                success += 1
                stats[symbol] = len(df)
                logger.info("[OK] %s: %d条已保存", symbol, len(df))
            else:
                failed += 1
                stats[symbol] = 0
                logger.warning("[FAIL] %s: 无数据", symbol)
        except Exception as e:
            logger.error("[ERR] %s: %s", symbol, e)
            failed += 1
            stats[symbol] = 0

        # 礼貌延迟，避免请求过快
        time.sleep(0.5)

    # ---------- 打印统计 ----------
    logger.info("=" * 60)
    logger.info("下载完成: 成功%d, 失败%d", success, failed)
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("【各标的数据量统计】")
    print("=" * 60)
    for sym, cnt in sorted(stats.items()):
        status = "[OK]" if cnt > 0 else "[FAIL]"
        print(f"  {status} {sym}: {cnt}条")

    # 数据库总体统计
    print("\n" + "=" * 60)
    print("【数据库 daily_prices 总体统计】")
    print("=" * 60)
    with store._connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, COUNT(*) as cnt FROM daily_prices GROUP BY symbol ORDER BY cnt DESC")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}条")

    return stats


if __name__ == "__main__":
    stats = batch_download_stocks(NEW_STOCKS)
