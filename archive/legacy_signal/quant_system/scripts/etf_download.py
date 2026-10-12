"""
ETF数据下载脚本 - 多源容灾版
主源: AkShare (fund_etf_hist_em) - 东方财富，数据最全
兜底1: 新浪财经 (sina_etf_download) - 稳定但条数有限(~300条)
兜底2: Baostock - ETF支持有限

修复内容:
1. 修复命令行参数传递bug (years参数误传给download_etf_history)
2. 增加重试机制 (指数退避 + 请求头伪装)
3. 增加自动切换备用数据源
4. 增加网络诊断和详细日志
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional

import akshare as ak
import pandas as pd
import requests

from data.data_store import DataStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("ETFDownload")

# ============================================================
# 配置
# ============================================================
MAX_RETRIES = 3          # akshare最大重试次数
RETRY_DELAY_BASE = 2     # 重试基础延迟(秒)
SINA_MAX_DAYS = 300      # 新浪财经最大返回条数

# 请求头伪装 - 模拟浏览器
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://quote.eastmoney.com/",
}


def _retry_with_backoff(func, max_retries=MAX_RETRIES, base_delay=RETRY_DELAY_BASE):
    """带指数退避的重试装饰器"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"第{attempt + 1}次尝试失败: {e}, {delay}秒后重试...")
                time.sleep(delay)
            else:
                raise
    return None


def _fetch_akshare_etf(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """内部函数: 用akshare获取ETF历史数据（带重试）"""
    code = symbol.split(".")[0]

    def _do_fetch():
        # 临时修改requests的默认headers（akshare内部使用requests）
        original_get = requests.get

        def _patched_get(url, **kwargs):
            kwargs.setdefault("headers", {})
            kwargs["headers"].update(HEADERS)
            kwargs.setdefault("timeout", 20)
            return original_get(url, **kwargs)

        requests.get = _patched_get
        try:
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        finally:
            requests.get = original_get
        return df

    return _retry_with_backoff(_do_fetch)


def _fetch_sina_etf(symbol: str, n_days: int = SINA_MAX_DAYS) -> pd.DataFrame:
    """内部函数: 用新浪财经获取ETF历史数据"""
    code, market = symbol.split(".")
    sina_symbol = f"{market.lower()}{code}"

    url = (
        f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={sina_symbol}"
        f"&scale=240&ma=5&datalen={n_days}"
    )

    def _do_fetch():
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

    return _retry_with_backoff(_do_fetch, max_retries=2, base_delay=1)


def download_etf_history(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    adjust: str = "qfq",
    prefer_source: str = "auto",  # 'auto', 'akshare', 'sina'
) -> pd.DataFrame:
    """
    下载单只ETF历史日K - 多源容灾

    Args:
        symbol: ETF代码，如 "562500.SH" 或 "159382.SZ"
        start_date: YYYYMMDD，默认1年前
        end_date: YYYYMMDD，默认昨天
        adjust: qfq=前复权, hfq=后复权, ""=不复权
        prefer_source: 首选数据源 ('auto'=自动选择, 'akshare', 'sina')

    Returns:
        标准格式DataFrame
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365 * 1)).strftime("%Y%m%d")

    sources_tried = []
    df = pd.DataFrame()

    # ---------- 尝试1: AkShare (主源) ----------
    if prefer_source in ("auto", "akshare"):
        try:
            logger.info("[%s] 尝试AkShare获取...", symbol)
            df = _fetch_akshare_etf(symbol, start_date, end_date, adjust)
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
    if df.empty and prefer_source in ("auto", "sina"):
        try:
            logger.info("[%s] 尝试新浪财经获取...", symbol)
            df = _fetch_sina_etf(symbol, n_days=SINA_MAX_DAYS)
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

    logger.info(
        "[%s] 最终获取: %d条 (%s ~ %s)",
        symbol, len(df), df["trade_date"].iloc[0], df["trade_date"].iloc[-1],
    )
    return df


def batch_download_etfs(
    symbols: List[str],
    years: int = 1,
    db_path: str = None,
):
    """批量下载ETF并入库"""
    store = DataStore(db_path)

    end = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")

    logger.info("=" * 60)
    logger.info("批量下载ETF | 数量: %d | %s ~ %s", len(symbols), start, end)
    logger.info("=" * 60)

    success = 0
    failed = 0

    for symbol in symbols:
        try:
            df = download_etf_history(symbol, start_date=start, end_date=end)
            if not df.empty:
                store.save_klines(df)
                success += 1
                logger.info("[OK] %s: %d条已保存", symbol, len(df))
            else:
                failed += 1
                logger.warning("[FAIL] %s: 无数据", symbol)
        except Exception as e:
            logger.error("[ERR] %s: %s", symbol, e)
            failed += 1

    logger.info("=" * 60)
    logger.info("ETF下载完成: 成功%d, 失败%d", success, failed)
    logger.info("=" * 60)


def download_user_etfs(years: int = 1):
    """下载用户指定的ETF列表"""
    etf_list = [
        "562500.SH",   # 机器人ETF
        "159382.SZ",   # AI创业板ETF
        "588790.SH",   # 科创智能ETF
        "159241.SZ",   # 用户指定
        "588200.SH",   # 用户指定
    ]
    batch_download_etfs(etf_list, years=years)


def diagnose_network():
    """网络诊断 - 测试各数据源连通性"""
    logger.info("=" * 60)
    logger.info("网络诊断")
    logger.info("=" * 60)

    # 测试东方财富
    try:
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        r = requests.get(url, timeout=5, headers=HEADERS)
        logger.info("东方财富API: HTTP %d (预期400/200, 非连接错误即正常)", r.status_code)
    except Exception as e:
        logger.warning("东方财富API: 连接失败 - %s", e)

    # 测试新浪财经
    try:
        url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sh562500&scale=240&ma=5&datalen=5"
        r = requests.get(url, timeout=5, headers=HEADERS)
        logger.info("新浪财经API: HTTP %d", r.status_code)
    except Exception as e:
        logger.warning("新浪财经API: 连接失败 - %s", e)

    # 测试akshare ETF实时行情（不同接口）
    try:
        df = ak.fund_etf_spot_em()
        logger.info("akshare ETF实时行情: 成功 (%d条)", len(df))
    except Exception as e:
        logger.warning("akshare ETF实时行情: 失败 - %s", e)

    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ETF数据下载（多源容灾版）")
    parser.add_argument("--symbol", type=str, help="下载单只ETF，如562500.SH")
    parser.add_argument("--years", type=int, default=1, help="下载年数")
    parser.add_argument("--user", action="store_true", help="下载用户持仓ETF列表")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "akshare", "sina"],
                        help="首选数据源")
    parser.add_argument("--diagnose", action="store_true", help="网络诊断模式")

    args = parser.parse_args()

    if args.diagnose:
        diagnose_network()
    elif args.user:
        download_user_etfs(years=args.years)
    elif args.symbol:
        # BUG修复: 之前错误地传了years参数给download_etf_history
        # 现在正确计算start_date/end_date后传入
        end = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=365 * args.years)).strftime("%Y%m%d")
        df = download_etf_history(args.symbol, start_date=start, end_date=end, prefer_source=args.source)
        if not df.empty:
            store = DataStore()
            store.save_klines(df)
            print(f"已保存 {args.symbol}: {len(df)}条")
        else:
            print(f"下载失败 {args.symbol}")
    else:
        print("用法:")
        print("  python etf_download.py --user                    # 下载用户ETF列表")
        print("  python etf_download.py --symbol 562500.SH        # 下载单只")
        print("  python etf_download.py --symbol 562500.SH --years 2")
        print("  python etf_download.py --diagnose                # 网络诊断")
        print("  python etf_download.py --symbol 562500.SH --source sina  # 强制使用新浪财经")
