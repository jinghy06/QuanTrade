"""
ETF数据下载脚本 - 使用新浪财经接口
akshare/东方财富不稳定时，新浪财经作为兜底方案
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import requests

from data.data_store import DataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SinaETF")


def download_etf_from_sina(
    symbol: str,
    n_days: int = 300,
) -> pd.DataFrame:
    """
    从新浪财经下载ETF历史K线

    Args:
        symbol: 项目标准格式，如 "562500.SH" -> sina格式 "sh562500"
        n_days: 获取条数（新浪最多约300条）

    Returns:
        标准格式DataFrame
    """
    code, market = symbol.split(".")
    sina_symbol = f"{market.lower()}{code}"

    url = (
        f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={sina_symbol}"
        f"&scale=240&ma=5&datalen={n_days}"
    )

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()

        if not data:
            logger.warning("新浪无数据: %s", symbol)
            return pd.DataFrame()

        records = []
        prev_close = None
        for item in data:
            close = float(item["close"])
            open_ = float(item["open"])
            high = float(item["high"])
            low = float(item["low"])
            volume = float(item["volume"])

            # 计算涨跌幅
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
                "amount": 0.0,  # 新浪无成交额数据
                "amplitude": amplitude,
                "pct_change": pct_change,
                "change": change,
                "turnover": 0.0,  # 新浪无换手率数据
            })

        df = pd.DataFrame(records)
        df = df.sort_values("trade_date").reset_index(drop=True)

        logger.info(
            "新浪ETF %s: %d条 (%s ~ %s)",
            symbol, len(df), df["trade_date"].iloc[0], df["trade_date"].iloc[-1],
        )
        return df

    except Exception as e:
        logger.error("新浪ETF %s 失败: %s", symbol, e)
        return pd.DataFrame()


def batch_download_user_etfs():
    """下载用户持仓ETF（排除REIT 180503）"""
    # 用户持仓（图片）: 562500, 159382, 588790
    # 排除REIT: 180503
    # 用户指定: 159241, 588200
    etf_list = [
        "562500.SH",   # 机器人ETF
        "159382.SZ",   # AI创业板ETF
        "588790.SH",   # 科创智能ETF
        "159241.SZ",   # 用户指定
        "588200.SH",   # 用户指定
    ]

    store = DataStore()
    logger.info("=" * 60)
    logger.info("下载用户ETF | 共%d只", len(etf_list))
    logger.info("=" * 60)

    success = 0
    failed = 0

    for symbol in etf_list:
        try:
            df = download_etf_from_sina(symbol, n_days=300)
            if not df.empty:
                store.save_klines(df)
                success += 1
                logger.info("[OK] %s: %d条已保存", symbol, len(df))
            else:
                failed += 1
        except Exception as e:
            logger.error("[ERR] %s: %s", symbol, e)
            failed += 1

    logger.info("=" * 60)
    logger.info("完成: 成功%d, 失败%d", success, failed)
    logger.info("=" * 60)

    # 显示统计
    check_etf_stats(store)


def check_etf_stats(store: DataStore = None):
    """查看ETF数据统计"""
    store = store or DataStore()

    print("\n" + "=" * 60)
    print("ETF数据库统计")
    print("=" * 60)

    with store._connect() as conn:
        cursor = conn.cursor()

        # ETF列表
        cursor.execute("""
            SELECT DISTINCT symbol FROM daily_prices
            WHERE symbol LIKE '5%' OR symbol LIKE '1%'
            ORDER BY symbol
        """)
        etfs = [r[0] for r in cursor.fetchall()]
        print(f"\nETF数量: {len(etfs)}")
        for e in etfs:
            cursor.execute(
                "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM daily_prices WHERE symbol = ?",
                (e,),
            )
            cnt, min_d, max_d = cursor.fetchone()
            print(f"  {e}: {cnt}条 ({min_d} ~ {max_d})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ETF数据下载（新浪财经）")
    parser.add_argument("--user", action="store_true", help="下载用户ETF列表")
    parser.add_argument("--symbol", type=str, help="下载单只，如562500.SH")
    parser.add_argument("--days", type=int, default=300, help="获取条数")
    parser.add_argument("--stats", action="store_true", help="查看统计")

    args = parser.parse_args()

    if args.user:
        batch_download_user_etfs()
    elif args.symbol:
        df = download_etf_from_sina(args.symbol, n_days=args.days)
        if not df.empty:
            store = DataStore()
            store.save_klines(df)
            print(f"已保存 {args.symbol}: {len(df)}条")
    elif args.stats:
        check_etf_stats()
    else:
        print("用法:")
        print("  python sina_etf_download.py --user              # 下载用户ETF")
        print("  python sina_etf_download.py --symbol 562500.SH  # 下载单只")
        print("  python sina_etf_download.py --stats               # 查看统计")
