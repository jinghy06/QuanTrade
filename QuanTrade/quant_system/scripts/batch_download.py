"""
历史数据批量下载脚本 - 使用 akshare
下载沪深300成分股3年历史日K线数据到SQLite
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from config.settings import DB_PATH, DEFAULT_WATCHLIST
from data.data_fetcher import DataFetcher, init_stock_data
from data.data_store import DataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BatchDownload")


def download_watchlist(watchlist: list = None, years: int = 3, max_workers: int = 3):
    """
    批量下载股票池历史数据
    
    Args:
        watchlist: 股票代码列表，默认DEFAULT_WATCHLIST
        years: 下载年数
        max_workers: 并发数（AkShare有速率限制，建议3-5）
    """
    watchlist = watchlist or DEFAULT_WATCHLIST
    
    logger.info("=" * 60)
    logger.info("批量下载历史数据 | 股票池: %d只 | 年数: %d", len(watchlist), years)
    logger.info("=" * 60)
    
    # 使用data_fetcher中的init_stock_data（已有并发逻辑）
    init_stock_data(watchlist, db_path=str(DB_PATH), years=years)
    
    # 显示结果
    store = DataStore()
    check_stats(store)


def incremental_update(symbols: list = None):
    """
    增量更新：只下载缺失或陈旧的数据
    适合每日定时运行
    """
    symbols = symbols or DEFAULT_WATCHLIST
    store = DataStore()
    fetcher = DataFetcher()
    
    logger.info("=" * 60)
    logger.info("增量数据更新 | 股票池: %d只", len(symbols))
    logger.info("=" * 60)
    
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    
    to_update = []
    for symbol in symbols:
        try:
            date_range = store.get_date_range(symbol)
            if date_range[1]:
                latest_db = datetime.strptime(date_range[1], "%Y-%m-%d")
                days_behind = (datetime.now() - latest_db).days
                if days_behind > 2:
                    to_update.append((symbol, days_behind))
            else:
                to_update.append((symbol, 999))
        except Exception as e:
            logger.error("检查 %s 失败: %s", symbol, e)
            to_update.append((symbol, 999))
    
    if not to_update:
        logger.info("所有数据已是最新，无需更新")
        return
    
    logger.info("需要更新: %d只", len(to_update))
    
    updated = 0
    failed = 0
    
    def _fetch_one(item):
        symbol, days = item
        time.sleep(0.5)  # 速率限制
        try:
            start = (datetime.now() - timedelta(days=days + 5)).strftime("%Y%m%d")
            df = fetcher.get_daily_k(symbol, start_date=start, end_date=end_date, use_cache=True)
            if not df.empty:
                store.save_klines(df)
                return symbol, True
        except Exception as e:
            logger.error("更新 %s 失败: %s", symbol, e)
        return symbol, False
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_fetch_one, item): item for item in to_update}
        for future in as_completed(futures):
            sym, success = future.result()
            if success:
                updated += 1
                logger.info("[%d/%d] 更新完成 %s", updated, len(to_update), sym)
            else:
                failed += 1
    
    logger.info("增量更新完成: 成功%d, 失败%d", updated, failed)
    check_stats(store)


def check_stats(store: DataStore = None):
    """查看数据库统计"""
    store = store or DataStore()
    
    print("\n" + "=" * 60)
    print("数据库统计")
    print("=" * 60)
    
    with store._connect() as conn:
        cursor = conn.cursor()
        for table in ["daily_prices", "features", "signals"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table}: {count} rows")
        
        cursor.execute("SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol")
        symbols = [r[0] for r in cursor.fetchall()]
        print(f"\n  股票数: {len(symbols)}")
        
        if symbols:
            cursor.execute(
                "SELECT MIN(trade_date), MAX(trade_date) FROM daily_prices"
            )
            min_d, max_d = cursor.fetchone()
            print(f"  数据范围: {min_d} ~ {max_d}")
            
            # 每只股票的数据条数
            cursor.execute("""
                SELECT symbol, COUNT(*) as cnt 
                FROM daily_prices 
                GROUP BY symbol 
                ORDER BY cnt DESC 
                LIMIT 5
            """)
            print("\n  数据最多的5只股票:")
            for row in cursor.fetchall():
                print(f"    {row[0]}: {row[1]}条")


def download_single(symbol: str, years: int = 3):
    """下载单只股票历史数据"""
    fetcher = DataFetcher()
    store = DataStore()
    
    end = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")
    
    logger.info("下载 %s | %s ~ %s", symbol, start, end)
    df = fetcher.get_daily_k(symbol, start_date=start, end_date=end, use_cache=True)
    
    if not df.empty:
        store.save_klines(df)
        logger.info("保存 %d 条K线", len(df))
    else:
        logger.warning("无数据: %s", symbol)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="历史数据批量下载")
    parser.add_argument("--full", action="store_true", help="全量下载（3年历史）")
    parser.add_argument("--incremental", action="store_true", help="增量更新")
    parser.add_argument("--symbol", type=str, help="下载单只股票")
    parser.add_argument("--years", type=int, default=3, help="下载年数")
    parser.add_argument("--workers", type=int, default=3, help="并发数")
    parser.add_argument("--stats", action="store_true", help="查看统计")
    
    args = parser.parse_args()
    
    if args.full:
        download_watchlist(years=args.years, max_workers=args.workers)
    elif args.incremental:
        incremental_update()
    elif args.symbol:
        download_single(args.symbol, years=args.years)
    elif args.stats:
        check_stats()
    else:
        print("用法:")
        print("  python batch_download.py --full              # 全量下载")
        print("  python batch_download.py --incremental       # 增量更新")
        print("  python batch_download.py --symbol 000001.SZ # 下载单只")
        print("  python batch_download.py --stats             # 查看统计")
