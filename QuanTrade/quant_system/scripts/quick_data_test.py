"""
快速数据验证脚本 - 使用 kimi_finance_v2 获取实时数据
用于快速测试特征计算和预测流程，无需等待大量历史数据下载
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import sqlite3
from datetime import datetime

import pandas as pd

from config.settings import DB_PATH, DEFAULT_WATCHLIST
from data.data_store import DataStore
from features.feature_engine import FeatureEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuickDataTest")


def import_kimi_data(price_csv: str, tech_csv: str):
    """
    将 kimi_finance_v2 获取的实时数据导入 SQLite
    作为快速测试数据（仅1条记录，用于验证流程）
    """
    store = DataStore()
    
    # 读取价格数据
    df_price = pd.read_csv(price_csv)
    print("Price CSV columns:", df_price.columns.tolist())
    df_price.columns = [c.strip().lower() for c in df_price.columns]
    
    # 读取技术指标数据
    df_tech = pd.read_csv(tech_csv)
    print("Tech CSV columns:", df_tech.columns.tolist())
    df_tech.columns = [c.strip().lower() for c in df_tech.columns]
    
    # 合并（保留价格表的time列）
    df = df_price.merge(df_tech, left_on="ts_code", right_on="code", how="left", suffixes=("", "_tech"))
    
    # 转换为项目标准格式
    records = []
    for _, row in df.iterrows():
        symbol = row["ts_code"]
        # 解析时间
        time_str = str(row["time"])
        if len(time_str) == 12:  # 202606051347
            trade_date = time_str[:8]
        else:
            trade_date = datetime.now().strftime("%Y%m%d")
        
        records.append({
            "trade_date": trade_date,
            "symbol": symbol,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["vol"]),
            "amount": float(row["amount"]),
            "amplitude": abs(float(row.get("pct_change", 0))) * 0.5,  # 估算
            "pct_change": float(row.get("pct_change", 0)),
            "change": float(row["close"]) - float(row["open"]),
            "turnover": 0.0,  # kimi未提供
        })
    
    df_out = pd.DataFrame(records)
    store.save_klines(df_out)
    logger.info("kimi实时数据已导入: %d条", len(df_out))
    return df_out


def quick_test_pipeline(symbols: list = None):
    """
    快速测试完整流水线（用kimi数据或现有数据）
    """
    symbols = symbols or ["000001.SZ", "600519.SH", "300750.SZ"]
    store = DataStore()
    engine = FeatureEngine()
    
    logger.info("=" * 60)
    logger.info("快速测试流水线")
    logger.info("=" * 60)
    
    for symbol in symbols:
        try:
            # 1. 获取K线
            df_kline = store.get_kline(symbol, n_days=100)
            if len(df_kline) < 60:
                logger.warning("%s 数据不足60条 (%d条)，跳过", symbol, len(df_kline))
                continue
            
            logger.info("\n%s: %d条K线", symbol, len(df_kline))
            
            # 2. 计算特征
            df_feat = engine.compute_features(df_kline)
            if df_feat.empty:
                logger.warning("%s 特征计算失败", symbol)
                continue
            
            logger.info("  特征列: %d个", len(df_feat.columns))
            logger.info("  最新特征样本:")
            latest = df_feat.iloc[-1]
            for col in ["return_5d", "return_10d", "rsi_14", "macd_hist", "atr_14"]:
                if col in latest.index:
                    logger.info("    %s: %.4f", col, latest[col])
            
            # 3. 检查目标变量
            if "target_return_5d" in df_feat.columns:
                logger.info("  5日目标收益: %.4f", latest["target_return_5d"])
            
            # 4. 保存特征
            store.save_features(df_feat.tail(1))
            logger.info("  特征已保存到数据库")
            
        except Exception as e:
            logger.error("%s 测试失败: %s", symbol, e)
    
    logger.info("\n" + "=" * 60)
    logger.info("快速测试完成")
    logger.info("=" * 60)


def check_database_stats():
    """查看数据库统计"""
    store = DataStore()
    
    print("\n" + "=" * 60)
    print("数据库统计")
    print("=" * 60)
    
    with store._connect() as conn:
        cursor = conn.cursor()
        for table in ["daily_prices", "features", "signals"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table}: {count} rows")
        
        # 显示股票列表
        cursor.execute("SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol")
        symbols = [r[0] for r in cursor.fetchall()]
        print(f"\n  股票数: {len(symbols)}")
        print(f"  股票列表: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        
        # 显示日期范围
        if symbols:
            cursor.execute(
                "SELECT MIN(trade_date), MAX(trade_date) FROM daily_prices"
            )
            min_d, max_d = cursor.fetchone()
            print(f"  数据范围: {min_d} ~ {max_d}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="快速数据测试")
    parser.add_argument("--import-kimi", action="store_true", help="导入kimi实时数据")
    parser.add_argument("--price-csv", default="../../data_test/kimi_price.csv")
    parser.add_argument("--tech-csv", default="../../data_test/kimi_realtime.csv")
    parser.add_argument("--test", action="store_true", help="运行快速测试流水线")
    parser.add_argument("--stats", action="store_true", help="查看数据库统计")
    
    args = parser.parse_args()
    
    if args.import_kimi:
        import_kimi_data(args.price_csv, args.tech_csv)
    
    if args.test:
        quick_test_pipeline()
    
    if args.stats:
        check_database_stats()
    
    # 默认行为
    if not any([args.import_kimi, args.test, args.stats]):
        check_database_stats()
