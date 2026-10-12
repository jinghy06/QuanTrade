#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuanTrade 每日数据增量更新
==========================
每个交易日前自动运行，从 Baostock 下载最新 ETF 数据并追加到数据库。

用法:
    cd /root/QuanTrade
    python3 daily_update.py

定时任务示例（crontab）:
    30 8 * * 1-5 cd /root/QuanTrade && python3 daily_update.py >> logs/daily_update.log 2>&1
"""

import os
import sys
import time
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import baostock as bs

# 数据库路径（相对于项目根目录）
DB_PATH = "QuanTrade/quant_system/data/quant.db"
LOG_DIR = "logs"

# ETF 池（与 run_etf_system_v2.0.2c_final.py 保持一致）
ETF_POOL = [
    '562500', '515070', '159995', '159550', '516510',
    '512660', '512670', '515960',
    '515790', '516160', '561160', '159790',
    '512010', '159928', '512690', '515170',
    '512480', '588000', '159915', '513180',
    '512880', '512800', '512200',
]

BENCHMARK_SYMBOL = '510300'
GOLD_SYMBOL = '518880'


def ensure_log_dir():
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def get_baostock_code(symbol: str) -> str:
    """将 symbol 转换为 baostock 代码格式"""
    if symbol.startswith('5'):
        return f"sh.{symbol}"
    elif symbol.startswith(('1', '0', '3')):
        return f"sz.{symbol}"
    else:
        return f"sh.{symbol}"


def get_latest_date(conn: sqlite3.Connection, symbol: str) -> Optional[datetime]:
    """获取某只 ETF 在数据库中的最新日期"""
    cursor = conn.execute(
        "SELECT MAX(trade_date) FROM etf_daily_prices WHERE symbol=?",
        (symbol,)
    )
    row = cursor.fetchone()
    if row and row[0]:
        return pd.to_datetime(row[0])
    return None


def download_etf_baostock(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """从 Baostock 下载单只 ETF 数据"""
    bs_code = get_baostock_code(symbol)
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 前复权
        )

        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={'date': 'trade_date', 'pctChg': 'pct_change'})
        df['symbol'] = symbol
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[df['close'] > 0].copy()
        return df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']]

    except Exception as e:
        print(f"  下载 {symbol} 异常: {e}")
        return None


def update_etf_data(conn: sqlite3.Connection, end_date: str) -> dict:
    """增量更新所有 ETF 数据"""
    all_symbols = ETF_POOL + [BENCHMARK_SYMBOL]
    stats = {'updated': 0, 'no_new_data': 0, 'latest': 0, 'no_history': 0, 'fail': 0, 'new_rows': 0}

    for symbol in all_symbols:
        latest = get_latest_date(conn, symbol)
        if latest is None:
            print(f"  {symbol}: 数据库中无历史数据，跳过")
            stats['no_history'] += 1
            continue

        # 从最新日期的下一天开始下载
        start_date = (latest + timedelta(days=1)).strftime('%Y-%m-%d')

        if start_date > end_date:
            print(f"  {symbol}: 已是最新 ({latest.date()})")
            stats['latest'] += 1
            continue

        print(f"  {symbol}: 下载 {start_date} ~ {end_date} ...", end='', flush=True)
        df = download_etf_baostock(symbol, start_date, end_date)

        if df is not None and len(df) > 0:
            df.to_sql('etf_daily_prices', conn, if_exists='append', index=False)
            print(f" OK (+{len(df)}条)")
            stats['updated'] += 1
            stats['new_rows'] += len(df)
        else:
            print(" 无新数据")
            stats['no_new_data'] += 1

        time.sleep(0.3)  # 避免请求过快

    return stats


def update_gold_data():
    """更新黄金 ETF 数据（复用现有脚本）"""
    print("\n[3/3] 更新黄金 ETF 数据...")
    try:
        import download_gold_data
        download_gold_data.download_gold_etf()
    except Exception as e:
        print(f"  黄金数据更新失败: {e}")


def main():
    print("=" * 80)
    print(f"QuanTrade 每日数据更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    ensure_log_dir()

    # 检查数据库
    if not os.path.exists(DB_PATH):
        print(f"[错误] 数据库不存在: {DB_PATH}")
        sys.exit(1)

    end_date = datetime.now().strftime('%Y-%m-%d')

    # 登录 Baostock
    print("\n[1/3] 登录 Baostock...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"[错误] Baostock 登录失败: {lg.error_msg}")
        sys.exit(1)
    print("  登录成功")

    # 更新 ETF 数据
    print(f"\n[2/3] 增量更新 ETF 数据（截至 {end_date}）...")
    conn = sqlite3.connect(DB_PATH)
    stats = update_etf_data(conn, end_date)
    conn.close()
    bs.logout()

    print(f"\nETF更新统计:")
    print(f"  已更新: {stats['updated']} 只")
    print(f"  已最新: {stats['latest']} 只")
    print(f"  无新数据: {stats['no_new_data']} 只")
    print(f"  无历史数据: {stats['no_history']} 只")
    print(f"  新增行数: {stats['new_rows']} 条")

    # 更新黄金数据
    update_gold_data()

    print("\n" + "=" * 80)
    print("每日数据更新完成")
    print("=" * 80)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
