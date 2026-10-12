"""
A股量化信号系统 - SQLite数据库操作封装
核心三张表: daily_prices(日K), features(特征), signals(信号)
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import DB_PATH

logger = logging.getLogger(__name__)


class DataStore:
    """
    SQLite数据仓库封装
    对应计划书 Phase 0: "SQLite建表: daily_prices、features、signals"
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self._init_db()

    @contextmanager
    def _connect(self):
        """上下文管理器，自动提交和关闭"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使查询结果可通过列名访问
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表结构（幂等）"""
        with self._connect() as conn:
            cursor = conn.cursor()

            # ---------- 1. 日K数据表 ----------
            # 对应 data_fetcher.py 写入的原始数据
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

            # ---------- 2. 特征表 ----------
            # 对应 feature_engine.py 输出的机器学习特征
            # 含新增趋势特征 + 多步回归目标
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    trade_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    -- 动量特征
                    return_5d REAL,
                    return_10d REAL,
                    return_20d REAL,
                    rsi_14 REAL,
                    macd_dif REAL,
                    macd_dea REAL,
                    macd_hist REAL,
                    -- 波动率特征
                    std_5d REAL,
                    std_20d REAL,
                    atr_14 REAL,
                    -- 量价特征
                    volume_ma5 REAL,
                    volume_ma20 REAL,
                    obv REAL,
                    turnover_ma5 REAL,
                    -- 趋势特征（新增）
                    ma_alignment REAL,
                    price_position REAL,
                    trend_slope REAL,
                    vol_percentile REAL,
                    dist_to_support REAL,
                    dist_to_resistance REAL,
                    divergence_bear INTEGER,
                    -- 宏观代理
                    bond_yield_10y REAL,
                    -- RD-Agent生成因子（动态扩展）
                    rd_factor_1 REAL,
                    rd_factor_2 REAL,
                    rd_factor_3 REAL,
                    -- 多步回归目标（新增）
                    target_close_1d REAL,
                    target_close_3d REAL,
                    target_close_5d REAL,
                    target_close_10d REAL,
                    target_return_1d REAL,
                    target_return_3d REAL,
                    target_return_5d REAL,
                    target_return_10d REAL,
                    target_volatility_5d REAL,
                    -- 兼容旧目标
                    target_next_day_return REAL,
                    target_direction INTEGER,  -- 1:涨, 0:跌
                    -- 元信息
                    feature_version TEXT DEFAULT 'v1.0',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (trade_date, symbol)
                )
            """)

            # ---------- 3. 信号表 ----------
            # 对应 signal_bot.py 输出的交易信号
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    -- ML预测结果
                    signal_type TEXT,  -- 'buy', 'sell', 'hold', 'light_buy'
                    confidence REAL,  -- ML模型输出的概率
                    ml_up_prob REAL,  -- 上涨概率
                    ml_volatility TEXT,  -- '高', '中', '低'
                    -- 技术面摘要
                    tech_trend TEXT,
                    tech_rsi REAL,
                    tech_macd TEXT,
                    -- 宏观环境
                    macro_rate_env TEXT,
                    macro_sentiment TEXT,
                    -- 策略建议（JSON存储结构化建议）
                    strategy_suggestion TEXT,  -- JSON字符串
                    strategy_rationale TEXT,
                    strategy_risk_factors TEXT,  -- JSON数组字符串
                    -- 风控参数
                    target_price REAL,
                    stop_loss REAL,
                    position_pct REAL,
                    -- 实际执行（后续回填）
                    executed INTEGER DEFAULT 0,
                    executed_price REAL,
                    executed_at TEXT,
                    pnl REAL,
                    -- 元信息
                    model_version TEXT DEFAULT 'v1.0',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ---------- 索引优化 ----------
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_price_symbol ON daily_prices(symbol)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_price_date ON daily_prices(trade_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_feat_symbol ON features(symbol)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_feat_date ON features(trade_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_signal_date ON signals(trade_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_signal_symbol ON signals(symbol)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_signal_executed ON signals(executed)"
            )

            conn.commit()
            logger.info("数据库初始化完成: %s", self.db_path)

    # ============================================================
    # daily_prices 操作
    # ============================================================

    def get_kline(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        n_days: int = None,
    ) -> pd.DataFrame:
        """
        查询日K数据

        Args:
            symbol: 股票代码
            start_date/end_date: 日期范围 YYYYMMDD
            n_days: 取最近N天（优先级高于日期范围）

        Returns:
            DataFrame 按日期升序排列
        """
        with self._connect() as conn:
            if n_days:
                query = """
                    SELECT * FROM daily_prices
                    WHERE symbol = ?
                    ORDER BY trade_date DESC
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(symbol, n_days))
                df = df.sort_values("trade_date").reset_index(drop=True)
            else:
                params = [symbol]
                query = "SELECT * FROM daily_prices WHERE symbol = ?"
                if start_date:
                    query += " AND trade_date >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND trade_date <= ?"
                    params.append(end_date)
                query += " ORDER BY trade_date"
                df = pd.read_sql_query(query, conn, params=params)

        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        return df

    def save_klines(self, df: pd.DataFrame):
        """批量保存日K数据（自动处理重复，INSERT OR REPLACE）"""
        with self._connect() as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO daily_prices 
                    (trade_date, symbol, open, high, low, close, volume, amount, amplitude, pct_change, change, turnover, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    row.get('trade_date'),
                    row.get('symbol'),
                    row.get('open'),
                    row.get('high'),
                    row.get('low'),
                    row.get('close'),
                    row.get('volume'),
                    row.get('amount'),
                    row.get('amplitude'),
                    row.get('pct_change'),
                    row.get('change'),
                    row.get('turnover'),
                ))
        logger.info("保存 %d 条K线数据 (INSERT OR REPLACE)", len(df))

    # ============================================================
    # features 操作
    # ============================================================

    def get_features(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        with_target: bool = False,
        n_days: int = None,
    ) -> pd.DataFrame:
        """
        查询特征数据（用于模型训练和预测）

        Args:
            with_target: 是否包含target列（训练时使用）
            n_days: 取最近N天（优先级高于日期范围）
        """
        cols = "*" if with_target else "trade_date, symbol, " + \
            ", ".join([f"{c}" for c in self._get_feature_cols()])

        with self._connect() as conn:
            if n_days:
                query = f"""
                    SELECT {cols} FROM features
                    WHERE symbol = ?
                    ORDER BY trade_date DESC
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(symbol, n_days))
                df = df.sort_values("trade_date").reset_index(drop=True)
            else:
                params = [symbol]
                query = f"SELECT {cols} FROM features WHERE symbol = ?"
                if start_date:
                    query += " AND trade_date >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND trade_date <= ?"
                    params.append(end_date)
                query += " ORDER BY trade_date"
                df = pd.read_sql_query(query, conn, params=params)

        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        return df

    def save_features(self, df: pd.DataFrame):
        """批量保存特征数据"""
        with self._connect() as conn:
            df.to_sql("features", conn, if_exists="append", index=False)
        logger.info("保存 %d 条特征数据", len(df))

    def get_training_data(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        获取训练数据集（features + target）
        对应 ml_trainer.py 的数据输入
        """
        placeholders = ",".join(["?"] * len(symbols))
        query = f"""
            SELECT * FROM features
            WHERE symbol IN ({placeholders})
            AND trade_date >= ? AND trade_date <= ?
            AND target_direction IS NOT NULL
            ORDER BY trade_date, symbol
        """
        params = symbols + [start_date, end_date]
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        logger.info(
            "训练数据: %d条, %s ~ %s", len(df), start_date, end_date
        )
        return df

    def _get_feature_cols(self) -> List[str]:
        """获取所有特征列名（不含target和元信息）"""
        return [
            "return_5d", "return_10d", "return_20d",
            "rsi_14", "macd_dif", "macd_dea", "macd_hist",
            "std_5d", "std_20d", "atr_14",
            "volume_ma5", "volume_ma20", "obv", "turnover_ma5",
            "ma_alignment", "price_position", "trend_slope",
            "vol_percentile", "dist_to_support", "dist_to_resistance",
            "divergence_bear",
            "bond_yield_10y",
            "rd_factor_1", "rd_factor_2", "rd_factor_3",
        ]

    # ============================================================
    # signals 操作
    # ============================================================

    def save_signal(self, signal: Dict[str, Any]) -> int:
        """
        保存交易信号

        Args:
            signal: 符合结构化JSON格式的信号字典

        Returns:
            信号ID
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signals (
                    timestamp, trade_date, symbol, name,
                    signal_type, confidence, ml_up_prob, ml_volatility,
                    tech_trend, tech_rsi, tech_macd,
                    macro_rate_env, macro_sentiment,
                    strategy_suggestion, strategy_rationale, strategy_risk_factors,
                    target_price, stop_loss, position_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.get("timestamp", datetime.now().isoformat()),
                signal.get("trade_date", ""),
                signal.get("symbol", ""),
                signal.get("name", ""),
                signal.get("signal", ""),
                signal.get("confidence", 0),
                signal.get("ml_prediction", {}).get("up_prob", 0),
                signal.get("ml_prediction", {}).get("volatility", ""),
                signal.get("technical", {}).get("trend", ""),
                signal.get("technical", {}).get("rsi", 0),
                signal.get("technical", {}).get("macd", ""),
                signal.get("macro", {}).get("rate_env", ""),
                signal.get("macro", {}).get("market_sentiment", ""),
                json.dumps(signal.get("suggestion", {}), ensure_ascii=False),
                signal.get("suggestion", {}).get("rationale", ""),
                json.dumps(
                    signal.get("suggestion", {}).get("risk_factors", []),
                    ensure_ascii=False,
                ),
                signal.get("suggestion", {}).get("target_price", 0),
                signal.get("suggestion", {}).get("stop_loss", 0),
                signal.get("suggestion", {}).get("position_pct", 0),
            ))
            signal_id = cursor.lastrowid

        logger.info(
            "信号已保存 [ID:%d] %s %s 置信度%.2f",
            signal_id,
            signal.get("symbol"),
            signal.get("signal"),
            signal.get("confidence", 0),
        )
        return signal_id

    def get_recent_signals(
        self, symbol: str = None, n: int = 10
    ) -> pd.DataFrame:
        """获取最近N条信号"""
        with self._connect() as conn:
            query = """
                SELECT * FROM signals
                WHERE 1=1
            """
            params = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(n)
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_signal_stats(self, days: int = 30) -> pd.DataFrame:
        """
        获取信号统计（用于飞书"绩效"命令）
        返回近N天的信号准确率和性能指标
        """
        with self._connect() as conn:
            query = """
                SELECT
                    symbol,
                    signal_type,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN executed = 1 AND pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN executed = 1 AND pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    AVG(pnl) as avg_pnl
                FROM signals
                WHERE created_at >= datetime('now', '-{} days')
                GROUP BY symbol, signal_type
                ORDER BY count DESC
            """.format(days)
            df = pd.read_sql_query(query, conn)
        return df

    # ============================================================
    # 工具方法
    # ============================================================

    def has_today_signal(self, symbol: str) -> bool:
        """检查今日是否已推送过该 symbol 的信号"""
        today = datetime.now().strftime("%Y-%m-%d")
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM signals WHERE symbol = ? AND trade_date = ?",
                (symbol, today),
            )
            count = cursor.fetchone()[0]
        return count > 0

    def get_all_symbols(self) -> List[str]:
        """获取数据库中所有股票代码"""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_date_range(self, symbol: str) -> tuple:
        """获取某股票的数据日期范围"""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT MIN(trade_date), MAX(trade_date)
                FROM daily_prices WHERE symbol = ?
            """,
                (symbol,),
            )
            row = cursor.fetchone()
        return (row[0], row[1]) if row else (None, None)

    def close(self):
        """显式关闭资源（上下文管理器已自动处理）"""
        logger.debug("DataStore资源已释放")


if __name__ == "__main__":
    # 命令行快速测试数据库
    logging.basicConfig(level=logging.INFO)
    store = DataStore()

    # 显示表统计
    with store._connect() as conn:
        cursor = conn.cursor()
        for table in ["daily_prices", "features", "signals"]:
            cursor.execute(
                f"SELECT COUNT(*) FROM {table}"
            )
            count = cursor.fetchone()[0]
            print(f"  {table}: {count} rows")
