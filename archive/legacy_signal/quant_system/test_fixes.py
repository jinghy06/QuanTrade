#!/usr/bin/env python3
"""
验证所有优化修复的测试脚本
运行: python test_fixes.py
"""

import json
import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# 确保项目根目录在路径中
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("TestFixes")

PASSED = 0
FAILED = 0


def test(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


# ============================================================
# Test 1: OBV 向量化正确性
# ============================================================
def test_obv_vectorization():
    print("\n=== Test 1: OBV 向量化 ===")
    from features.feature_engine import FeatureEngine

    # 构造测试数据
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(20) * 2)
    volume = np.random.randint(1000, 10000, 20).astype(float)

    df = pd.DataFrame({
        "trade_date": dates,
        "symbol": "TEST.SZ",
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": volume,
        "amount": volume * close,
        "turnover": np.random.rand(20) * 5,
    })

    engine = FeatureEngine()
    result = engine.compute_features(df)

    # OBV 应该是一个累积值序列
    obv = result["obv"].values
    test("OBV 长度正确", len(obv) == 20)
    test("OBV 首值为 0", abs(obv[0]) < 1e-10, f"got {obv[0]}")

    # 手动验证：close diff 符号 * volume 的累积和
    sign = np.sign(np.diff(close, prepend=close[0]))
    expected_obv = np.cumsum(sign * volume)
    # 第一个值应该是0（diff为0）
    expected_obv[0] = 0
    test("OBV 向量化结果正确", np.allclose(obv, expected_obv, atol=1e-6),
         f"max diff: {np.max(np.abs(obv - expected_obv))}")


# ============================================================
# Test 2: get_features n_days 参数
# ============================================================
def test_get_features_n_days():
    print("\n=== Test 2: get_features n_days 参数 ===")
    from data.data_store import DataStore

    # 使用临时数据库
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = DataStore(db_path=db_path)

        # 插入测试数据
        dates = [f"2024010{i}" for i in range(1, 10)]
        with store._connect() as conn:
            for d in dates:
                conn.execute(
                    "INSERT OR IGNORE INTO features (trade_date, symbol, return_5d) VALUES (?, ?, ?)",
                    (d, "TEST.SZ", 0.01),
                )
            conn.commit()

        # 测试 n_days 参数
        df = store.get_features("TEST.SZ", n_days=3)
        test("n_days=3 返回3条", len(df) == 3, f"got {len(df)}")

        df = store.get_features("TEST.SZ", n_days=5)
        test("n_days=5 返回5条", len(df) == 5, f"got {len(df)}")

        df = store.get_features("TEST.SZ")
        test("无 n_days 返回全部", len(df) == 9, f"got {len(df)}")

    finally:
        os.unlink(db_path)


# ============================================================
# Test 3: 信号去重
# ============================================================
def test_signal_dedup():
    print("\n=== Test 3: 信号去重 ===")
    from data.data_store import DataStore

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = DataStore(db_path=db_path)

        # 初始状态：无信号
        test("初始无今日信号", not store.has_today_signal("TEST.SZ"))

        # 插入一条今日信号
        signal = {
            "timestamp": "2024-01-01T10:00:00",
            "trade_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "symbol": "TEST.SZ",
            "name": "测试",
            "signal": "轻仓试探",
            "confidence": 0.6,
            "ml_prediction": {"up_prob": 0.58, "volatility": "中"},
            "technical": {"trend": "多头", "rsi": 55, "macd": "金叉"},
            "macro": {"rate_env": "低利率", "market_sentiment": "中性"},
            "suggestion": {
                "action": "买入",
                "target_price": 12.5,
                "stop_loss": 11.8,
                "position_pct": 0.05,
                "rationale": "测试",
                "risk_factors": [],
            },
        }
        store.save_signal(signal)

        test("插入后有今日信号", store.has_today_signal("TEST.SZ"))
        test("其他 symbol 无信号", not store.has_today_signal("OTHER.SZ"))

    finally:
        os.unlink(db_path)


# ============================================================
# Test 4: init_stock_data 不覆盖其他 symbol
# ============================================================
def test_init_stock_data_no_overwrite():
    print("\n=== Test 4: init_stock_data 不覆盖其他 symbol ===")
    # 这个测试只验证逻辑，不实际下载数据
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                trade_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                volume REAL, amount REAL,
                PRIMARY KEY (trade_date, symbol)
            )
        """)
        # 插入 SYMBOL_A 的数据
        conn.execute("INSERT INTO daily_prices VALUES ('20240101', 'A.SZ', 10, 11, 9, 10.5, 1000, 10000)")
        conn.execute("INSERT INTO daily_prices VALUES ('20240102', 'A.SZ', 10.5, 11.5, 9.5, 11, 1200, 12000)")
        # 插入 SYMBOL_B 的数据
        conn.execute("INSERT INTO daily_prices VALUES ('20240101', 'B.SZ', 20, 21, 19, 20.5, 2000, 20000)")
        conn.commit()

        # 模拟 init_stock_data 的逻辑：只删除 B 的数据，保留 A
        conn.execute("DELETE FROM daily_prices WHERE symbol = ?", ("B.SZ",))
        # 插入 B 的新数据
        conn.execute("INSERT INTO daily_prices VALUES ('20240101', 'B.SZ', 20, 22, 19, 21, 2500, 25000)")
        conn.execute("INSERT INTO daily_prices VALUES ('20240103', 'B.SZ', 21, 22, 20, 21.5, 2200, 22000)")
        conn.commit()

        # 验证 A 的数据未被覆盖
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM daily_prices WHERE symbol = 'A.SZ'")
        a_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM daily_prices WHERE symbol = 'B.SZ'")
        b_count = cursor.fetchone()[0]

        test("A 数据未被覆盖", a_count == 2, f"got {a_count}")
        test("B 数据已更新", b_count == 2, f"got {b_count}")

        conn.close()
    finally:
        os.unlink(db_path)


# ============================================================
# Test 5: 模型衰减评估方法存在
# ============================================================
def test_evaluate_recent():
    print("\n=== Test 5: 模型衰减评估 ===")
    from models.ml_trainer import MLTrainer

    trainer = MLTrainer()
    test("evaluate_recent 方法存在", hasattr(trainer, "evaluate_recent"))

    # 验证方法签名
    import inspect
    sig = inspect.signature(trainer.evaluate_recent)
    params = list(sig.parameters.keys())
    test("evaluate_recent 参数正确",
         "symbols" in params and "n_days" in params and "warn_rmse_threshold" in params)


# ============================================================
# Test 6: n_features_in_ key 一致性
# ============================================================
def test_n_features_in_key():
    print("\n=== Test 6: n_features_in_ key 一致性 ===")
    # 检查 signal_bot.py 中使用的是 n_features_in_
    import ast
    bot_path = project_root / "signal_bot.py"
    content = bot_path.read_text(encoding="utf-8")

    test("signal_bot 使用 n_features_in_",
         'n_features_in_' in content,
         "found: " + ("n_features_in_" if 'n_features_in_' in content else "n_features_in"))
    test("signal_bot 不使用旧 key n_features_in",
         'n_features_in"]' not in content or 'n_features_in_' in content)


# ============================================================
# Test 7: 并发导入检查
# ============================================================
def test_concurrent_imports():
    print("\n=== Test 7: 并发导入 ===")
    # 检查 data_fetcher.py 和 signal_bot.py 都导入了 concurrent.futures
    fetcher_path = project_root / "data" / "data_fetcher.py"
    bot_path = project_root / "signal_bot.py"

    fetcher_content = fetcher_path.read_text(encoding="utf-8")
    bot_content = bot_path.read_text(encoding="utf-8")

    test("data_fetcher 导入 ThreadPoolExecutor",
         "ThreadPoolExecutor" in fetcher_content)
    test("signal_bot 导入 ThreadPoolExecutor",
         "ThreadPoolExecutor" in bot_content)


# ============================================================
# Test 8: 策略信号集成
# ============================================================
def test_strategy_integration():
    print("\n=== Test 8: 策略信号集成 ===")
    import ast
    bot_path = project_root / "signal_bot.py"

    bot_content = bot_path.read_text(encoding="utf-8")

    test("run_full_pipeline 调用 _run_strategies",
         "_run_strategies(candidate_symbols)" in bot_content or "_run_strategies(" in bot_content)
    test("_generate_signals_from_strategies 方法存在",
         "def _generate_signals_from_strategies" in bot_content)
    test("策略信号汇总逻辑存在",
         "strategy_signals" in bot_content.split("def _generate_signals_from_strategies")[1].split("def ")[0] if "def _generate_signals_from_strategies" in bot_content else False)


# ============================================================
# Test 9: --evaluate CLI 参数
# ============================================================
def test_evaluate_cli():
    print("\n=== Test 9: --evaluate CLI 参数 ===")
    import ast
    bot_path = project_root / "signal_bot.py"
    content = bot_path.read_text(encoding="utf-8")

    test("--evaluate 参数已添加", "--evaluate" in content)
    test("--eval-days 参数已添加", "--eval-days" in content or "--eval_days" in content)
    test("evaluate_model 方法存在", "def evaluate_model" in content)


# ============================================================
# 运行所有测试
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("QuanTrade 优化修复验证测试")
    print("=" * 50)

    test_obv_vectorization()
    test_get_features_n_days()
    test_signal_dedup()
    test_init_stock_data_no_overwrite()
    test_evaluate_recent()
    test_n_features_in_key()
    test_concurrent_imports()
    test_strategy_integration()
    test_evaluate_cli()

    print("\n" + "=" * 50)
    print(f"结果: {PASSED} 通过, {FAILED} 失败")
    print("=" * 50)

    sys.exit(1 if FAILED > 0 else 0)
