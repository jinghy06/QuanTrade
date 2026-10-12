"""
Future Klines v4 — 未来20日价格路径Monte Carlo模拟
- 基于几何布朗运动 + 情景漂移调整
- 读取 scenario_signals_v4 最新决策
- 保存到 future_klines_v4 表
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DB_PATH = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
TEST_ETFS = ['562500.SH', '588200.SH', '588790.SH', '159382.SZ', '159241.SZ']
N_PATHS = 100
HORIZON = 20

print("=" * 70)
print("【Future Klines v4 — Monte Carlo未来价格模拟】")
print("=" * 70)

# ==================== 1. 读取数据 ====================
conn = sqlite3.connect(DB_PATH)

# 读取 scenario_signals_v4 最新一天
placeholders = ','.join([f"'{s}'" for s in TEST_ETFS])
df_scenario = pd.read_sql_query(
    f"""
    SELECT s.* FROM scenario_signals_v4 s
    INNER JOIN (
        SELECT symbol, MAX(trade_date) as max_date FROM scenario_signals_v4
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
    ) m ON s.symbol = m.symbol AND s.trade_date = m.max_date
    """,
    conn
)

# 读取 daily_prices 用于计算历史参数
df_prices = pd.read_sql_query(
    f"""
    SELECT * FROM daily_prices
    WHERE symbol IN ({placeholders})
    ORDER BY symbol, trade_date
    """,
    conn
)

conn.close()

if df_scenario.empty:
    print("[错误] scenario_signals_v4 为空，请先运行 scenario_engine_v4.py")
    sys.exit(1)

if df_prices.empty:
    print("[错误] daily_prices 为空")
    sys.exit(1)

df_scenario['trade_date'] = pd.to_datetime(df_scenario['trade_date'], format='mixed')
df_prices['trade_date'] = pd.to_datetime(df_prices['trade_date'], format='mixed')

print(f"[数据] scenario_signals_v4 最新记录: {len(df_scenario)} 条")
print(f"[数据] daily_prices 历史记录: {len(df_prices)} 条")

# ==================== 2. Monte Carlo 模拟 ====================
all_results = []

for sym in TEST_ETFS:
    sym_scenario = df_scenario[df_scenario['symbol'] == sym]
    sym_prices = df_prices[df_prices['symbol'] == sym].sort_values('trade_date')

    if len(sym_scenario) == 0:
        print(f"[跳过] {sym}: 无情景信号")
        continue
    if len(sym_prices) < 60:
        print(f"[跳过] {sym}: 历史数据不足60天 ({len(sym_prices)})")
        continue

    # 取最新记录（去重后）
    latest_scenario = sym_scenario.sort_values('trade_date').iloc[-1]
    forecast_date = latest_scenario['trade_date']
    scenario_decision = latest_scenario['scenario_decision']
    last_close = latest_scenario['close']

    # 计算GBM参数
    returns = sym_prices['close'].pct_change().dropna()
    mu = returns.tail(60).mean()  # 近60日平均日收益
    sigma = returns.tail(20).std()  # 近20日波动率

    if sigma == 0 or np.isnan(sigma):
        sigma = returns.std()
    if np.isnan(mu):
        mu = 0.0

    # 情景漂移调整
    mu_adj = mu
    if scenario_decision == 'favorable':
        mu_adj += 0.001
    elif scenario_decision == 'adverse':
        mu_adj -= 0.001
    # base: 保持原mu

    print(f"\n>>> {sym} | 基准日: {forecast_date.strftime('%Y-%m-%d')} | 决策: {scenario_decision}")
    print(f"  最新收盘: {last_close:.3f} | μ={mu:.5f} σ={sigma:.5f} | 调整后μ={mu_adj:.5f}")

    # 生成100条路径
    dt = 1.0
    paths = np.zeros((N_PATHS, HORIZON))
    paths[:, 0] = last_close

    for t in range(1, HORIZON):
        Z = np.random.standard_normal(N_PATHS)
        # GBM: S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu_adj - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    # 计算每日 OHLC（基于路径）
    # open = 前日close, close = 当日close
    # high = close * (1 + 0.3*sigma), low = close * (1 - 0.3*sigma)
    # 对每条路径分别计算
    all_opens = np.zeros((N_PATHS, HORIZON))
    all_highs = np.zeros((N_PATHS, HORIZON))
    all_lows = np.zeros((N_PATHS, HORIZON))
    all_closes = paths.copy()

    for i in range(N_PATHS):
        for t in range(HORIZON):
            if t == 0:
                all_opens[i, t] = last_close
            else:
                all_opens[i, t] = all_closes[i, t-1]
            c = all_closes[i, t]
            all_highs[i, t] = c * (1 + 0.3 * sigma)
            all_lows[i, t] = c * (1 - 0.3 * sigma)

    # 汇总统计: 中位数 + 10%/90%分位数
    for t in range(HORIZON):
        day = t + 1
        row = {
            'symbol': sym,
            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
            'horizon_day': day,
            'median_open': float(np.median(all_opens[:, t])),
            'median_high': float(np.median(all_highs[:, t])),
            'median_low': float(np.median(all_lows[:, t])),
            'median_close': float(np.median(all_closes[:, t])),
            'p10_open': float(np.percentile(all_opens[:, t], 10)),
            'p10_high': float(np.percentile(all_highs[:, t], 10)),
            'p10_low': float(np.percentile(all_lows[:, t], 10)),
            'p10_close': float(np.percentile(all_closes[:, t], 10)),
            'p90_open': float(np.percentile(all_opens[:, t], 90)),
            'p90_high': float(np.percentile(all_highs[:, t], 90)),
            'p90_low': float(np.percentile(all_lows[:, t], 90)),
            'p90_close': float(np.percentile(all_closes[:, t], 90)),
            'scenario_decision': scenario_decision,
        }
        all_results.append(row)

    # 打印关键节点
    print(f"  第5日 中位数收盘: {np.median(all_closes[:, 4]):.3f} (P10={np.percentile(all_closes[:, 4], 10):.3f}, P90={np.percentile(all_closes[:, 4], 90):.3f})")
    print(f"  第10日中位数收盘: {np.median(all_closes[:, 9]):.3f} (P10={np.percentile(all_closes[:, 9], 10):.3f}, P90={np.percentile(all_closes[:, 9], 90):.3f})")
    print(f"  第20日中位数收盘: {np.median(all_closes[:, 19]):.3f} (P10={np.percentile(all_closes[:, 19], 10):.3f}, P90={np.percentile(all_closes[:, 19], 90):.3f})")

# ==================== 3. 保存到数据库 ====================
if not all_results:
    print("[错误] 没有生成任何模拟结果")
    sys.exit(1)

df_result = pd.DataFrame(all_results)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 删除旧表
cursor.execute("DROP TABLE IF EXISTS future_klines_v4")

# 创建新表
create_sql = """
CREATE TABLE future_klines_v4 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    forecast_date TEXT,
    horizon_day INTEGER,
    median_open REAL,
    median_high REAL,
    median_low REAL,
    median_close REAL,
    p10_open REAL,
    p10_high REAL,
    p10_low REAL,
    p10_close REAL,
    p90_open REAL,
    p90_high REAL,
    p90_low REAL,
    p90_close REAL,
    scenario_decision TEXT
)
"""
cursor.execute(create_sql)
conn.commit()

# 插入数据
for _, row in df_result.iterrows():
    cursor.execute("""
        INSERT INTO future_klines_v4 (
            symbol, forecast_date, horizon_day,
            median_open, median_high, median_low, median_close,
            p10_open, p10_high, p10_low, p10_close,
            p90_open, p90_high, p90_low, p90_close,
            scenario_decision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row['symbol'], row['forecast_date'], row['horizon_day'],
        row['median_open'], row['median_high'], row['median_low'], row['median_close'],
        row['p10_open'], row['p10_high'], row['p10_low'], row['p10_close'],
        row['p90_open'], row['p90_high'], row['p90_low'], row['p90_close'],
        row['scenario_decision']
    ))

conn.commit()
conn.close()

print(f"\n[保存] future_klines_v4: {len(df_result)} 条记录已写入数据库")

print("\n" + "=" * 70)
print("【Future Klines v4 完成】")
print("=" * 70)
