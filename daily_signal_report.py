#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuanTrade 每日交易信号报告
==========================
基于最新数据生成未来可能的建仓/加仓/减仓建议。

用法:
    cd /root/QuanTrade
    python3 daily_signal_report.py
    python3 daily_signal_report.py --output reports/signal_2026-06-20.txt
"""

import os
import sys
import argparse
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DB_PATH = "QuanTrade/quant_system/data/quant.db"
REPORT_DIR = "reports"

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

EMOTION_ETFS = ['562500', '515070', '159995', '159550', '516510', '512660', '512670', '515960',
                '588000', '159915', '513180', '512480']


def ensure_report_dir():
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)


def load_all_data():
    """从数据库加载所有 ETF 和黄金数据"""
    conn = sqlite3.connect(DB_PATH)
    all_prices = {}

    for symbol in ETF_POOL + [BENCHMARK_SYMBOL]:
        df = pd.read_sql(
            f"SELECT * FROM etf_daily_prices WHERE symbol='{symbol}' ORDER BY trade_date",
            conn
        )
        if len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')
            all_prices[symbol] = df

    gold_df = pd.read_sql("SELECT * FROM gold_daily_prices ORDER BY date", conn)
    if len(gold_df) > 0:
        gold_df['date'] = pd.to_datetime(gold_df['date'])
        gold_df = gold_df.set_index('date')
    else:
        gold_df = None

    conn.close()
    return all_prices, gold_df


# ============================================================
# 核心计算函数（从 run_etf_system_v2.0.2c_final.py 提取）
# ============================================================
def calculate_momentum_score(etf_df: pd.DataFrame, benchmark_df: pd.DataFrame,
                             symbol: str = None, market_bottom: bool = False) -> float:
    """计算 ETF 动量得分"""
    if len(etf_df) < 120:
        return -999

    close = etf_df['close']
    mom_120 = close.iloc[-1] / close.iloc[-120] - 1
    mom_60 = close.iloc[-1] / close.iloc[-60] - 1

    is_bottom = mom_120 < -0.15
    DEFENSIVE_ETFS = ['515960', '512010', '512690', '512880', '512800', '512200']

    if (is_bottom or market_bottom) and symbol and symbol in DEFENSIVE_ETFS and mom_120 > -0.05:
        return -100

    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1] if len(etf_df) >= 10 else ma5
    trend_score = 0.6 * (1 if ma5 > ma10 else 0) + 0.4 * (1 if close.iloc[-1] > ma5 else 0)

    if is_bottom:
        score_oversold = np.clip(-mom_120 / 0.30, -1, 1)
        score_60 = np.clip(-mom_60 / 0.20, -1, 1)
        score_relative = 0
        if benchmark_df is not None and len(benchmark_df) > 20:
            bench = benchmark_df['close'].reindex(etf_df.index, method='ffill')
            bench_mom_20 = bench.iloc[-1] / bench.iloc[-20] - 1
            etf_mom_20 = close.iloc[-1] / close.iloc[-20] - 1
            score_relative = np.clip((bench_mom_20 - etf_mom_20) / 0.20, -1, 1)
        return score_oversold * 0.40 + score_60 * 0.30 + score_relative * 0.20 + trend_score * 0.10
    else:
        score_120 = np.clip(mom_120 / 0.5, -1, 1)
        score_60 = np.clip(mom_60 / 0.3, -1, 1)
        score_relative = 0
        if benchmark_df is not None and len(benchmark_df) > 120:
            bench = benchmark_df['close'].reindex(etf_df.index, method='ffill')
            etf_mom_20 = close.iloc[-1] / close.iloc[-20] - 1
            bench_mom_20 = bench.iloc[-1] / bench.iloc[-20] - 1
            score_relative = np.clip((etf_mom_20 - bench_mom_20) / 0.2, -1, 1)
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        trend_score = (0.4 * (1 if ma5 > ma20 else 0) +
                       0.3 * (1 if ma20 > ma60 else 0) +
                       0.3 * (1 if close.iloc[-1] > ma5 else 0))
        return score_120 * 0.40 + score_60 * 0.30 + score_relative * 0.20 + trend_score * 0.10


def calculate_three_factors(benchmark_df: pd.DataFrame, gold_df: Optional[pd.DataFrame]) -> dict:
    """计算情绪/地缘/政策三因子"""
    if benchmark_df is None or len(benchmark_df) < 60:
        return {'sentiment': 0, 'geopolitical': 0, 'policy': 0, 'combined': 0}

    close = benchmark_df['close']
    returns = close.pct_change()

    vol_20 = returns.rolling(20).std().iloc[-1]
    vol_ma60 = returns.rolling(60).std().iloc[-1]
    vol_signal = -0.5 if vol_20 > vol_ma60 * 1.5 else (0.5 if vol_20 < vol_ma60 * 0.7 else 0)
    mom_5 = close.pct_change(5).iloc[-1]
    mom_60 = close.pct_change(60).iloc[-1]
    mom_signal = 0.5 if mom_5 > 0 and mom_60 > 0 else (-0.5 if mom_5 < 0 and mom_60 < 0 else 0)
    sentiment = np.clip((vol_signal + mom_signal) / 2, -1, 1)

    geo = 0
    if gold_df is not None and len(gold_df) > 60:
        g_close = gold_df['close']
        g_mom_5 = g_close.pct_change(5).iloc[-1]
        g_mom_20 = g_close.pct_change(20).iloc[-1]
        g_ma20 = g_close.rolling(20).mean().iloc[-1]
        if g_mom_5 > 0.02 and g_mom_20 > 0.05 and g_close.iloc[-1] > g_ma20:
            geo = -0.8
        elif g_mom_5 > 0.02:
            geo = -0.4
        elif g_mom_5 < -0.02:
            geo = 0.4

    ma120 = close.rolling(120).mean().iloc[-1]
    trend_score = np.clip((close.iloc[-1] - ma120) / ma120 * 5, -1, 1)
    high_60 = close.rolling(60).max()
    low_60 = close.rolling(60).min()
    position = (close.iloc[-1] - low_60.iloc[-1]) / (high_60.iloc[-1] - low_60.iloc[-1] + 1e-8)
    breadth_score = 0.5 if position > 0.8 else (-0.5 if position < 0.2 else 0)
    policy = np.clip((trend_score + breadth_score) / 2, -1, 1)

    combined = sentiment * 0.4 + geo * 0.3 + policy * 0.3
    return {'sentiment': sentiment, 'geopolitical': geo, 'policy': policy,
            'combined': np.clip(combined, -1, 1)}


def detect_bubble(etf_df: pd.DataFrame) -> dict:
    """检测情绪泡沫"""
    if len(etf_df) < 60:
        return {'is_bubble': False, 'bubble_score': 0, 'rsi': 50, 'return_60d': 0}

    close = etf_df['close']
    volume = etf_df.get('volume', pd.Series(0, index=etf_df.index))

    return_60d = close.iloc[-1] / close.iloc[-60] - 1
    vol_ma20 = volume.rolling(20).mean().iloc[-1] if volume.sum() > 0 else 1
    vol_current = volume.iloc[-1] if volume.sum() > 0 else 0
    turnover_spike = vol_current / (vol_ma20 + 1e-8) > 3

    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    vol_5d = close.pct_change().rolling(5).std().iloc[-1]
    vol_60d = close.pct_change().rolling(60).std().iloc[-1]
    vol_spike = vol_5d / (vol_60d + 1e-8) > 2

    bubble_score = 0
    if return_60d > 0.5: bubble_score += 0.3
    if turnover_spike: bubble_score += 0.2
    if rsi > 80: bubble_score += 0.3
    if vol_spike: bubble_score += 0.2

    return {'is_bubble': bubble_score >= 0.5, 'bubble_score': bubble_score,
            'rsi': rsi, 'return_60d': return_60d}


# ============================================================
# 信号分类
# ============================================================
def classify_signal(score: float, bubble: dict, factors: dict,
                    close: float, ma20: float, ma60: float, ma120: float,
                    mom_120: float, mom_60: float, symbol: str) -> tuple:
    """根据指标生成交易信号"""
    # 减仓/清仓信号（优先级最高）
    if bubble['is_bubble']:
        reason = (f"RSI={bubble['rsi']:.1f} 超买，60日涨幅 {bubble['return_60d']:.1%}，"
                  f"泡沫得分 {bubble['bubble_score']:.2f}")
        return "减仓", reason, "强"

    if factors['sentiment'] > 0.5 and symbol in EMOTION_ETFS:
        return "减仓", f"市场情绪过热 ({factors['sentiment']:.2f})，情绪型 ETF 建议获利了结", "中"

    # 加仓/建仓信号
    is_bottom = mom_120 < -0.15

    if is_bottom and mom_60 < -0.10:
        return "试探建仓", f"120日跌幅 {mom_120:.1%}，60日跌幅 {mom_60:.1%}，严重超跌", "强"

    if score > 0.6 and close > ma20 > ma60:
        return "加仓/持有", f"动量得分 {score:.2f}，趋势多头排列，处于强势上升期", "中"

    if score > 0.4 and close > ma20:
        return "建仓", f"动量得分 {score:.2f}，突破20日均线，趋势转强", "中"

    if mom_120 < -0.10 and close < ma20 < ma60:
        return "观望/等待", f"处于下降通道，120日跌幅 {mom_120:.1%}", "弱"

    return "观望", f"动量得分 {score:.2f}，无明显建仓/减仓信号", "-"


# ============================================================
# 报告生成
# ============================================================
def generate_report(output_path: Optional[str] = None) -> str:
    """生成每日信号报告"""
    all_prices, gold_df = load_all_data()
    benchmark_df = all_prices.get(BENCHMARK_SYMBOL)

    if benchmark_df is None or len(benchmark_df) < 120:
        raise ValueError("基准数据不足，无法计算信号")

    latest_date = benchmark_df.index[-1]
    factors = calculate_three_factors(benchmark_df, gold_df)

    lines = []
    lines.append("=" * 80)
    lines.append(f"QuanTrade 每日交易信号报告 - {latest_date.strftime('%Y-%m-%d')}")
    lines.append("=" * 80)
    lines.append("")

    # 市场环境
    lines.append("【市场环境】")
    lines.append(f"  最新日期: {latest_date.date()}")
    lines.append(f"  情绪因子: {factors['sentiment']:.2f} ({'过热' if factors['sentiment'] > 0.5 else '偏冷' if factors['sentiment'] < -0.5 else '中性'})")
    lines.append(f"  地缘因子: {factors['geopolitical']:.2f} ({'风险高' if factors['geopolitical'] < -0.5 else '风险低'})")
    lines.append(f"  政策因子: {factors['policy']:.2f} ({'宽松/向好' if factors['policy'] > 0.5 else '收紧/承压' if factors['policy'] < -0.5 else '中性'})")
    lines.append(f"  综合因子: {factors['combined']:.2f}")
    lines.append("")

    # 大盘状态
    bench_close = benchmark_df['close'].iloc[-1]
    bench_ma20 = benchmark_df['close'].rolling(20).mean().iloc[-1]
    bench_ma60 = benchmark_df['close'].rolling(60).mean().iloc[-1]
    bench_ma120 = benchmark_df['close'].rolling(120).mean().iloc[-1]
    bench_mom_5 = benchmark_df['close'].pct_change(5).iloc[-1]
    bench_mom_60 = benchmark_df['close'].pct_change(60).iloc[-1]
    bench_mom_120 = benchmark_df['close'].pct_change(120).iloc[-1]

    lines.append("【大盘状态】")
    lines.append(f"  沪深300: {bench_close:.3f}")
    lines.append(f"  5日涨幅: {bench_mom_5:.2%}")
    lines.append(f"  60日涨幅: {bench_mom_60:.2%}")
    lines.append(f"  120日涨幅: {bench_mom_120:.2%}")
    lines.append(f"  均线位置: 价格{'高于' if bench_close > bench_ma20 else '低于'}20日线, "
                 f"{'高于' if bench_close > bench_ma60 else '低于'}60日线, "
                 f"{'高于' if bench_close > bench_ma120 else '低于'}120日线")
    lines.append("")

    # ETF 信号
    signals = []
    market_bottom = bench_mom_120 < -0.15

    for symbol in ETF_POOL:
        if symbol not in all_prices or len(all_prices[symbol]) < 120:
            continue

        df = all_prices[symbol]
        close = df['close'].iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma60 = df['close'].rolling(60).mean().iloc[-1]
        ma120 = df['close'].rolling(120).mean().iloc[-1]
        mom_120 = df['close'].pct_change(120).iloc[-1]
        mom_60 = df['close'].pct_change(60).iloc[-1]

        score = calculate_momentum_score(df, benchmark_df, symbol, market_bottom)
        bubble = detect_bubble(df)

        signal, reason, strength = classify_signal(
            score, bubble, factors, close, ma20, ma60, ma120,
            mom_120, mom_60, symbol
        )

        signals.append({
            'symbol': symbol,
            'signal': signal,
            'strength': strength,
            'score': score,
            'close': close,
            'mom_120': mom_120,
            'mom_60': mom_60,
            'rsi': bubble['rsi'],
            'bubble_score': bubble['bubble_score'],
            'reason': reason,
        })

    sig_df = pd.DataFrame(signals)

    # 按信号类型分组输出
    signal_order = ["减仓", "加仓/持有", "建仓", "试探建仓", "观望/等待", "观望"]
    for sig_type in signal_order:
        subset = sig_df[sig_df['signal'] == sig_type]
        if len(subset) == 0:
            continue

        lines.append(f"【{sig_type}信号】共 {len(subset)} 只")
        lines.append("-" * 80)

        for _, row in subset.sort_values('score', ascending=False).iterrows():
            lines.append(f"  {row['symbol']:>6s} | 收盘价 {row['close']:>7.3f} | "
                         f"120日 {row['mom_120']:>+7.1%} | 60日 {row['mom_60']:>+7.1%} | "
                         f"RSI {row['rsi']:>5.1f} | 强度 {row['strength']:>2s}")
            lines.append(f"         原因: {row['reason']}")
            lines.append("")

    # 操作建议摘要
    lines.append("=" * 80)
    lines.append("【操作建议摘要】")
    lines.append("=" * 80)

    buy_count = len(sig_df[sig_df['signal'].isin(['建仓', '试探建仓'])])
    add_count = len(sig_df[sig_df['signal'] == '加仓/持有'])
    reduce_count = len(sig_df[sig_df['signal'] == '减仓'])

    lines.append(f"  建议建仓: {buy_count} 只")
    lines.append(f"  建议加仓/持有: {add_count} 只")
    lines.append(f"  建议减仓: {reduce_count} 只")
    lines.append(f"  观望: {len(sig_df) - buy_count - add_count - reduce_count} 只")
    lines.append("")

    if factors['combined'] < -0.5:
        lines.append("  ⚠️ 市场环境偏空，建议控制仓位，谨慎建仓")
    elif factors['combined'] > 0.5:
        lines.append("  ✅ 市场环境偏多，可积极跟随趋势")
    else:
        lines.append("  ➡️ 市场环境中性，按个股信号操作")

    lines.append("")
    lines.append("=" * 80)
    lines.append("免责声明: 本报告仅供学习研究，不构成投资建议。")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # 输出到文件
    if output_path is None:
        output_path = os.path.join(REPORT_DIR, f"signal_{latest_date.strftime('%Y-%m-%d')}.txt")

    ensure_report_dir()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n报告已保存: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="生成 QuanTrade 每日交易信号报告")
    parser.add_argument("--output", default=None, help="报告输出路径")
    args = parser.parse_args()

    generate_report(args.output)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
