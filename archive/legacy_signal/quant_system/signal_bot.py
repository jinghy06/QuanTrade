#!/usr/bin/env python3
"""
A股量化信号系统 - 树莓派主入口
定时信号生成 + 飞书推送
对应 plan.md Phase 5-6: "树莓派部署 + 模拟验证"

使用方法:
    # 单次运行（测试）
    python signal_bot.py --once

    # 定时模式（生产）
    python signal_bot.py --schedule

    # 分析单只股票
    python signal_bot.py --symbol 000001.SZ

    # 指定股票池文件
    python signal_bot.py --watchlist watchlist.txt

    # 使用自定义策略目录
    python signal_bot.py --once --strategy-dir strategies/custom

    # 训练模型
    python signal_bot.py --train
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import schedule

# 确保项目根目录在路径中
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config.settings import (
    DB_PATH,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    MODELS_DIR,
    SCHEDULE_TIMES,
    SIGNAL_THRESHOLD,
    WATCHLIST,
)
from data.data_fetcher import DataFetcher
from data.data_store import DataStore
from features.feature_engine import FeatureEngine
from models.ml_trainer import MLTrainer
from notify.feishu_bot import FeishuBot
from trend_forecaster import TrendForecaster
from kline_plotter import KlinePlotter

# 策略系统集成
from strategies.loader import StrategyLoader
from strategies.registry import StrategyRegistry

# ---------- 日志配置 ----------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("SignalBot")


class SignalBot:
    """
    量化信号机器人
    整合数据获取、特征计算、ML预测、策略信号、飞书推送全流程
    """

    def __init__(self, strategy_dir: str = None):
        self.store = DataStore()
        self.fetcher = DataFetcher()
        self.feature_engine = FeatureEngine()
        self.ml_model = MLTrainer()
        self.feishu = FeishuBot()
        self.trend_forecaster = TrendForecaster()
        self.kline_plotter = KlinePlotter(save_dir=str(MODELS_DIR.parent / "plots"))

        # 策略系统
        self.strategy_loader = StrategyLoader()
        self.strategy_loader.load_all(custom_dir=strategy_dir)
        self.strategy_registry = self.strategy_loader.get_registry()
        logger.info("策略系统加载完成: %d个策略", len(self.strategy_registry))

        # 加载ML模型（多视野回归版）
        self._load_models()

    def _load_models(self):
        """加载多视野回归模型"""
        config_path = MODELS_DIR / "model_config.json"
        if config_path.exists():
            try:
                config = self.ml_model.load_models(str(config_path))
                n_models = len(config.get("horizons", []))
                logger.info(
                    "多视野模型加载成功 | 模型数: %d | 特征数: %d",
                    n_models, config.get("n_features_in_", 0),
                )
            except Exception as e:
                logger.error("模型加载失败: %s", e)
                logger.warning("将以纯策略模式运行（无ML信号）")
        else:
            logger.warning("未找到模型配置文件，请先训练模型")

    # ============================================================
    # 核心流程
    # ============================================================

    def run_full_pipeline(self, watchlist: list = None):
        """
        完整信号流水线
        对应 plan.md 数据流与信号流
        """
        symbols = watchlist or WATCHLIST
        logger.info("=" * 60)
        logger.info("信号流水线启动 | 股票池: %d只", len(symbols))
        logger.info("=" * 60)

        # Step 1: 数据同步
        df_macro = self._sync_data(symbols)

        # Step 2: 特征计算
        self._compute_features(symbols, df_macro)

        # Step 3: ML预测 -> 筛选候选
        candidates = self._ml_screening(symbols)

        if not candidates:
            logger.info("无候选股票通过ML筛选（阈值>%.2f）", SIGNAL_THRESHOLD)
            self.feishu.send_text(
                f"📋 {datetime.now().strftime('%Y-%m-%d')} 无交易信号\n"
                f"（没有股票ML上涨概率超过{SIGNAL_THRESHOLD:.0%}）"
            )
            return

        logger.info("ML筛选通过: %d只", len(candidates))

        # Step 3.5: 运行技术策略
        candidate_symbols = [c["symbol"] for c in candidates]
        strategy_signals = self._run_strategies(candidate_symbols)

        # Step 4: 基于策略信号生成交易信号
        signals = self._generate_signals_from_strategies(candidates, strategy_signals)

        # Step 5: 保存信号 & 飞书推送
        self._publish_signals(signals)

        logger.info("流水线完成 | 生成信号: %d条", len(signals))

    def _sync_data(self, symbols: list) -> "pd.DataFrame|None":
        """
        Step 1: 数据同步
        - 增量更新日K数据（并发拉取）
        - 获取宏观数据
        """
        logger.info("[1/5] 数据同步...")

        # 获取宏观数据
        df_macro = self.fetcher.get_macro_bond_yield()

        # 筛选需要更新的 symbol
        to_update = []
        for symbol in symbols:
            try:
                date_range = self.store.get_date_range(symbol)
                if date_range[1]:
                    latest_db = datetime.strptime(date_range[1], "%Y-%m-%d")
                    if (datetime.now() - latest_db).days <= 2:
                        continue
                to_update.append(symbol)
            except Exception as e:
                logger.error("检查数据日期失败 %s: %s", symbol, e)

        if not to_update:
            logger.info("[1/5] 数据同步完成 | 所有数据已是最新")
            return df_macro

        # 并发拉取
        updated = 0

        def _fetch_one(sym):
            time.sleep(0.3)  # 速率限制
            df = self.fetcher.get_daily_k(sym, use_cache=True)
            if not df.empty:
                self.store.save_klines(df)
                return sym
            return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_fetch_one, s): s for s in to_update}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    result = future.result()
                    if result:
                        updated += 1
                except Exception as e:
                    logger.error("数据同步失败 %s: %s", sym, e)

        logger.info("[1/5] 数据同步完成 | 更新: %d只", updated)
        return df_macro

    def _compute_features(self, symbols: list, df_macro=None):
        """
        Step 2: 特征计算
        为每只股票计算最新特征
        """
        logger.info("[2/5] 特征计算...")

        computed = 0
        for symbol in symbols:
            try:
                df_price = self.store.get_kline(symbol, n_days=60)
                if len(df_price) < 20:
                    continue

                df_feat = self.feature_engine.compute_features(df_price, df_macro)
                if not df_feat.empty:
                    self.store.save_features(df_feat.tail(1))
                    computed += 1

            except Exception as e:
                logger.error("特征计算失败 %s: %s", symbol, e)

        logger.info("[2/5] 特征计算完成 | %d只", computed)

    def _ml_screening(self, symbols: list) -> list:
        """
        Step 3: ML多视野趋势筛选
        基于未来1/3/5/10日走势预测筛选候选股票，生成预测K线图
        """
        logger.info("[3/5] ML多视野趋势筛选...")

        if not self.ml_model.models:
            logger.warning("ML模型未加载，跳过筛选，全量进入策略分析")
            return [{"symbol": s, "trend": {"overall_direction": "unknown"}} for s in symbols]

        candidates = []
        for symbol in symbols:
            try:
                df_feat = self.store.get_features(symbol, n_days=5)
                if df_feat.empty:
                    continue

                # 获取最新价格
                df_kline = self.store.get_kline(symbol, n_days=60)
                current_price = df_kline["close"].iloc[-1] if not df_kline.empty else 0

                # 多视野趋势预测
                trend_pred = self.ml_model.predict_trend(df_feat, current_price)
                anchors = trend_pred.get("anchors", {})

                # 走势分类
                trend_info = self.trend_forecaster.classify_trend(
                    {d: a["price"] for d, a in anchors.items()},
                    current_price,
                )

                # 筛选逻辑：5日预期收益为正 或 走势为"先抑后扬"
                r5 = trend_info.get("returns", {}).get("5d", 0)
                trend_type = trend_info.get("trend_type", "")
                if r5 > 0 or trend_type in ["先抑后扬", "强势上涨"]:
                    # 生成预测K线图
                    forecast_df = self.trend_forecaster.generate_future_kline(
                        current_price=current_price,
                        anchors={d: a["price"] for d, a in anchors.items()},
                        predicted_volatility=trend_pred.get("predicted_volatility", 0.02),
                        n_days=10,
                    )
                    plot_path = None
                    try:
                        plot_path = self.kline_plotter.plot_with_forecast(
                            history_df=df_kline,
                            forecast_df=forecast_df,
                            symbol=symbol,
                            trend_info=trend_info,
                        )
                    except Exception as e:
                        logger.warning("预测K线图生成失败 %s: %s", symbol, e)

                    candidates.append({
                        "symbol": symbol,
                        "current_price": current_price,
                        "trend": trend_info,
                        "anchors": anchors,
                        "predicted_volatility": trend_pred.get("predicted_volatility", 0),
                        "forecast_df": forecast_df,
                        "plot_path": plot_path,
                    })
                    logger.debug(
                        "ML通过: %s, 5d=%.2f%%, 类型=%s",
                        symbol, r5, trend_type,
                    )

            except Exception as e:
                logger.error("ML趋势预测失败 %s: %s", symbol, e)

        # 按5日预期收益降序
        candidates.sort(
            key=lambda x: x["trend"].get("returns", {}).get("5d", -999),
            reverse=True,
        )
        candidates = candidates[:20]

        logger.info("[3/5] ML趋势筛选完成 | 候选: %d只", len(candidates))
        return candidates

    def _generate_signals_from_strategies(self, candidates: list, strategy_signals: dict = None) -> list:
        """
        Step 4: 基于策略信号生成交易信号
        """
        logger.info("[4/5] 策略信号汇总...")

        signals = []
        for i, cand in enumerate(candidates, 1):
            symbol = cand["symbol"]
            try:
                logger.info("[%d/%d] 汇总策略信号 %s...", i, len(candidates), symbol)
                sym_strategy = strategy_signals.get(symbol) if strategy_signals else None

                if sym_strategy:
                    combined = sym_strategy["combined"]
                    signal = {
                        "symbol": symbol,
                        "signal": combined.action,
                        "confidence": combined.confidence,
                        "suggestion": {
                            "action": combined.action,
                            "target_price": 0.0,
                            "stop_loss": 0.0,
                            "position_pct": 0.0,
                            "rationale": combined.rationale,
                            "risk_factors": [],
                        },
                        "holding_advice": None,
                        "trend_prediction": cand.get("trend"),
                        "forecast_plot": cand.get("plot_path"),
                        "current_price": cand.get("current_price"),
                        "strategy_details": {
                            "n_triggered": sym_strategy["n_triggered"],
                            "results": {name: {
                                "action": sig.action,
                                "confidence": sig.confidence,
                                "triggered": sig.triggered,
                            } for name, sig in sym_strategy["results"].items()},
                        },
                    }
                    signals.append(signal)
                else:
                    # 无策略信号，仅基于ML趋势预测生成弱信号
                    trend = cand.get("trend", {})
                    returns_5d = trend.get("returns", {}).get("5d", 0)
                    if returns_5d > 0:
                        signal = {
                            "symbol": symbol,
                            "signal": "轻仓试探",
                            "confidence": min(0.55, 0.5 + returns_5d),
                            "suggestion": {
                                "action": "基于ML趋势预测轻仓试探",
                                "target_price": 0.0,
                                "stop_loss": 0.0,
                                "position_pct": 0.0,
                                "rationale": f"ML预测5日收益率{returns_5d:.2%}，趋势向好",
                                "risk_factors": ["无策略确认", "仅ML预测"],
                            },
                            "holding_advice": None,
                            "trend_prediction": trend,
                            "forecast_plot": cand.get("plot_path"),
                            "current_price": cand.get("current_price"),
                        }
                        signals.append(signal)

            except Exception as e:
                logger.error("策略信号汇总失败 %s: %s", symbol, e)

        logger.info("[4/5] 策略信号汇总完成 | 信号: %d条", len(signals))
        return signals

    def _run_strategies(self, symbols: list) -> dict:
        """
        Step 4.5: 运行所有已注册的策略
        返回 {symbol: aggregated_signal}
        """
        logger.info("[4.5/5] 策略信号生成 (%d个策略)...", len(self.strategy_registry))
        strategy_signals = {}

        for symbol in symbols:
            try:
                df_kline = self.store.get_kline(symbol, n_days=60)
                if len(df_kline) < 30:
                    continue

                results = self.strategy_registry.run_all(df_kline, symbol=symbol)
                triggered = {k: v for k, v in results.items() if v.triggered}

                if triggered:
                    combined = self.strategy_registry.aggregate_voting(
                        results, method="confidence_weighted"
                    )
                    strategy_signals[symbol] = {
                        "results": results,
                        "combined": combined,
                        "n_triggered": len(triggered),
                    }
                    logger.info(
                        "[%s] %d个策略触发 -> 聚合: %s (conf=%.2f)",
                        symbol, len(triggered), combined.action, combined.confidence
                    )

            except Exception as e:
                logger.error("策略运行失败 %s: %s", symbol, e)

        logger.info("[4.5/5] 策略信号完成: %d只股票触发", len(strategy_signals))
        return strategy_signals

    def _publish_signals(self, signals: list):
        """
        Step 5: 信号发布
        - 保存到数据库
        - 飞书推送
        - 防重复推送（同日同 symbol 跳过）
        """
        logger.info("[5/5] 信号发布...")

        published = 0
        skipped = 0
        for signal in signals:
            try:
                symbol = signal.get("symbol", "")

                # 去重：今日已推送过该 symbol 则跳过
                if self.store.has_today_signal(symbol):
                    logger.info("跳过重复信号: %s (今日已推送)", symbol)
                    skipped += 1
                    continue

                # 保存到数据库
                signal_id = self.store.save_signal(signal)
                signal["id"] = signal_id

                # 飞书推送
                self.feishu.send_signal_card(signal)
                published += 1

                logger.info(
                    "信号发布: %s %s (置信度%.2f)",
                    signal.get("symbol"),
                    signal.get("signal"),
                    signal.get("confidence", 0),
                )

            except Exception as e:
                logger.error("信号发布失败: %s", e)

        # 发送每日汇总
        if published > 1:
            self.feishu.send_daily_summary(signals)

        logger.info("[5/5] 信号发布完成 | 发布: %d, 跳过: %d", published, skipped)

    # ============================================================
    # 定时任务
    # ============================================================

    def schedule_jobs(self):
        """设置定时任务"""
        for t in SCHEDULE_TIMES:
            schedule.every().day.at(t).do(self.run_full_pipeline)
            logger.info("定时任务已设置: 每日 %s", t)

        logger.info("进入定时循环，按Ctrl+C退出...")
        while True:
            schedule.run_pending()
            time.sleep(60)

    # ============================================================
    # CLI命令
    # ============================================================

    def analyze_single(self, symbol: str):
        """分析单只股票（基于策略信号）"""
        logger.info("单股分析: %s", symbol)
        strategy_signals = self._run_strategies([symbol])
        sym_strategy = strategy_signals.get(symbol)

        if sym_strategy:
            combined = sym_strategy["combined"]
            signal = {
                "symbol": symbol,
                "signal": combined.action,
                "confidence": combined.confidence,
                "suggestion": {
                    "action": combined.action,
                    "target_price": 0.0,
                    "stop_loss": 0.0,
                    "position_pct": 0.0,
                    "rationale": combined.rationale,
                    "risk_factors": [],
                },
                "holding_advice": None,
                "strategy_details": {
                    "n_triggered": sym_strategy["n_triggered"],
                    "results": {name: {
                        "action": sig.action,
                        "confidence": sig.confidence,
                        "triggered": sig.triggered,
                    } for name, sig in sym_strategy["results"].items()},
                },
            }
        else:
            signal = {
                "symbol": symbol,
                "signal": "观望",
                "confidence": 0.0,
                "suggestion": {
                    "action": "无策略信号触发，建议观望",
                    "target_price": 0.0,
                    "stop_loss": 0.0,
                    "position_pct": 0.0,
                    "rationale": "当前无任何策略触发信号",
                    "risk_factors": [],
                },
                "holding_advice": None,
            }

        print(json.dumps(signal, indent=2, ensure_ascii=False))
        self.feishu.send_signal_card(signal)
        return signal

    def train_model(self, symbols: list = None, tune: bool = False):
        """训练ML模型"""
        symbols = symbols or WATCHLIST[:50]  # 训练用股票池
        logger.info("开始训练模型 | 股票池: %d只", len(symbols))

        # 1. 获取数据并计算特征
        df_macro = self.fetcher.get_macro_bond_yield()
        self.feature_engine.batch_compute(symbols, df_macro)

        # 2. 训练
        end_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365 * 3)).strftime(
            "%Y%m%d"
        )

        result = self.ml_model.train(
            symbols, start_date=start_date, end_date=end_date, tune_hyperparams=tune
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    def evaluate_model(self, symbols: list = None, n_days: int = 30):
        """评估模型近期表现，检测衰减（回归版）"""
        symbols = symbols or WATCHLIST[:50]
        logger.info("模型评估 | 股票池: %d只 | 近%d天", len(symbols), n_days)
        result = self.ml_model.evaluate_recent(symbols, n_days=n_days)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        if result.get("overall_needs_retrain"):
            worst = max(
                result.get("horizon_results", {}).values(),
                key=lambda x: x.get("rmse", 0),
            )
            self.feishu.send_error_alert(
                f"模型衰减告警: 近{n_days}天最差RMSE {worst['rmse']:.2%}，建议重新训练"
            )
        return result


def main():
    parser = argparse.ArgumentParser(
        description="A股量化信号系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --once                    # 单次运行完整流水线
  %(prog)s --schedule                # 定时模式（每日9:00和17:30）
  %(prog)s --symbol 000001.SZ        # 分析单只股票
  %(prog)s --train                   # 训练ML模型
  %(prog)s --train --tune            # 训练（含超参调优）
  %(prog)s --once --watchlist list.txt  # 使用自定义股票池
        """,
    )

    parser.add_argument("--once", action="store_true", help="单次运行")
    parser.add_argument("--schedule", action="store_true", help="定时模式")
    parser.add_argument("--symbol", type=str, help="分析单只股票")
    parser.add_argument("--train", action="store_true", help="训练ML模型")
    parser.add_argument("--tune", action="store_true", help="超参数调优")
    parser.add_argument(
        "--watchlist", type=str, help="股票池文件（每行一个代码）"
    )
    parser.add_argument(
        "--strategy-dir", type=str, default=None,
        help="自定义策略目录（默认加载内置策略）"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="评估模型近30天表现，检测衰减"
    )
    parser.add_argument(
        "--eval-days", type=int, default=30,
        help="模型评估天数（默认30天）"
    )

    args = parser.parse_args()

    # 加载自定义股票池
    watchlist = None
    if args.watchlist:
        with open(args.watchlist, "r") as f:
            watchlist = [line.strip() for line in f if line.strip()]
        logger.info("加载自定义股票池: %d只", len(watchlist))

    # 创建SignalBot（传入策略目录）
    bot = SignalBot(strategy_dir=args.strategy_dir)

    if args.symbol:
        # 单股分析
        bot.analyze_single(args.symbol)

    elif args.train:
        # 训练模型
        bot.train_model(watchlist, tune=args.tune)

    elif args.evaluate:
        # 评估模型
        bot.evaluate_model(watchlist, n_days=args.eval_days)

    elif args.schedule:
        # 定时模式
        bot.schedule_jobs()

    else:
        # 默认单次运行
        bot.run_full_pipeline(watchlist)


if __name__ == "__main__":
    main()
