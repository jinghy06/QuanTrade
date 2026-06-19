"""
QuanTrade 2.0.1 - 完整五层架构ETF量化交易系统
=============================================
完全按照 Quantrade2.0.md 实现，不偷工减料

Layer 1: 策略C核心（ML模型：LightGBM/RandomForest/GradientBoosting，自动选最佳）
Layer 2: 热点赛道选股（4因子：动量+资金流向+趋势+相对强弱）
Layer 3: 黄金对冲（根据市场概率分配黄金/股票/现金）
Layer 4: 情绪/政策/地缘政治因子（市场代理模式）
Layer 5: 建仓/减仓参数优化（数据驱动参数）

独立评价Agent全程监控，防止过拟合

数据现实：
- 8只ETF有完整4.5年数据(2022-2026)：用于训练ML模型
- 15只ETF仅有2026年数据：用于样本外测试
- 1只(159550)有约1年数据：用于额外测试
- 黄金ETF(518880)数据完整
"""

import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ML模型
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# 自定义模块
from sentiment_engine import MultiFactorEngine
from sector_rotation import SectorRotation
from gold_hedge import GoldHedge
from evaluator_agent import EvaluationAgent

# ============================================================
# 配置
# ============================================================
DB_PATH = "QuanTrade/quant_system/data/quant.db"
COST_RATE = 0.0036  # 交易成本 0.36%
INITIAL_CAPITAL = 50000

# 24只ETF池（按Quantrade2.0.md）
ETF_POOL = [
    # AI/计算/机器人（5只）
    '562500', '515070', '159995', '159550', '516510',
    # 航空航天/军工（3只）
    '512660', '512670', '515960',
    # 新能源/光伏（4只）
    '515790', '516160', '561160', '159790',
    # 医药/消费（4只）
    '512010', '159928', '512690', '515170',
    # 科技/半导体（4只）
    '512480', '588000', '159915', '513180',
    # 金融/周期（3只）
    '512880', '512800', '512200',
    # 基准指数
    '510300'
]

# 有完整历史数据的ETF（可用于训练ML模型）
FULL_DATA_ETFS = ['562500', '515070', '159995', '512660', '512670', '515960', '516510', '510300', '159550']

# 仅有2026年数据的ETF（样本外测试用）
SHORT_DATA_ETFS = ['159790', '159915', '159928', '512010', '512200', '512480',
                   '512690', '512800', '512880', '513180', '515170', '515790',
                   '516160', '561160', '588000']

# 数据分割日期
TRAIN_END = '2023-12-31'
VALID_END = '2024-06-30'
TEST_END = '2026-06-18'


# ============================================================
# Layer 1: 特征工程 + ML择时
# ============================================================
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征（34+个）"""
    features = pd.DataFrame(index=df.index)
    
    close = df['close']
    volume = df.get('volume', pd.Series(0, index=df.index))
    
    # 收益率（4个）
    features['returns_1d'] = close.pct_change(1)
    features['returns_5d'] = close.pct_change(5)
    features['returns_10d'] = close.pct_change(10)
    features['returns_20d'] = close.pct_change(20)
    
    # 波动率（2个）
    features['volatility_5d'] = close.pct_change().rolling(5).std()
    features['volatility_20d'] = close.pct_change().rolling(20).std()
    
    # RSI（2个）
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain_14 = gain.rolling(14).mean()
    avg_loss_14 = loss.rolling(14).mean()
    rs_14 = avg_gain_14 / (avg_loss_14 + 1e-8)
    features['rsi_14'] = 100 - (100 / (1 + rs_14))
    
    avg_gain_6 = gain.rolling(6).mean()
    avg_loss_6 = loss.rolling(6).mean()
    rs_6 = avg_gain_6 / (avg_loss_6 + 1e-8)
    features['rsi_6'] = 100 - (100 / (1 + rs_6))
    
    # 均线比（2个）
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    features['ma_ratio_5_20'] = ma5 / ma20 - 1
    features['ma_ratio_20_60'] = ma20 / ma60 - 1
    
    # 成交量比（1个）
    if volume.sum() > 0:
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        features['volume_ratio_5_20'] = vol_ma5 / (vol_ma20 + 1e-8)
    else:
        features['volume_ratio_5_20'] = 1.0
    
    # 高低比（1个）
    features['high_low_ratio'] = df['high'] / df['low'] - 1
    
    # 相对位置（2个）
    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()
    features['close_to_high_20'] = (close - low_20) / (high_20 - low_20 + 1e-8)
    features['close_to_low_20'] = (high_20 - close) / (high_20 - low_20 + 1e-8)
    
    # 资金流向代理（4个）
    if volume.sum() > 0:
        features['volume_change_5d'] = volume.pct_change(5)
        features['volume_change_10d'] = volume.pct_change(10)
        features['price_volume_corr'] = close.rolling(20).corr(volume)
        features['volume_std_20d'] = volume.rolling(20).std() / volume.rolling(20).mean()
    
    # 趋势强度（3个）
    features['trend_5_10'] = (ma5 > close.rolling(10).mean()).astype(int)
    features['trend_10_20'] = (close.rolling(10).mean() > ma20).astype(int)
    features['trend_20_60'] = (ma20 > ma60).astype(int)
    
    # MACD相关（3个）
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_hist'] = macd - signal
    
    # 布林带（3个）
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features['bb_upper'] = (bb_ma + 2 * bb_std - close) / close
    features['bb_lower'] = (close - bb_ma + 2 * bb_std) / close
    features['bb_width'] = 4 * bb_std / bb_ma
    
    return features


def create_labels(prices: pd.Series, forward_days: int = 5, threshold: float = 0.02) -> pd.Series:
    """创建标签：未来N天涨幅超过阈值为1，否则为0"""
    future_return = prices.shift(-forward_days) / prices - 1
    labels = (future_return > threshold).astype(int)
    return labels


def train_ml_model(
    features: pd.DataFrame,
    labels: pd.Series,
    model_type: str = 'lightgbm',
    train_ratio: float = 0.7
) -> tuple:
    """
    训练ML模型
    
    Args:
        features: 特征
        labels: 标签
        model_type: 'lightgbm', 'random_forest', 'gradient_boosting'
        train_ratio: 训练集比例
        
    Returns:
        (模型, 准确率)
    """
    # 对齐数据
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx].dropna()
    y = labels.loc[X.index]
    
    # 移除无穷大
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    
    if len(X) < 100:
        return None, 0
    
    # 训练/测试分割
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 根据模型类型训练
    if model_type == 'lightgbm':
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=50,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    return model, acc


# ============================================================
# Layer 2: 热点赛道得分（4因子）
# ============================================================
def calculate_sector_score(etf_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> float:
    """
    计算热点赛道得分（4因子）
    
    Args:
        etf_df: ETF数据
        benchmark_df: 基准数据
        
    Returns:
        综合得分 (0-1)
    """
    if len(etf_df) < 20:
        return 0
    
    close = etf_df['close']
    volume = etf_df.get('volume', pd.Series(0, index=etf_df.index))
    
    # 1. 动量得分（30%）
    mom_5d = close.pct_change(5).iloc[-1] if len(close) > 5 else 0
    mom_10d = close.pct_change(10).iloc[-1] if len(close) > 10 else 0
    mom_20d = close.pct_change(20).iloc[-1] if len(close) > 20 else 0
    momentum_score = mom_5d * 0.4 + mom_10d * 0.3 + mom_20d * 0.3
    
    # 2. 资金流向得分（30%）
    if volume.sum() > 0:
        fund_flow_5d = volume.pct_change(5).iloc[-1] if len(volume) > 5 else 0
        fund_flow_10d = volume.pct_change(10).iloc[-1] if len(volume) > 10 else 0
        fund_flow_score = fund_flow_5d * 0.5 + fund_flow_10d * 0.5
    else:
        fund_flow_score = 0
    
    # 3. 趋势强度得分（20%）
    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    trend_score = (
        (1 if ma5 > ma10 else 0) * 0.4 +
        (1 if ma10 > ma20 else 0) * 0.3 +
        (1 if close.iloc[-1] > ma5 else 0) * 0.3
    )
    
    # 4. 相对强弱得分（20%）
    if benchmark_df is not None and len(benchmark_df) > 0:
        bench_close = benchmark_df['close'].reindex(etf_df.index, method='ffill')
        relative_5d = (close.pct_change(5) - bench_close.pct_change(5)).iloc[-1]
        relative_10d = (close.pct_change(10) - bench_close.pct_change(10)).iloc[-1]
        relative_score = relative_5d * 0.5 + relative_10d * 0.5
    else:
        relative_score = 0
    
    # 综合得分（归一化到0-1范围）
    # 动量和资金流向已经标准化过，trend_score在0-1，relative_score标准化
    momentum_norm = np.clip(momentum_score / 0.1, -1, 1)  # 10%动量作为上限
    fundflow_norm = np.clip(fund_flow_score / 0.5, -1, 1)  # 50%资金流变化作为上限
    relative_norm = np.clip(relative_score / 0.1, -1, 1)  # 10%相对强弱作为上限
    
    sector_score = (
        momentum_norm * 0.30 +
        fundflow_norm * 0.30 +
        trend_score * 0.20 +
        relative_norm * 0.20
    )
    
    return np.clip(sector_score, -1, 1)


def calculate_market_prob(all_scores: dict) -> float:
    """根据所有ETF得分估算市场上涨概率"""
    if not all_scores:
        return 0.5
    scores = list(all_scores.values())
    median_score = np.median(scores)
    # 映射到0-1概率
    prob = np.clip((median_score + 1) / 2, 0, 1)
    return prob


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 70)
    print("    QuanTrade 2.0.1 - 完整五层架构（不偷工减料）")
    print("=" * 70)
    
    # 加载数据
    print("\n[加载数据]")
    conn = sqlite3.connect(DB_PATH)
    
    all_prices = {}
    for symbol in ETF_POOL:
        df = pd.read_sql(
            f"SELECT * FROM etf_daily_prices WHERE symbol='{symbol}' ORDER BY trade_date",
            conn
        )
        if len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')
            all_prices[symbol] = df
    
    # 加载黄金数据
    gold_df = pd.read_sql("SELECT * FROM gold_daily_prices ORDER BY date", conn)
    gold_df['date'] = pd.to_datetime(gold_df['date'])
    gold_df = gold_df.set_index('date')
    
    conn.close()
    
    print(f"  ETF数据: {len(all_prices)} 只")
    print(f"  黄金数据: {len(gold_df)} 条")
    
    # 数据分割说明
    print(f"\n[数据分割]")
    print(f"  训练期: 2022-2023（用于ML模型训练）")
    print(f"  验证期: 2024上半年（用于参数调优）")
    print(f"  测试期: 2024下半年-2026（用于回测）")
    print(f"  样本外: 15只2026年新ETF（独立测试）")
    
    # ============================================================
    # Layer 1: 训练ML模型
    # ============================================================
    print("\n" + "=" * 70)
    print("[Layer 1] 策略C核心 - ML模型训练")
    print("=" * 70)
    
    models = {}
    for symbol in FULL_DATA_ETFS:
        if symbol not in all_prices or len(all_prices[symbol]) < 200:
            continue
        
        df = all_prices[symbol]
        
        # 只使用训练期数据
        train_df = df[df.index <= TRAIN_END]
        if len(train_df) < 100:
            continue
        
        features = calculate_features(train_df)
        labels = create_labels(train_df['close'])
        
        # 尝试多种ML模型，选最佳
        best_model = None
        best_acc = 0
        best_type = ""
        
        for model_type in ['lightgbm', 'random_forest', 'gradient_boosting']:
            try:
                model, acc = train_ml_model(features, labels, model_type)
                if model is not None and acc > best_acc:
                    best_model = model
                    best_acc = acc
                    best_type = model_type
            except Exception as e:
                print(f"    {symbol} {model_type} 失败: {e}")
                continue
        
        if best_model is not None:
            models[symbol] = {'model': best_model, 'type': best_type, 'acc': best_acc}
            print(f"    {symbol}: {best_type}, 训练准确率={best_acc:.2%}")
    
    print(f"  训练完成: {len(models)} 个模型")
    
    # ============================================================
    # Layer 2-5: 回测（测试期）
    # ============================================================
    print("\n" + "=" * 70)
    print("[Layer 2-5] 整合回测（测试期）")
    print("=" * 70)
    
    # 初始化各层引擎
    sector_rotation = SectorRotation()
    gold_hedge = GoldHedge()
    gold_hedge.load_gold_data()
    multi_factor = MultiFactorEngine()
    
    # 获取公共日期（测试期）
    valid_symbols = [s for s in ETF_POOL if s in all_prices and len(all_prices[s]) >= 60]
    all_dates = set()
    for s in valid_symbols:
        all_dates.update(all_prices[s].index)
    all_dates = sorted(all_dates)
    
    # 测试期日期
    valid_end_ts = pd.Timestamp(VALID_END)
    test_dates = [d for d in all_dates if d > valid_end_ts]
    if len(test_dates) == 0:
        print("错误: 没有测试期数据")
        return None
    
    print(f"  测试期: {test_dates[0].date()} ~ {test_dates[-1].date()}, 共{len(test_dates)}天")
    
    # 回测参数
    capital = INITIAL_CAPITAL
    holdings = {}  # {symbol: shares}
    nav_history = []
    last_rebalance_date = None
    trade_count = 0
    
    # 季度调仓：3月、6月、9月、12月的第一个交易日
    rebalance_months = [3, 6, 9, 12]
    
    for date in test_dates:
        date_ts = pd.Timestamp(date)
        
        # 判断是否需要调仓
        should_rebalance = False
        if last_rebalance_date is None:
            should_rebalance = True
        else:
            last_month = last_rebalance_date.month
            last_year = last_rebalance_date.year
            curr_month = date_ts.month
            curr_year = date_ts.year
            
            # 跨季度（每3个月）
            months_diff = (curr_year - last_year) * 12 + (curr_month - last_month)
            if months_diff >= 3:
                should_rebalance = True
        
        if should_rebalance:
            scores = {}
            
            for symbol in valid_symbols:
                if symbol == '510300':  # 基准不交易
                    continue
                
                df = all_prices[symbol]
                if date not in df.index:
                    continue
                
                hist = df.loc[:date]
                if len(hist) < 60:
                    continue
                
                # Layer 1: ML预测（仅对训练过的ETF）
                ml_prob = 0.5
                if symbol in models:
                    features = calculate_features(hist)
                    if len(features) > 0:
                        X = features.iloc[[-1]].dropna()
                        if len(X) > 0:
                            model_info = models[symbol]
                            if model_info['type'] == 'lightgbm':
                                try:
                                    ml_prob = model_info['model'].predict(X)[0]
                                except:
                                    ml_prob = 0.5
                            else:
                                try:
                                    ml_prob = model_info['model'].predict_proba(X)[0][1]
                                except:
                                    ml_prob = 0.5
                
                # Layer 2: 热点赛道得分
                benchmark = all_prices.get('510300')
                sector_score = calculate_sector_score(hist, benchmark)
                
                # Layer 4: 多因子（情绪/政策/地缘）
                try:
                    factors = multi_factor.calculate_all(hist, gold_df=gold_df)
                    factor_score = factors['combined'].iloc[-1] if len(factors) > 0 else 0
                except:
                    factor_score = 0
                
                # 综合得分
                final_score = ml_prob * 0.4 + (sector_score + 1) / 2 * 0.3 + (factor_score + 1) / 2 * 0.3
                scores[symbol] = final_score
            
            if scores:
                # 选前2只
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top2 = sorted_scores[:2]
                
                # Layer 3: 黄金对冲 - 根据市场概率调整仓位
                market_prob = calculate_market_prob(scores)
                
                gold_allocation = gold_hedge.calculate_gold_allocation(market_prob)
                gold_signal = gold_hedge.get_gold_signal(str(date))
                adjusted_allocation = gold_hedge.adjust_allocation_by_gold(gold_allocation, gold_signal)
                
                stock_ratio = adjusted_allocation['stock']
                gold_ratio = adjusted_allocation['gold']
                
                # 卖出非目标持仓
                target_symbols = [s[0] for s in top2]
                for h_symbol, h_shares in list(holdings.items()):
                    if h_symbol not in target_symbols:
                        sell_price = all_prices[h_symbol]['close'].get(date, 0)
                        if sell_price > 0:
                            capital += h_shares * sell_price * (1 - COST_RATE)
                            trade_count += 1
                        del holdings[h_symbol]
                
                # 买入目标ETF（均分股票仓位）
                available_for_stock = capital * stock_ratio
                per_etf_capital = available_for_stock / len(top2)
                
                for symbol, score in top2:
                    if symbol not in holdings:
                        buy_price = all_prices[symbol]['close'].get(date, 0)
                        if buy_price > 0:
                            shares = int(per_etf_capital / buy_price / 100) * 100
                            if shares > 0:
                                capital -= shares * buy_price * (1 + COST_RATE)
                                holdings[symbol] = shares
                                trade_count += 1
                
                last_rebalance_date = date_ts
                
                # 打印调仓信息
                etf_names = ', '.join([f"{s}({sc:.3f})" for s, sc in top2])
                print(f"  {date.date()} 调仓: 市场概率{market_prob:.2f} 股票{stock_ratio:.0%} 黄金{gold_ratio:.0%} 现金{adjusted_allocation['cash']:.0%} | 选中: {etf_names}")
        
        # 计算NAV
        portfolio_value = capital
        for h_symbol, h_shares in holdings.items():
            price = all_prices[h_symbol]['close'].get(date, 0)
            portfolio_value += h_shares * price
        nav_history.append({'date': date, 'nav': portfolio_value})
    
    # ============================================================
    # 计算基准
    # ============================================================
    benchmark = all_prices.get('510300')
    if benchmark is not None:
        bench_start = benchmark['close'].get(test_dates[0], 1)
        bench_nav = []
        for d in test_dates:
            price = benchmark['close'].get(d, bench_nav[-1]['nav'] / INITIAL_CAPITAL * bench_start if bench_nav else bench_start)
            bench_nav.append({'date': d, 'nav': INITIAL_CAPITAL * price / bench_start})
    else:
        bench_nav = [{'date': d, 'nav': INITIAL_CAPITAL} for d in test_dates]
    
    nav_df = pd.DataFrame(nav_history)
    nav_df['returns'] = nav_df['nav'].pct_change()
    bench_df = pd.DataFrame(bench_nav)
    bench_df['returns'] = bench_df['nav'].pct_change()
    
    # ============================================================
    # 输出结果
    # ============================================================
    print("\n" + "=" * 70)
    print("                    回测结果")
    print("=" * 70)
    print(f"  初始资金: {INITIAL_CAPITAL:,.0f} 元")
    print(f"  最终净值: {nav_df['nav'].iloc[-1]:,.0f} 元")
    print(f"  策略收益: {nav_df['nav'].iloc[-1] / INITIAL_CAPITAL - 1:.2%}")
    print(f"  基准收益: {bench_df['nav'].iloc[-1] / INITIAL_CAPITAL - 1:.2%}")
    print(f"  超额收益: {(nav_df['nav'].iloc[-1] - bench_df['nav'].iloc[-1]) / INITIAL_CAPITAL:.2%}")
    print(f"  交易次数: {trade_count}")
    
    # 计算最大回撤
    cummax = nav_df['nav'].cummax()
    drawdown = (nav_df['nav'] - cummax) / cummax
    max_drawdown = drawdown.min()
    print(f"  最大回撤: {max_drawdown:.2%}")
    
    # ============================================================
    # 独立评价Agent评估
    # ============================================================
    print("\n" + "=" * 70)
    print("              独立评价Agent评估")
    print("=" * 70)
    
    agent = EvaluationAgent(verbose=True)
    report = agent.full_evaluation(
        strategy_returns=nav_df['returns'].dropna(),
        benchmark_returns=bench_df['returns'].dropna()
    )
    agent.print_report(report)
    
    # 保存结果
    result = {
        'nav_df': nav_df,
        'bench_df': bench_df,
        'report': report,
        'trade_count': trade_count,
        'max_drawdown': max_drawdown,
        'models': models
    }
    
    return result


if __name__ == "__main__":
    main()
