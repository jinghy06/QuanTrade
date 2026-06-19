"""
QuanTrade 2.0 - 五层架构ETF量化交易系统
========================================
Layer 1: Strategy C核心（择时+轮动）- LightGBM
Layer 2: 热点板块轮动评分
Layer 3: 黄金对冲（避险资产切换）
Layer 4: 情绪/政策/地缘政治多因子
Layer 5: 数据驱动优化（网格搜索）

独立评价Agent全程监控，防止过拟合
"""

import numpy as np
import pandas as pd
import sqlite3
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from sentiment_engine import MultiFactorEngine
from sector_rotation import SectorRotation
from evaluator_agent import EvaluationAgent


# ============================================================
# 配置
# ============================================================
DB_PATH = "QuanTrade/quant_system/data/quant.db"
COST_RATE = 0.0036  # 交易成本 0.36%

# 24只ETF池
ETF_POOL = [
    # 宽基指数
    '510300', '159915', '512010', '588000', '159928',
    # 军工航天
    '512660', '512690', '515790',
    # AI/芯片/科技
    '515070', '512480', '159995', '516160',
    # 医药/消费
    '512010', '159928',
    # 新能源/制造
    '515170', '561160', '562500',
    # 金融/红利
    '512800', '512670',
    # 黄金对冲
    '518880',
    # 其他
    '512200', '512880', '513180', '159790', '159550'
]

# 特征列
FEATURE_COLS = [
    'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d',
    'volatility_5d', 'volatility_20d',
    'rsi_14', 'rsi_6',
    'ma_ratio_5_20', 'ma_ratio_20_60',
    'volume_ratio_5_20',
    'high_low_ratio',
    'close_to_high_20', 'close_to_low_20'
]


# ============================================================
# Layer 1: 特征工程 + LightGBM择时
# ============================================================
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征"""
    features = pd.DataFrame(index=df.index)
    
    close = df['close']
    volume = df.get('volume', pd.Series(0, index=df.index))
    
    # 收益率
    features['returns_1d'] = close.pct_change(1)
    features['returns_5d'] = close.pct_change(5)
    features['returns_10d'] = close.pct_change(10)
    features['returns_20d'] = close.pct_change(20)
    
    # 波动率
    features['volatility_5d'] = close.pct_change().rolling(5).std()
    features['volatility_20d'] = close.pct_change().rolling(20).std()
    
    # RSI
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
    
    # 均线比
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    features['ma_ratio_5_20'] = ma5 / ma20 - 1
    features['ma_ratio_20_60'] = ma20 / ma60 - 1
    
    # 成交量比
    if volume.sum() > 0:
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        features['volume_ratio_5_20'] = vol_ma5 / (vol_ma20 + 1e-8)
    else:
        features['volume_ratio_5_20'] = 1.0
    
    # 高低比
    features['high_low_ratio'] = df['high'] / df['low'] - 1
    
    # 相对位置
    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()
    features['close_to_high_20'] = (close - low_20) / (high_20 - low_20 + 1e-8)
    features['close_to_low_20'] = (high_20 - close) / (high_20 - low_20 + 1e-8)
    
    return features


def create_labels(prices: pd.Series, forward_days: int = 5, threshold: float = 0.02) -> pd.Series:
    """创建标签：未来N天涨幅超过阈值为1，否则为0"""
    future_return = prices.shift(-forward_days) / prices - 1
    labels = (future_return > threshold).astype(int)
    return labels


def train_timing_model(
    features: pd.DataFrame, 
    labels: pd.Series,
    train_ratio: float = 0.7
) -> tuple:
    """训练择时模型"""
    # 对齐数据
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx].dropna()
    y = labels.loc[X.index]
    
    # 移除无穷大
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    
    if len(X) < 100:
        return None, None
    
    # 训练/测试分割
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # LightGBM参数（保守设置，防过拟合）
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,           # 较少的叶子数
        'learning_rate': 0.05,      # 较低学习率
        'feature_fraction': 0.7,    # 特征采样
        'bagging_fraction': 0.8,    # 数据采样
        'bagging_freq': 5,
        'min_child_samples': 50,    # 最小样本数
        'reg_alpha': 0.1,           # L1正则
        'reg_lambda': 0.1,          # L2正则
        'verbose': -1,
        'n_jobs': -1
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
    
    # 评估
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    
    return model, acc


# ============================================================
# Layer 2-4: 多因子信号
# ============================================================
def calculate_multi_factor_signals(
    all_prices: dict,
    gold_df: pd.DataFrame,
    etf_symbol: str
) -> pd.DataFrame:
    """计算多因子信号"""
    
    # Layer 2: 板块轮动
    rotation = SectorRotation()
    rotation_scores = rotation.calculate_rotation_score(all_prices)
    
    # Layer 4: 情绪/政策/地缘政治
    multi_factor = MultiFactorEngine()
    
    if etf_symbol in all_prices:
        etf_df = all_prices[etf_symbol]
        factors = multi_factor.calculate_all(etf_df, gold_df=gold_df)
    else:
        factors = pd.DataFrame()
    
    return rotation_scores, factors


# ============================================================
# 信号融合
# ============================================================
def fuse_signals(
    timing_signal: pd.Series,
    rotation_score: pd.Series,
    multi_factor: pd.DataFrame,
    weights: dict = None
) -> pd.Series:
    """
    融合5层信号
    
    权重分配：
    - Layer 1 择时信号: 40%
    - Layer 2 轮动信号: 25%
    - Layer 3 黄金对冲: 10% (在multi_factor中)
    - Layer 4 多因子: 25%
    """
    if weights is None:
        weights = {
            'timing': 0.40,
            'rotation': 0.25,
            'factor': 0.35
        }
    
    # 对齐所有信号
    common_idx = timing_signal.index
    if len(rotation_score) > 0:
        common_idx = common_idx.intersection(rotation_score.index)
    if len(multi_factor) > 0:
        common_idx = common_idx.intersection(multi_factor.index)
    
    # 归一化到 [-1, 1]
    timing_norm = timing_signal.reindex(common_idx).fillna(0)
    timing_norm = timing_norm * 2 - 1  # [0,1] -> [-1,1]
    
    rotation_norm = rotation_score.reindex(common_idx).fillna(0)
    
    if 'combined' in multi_factor.columns:
        factor_norm = multi_factor['combined'].reindex(common_idx).fillna(0)
    else:
        factor_norm = pd.Series(0, index=common_idx)
    
    # 加权融合
    fused = (
        timing_norm * weights['timing'] +
        rotation_norm * weights['rotation'] +
        factor_norm * weights['factor']
    )
    
    return fused.clip(-1, 1)


# ============================================================
# 回测引擎
# ============================================================
def backtest_v2(
    all_prices: dict,
    gold_df: pd.DataFrame,
    initial_capital: float = 50000,
    rebalance_freq: str = 'monthly'  # 改为月度调仓
) -> dict:
    """
    QuanTrade 2.0 回测 (改进版)
    
    策略逻辑：
    1. 月度调仓（减少交易成本）
    2. 基于动量+趋势选择ETF
    3. 结合多因子信号过滤
    4. 持仓1只ETF（小资金集中）
    """
    print("\n" + "=" * 60)
    print("QuanTrade 2.0 回测 (改进版)")
    print("=" * 60)
    
    # 准备数据
    print("\n[1/5] 准备数据...")
    
    # 选择有足够数据的ETF（降低阈值到100天）
    valid_symbols = [s for s in ETF_POOL if s in all_prices and len(all_prices[s]) >= 100]
    print(f"  有效ETF: {len(valid_symbols)} 只")
    
    # 获取公共日期范围
    all_dates = set()
    for s in valid_symbols:
        all_dates.update(all_prices[s].index)
    all_dates = sorted(all_dates)
    
    # 从第60天开始（需要历史数据计算特征）
    start_idx = 60
    backtest_dates = all_dates[start_idx:]
    
    # 计算轮动得分
    print("\n[2/5] 计算轮动得分...")
    rotation = SectorRotation()
    rotation_scores = rotation.calculate_rotation_score(all_prices)
    
    # 计算多因子信号
    print("\n[3/5] 计算多因子信号...")
    multi_factor_engine = MultiFactorEngine()
    
    # 按月分组
    backtest_df = pd.DataFrame({'date': backtest_dates})
    backtest_df['month'] = backtest_df['date'].dt.month
    backtest_df['year'] = backtest_df['date'].dt.year
    months = backtest_df.groupby(['year', 'month'])['date'].first().values
    
    # 回测
    print("\n[4/5] 执行回测...")
    capital = initial_capital
    holdings = {}  # {symbol: shares}
    nav_history = []
    trade_count = 0
    
    for month_start in months:
        month_start = pd.Timestamp(month_start)
        
        # 找到该月的交易日
        month_dates = [d for d in backtest_dates 
                       if pd.Timestamp(d).year == month_start.year 
                       and pd.Timestamp(d).month == month_start.month]
        
        if not month_dates:
            continue
        
        target_date = month_dates[0]
        
        # 计算每只ETF的综合得分
        scores = {}
        for symbol in valid_symbols:
            df = all_prices[symbol]
            
            # 确保有足够的历史数据
            if target_date not in df.index:
                continue
            
            hist = df.loc[:target_date]
            if len(hist) < 20:
                continue
            
            close = hist['close']
            
            # Layer 1: 动量得分（简化版，不用ML）
            mom_5d = close.pct_change(5).iloc[-1] if len(close) > 5 else 0
            mom_20d = close.pct_change(20).iloc[-1] if len(close) > 20 else 0
            
            # 趋势得分
            ma5 = close.rolling(5).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            trend = 1 if ma5 > ma20 else -1
            
            # Layer 2: 轮动得分
            rot_score = 0
            if target_date in rotation_scores.index and symbol in rotation_scores.columns:
                rot_score = rotation_scores.loc[target_date, symbol]
            
            # Layer 4: 多因子
            factor_score = 0
            factors = multi_factor_engine.calculate_all(hist, gold_df=gold_df)
            if len(factors) > 0:
                factor_score = factors['combined'].iloc[-1]
            
            # 综合得分
            score = mom_5d * 0.2 + mom_20d * 0.2 + trend * 0.1 + rot_score * 0.25 + factor_score * 0.25
            scores[symbol] = score
        
        if not scores:
            # 计算NAV
            for day in month_dates:
                portfolio_value = capital
                for h_symbol, h_shares in holdings.items():
                    price = all_prices[h_symbol]['close'].get(day, 0)
                    portfolio_value += h_shares * price
                nav_history.append({'date': day, 'nav': portfolio_value})
            continue
        
        # 选择最强ETF
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_symbol = sorted_scores[0][0]
        best_score = sorted_scores[0][1]
        
        # 决策：只在看多时持仓
        if best_score > 0:
            # 卖出当前持仓（如果换仓）
            for h_symbol, h_shares in list(holdings.items()):
                if h_symbol != best_symbol:
                    sell_price = all_prices[h_symbol]['close'].get(target_date, 0)
                    if sell_price > 0:
                        capital += h_shares * sell_price * (1 - COST_RATE)
                        trade_count += 1
                    del holdings[h_symbol]
            
            # 买入最佳ETF
            if best_symbol not in holdings:
                buy_price = all_prices[best_symbol]['close'].get(target_date, 0)
                if buy_price > 0:
                    shares = int(capital / buy_price / 100) * 100
                    if shares > 0:
                        cost = shares * buy_price * (1 + COST_RATE)
                        capital -= cost
                        holdings[best_symbol] = shares
                        trade_count += 1
        else:
            # 清仓
            for h_symbol, h_shares in list(holdings.items()):
                sell_price = all_prices[h_symbol]['close'].get(target_date, 0)
                if sell_price > 0:
                    capital += h_shares * sell_price * (1 - COST_RATE)
                    trade_count += 1
            holdings = {}
        
        # 计算每日NAV
        for day in month_dates:
            portfolio_value = capital
            for h_symbol, h_shares in holdings.items():
                price = all_prices[h_symbol]['close'].get(day, 0)
                portfolio_value += h_shares * price
            nav_history.append({'date': day, 'nav': portfolio_value})
    
    # 计算基准
    print("\n[5/5] 计算基准...")
    
    # 沪深300作为基准
    benchmark = all_prices.get('510300')
    if benchmark is not None:
        bench_start = benchmark['close'].get(backtest_dates[0], 1)
        bench_nav = [{
            'date': backtest_dates[0],
            'nav': initial_capital
        }]
        for d in backtest_dates[1:]:
            price = benchmark['close'].get(d, bench_nav[-1]['nav'] / initial_capital * bench_start)
            bench_nav.append({
                'date': d,
                'nav': initial_capital * price / bench_start
            })
    else:
        bench_nav = nav_history.copy()
    
    # 整理结果
    nav_df = pd.DataFrame(nav_history)
    nav_df['returns'] = nav_df['nav'].pct_change()
    
    bench_df = pd.DataFrame(bench_nav)
    bench_df['returns'] = bench_df['nav'].pct_change()
    
    results = {
        'nav': nav_df,
        'benchmark': bench_df,
        'final_nav': nav_df['nav'].iloc[-1],
        'total_return': nav_df['nav'].iloc[-1] / initial_capital - 1,
        'benchmark_return': bench_df['nav'].iloc[-1] / initial_capital - 1,
        'trade_count': trade_count,
        'initial_capital': initial_capital
    }
    
    return results


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 70)
    print("           QuanTrade 2.0 - 五层架构ETF量化交易系统")
    print("=" * 70)
    
    # 加载数据
    print("\n[加载数据]")
    conn = sqlite3.connect(DB_PATH)
    
    # 加载所有ETF
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
    
    # 加载黄金
    gold_df = pd.read_sql("SELECT * FROM gold_daily_prices ORDER BY date", conn)
    gold_df['date'] = pd.to_datetime(gold_df['date'])
    gold_df = gold_df.set_index('date')
    
    conn.close()
    
    print(f"  ETF数据: {len(all_prices)} 只")
    print(f"  黄金数据: {len(gold_df)} 条")
    
    # 运行回测
    results = backtest_v2(
        all_prices=all_prices,
        gold_df=gold_df,
        initial_capital=50000,
        rebalance_freq='monthly'
    )
    
    # 打印结果
    print("\n" + "=" * 70)
    print("                    回测结果")
    print("=" * 70)
    print(f"  初始资金: {results['initial_capital']:,.0f} 元")
    print(f"  最终净值: {results['final_nav']:,.0f} 元")
    print(f"  策略收益: {results['total_return']:.2%}")
    print(f"  基准收益: {results['benchmark_return']:.2%}")
    print(f"  超额收益: {results['total_return'] - results['benchmark_return']:.2%}")
    print(f"  交易次数: {results['trade_count']}")
    
    # 使用独立评价Agent评估
    print("\n" + "=" * 70)
    print("              独立评价Agent评估")
    print("=" * 70)
    
    agent = EvaluationAgent(verbose=True)
    report = agent.full_evaluation(
        strategy_returns=results['nav']['returns'].dropna(),
        benchmark_returns=results['benchmark']['returns'].dropna()
    )
    agent.print_report(report)
    
    return results, report


if __name__ == "__main__":
    results, report = main()
