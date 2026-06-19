"""
QuanTrade 2.0.2 - 完整重构版
==========================
核心改动：
1. 基准 = 24只ETF等权买入持有
2. 热点赛道选股增强（60/120日动量 + 板块相对强弱）
3. 融入用户持仓策略（大跌建仓/分批/情绪高点减仓）
4. 黄金ETF对冲优化
5. 三因子分析（市场情绪/国际政治/国内政策）
6. 情绪泡沫 vs 业绩支撑识别

用户策略量化：
- 大跌建仓：大盘跌3% + ETF跌10% → 试探建仓10%
- 分批建仓：每跌5%加仓20%，反弹确认加满
- 上升潮加仓：3天连涨/突破均线 → 加仓
- 目标收益减仓：涨20% → 减仓30%
- 情绪高点减仓：情绪>0.8 → 减仓50%
- 区分泡沫：情绪ETF（AI/机器人）多减，业绩ETF（消费/医药）少减
"""

import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# ============================================================
# 配置
# ============================================================
DB_PATH = "QuanTrade/quant_system/data/quant.db"
COST_RATE = 0.0036
INITIAL_CAPITAL = 50000

# 23只ETF + 1只基准（沪深300用于特征计算，不作为持仓）
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
]

# 用于特征计算的基准指数
BENCHMARK_SYMBOL = '510300'

# 黄金ETF
GOLD_SYMBOL = '518880'

# ETF分类（情绪驱动 vs 业绩驱动）
EMOTION_ETFS = ['562500', '515070', '159995', '159550', '516510', '512660', '512670', '515960', '588000', '159915', '513180', '512480']
FUNDAMENTAL_ETFS = ['512010', '159928', '512690', '515170', '515790', '516160', '561160', '159790', '512880', '512800', '512200']

# 用户策略参数
MARKET_CRASH_THRESHOLD = -0.03      # 大盘跌3%算大跌
ETF_CRASH_THRESHOLD = -0.10         # ETF跌10%算大跌
PROBE_POSITION = 0.10               # 试探建仓10%
ADD_POSITION_PER_DROP = 0.20        # 每跌5%加仓20%
UPSURGE_ADD_POSITION = 0.30         # 上升潮加仓30%
PROFIT_TARGET = 0.20                # 目标收益20%
PROFIT_REDUCE = 0.30               # 达到目标减仓30%
EMOTION_REDUCE = 0.50               # 情绪高点减仓50%
EMOTION_THRESHOLD = 0.80            # 情绪高点阈值

# ============================================================
# 数据加载
# ============================================================
def load_all_data():
    """加载所有ETF和黄金数据"""
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
    
    # 黄金数据
    gold_df = pd.read_sql("SELECT * FROM gold_daily_prices ORDER BY date", conn)
    gold_df['date'] = pd.to_datetime(gold_df['date'])
    gold_df = gold_df.set_index('date')
    
    conn.close()
    return all_prices, gold_df


# ============================================================
# Layer 1: 特征工程（增强版）
# ============================================================
def calculate_enhanced_features(df: pd.DataFrame, benchmark_df: pd.DataFrame = None) -> pd.DataFrame:
    """计算增强版技术指标特征"""
    features = pd.DataFrame(index=df.index)
    close = df['close']
    volume = df.get('volume', pd.Series(0, index=df.index))
    
    # 1. 收益率（4个）
    features['returns_1d'] = close.pct_change(1)
    features['returns_5d'] = close.pct_change(5)
    features['returns_10d'] = close.pct_change(10)
    features['returns_20d'] = close.pct_change(20)
    
    # 2. 中长期动量（新增！）
    features['returns_60d'] = close.pct_change(60)
    features['returns_120d'] = close.pct_change(120)
    
    # 3. 波动率（2个）
    features['volatility_5d'] = close.pct_change().rolling(5).std()
    features['volatility_20d'] = close.pct_change().rolling(20).std()
    
    # 4. RSI（2个）
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain_14 = gain.rolling(14).mean()
    avg_loss_14 = loss.rolling(14).mean()
    rs_14 = avg_gain_14 / (avg_loss_14 + 1e-8)
    features['rsi_14'] = 100 - (100 / (1 + rs_14))
    
    # 5. 均线比（2个）
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    features['ma_ratio_5_20'] = ma5 / ma20 - 1
    features['ma_ratio_20_60'] = ma20 / ma60 - 1
    
    # 6. 板块相对强弱（新增！vs 沪深300）
    if benchmark_df is not None and len(benchmark_df) > 0:
        bench_close = benchmark_df['close'].reindex(df.index, method='ffill')
        relative = close / bench_close
        features['relative_20d'] = relative.pct_change(20)
        features['relative_60d'] = relative.pct_change(60)
    
    # 7. 成交量比
    if volume.sum() > 0:
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        features['volume_ratio_5_20'] = vol_ma5 / (vol_ma20 + 1e-8)
        features['volume_change_5d'] = volume.pct_change(5)
    
    # 8. 趋势强度
    features['trend_5_20'] = (ma5 > ma20).astype(int)
    features['trend_20_60'] = (ma20 > ma60).astype(int)
    
    # 9. MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features['macd_hist'] = macd - signal
    
    # 10. 布林带
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features['bb_position'] = (close - bb_ma) / (2 * bb_std + 1e-8)
    
    # 11. 高低位置
    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()
    features['close_to_high_20'] = (close - low_20) / (high_20 - low_20 + 1e-8)
    
    return features


def create_labels(prices: pd.Series, forward_days: int = 20, threshold: float = 0.05) -> pd.Series:
    """标签：未来N天涨超阈值=1，否则=0"""
    future_return = prices.shift(-forward_days) / prices - 1
    labels = (future_return > threshold).astype(int)
    return labels


# ============================================================
# Layer 1: ML模型训练（自动选最佳）
# ============================================================
def train_best_model(features: pd.DataFrame, labels: pd.Series, train_ratio: float = 0.7):
    """训练最佳ML模型"""
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx].dropna()
    y = labels.loc[X.index]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    
    if len(X) < 100:
        return None, 0, "none"
    
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    best_model = None
    best_acc = 0
    best_type = ""
    
    # LightGBM
    try:
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
            'num_leaves': 15, 'learning_rate': 0.05, 'feature_fraction': 0.7,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1,
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        model = lgb.train(params, train_data, num_boost_round=200,
                          valid_sets=[test_data],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_model, best_acc, best_type = model, acc, 'lightgbm'
    except Exception as e:
        pass
    
    # RandomForest
    try:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=50, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_model, best_acc, best_type = model, acc, 'random_forest'
    except:
        pass
    
    # GradientBoosting
    try:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, min_samples_split=50, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_model, best_acc, best_type = model, acc, 'gradient_boosting'
    except:
        pass
    
    return best_model, best_acc, best_type


# ============================================================
# Layer 2: 热点赛道选股（增强版）
# ============================================================
def calculate_sector_score_enhanced(etf_df: pd.DataFrame, benchmark_df: pd.DataFrame, all_etfs: dict) -> float:
    """
    增强版热点赛道得分
    权重：中期动量30% + 板块相对强弱20% + 趋势强度15% + 短期动量20% + 资金流向15%
    """
    if len(etf_df) < 120:
        return 0
    
    close = etf_df['close']
    volume = etf_df.get('volume', pd.Series(0, index=etf_df.index))
    
    # 1. 中期动量（30%）- 牛市关键因子
    mom_60d = close.pct_change(60).iloc[-1]
    mom_120d = close.pct_change(120).iloc[-1]
    momentum_score = np.clip(mom_60d / 0.3, -1, 1) * 0.6 + np.clip(mom_120d / 0.5, -1, 1) * 0.4
    
    # 2. 板块相对强弱（20%）- vs 沪深300
    relative_score = 0
    if benchmark_df is not None and len(benchmark_df) > 0:
        bench_close = benchmark_df['close'].reindex(etf_df.index, method='ffill')
        relative_20d = (close.pct_change(20) - bench_close.pct_change(20)).iloc[-1]
        relative_60d = (close.pct_change(60) - bench_close.pct_change(60)).iloc[-1]
        relative_score = np.clip(relative_20d / 0.15, -1, 1) * 0.4 + np.clip(relative_60d / 0.3, -1, 1) * 0.6
    
    # 3. 趋势强度（15%）
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    trend_score = (
        (1 if ma5 > ma20 else 0) * 0.4 +
        (1 if ma20 > ma60 else 0) * 0.3 +
        (1 if close.iloc[-1] > ma5 else 0) * 0.3
    )
    
    # 4. 短期动量（20%）
    mom_5d = close.pct_change(5).iloc[-1]
    mom_20d = close.pct_change(20).iloc[-1]
    short_mom_score = np.clip(mom_5d / 0.1, -1, 1) * 0.4 + np.clip(mom_20d / 0.2, -1, 1) * 0.6
    
    # 5. 资金流向（15%）
    fund_score = 0
    if volume.sum() > 0:
        vol_change_5d = volume.pct_change(5).iloc[-1]
        price_vol_corr = close.rolling(20).corr(volume).iloc[-1]
        fund_score = np.clip(vol_change_5d / 0.5, -1, 1) * 0.5 + np.clip(price_vol_corr, -1, 1) * 0.5
    
    # 综合得分
    sector_score = (
        momentum_score * 0.30 +
        relative_score * 0.20 +
        trend_score * 0.15 +
        short_mom_score * 0.20 +
        fund_score * 0.15
    )
    
    return np.clip(sector_score, -1, 1)


# ============================================================
# Layer 3: 黄金ETF对冲
# ============================================================
def calculate_gold_allocation(market_crash: bool, market_panic: bool, market_bull: bool) -> dict:
    """黄金仓位分配"""
    if market_panic:
        return {'gold': 0.30, 'stock': 0.00, 'cash': 0.70}
    elif market_crash:
        return {'gold': 0.20, 'stock': 0.50, 'cash': 0.30}
    elif market_bull:
        return {'gold': 0.00, 'stock': 0.90, 'cash': 0.10}
    else:
        return {'gold': 0.10, 'stock': 0.70, 'cash': 0.20}


# ============================================================
# Layer 4: 三因子分析（市场代理版）
# ============================================================
def calculate_market_sentiment(benchmark_df: pd.DataFrame) -> float:
    """
    市场情绪因子 [-1, 1]
    正值=乐观，负值=恐慌
    """
    if benchmark_df is None or len(benchmark_df) < 60:
        return 0
    
    close = benchmark_df['close']
    returns = close.pct_change()
    
    # 1. 波动率情绪：波动率上升=恐慌
    vol_20 = returns.rolling(20).std().iloc[-1]
    vol_ma60 = vol_20.rolling(60).mean().iloc[-1] if hasattr(vol_20, 'rolling') else vol_20
    vol_score = -1 if vol_20 > vol_ma60 * 1.5 else (1 if vol_20 < vol_ma60 * 0.7 else 0)
    
    # 2. 动量情绪：短期vs长期
    mom_5 = close.pct_change(5).iloc[-1]
    mom_60 = close.pct_change(60).iloc[-1]
    mom_score = 1 if mom_5 > 0 and mom_60 > 0 else (-1 if mom_5 < 0 and mom_60 < 0 else 0)
    
    # 3. 成交量情绪
    volume = benchmark_df.get('volume', pd.Series(0, index=benchmark_df.index))
    if volume.sum() > 0:
        vol_change = volume.pct_change(5).iloc[-1]
        price_change = close.pct_change(5).iloc[-1]
        vol_price_score = 1 if vol_change > 0.3 and price_change > 0 else (-1 if vol_change > 0.3 and price_change < 0 else 0)
    else:
        vol_price_score = 0
    
    return np.clip((vol_score + mom_score + vol_price_score) / 3, -1, 1)


def calculate_geopolitical_factor(gold_df: pd.DataFrame) -> float:
    """
    地缘政治风险因子 [-1, 1]
    负值=风险高，正值=风险低
    """
    if gold_df is None or len(gold_df) < 60:
        return 0
    
    close = gold_df['close']
    mom_5 = close.pct_change(5).iloc[-1]
    mom_20 = close.pct_change(20).iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    
    # 黄金大涨 = 避险情绪上升 = 地缘风险上升 = 负值
    if mom_5 > 0.02 and mom_20 > 0.05 and close.iloc[-1] > ma20:
        return -0.8
    elif mom_5 > 0.02:
        return -0.4
    elif mom_5 < -0.02 and mom_20 < -0.05:
        return 0.8
    elif mom_5 < -0.02:
        return 0.4
    return 0


def calculate_policy_factor(benchmark_df: pd.DataFrame) -> float:
    """
    政策因子 [-1, 1]
    正值=政策利好，负值=政策收紧
    """
    if benchmark_df is None or len(benchmark_df) < 120:
        return 0
    
    close = benchmark_df['close']
    
    # 1. 趋势强度：长期趋势方向
    ma120 = close.rolling(120).mean().iloc[-1]
    trend = (close.iloc[-1] - ma120) / ma120
    trend_score = np.clip(trend * 5, -1, 1)
    
    # 2. 市场宽度：用价格相对位置代理
    high_60 = close.rolling(60).max()
    low_60 = close.rolling(60).min()
    position = (close.iloc[-1] - low_60.iloc[-1]) / (high_60.iloc[-1] - low_60.iloc[-1] + 1e-8)
    breadth_score = 1 if position > 0.8 else (-1 if position < 0.2 else 0)
    
    # 3. 趋势稳定性
    returns = close.pct_change()
    autocorr = returns.rolling(60).apply(lambda x: x.autocorr(lag=1) if len(x) > 10 else 0, raw=False).iloc[-1]
    stability_score = 0.5 if autocorr > 0.1 else (-0.5 if autocorr < -0.1 else 0)
    
    return np.clip((trend_score + breadth_score + stability_score) / 3, -1, 1)


def calculate_three_factors(benchmark_df: pd.DataFrame, gold_df: pd.DataFrame) -> dict:
    """三因子综合得分"""
    sentiment = calculate_market_sentiment(benchmark_df)
    geopolitical = calculate_geopolitical_factor(gold_df)
    policy = calculate_policy_factor(benchmark_df)
    
    combined = sentiment * 0.4 + geopolitical * 0.3 + policy * 0.3
    
    return {
        'sentiment': sentiment,
        'geopolitical': geopolitical,
        'policy': policy,
        'combined': np.clip(combined, -1, 1)
    }


# ============================================================
# Layer 5: 情绪泡沫 vs 业绩支撑识别
# ============================================================
def is_emotion_etf(symbol: str) -> bool:
    """判断是否为情绪驱动型ETF"""
    return symbol in EMOTION_ETFS


def detect_bubble(etf_df: pd.DataFrame) -> dict:
    """
    检测ETF是否为泡沫状态
    Returns: {'is_bubble': bool, 'bubble_score': float, 'reason': str}
    """
    if len(etf_df) < 60:
        return {'is_bubble': False, 'bubble_score': 0, 'reason': '数据不足'}
    
    close = etf_df['close']
    volume = etf_df.get('volume', pd.Series(0, index=etf_df.index))
    
    # 1. 短期涨幅 > 50%（60天内）
    return_60d = close.iloc[-1] / close.iloc[-60] - 1
    
    # 2. 换手率 > 均值3倍（用成交量代理）
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    vol_current = volume.iloc[-1]
    turnover_spike = vol_current / (vol_ma20 + 1e-8) > 3
    
    # 3. RSI > 80（超买）
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    
    # 4. 波动率急剧放大
    vol_5d = close.pct_change().rolling(5).std().iloc[-1]
    vol_60d = close.pct_change().rolling(60).std().iloc[-1]
    vol_spike = vol_5d / (vol_60d + 1e-8) > 2
    
    # 泡沫评分
    bubble_score = 0
    reasons = []
    
    if return_60d > 0.5:
        bubble_score += 0.3
        reasons.append(f'60日涨幅{return_60d:.1%}')
    if turnover_spike:
        bubble_score += 0.2
        reasons.append('成交量暴增3倍+')
    if rsi > 80:
        bubble_score += 0.3
        reasons.append(f'RSI超买{rsi:.1f}')
    if vol_spike:
        bubble_score += 0.2
        reasons.append('波动率急剧放大')
    
    is_bubble = bubble_score >= 0.5
    reason = ' + '.join(reasons) if reasons else '无泡沫信号'
    
    return {
        'is_bubble': is_bubble,
        'bubble_score': bubble_score,
        'reason': reason,
        'rsi': rsi,
        'return_60d': return_60d
    }


def detect_fundamental_strength(etf_df: pd.DataFrame) -> dict:
    """
    检测业绩支撑强度
    Returns: {'is_strong': bool, 'strength_score': float, 'reason': str}
    """
    if len(etf_df) < 60:
        return {'is_strong': False, 'strength_score': 0, 'reason': '数据不足'}
    
    close = etf_df['close']
    
    # 1. 趋势稳定（60日均线向上，波动率适中）
    ma60 = close.rolling(60).mean()
    trend_up = ma60.iloc[-1] > ma60.iloc[-20]
    
    # 2. RSI 在 40-70 区间（健康）
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    rsi_healthy = 40 < rsi < 70
    
    # 3. 波动率适中
    vol_20d = close.pct_change().rolling(20).std().iloc[-1]
    vol_healthy = 0.01 < vol_20d < 0.03
    
    # 4. 持续上涨但非暴涨（60日涨10-30%）
    return_60d = close.iloc[-1] / close.iloc[-60] - 1
    return_healthy = 0.10 < return_60d < 0.30
    
    strength_score = 0
    reasons = []
    
    if trend_up:
        strength_score += 0.3
        reasons.append('60日均线向上')
    if rsi_healthy:
        strength_score += 0.2
        reasons.append(f'RSI健康{rsi:.1f}')
    if vol_healthy:
        strength_score += 0.2
        reasons.append('波动率适中')
    if return_healthy:
        strength_score += 0.3
        reasons.append(f'60日涨{return_60d:.1%}')
    
    is_strong = strength_score >= 0.6
    reason = ' + '.join(reasons) if reasons else '业绩支撑信号弱'
    
    return {
        'is_strong': is_strong,
        'strength_score': strength_score,
        'reason': reason,
        'rsi': rsi,
        'return_60d': return_60d
    }


# ============================================================
# Layer 6: 用户持仓策略融入
# ============================================================
class UserPositionStrategy:
    """用户持仓策略管理器"""
    
    def __init__(self, initial_capital=50000):
        self.capital = initial_capital
        self.holdings = {}  # {symbol: {'shares': int, 'cost_basis': float, 'entry_dates': list, 'profit_target': float}}
        self.trade_count = 0
    
    def check_crash_entry(self, market_return_1d: float, etf_return_1d: float) -> bool:
        """检查是否满足大跌建仓条件"""
        return market_return_1d < MARKET_CRASH_THRESHOLD and etf_return_1d < ETF_CRASH_THRESHOLD
    
    def check_upsurge(self, etf_df: pd.DataFrame) -> bool:
        """检查是否进入上升潮（3天连涨 或 突破20日均线）"""
        if len(etf_df) < 3:
            return False
        close = etf_df['close']
        returns_3d = close.pct_change(3).iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        return returns_3d > 0.05 or close.iloc[-1] > ma20 * 1.02
    
    def check_profit_target(self, symbol: str, current_price: float) -> bool:
        """检查是否达到目标收益"""
        if symbol not in self.holdings:
            return False
        cost = self.holdings[symbol]['cost_basis']
        return (current_price - cost) / cost > PROFIT_TARGET
    
    def check_emotion_peak(self, factors: dict) -> bool:
        """检查是否情绪高点"""
        return factors['sentiment'] > EMOTION_THRESHOLD
    
    def calculate_reduce_ratio(self, symbol: str, bubble_info: dict, emotion_peak: bool, profit_target: bool) -> float:
        """
        计算减仓比例
        - 情绪高点 + 泡沫ETF：减仓50%（甚至更多）
        - 情绪高点 + 业绩ETF：减仓30%
        - 达到目标收益：减仓30%
        - 叠加：最大减仓80%
        """
        reduce = 0.0
        
        if profit_target:
            reduce += PROFIT_REDUCE
        
        if emotion_peak:
            if is_emotion_etf(symbol) and bubble_info['is_bubble']:
                reduce += EMOTION_REDUCE * 1.2  # 泡沫ETF多减
                print(f"    [泡沫预警] {symbol} 情绪高点+泡沫，大幅减仓")
            elif is_emotion_etf(symbol):
                reduce += EMOTION_REDUCE
            else:
                reduce += EMOTION_REDUCE * 0.6  # 业绩ETF少减
        
        return min(reduce, 0.80)
    
    def buy(self, symbol: str, price: float, ratio: float, date: str):
        """买入"""
        amount = self.capital * ratio
        shares = int(amount / price / 100) * 100
        if shares > 0:
            cost = shares * price * (1 + COST_RATE)
            self.capital -= cost
            if symbol in self.holdings:
                old_shares = self.holdings[symbol]['shares']
                old_cost = self.holdings[symbol]['cost_basis'] * old_shares
                new_cost_basis = (old_cost + shares * price) / (old_shares + shares)
                self.holdings[symbol]['shares'] += shares
                self.holdings[symbol]['cost_basis'] = new_cost_basis
                self.holdings[symbol]['entry_dates'].append(date)
            else:
                self.holdings[symbol] = {
                    'shares': shares,
                    'cost_basis': price,
                    'entry_dates': [date],
                }
            self.trade_count += 1
            return shares
        return 0
    
    def sell(self, symbol: str, price: float, ratio: float, date: str):
        """卖出（按比例）"""
        if symbol not in self.holdings:
            return 0
        shares_to_sell = int(self.holdings[symbol]['shares'] * ratio / 100) * 100
        if shares_to_sell > 0:
            proceeds = shares_to_sell * price * (1 - COST_RATE)
            self.capital += proceeds
            self.holdings[symbol]['shares'] -= shares_to_sell
            if self.holdings[symbol]['shares'] <= 0:
                del self.holdings[symbol]
            self.trade_count += 1
            return shares_to_sell
        return 0
    
    def get_portfolio_value(self, all_prices: dict, date) -> float:
        """计算组合市值"""
        value = self.capital
        for symbol, info in self.holdings.items():
            if symbol in all_prices and date in all_prices[symbol].index:
                price = all_prices[symbol]['close'].get(date, 0)
                value += info['shares'] * price
        return value


# ============================================================
# 等权基准计算
# ============================================================
def calculate_equal_weight_benchmark(all_prices: dict, dates: list) -> pd.DataFrame:
    """计算等权买入持有基准"""
    start_date = dates[0]
    valid_symbols = [s for s in ETF_POOL if s in all_prices and start_date in all_prices[s].index]
    
    if not valid_symbols:
        return pd.DataFrame({'date': dates, 'nav': [INITIAL_CAPITAL] * len(dates)})
    
    per_etf_cap = INITIAL_CAPITAL / len(valid_symbols)
    holdings = {}
    capital = INITIAL_CAPITAL
    
    # 第一天等权买入
    for symbol in valid_symbols:
        buy_price = all_prices[symbol]['close'].get(start_date, 0)
        if buy_price > 0:
            shares = int(per_etf_cap / buy_price / 100) * 100
            if shares > 0:
                capital -= shares * buy_price * (1 + COST_RATE)
                holdings[symbol] = shares
    
    nav_history = []
    for date in dates:
        nav = capital
        for symbol, shares in holdings.items():
            if symbol in all_prices and date in all_prices[symbol].index:
                price = all_prices[symbol]['close'].get(date, 0)
                nav += shares * price
        nav_history.append({'date': date, 'nav': nav})
    
    return pd.DataFrame(nav_history)


# ============================================================
# 主回测程序
# ============================================================
def main():
    print("=" * 70)
    print("    QuanTrade 2.0.2 - 用户策略重构版")
    print("=" * 70)
    
    # 加载数据
    print("\n[加载数据]")
    all_prices, gold_df = load_all_data()
    benchmark_df = all_prices.get(BENCHMARK_SYMBOL)
    
    print(f"  ETF数据: {len([s for s in ETF_POOL if s in all_prices])} / {len(ETF_POOL)} 只")
    print(f"  基准数据: {BENCHMARK_SYMBOL}, {len(benchmark_df)} 条")
    print(f"  黄金数据: {len(gold_df)} 条")
    
    # 获取公共日期
    all_dates = set()
    for s in ETF_POOL:
        if s in all_prices:
            all_dates.update(all_prices[s].index)
    all_dates = sorted(all_dates)
    
    if len(all_dates) < 200:
        print("错误: 数据不足")
        return None
    
    # 分割训练/测试期
    train_end = pd.Timestamp('2023-12-31')
    test_dates = [d for d in all_dates if d > train_end]
    
    print(f"  测试期: {test_dates[0].date()} ~ {test_dates[-1].date()}, 共{len(test_dates)}天")
    
    # ============================================================
    # 训练ML模型
    # ============================================================
    print("\n" + "=" * 70)
    print("[Layer 1] ML模型训练")
    print("=" * 70)
    
    models = {}
    for symbol in ETF_POOL:
        if symbol not in all_prices or len(all_prices[symbol]) < 200:
            continue
        
        df = all_prices[symbol]
        train_df = df[df.index <= train_end]
        if len(train_df) < 100:
            continue
        
        features = calculate_enhanced_features(train_df, benchmark_df)
        labels = create_labels(train_df['close'], forward_days=20, threshold=0.05)
        
        model, acc, model_type = train_best_model(features, labels)
        if model is not None:
            models[symbol] = {'model': model, 'type': model_type, 'acc': acc}
            print(f"    {symbol}: {model_type}, 准确率={acc:.2%}")
    
    print(f"  训练完成: {len(models)} 个模型")
    
    # ============================================================
    # 等权基准
    # ============================================================
    print("\n" + "=" * 70)
    print("[基准] 等权买入持有")
    print("=" * 70)
    
    bench_nav_df = calculate_equal_weight_benchmark(all_prices, test_dates)
    bench_total_return = bench_nav_df['nav'].iloc[-1] / INITIAL_CAPITAL - 1
    print(f"  等权基准总收益: {bench_total_return:.2%}")
    
    # ============================================================
    # 策略回测（用户策略 + 热点赛道 + 三因子）
    # ============================================================
    print("\n" + "=" * 70)
    print("[策略回测] 用户策略 + 热点赛道 + 三因子")
    print("=" * 70)
    
    strategy = UserPositionStrategy(INITIAL_CAPITAL)
    nav_history = []
    
    # 月度调仓（每月第一个交易日）
    rebalance_dates = []
    last_month = None
    for date in test_dates:
        if last_month != (date.year, date.month):
            rebalance_dates.append(date)
            last_month = (date.year, date.month)
    
    # 同时加入事件驱动（大跌/情绪高点）
    last_rebalance = None
    
    for i, date in enumerate(test_dates):
        date_ts = pd.Timestamp(date)
        
        # 获取当前市场状态
        market_return = 0
        if benchmark_df is not None and date in benchmark_df.index:
            if i > 0:
                prev_date = test_dates[i-1]
                if prev_date in benchmark_df.index:
                    market_return = benchmark_df['close'].get(date, 0) / benchmark_df['close'].get(prev_date, 1) - 1
        
        # 计算三因子
        benchmark_hist = benchmark_df.loc[:date] if benchmark_df is not None else None
        gold_hist = gold_df.loc[:date] if gold_df is not None else None
        factors = calculate_three_factors(benchmark_hist, gold_hist)
        
        # 判断是否调仓
        should_rebalance = False
        if last_rebalance is None:
            should_rebalance = True
        elif date in rebalance_dates:
            should_rebalance = True
        # 事件驱动：情绪高点，需要减仓
        elif factors['sentiment'] > EMOTION_THRESHOLD and strategy.holdings:
            should_rebalance = True
            print(f"  {date.date()} 情绪高点触发调仓 (情绪={factors['sentiment']:.2f})")
        # 事件驱动：大跌，需要建仓
        elif market_return < MARKET_CRASH_THRESHOLD and not strategy.holdings:
            should_rebalance = True
            print(f"  {date.date()} 大跌触发建仓 (沪指={market_return:.2%})")
        
        if should_rebalance:
            # 1. 计算所有ETF得分
            scores = {}
            for symbol in ETF_POOL:
                if symbol not in all_prices or date not in all_prices[symbol].index:
                    continue
                
                hist = all_prices[symbol].loc[:date]
                if len(hist) < 60:
                    continue
                
                # ML预测
                ml_score = 0.5
                if symbol in models:
                    features = calculate_enhanced_features(hist, benchmark_df)
                    if len(features) > 0:
                        X = features.iloc[[-1]].dropna()
                        if len(X) > 0:
                            try:
                                model_info = models[symbol]
                                if model_info['type'] == 'lightgbm':
                                    ml_score = model_info['model'].predict(X)[0]
                                else:
                                    ml_score = model_info['model'].predict_proba(X)[0][1]
                            except:
                                pass
                
                # 热点赛道得分
                sector_score = calculate_sector_score_enhanced(hist, benchmark_df, all_prices)
                
                # 三因子调整
                factor_adjust = factors['combined'] * 0.2  # 三因子影响±20%
                
                # 综合得分
                final_score = ml_score * 0.3 + (sector_score + 1) / 2 * 0.5 + (1 + factor_adjust) / 2 * 0.2
                scores[symbol] = final_score
            
            if scores:
                # 排序选前5-8只（增加持仓数量，避免错过强势赛道）
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_n = min(8, len(sorted_scores))
                top_etfs = sorted_scores[:top_n]
                
                # 不在前N的ETF，减仓到最小仓位
                target_symbols = [s[0] for s in top_etfs]
                for symbol in list(strategy.holdings.keys()):
                    if symbol not in target_symbols:
                        price = all_prices[symbol]['close'].get(date, 0)
                        if price > 0:
                            strategy.sell(symbol, price, 1.0, str(date))
                            print(f"    {date.date()} 卖出 {symbol} (不在前{top_n})")
                
                # 3. 检查减仓信号（目标收益/情绪高点）
                for symbol, info in list(strategy.holdings.items()):
                    price = all_prices[symbol]['close'].get(date, 0)
                    if price <= 0:
                        continue
                    
                    bubble_info = detect_bubble(all_prices[symbol].loc[:date])
                    profit_target = strategy.check_profit_target(symbol, price)
                    emotion_peak = strategy.check_emotion_peak(factors)
                    
                    reduce_ratio = strategy.calculate_reduce_ratio(symbol, bubble_info, emotion_peak, profit_target)
                    
                    if reduce_ratio > 0:
                        strategy.sell(symbol, price, reduce_ratio, str(date))
                        reason = []
                        if profit_target:
                            reason.append("目标收益")
                        if emotion_peak:
                            reason.append("情绪高点")
                        if bubble_info['is_bubble']:
                            reason.append(f"泡沫({bubble_info['reason']})")
                        print(f"    {date.date()} 减仓 {symbol} {reduce_ratio:.0%} ({', '.join(reason)})")
                
                # 4. 建仓/加仓（用户策略 + 热点赛道）
                for symbol, score in top_etfs:
                    price = all_prices[symbol]['close'].get(date, 0)
                    if price <= 0:
                        continue
                    
                    hist = all_prices[symbol].loc[:date]
                    if len(hist) < 2:
                        continue
                    
                    etf_return_1d = hist['close'].pct_change().iloc[-1]
                    
                    if symbol not in strategy.holdings:
                        # 未持仓：检查是否大跌建仓
                        if strategy.check_crash_entry(market_return, etf_return_1d):
                            strategy.buy(symbol, price, PROBE_POSITION, str(date))
                            print(f"    {date.date()} 试探建仓 {symbol} @ {price:.3f} (沪指{market_return:.2%}, ETF{etf_return_1d:.2%})")
                        elif score > 0.5 and factors['sentiment'] > 0.1:
                            # 强势+情绪中性以上，直接建仓（等权分配到各标的）
                            per_etf_ratio = 0.80 / top_n  # 80%股票仓位，均分
                            strategy.buy(symbol, price, per_etf_ratio, str(date))
                            print(f"    {date.date()} 热点建仓 {symbol} @ {price:.3f} (得分{score:.3f})")
                    else:
                        # 已持仓：检查上升潮加仓
                        if strategy.check_upsurge(hist):
                            current_value = info['shares'] * price
                            total_value = strategy.get_portfolio_value(all_prices, date)
                            current_ratio = current_value / total_value if total_value > 0 else 0
                            per_etf_target = 0.80 / top_n
                            if current_ratio < per_etf_target * 0.8:
                                add_ratio = min(UPSURGE_ADD_POSITION, per_etf_target - current_ratio)
                                if add_ratio > 0.02:
                                    strategy.buy(symbol, price, add_ratio, str(date))
                                    print(f"    {date.date()} 上升潮加仓 {symbol} @ {price:.3f} (加{add_ratio:.1%})")
                
                last_rebalance = date_ts
            else:
                # 没有得分，空仓观望
                pass
        
        # 计算NAV
        nav = strategy.get_portfolio_value(all_prices, date)
        nav_history.append({'date': date, 'nav': nav})
    
    # ============================================================
    # 计算结果
    # ============================================================
    nav_df = pd.DataFrame(nav_history)
    nav_df['returns'] = nav_df['nav'].pct_change()
    bench_nav_df['returns'] = bench_nav_df['nav'].pct_change()
    
    strategy_total = nav_df['nav'].iloc[-1] / INITIAL_CAPITAL - 1
    benchmark_total = bench_nav_df['nav'].iloc[-1] / INITIAL_CAPITAL - 1
    excess_return = strategy_total - benchmark_total
    
    # 最大回撤
    cummax = nav_df['nav'].cummax()
    drawdown = (nav_df['nav'] - cummax) / cummax
    max_dd = drawdown.min()
    
    print("\n" + "=" * 70)
    print("                    回测结果")
    print("=" * 70)
    print(f"  初始资金: {INITIAL_CAPITAL:,.0f} 元")
    print(f"  策略最终净值: {nav_df['nav'].iloc[-1]:,.0f} 元")
    print(f"  基准最终净值: {bench_nav_df['nav'].iloc[-1]:,.0f} 元")
    print(f"  策略收益: {strategy_total:.2%}")
    print(f"  基准收益: {benchmark_total:.2%}")
    print(f"  超额收益: {excess_return:.2%}")
    print(f"  最大回撤: {max_dd:.2%}")
    print(f"  交易次数: {strategy.trade_count}")
    
    # ============================================================
    # 评价Agent
    # ============================================================
    print("\n" + "=" * 70)
    print("              独立评价Agent评估")
    print("=" * 70)
    
    from evaluator_agent import EvaluationAgent
    agent = EvaluationAgent(verbose=True)
    
    # 对齐数据
    common_dates = nav_df['date'].isin(bench_nav_df['date'])
    strategy_returns = nav_df.loc[common_dates, 'returns'].dropna()
    bench_returns = bench_nav_df.set_index('date')['returns'].reindex(nav_df.loc[common_dates, 'date']).dropna()
    
    report = agent.full_evaluation(
        strategy_returns=strategy_returns,
        benchmark_returns=bench_returns
    )
    agent.print_report(report)
    
    return {
        'nav_df': nav_df,
        'bench_nav_df': bench_nav_df,
        'report': report,
        'strategy_total': strategy_total,
        'benchmark_total': benchmark_total,
        'excess_return': excess_return,
        'max_dd': max_dd,
        'trade_count': strategy.trade_count
    }


if __name__ == "__main__":
    main()
