"""
QuanTrade 2.0 - 情绪/政策/地缘政治引擎
======================================
Layer 4: 多因子分析层

设计原则：
- 无法实时获取新闻时，用市场数据作为情绪代理
- 基于技术指标构建情绪、政策、地缘政治评分
- 所有评分归一化到 [-1, 1] 区间
- 正值=看多，负值=看空，0=中性

市场代理指标：
1. 情绪代理：波动率、涨跌比、成交量变化
2. 政策代理：大盘趋势、板块轮动速度
3. 地缘代理：黄金走势（避险情绪）、汇率波动
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class SentimentEngine:
    """情绪引擎 - 用市场数据代理情绪"""
    
    def __init__(self):
        self.name = "SentimentEngine"
    
    def calculate(self, prices_df: pd.DataFrame) -> pd.Series:
        """
        计算情绪得分
        
        Args:
            prices_df: 包含 close, volume 列的DataFrame
            
        Returns:
            情绪得分 [-1, 1]
        """
        scores = pd.DataFrame(index=prices_df.index)
        
        # 1. 波动率情绪：波动率上升=恐慌，下降=平静
        returns = prices_df['close'].pct_change()
        volatility = returns.rolling(20).std()
        vol_ma = volatility.rolling(60).mean()
        vol_signal = np.where(volatility > vol_ma * 1.5, -0.5,
                     np.where(volatility < vol_ma * 0.7, 0.5, 0))
        scores['volatility'] = vol_signal
        
        # 2. 动量情绪：短期vs长期动量
        mom_short = prices_df['close'].pct_change(5)
        mom_long = prices_df['close'].pct_change(20)
        mom_signal = np.where((mom_short > 0) & (mom_long > 0), 0.5,
                     np.where((mom_short < 0) & (mom_long < 0), -0.5, 0))
        scores['momentum'] = mom_signal
        
        # 3. 成交量情绪：放量上涨=乐观，放量下跌=恐慌
        if 'volume' in prices_df.columns and prices_df['volume'].sum() > 0:
            vol_change = prices_df['volume'].pct_change(5)
            price_change = prices_df['close'].pct_change(5)
            vol_price_signal = np.where((vol_change > 0.3) & (price_change > 0), 0.5,
                               np.where((vol_change > 0.3) & (price_change < 0), -0.5, 0))
            scores['volume_price'] = vol_price_signal
        else:
            scores['volume_price'] = 0
        
        # 4. 均线情绪：价格在均线上方=乐观
        ma20 = prices_df['close'].rolling(20).mean()
        ma60 = prices_df['close'].rolling(60).mean()
        ma_signal = np.where((prices_df['close'] > ma20) & (ma20 > ma60), 0.5,
                    np.where((prices_df['close'] < ma20) & (ma20 < ma60), -0.5, 0))
        scores['ma_position'] = ma_signal
        
        # 综合情绪得分（等权平均）
        final_score = scores.mean(axis=1)
        return final_score.clip(-1, 1)


class PolicyEngine:
    """政策引擎 - 用市场结构代理政策方向"""
    
    def __init__(self):
        self.name = "PolicyEngine"
    
    def calculate(self, prices_df: pd.DataFrame, benchmark_df: pd.DataFrame = None) -> pd.Series:
        """
        计算政策得分
        
        政策代理逻辑：
        - 大盘持续上涨 + 板块轮动活跃 = 政策支持
        - 大盘持续下跌 + 成交量萎缩 = 政策收紧
        - 黄金上涨 = 避险/货币宽松
        """
        scores = pd.DataFrame(index=prices_df.index)
        
        # 1. 趋势强度：长期趋势的方向和强度
        ma120 = prices_df['close'].rolling(120).mean()
        trend = (prices_df['close'] - ma120) / ma120
        trend_signal = np.where(trend > 0.1, 0.5,
                       np.where(trend < -0.1, -0.5, trend * 5))
        scores['trend'] = np.clip(trend_signal, -1, 1)
        
        # 2. 市场宽度：用价格相对位置代理
        high_60 = prices_df['close'].rolling(60).max()
        low_60 = prices_df['close'].rolling(60).min()
        position = (prices_df['close'] - low_60) / (high_60 - low_60 + 1e-8)
        breadth_signal = np.where(position > 0.8, 0.5,
                         np.where(position < 0.2, -0.5, 0))
        scores['breadth'] = breadth_signal
        
        # 3. 趋势稳定性：用收益率的自相关性代理
        returns = prices_df['close'].pct_change()
        autocorr = returns.rolling(60).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 10 else 0, raw=False
        )
        stability_signal = np.where(autocorr > 0.1, 0.3,  # 正自相关=趋势延续
                           np.where(autocorr < -0.1, -0.3, 0))
        scores['stability'] = stability_signal
        
        # 综合政策得分
        final_score = scores.mean(axis=1)
        return final_score.clip(-1, 1)


class GeopoliticalEngine:
    """地缘政治引擎 - 用避险资产代理地缘风险"""
    
    def __init__(self):
        self.name = "GeopoliticalEngine"
    
    def calculate(self, prices_df: pd.DataFrame, gold_df: pd.DataFrame = None) -> pd.Series:
        """
        计算地缘政治风险得分
        
        代理逻辑：
        - 黄金上涨 = 避险情绪上升 = 地缘风险上升
        - 黄金下跌 = 风险偏好上升 = 地缘风险下降
        - 注意：对A股而言，地缘风险高时应该减仓
        
        Returns:
            得分 [-1, 1]，负值=风险高应减仓，正值=风险低可加仓
        """
        scores = pd.DataFrame(index=prices_df.index)
        
        # 1. 用A股自身的波动率代理风险
        returns = prices_df['close'].pct_change()
        
        # 尾部风险：用下行波动率
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_vol = downside_returns.rolling(20).std()
        downside_ma = downside_vol.rolling(60).mean()
        
        risk_signal = np.where(downside_vol > downside_ma * 1.5, -0.5,  # 下行风险高
                      np.where(downside_vol < downside_ma * 0.7, 0.5, 0))  # 下行风险低
        scores['downside_risk'] = risk_signal
        
        # 2. 最大回撤速度：快速下跌=恐慌
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.rolling(20).max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        dd_signal = np.where(drawdown < -0.05, -0.5,  # 快速下跌
                   np.where(drawdown > -0.01, 0.3, 0))
        scores['drawdown'] = dd_signal
        
        # 3. 如果有黄金数据，用黄金走势
        if gold_df is not None and len(gold_df) > 0:
            # 对齐日期
            gold_aligned = gold_df.reindex(prices_df.index, method='ffill')
            if 'close' in gold_aligned.columns:
                gold_returns = gold_aligned['close'].pct_change(5)
                # 黄金上涨=避险=对A股不利
                gold_signal = np.where(gold_returns > 0.02, -0.4,
                             np.where(gold_returns < -0.02, 0.4, 0))
                scores['gold_hedge'] = gold_signal
        
        # 综合地缘政治得分（注意：负值=风险高）
        final_score = scores.mean(axis=1)
        return final_score.clip(-1, 1)


class MultiFactorEngine:
    """多因子综合引擎 - 整合情绪/政策/地缘政治"""
    
    def __init__(self):
        self.sentiment = SentimentEngine()
        self.policy = PolicyEngine()
        self.geopolitical = GeopoliticalEngine()
    
    def calculate_all(
        self, 
        prices_df: pd.DataFrame,
        gold_df: pd.DataFrame = None,
        weights: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        计算所有因子得分
        
        Args:
            prices_df: ETF价格数据
            gold_df: 黄金ETF数据（可选）
            weights: 因子权重 {'sentiment': 0.4, 'policy': 0.3, 'geopolitical': 0.3}
            
        Returns:
            DataFrame包含各因子得分和综合得分
        """
        if weights is None:
            weights = {'sentiment': 0.4, 'policy': 0.3, 'geopolitical': 0.3}
        
        result = pd.DataFrame(index=prices_df.index)
        
        # 计算各因子
        result['sentiment'] = self.sentiment.calculate(prices_df)
        result['policy'] = self.policy.calculate(prices_df)
        result['geopolitical'] = self.geopolitical.calculate(prices_df, gold_df)
        
        # 综合得分
        result['combined'] = (
            result['sentiment'] * weights['sentiment'] +
            result['policy'] * weights['policy'] +
            result['geopolitical'] * weights['geopolitical']
        )
        
        # 信号强度分类
        result['signal'] = np.where(result['combined'] > 0.3, 2,    # 强烈看多
                           np.where(result['combined'] > 0.1, 1,     # 看多
                           np.where(result['combined'] < -0.3, -2,   # 强烈看空
                           np.where(result['combined'] < -0.1, -1,   # 看空
                           0))))                                       # 中性
        
        return result
    
    def get_summary(self, factor_df: pd.DataFrame) -> Dict:
        """获取因子分析摘要"""
        latest = factor_df.iloc[-1] if len(factor_df) > 0 else None
        
        if latest is None:
            return {}
        
        return {
            'sentiment': latest['sentiment'],
            'policy': latest['policy'],
            'geopolitical': latest['geopolitical'],
            'combined': latest['combined'],
            'signal': latest['signal'],
            'signal_text': {
                2: '强烈看多', 1: '看多', 0: '中性', -1: '看空', -2: '强烈看空'
            }.get(int(latest['signal']), '未知')
        }


def test_engine():
    """测试引擎"""
    import sqlite3
    
    print("=" * 60)
    print("QuanTrade 2.0 - 多因子引擎测试")
    print("=" * 60)
    
    # 加载数据
    conn = sqlite3.connect("QuanTrade/quant_system/data/quant.db")
    
    # 加载沪深300ETF作为基准
    df = pd.read_sql("SELECT * FROM etf_daily_prices WHERE symbol='510300' ORDER BY trade_date", conn)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date')
    
    # 加载黄金数据
    gold = pd.read_sql("SELECT * FROM gold_daily_prices ORDER BY date", conn)
    gold['date'] = pd.to_datetime(gold['date'])
    gold = gold.set_index('date')
    
    conn.close()
    
    print(f"\n沪深300数据: {len(df)} 条")
    print(f"黄金数据: {len(gold)} 条")
    
    # 计算因子
    engine = MultiFactorEngine()
    factors = engine.calculate_all(df, gold_df=gold)
    
    print(f"\n因子计算完成: {len(factors)} 条")
    print(f"\n最新因子得分:")
    summary = engine.get_summary(factors)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    
    # 统计信号分布
    print(f"\n信号分布:")
    signal_counts = factors['signal'].value_counts().sort_index()
    for signal, count in signal_counts.items():
        pct = count / len(factors) * 100
        label = {2: '强烈看多', 1: '看多', 0: '中性', -1: '看空', -2: '强烈看空'}.get(signal, '未知')
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    return factors


if __name__ == "__main__":
    test_engine()
