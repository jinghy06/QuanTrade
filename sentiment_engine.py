"""
QuanTrade 精细情绪检测模块 (Sentiment Engine)
基于现有ETF价格/成交量数据构建合成情绪指数
无需外部API，完全基于技术指标

情绪指数范围: [-1, 1]
- 1.0: 极度乐观（全仓追涨）
- 0.8: 狂热（"连宝妈都入场"）→ 触发减仓
- 0.5: 乐观
- 0.0: 中性
- -0.5: 悲观
- -0.8: 恐慌（"连大妈都割肉"）→ 触发加仓
- -1.0: 极度恐慌（全仓割肉）
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class SentimentEngine:
    """
    合成情绪引擎
    基于多维度技术指标构建市场情绪指数
    """
    
    def __init__(self, history_window: int = 60):
        self.history_window = history_window
        self.history = []  # 情绪历史，用于计算极端值
    
    def calculate(self, all_prices: Dict[str, pd.DataFrame], 
                  benchmark_df: Optional[pd.DataFrame] = None,
                  date=None) -> dict:
        """
        计算当前市场情绪指数
        
        参数:
            all_prices: {symbol: DataFrame} 所有ETF的历史价格
            benchmark_df: 基准指数（如沪深300）
            date: 当前日期
        
        返回:
            {
                'sentiment': float,  # 综合情绪指数 [-1, 1]
                'components': dict,  # 各子维度得分
                'level': str,        # 情绪等级
                'is_extreme': bool,  # 是否极端（>0.8或<-0.8）
                'action': str        # 建议操作
            }
        """
        components = {}
        
        # 1. 波动率情绪 (Volatility Sentiment)
        # 原理: 高波动通常伴随恐慌或狂热
        vol_sentiment = self._calc_volatility_sentiment(benchmark_df)
        components['volatility'] = vol_sentiment
        
        # 2. 动量情绪 (Momentum Sentiment)
        # 原理: 持续上涨=乐观，持续下跌=悲观
        mom_sentiment = self._calc_momentum_sentiment(benchmark_df)
        components['momentum'] = mom_sentiment
        
        # 3. 成交量情绪 (Volume Sentiment)
        # 原理: 放量上涨=乐观，放量下跌=恐慌，缩量=观望
        vol_sentiment = self._calc_volume_sentiment(all_prices)
        components['volume'] = vol_sentiment
        
        # 4. 广度情绪 (Breadth Sentiment)
        # 原理: 上涨家数占比越高，情绪越乐观
        breadth_sentiment = self._calc_breadth_sentiment(all_prices)
        components['breadth'] = breadth_sentiment
        
        # 5. 趋势一致性 (Trend Consensus)
        # 原理: 短期/中期/长期趋势同向=强情绪，背离=犹豫
        trend_sentiment = self._calc_trend_consensus(benchmark_df)
        components['trend'] = trend_sentiment
        
        # 加权合成
        weights = {
            'volatility': 0.25,   # 波动率权重最高（恐慌/狂热的直接信号）
            'momentum': 0.25,     # 动量趋势
            'volume': 0.20,       # 成交量确认
            'breadth': 0.15,      # 市场广度
            'trend': 0.15         # 趋势一致性
        }
        
        sentiment = sum(components[k] * weights[k] for k in weights)
        sentiment = np.clip(sentiment, -1, 1)
        
        # 记录历史
        self.history.append(sentiment)
        if len(self.history) > self.history_window:
            self.history.pop(0)
        
        # 判断极端值
        is_extreme = abs(sentiment) > 0.8
        
        # 情绪等级
        if sentiment > 0.8:
            level = "极度乐观"
            action = "减仓50%"
        elif sentiment > 0.6:
            level = "乐观"
            action = "减仓30%"
        elif sentiment > 0.3:
            level = "偏乐观"
            action = "持有"
        elif sentiment > -0.3:
            level = "中性"
            action = "持有"
        elif sentiment > -0.6:
            level = "偏悲观"
            action = "观望"
        elif sentiment > -0.8:
            level = "悲观"
            action = "加仓20%"
        else:
            level = "极度恐慌"
            action = "加仓50%"
        
        return {
            'sentiment': sentiment,
            'components': components,
            'level': level,
            'is_extreme': is_extreme,
            'action': action
        }
    
    def _calc_volatility_sentiment(self, benchmark_df: Optional[pd.DataFrame]) -> float:
        """
        波动率情绪: 高波动=恐慌/狂热
        - 波动率快速上升 = 恐慌 (-1)
        - 波动率高位回落 = 乐观 (+1)
        - 波动率低位 = 中性 (0)
        """
        if benchmark_df is None or len(benchmark_df) < 20:
            return 0.0
        
        returns = benchmark_df['close'].pct_change().dropna()
        if len(returns) < 20:
            return 0.0
        
        vol_5d = returns.iloc[-5:].std() * np.sqrt(252)
        vol_20d = returns.iloc[-20:].std() * np.sqrt(252)
        vol_60d = returns.iloc[-60:].std() * np.sqrt(252) if len(returns) >= 60 else vol_20d
        
        # 波动率变化率
        vol_change = (vol_5d - vol_20d) / (vol_20d + 1e-8)
        
        # 波动率位置（相对于60日历史）
        vol_position = (vol_5d - vol_60d) / (vol_60d + 1e-8)
        
        # 恐慌: 波动率急剧上升
        # 狂热: 波动率高位但开始下降（价格仍在涨）
        if vol_change > 0.5:  # 波动率急剧上升
            return -0.8  # 恐慌
        elif vol_change > 0.2:
            return -0.5  # 担忧
        elif vol_position < -0.3:  # 波动率显著低于均值
            # 检查是否在上涨（狂热）
            price_mom_5d = benchmark_df['close'].iloc[-1] / benchmark_df['close'].iloc[-5] - 1
            if price_mom_5d > 0.05:  # 5天涨5%+
                return 0.8  # 狂热（低波动+快速上涨=压抑的乐观）
            return 0.3  # 平静
        else:
            return 0.0
    
    def _calc_momentum_sentiment(self, benchmark_df: Optional[pd.DataFrame]) -> float:
        """
        动量情绪: 持续动量=强情绪
        - 5日/20日/60日全部上涨 = 极度乐观 (+1)
        - 5日/20日/60日全部下跌 = 极度悲观 (-1)
        - 混合 = 犹豫 (0)
        """
        if benchmark_df is None or len(benchmark_df) < 60:
            return 0.0
        
        close = benchmark_df['close']
        mom_5d = close.iloc[-1] / close.iloc[-5] - 1
        mom_20d = close.iloc[-1] / close.iloc[-20] - 1
        mom_60d = close.iloc[-1] / close.iloc[-60] - 1
        
        # 同向动量
        if mom_5d > 0 and mom_20d > 0 and mom_60d > 0:
            # 全部上涨
            strength = min(mom_5d * 10, 1.0)  # 5日涨10%=满分
            return 0.5 + strength * 0.5  # [0.5, 1.0]
        elif mom_5d < 0 and mom_20d < 0 and mom_60d < 0:
            # 全部下跌
            strength = min(abs(mom_5d) * 10, 1.0)
            return -0.5 - strength * 0.5  # [-1.0, -0.5]
        elif mom_5d > 0 and mom_20d < 0:
            # 短期反弹，中期下跌 = 犹豫偏乐观
            return 0.2
        elif mom_5d < 0 and mom_20d > 0:
            # 短期回调，中期上涨 = 犹豫偏悲观
            return -0.2
        else:
            return 0.0
    
    def _calc_volume_sentiment(self, all_prices: Dict[str, pd.DataFrame]) -> float:
        """
        成交量情绪: 放量确认趋势
        - 放量上涨 = 乐观 (+1)
        - 放量下跌 = 恐慌 (-1)
        - 缩量 = 观望 (0)
        """
        if not all_prices:
            return 0.0
        
        volume_signals = []
        for symbol, df in all_prices.items():
            if len(df) < 20 or 'volume' not in df.columns:
                continue
            
            vol_5d = df['volume'].iloc[-5:].mean()
            vol_20d = df['volume'].iloc[-20:].mean()
            vol_ratio = vol_5d / (vol_20d + 1e-8)
            
            price_change = df['close'].iloc[-1] / df['close'].iloc[-5] - 1
            
            if vol_ratio > 1.5 and price_change > 0.02:
                volume_signals.append(0.8)  # 放量上涨
            elif vol_ratio > 1.5 and price_change < -0.02:
                volume_signals.append(-0.8)  # 放量下跌
            elif vol_ratio < 0.7:
                volume_signals.append(0.0)  # 缩量观望
            else:
                volume_signals.append(price_change * 5)  # 正常量，按价格变化
        
        if not volume_signals:
            return 0.0
        return np.clip(np.mean(volume_signals), -1, 1)
    
    def _calc_breadth_sentiment(self, all_prices: Dict[str, pd.DataFrame]) -> float:
        """
        市场广度情绪: 上涨家数占比
        - 80%+上涨 = 极度乐观
        - 20%-上涨 = 极度悲观
        """
        if not all_prices:
            return 0.0
        
        rising_count = 0
        total_count = 0
        
        for symbol, df in all_prices.items():
            if len(df) < 5:
                continue
            total_count += 1
            price_change = df['close'].iloc[-1] / df['close'].iloc[-5] - 1
            if price_change > 0:
                rising_count += 1
        
        if total_count == 0:
            return 0.0
        
        breadth = rising_count / total_count
        # 映射到 [-1, 1]
        sentiment = (breadth - 0.5) * 2
        return np.clip(sentiment, -1, 1)
    
    def _calc_trend_consensus(self, benchmark_df: Optional[pd.DataFrame]) -> float:
        """
        趋势一致性: 多时间周期趋势同向=强情绪
        - 5日/10日/20日/60日均线多头排列 = 乐观
        - 空头排列 = 悲观
        - 混乱 = 中性
        """
        if benchmark_df is None or len(benchmark_df) < 60:
            return 0.0
        
        close = benchmark_df['close']
        ma5 = close.iloc[-5:].mean()
        ma10 = close.iloc[-10:].mean()
        ma20 = close.iloc[-20:].mean()
        ma60 = close.iloc[-60:].mean()
        
        current = close.iloc[-1]
        
        # 多头排列得分
        bull_score = 0
        if current > ma5: bull_score += 0.25
        if ma5 > ma10: bull_score += 0.25
        if ma10 > ma20: bull_score += 0.25
        if ma20 > ma60: bull_score += 0.25
        
        # 空头排列得分
        bear_score = 0
        if current < ma5: bear_score += 0.25
        if ma5 < ma10: bear_score += 0.25
        if ma10 < ma20: bear_score += 0.25
        if ma20 < ma60: bear_score += 0.25
        
        # 映射到 [-1, 1]
        if bull_score > bear_score:
            return bull_score
        else:
            return -bear_score
    
    def get_history_extreme(self, threshold: float = 0.8) -> list:
        """获取历史极端情绪点"""
        return [(i, s) for i, s in enumerate(self.history) if abs(s) > threshold]


# ========== 集成到策略中的使用示例 ==========

def sentiment_factor_for_strategy(benchmark_df: pd.DataFrame, all_prices: Dict[str, pd.DataFrame]) -> dict:
    """
    为策略提供的情绪因子接口
    返回情绪值和是否触发极端操作
    """
    engine = SentimentEngine()
    result = engine.calculate(all_prices, benchmark_df)
    
    return {
        'sentiment': result['sentiment'],
        'is_emotion_high': result['sentiment'] > 0.8,   # 情绪高点 -> 减仓
        'is_emotion_low': result['sentiment'] < -0.8,   # 情绪低点 -> 加仓
        'level': result['level'],
        'action': result['action'],
        'components': result['components']
    }


if __name__ == '__main__':
    # 测试示例
    print("Sentiment Engine Test")
    print("=" * 60)
    
    # 创建模拟数据
    dates = pd.date_range('2024-01-01', '2026-06-01', freq='D')
    np.random.seed(42)
    
    # 模拟上涨行情（低波动+持续上涨）
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, len(dates)))
    benchmark = pd.DataFrame({'close': prices}, index=dates)
    benchmark['volume'] = 1000000 + np.random.randint(-200000, 200000, len(dates))
    
    all_prices = {
        '510300': benchmark.copy(),
        '159995': benchmark.copy() * 1.1,
    }
    
    engine = SentimentEngine()
    
    # 测试多个日期
    test_dates = ['2024-03-01', '2024-06-01', '2024-12-01', '2025-06-01', '2025-12-01', '2026-03-01']
    for d in test_dates:
        if d in benchmark.index:
            result = engine.calculate(all_prices, benchmark, d)
            print(f"\n{d}: sentiment={result['sentiment']:.2f} [{result['level']}] -> {result['action']}")
            print(f"  components: {result['components']}")
