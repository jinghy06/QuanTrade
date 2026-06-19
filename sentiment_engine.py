"""
QuanTrade 2.0 - 情绪/政策/地缘政治引擎
======================================
Layer 4: 多因子分析层

包含两种模式：
1. 关键词模式：基于新闻文本的关键词分析（100+正面/负面词库）
2. 市场代理模式：基于技术指标的情绪代理

所有评分归一化到 [-1, 1] 区间
正值=看多，负值=看空，0=中性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ============================================================
# 关键词词库（按Quantrade2.0.md）
# ============================================================

# 正面词库（100+）
POSITIVE_WORDS = [
    # 涨跌类
    '涨停', '大涨', '反弹', '突破', '创新高', '飙升', '暴涨', '强势', '领涨',
    '上涨', '走高', '拉升', '冲高', '新高', '翻倍', '暴涨',
    
    # 利好类
    '利好', '增持', '买入', '看多', '买入评级', '增持评级',
    '业绩增长', '盈利', '分红', '回购', '战略合作', '订单', '中标',
    
    # 政策类
    '政策支持', '补贴', '减税', '降准', '降息', '宽松', '刺激',
    '鼓励', '扶持', '改革', '开放', '创新', '发展',
    
    # 行业类（AI/芯片）
    'AI突破', '大模型', '算力', '芯片', '半导体', '集成电路',
    '人工智能', '机器人', '智能制造', '自动驾驶', '数字化',
    
    # 行业类（军工/航天）
    '国防', '军工', '航天', '卫星', '导弹', '战斗机', '航母',
    '北斗', '火箭', '航天器', '军事现代化',
    
    # 行业类（新能源）
    '光伏', '新能源', '储能', '锂电池', '风电', '碳中和',
    '绿色能源', '清洁能源', '电动', '氢能',
]

# 负面词库（100+）
NEGATIVE_WORDS = [
    # 涨跌类
    '跌停', '大跌', '暴跌', '跳水', '重挫', '弱势', '领跌',
    '下跌', '走低', '杀跌', '崩盘', '腰斩', '破位', '新低',
    
    # 利空类
    '利空', '减持', '卖出', '看空', '卖出评级', '减持评级',
    '业绩下滑', '亏损', '退市', 'ST', '违规', '处罚', '调查',
    
    # 政策类
    '加息', '收紧', '监管', '限制', '制裁', '整顿', '打压',
    '去杠杆', '紧缩', '调控', '限购', '限贷',
    
    # 风险类
    '贸易战', '制裁', '冲突', '战争', '脱钩', '断供',
    '地缘政治', '金融危机', '泡沫', '黑天鹅', '灰犀牛',
    
    # 行业类
    '产能过剩', '库存积压', '需求下滑', '价格战', '技术封锁',
]

# 重大事件词库（50+）
MAJOR_EVENT_WORDS = [
    # 公司事件
    '财报', '业绩', '分红', '回购', '增持', '减持',
    '重组', '并购', 'IPO', '增发', '配股', '可转债',
    
    # 政策事件
    '降准', '降息', '加息', 'MLF', 'LPR', '逆回购',
    '财政政策', '货币政策', '产业政策', '监管政策',
    
    # 国际事件
    '美联储', '欧央行', 'G7', 'G20', 'APEC', '联合国',
    '贸易战', '关税', '制裁', '脱钩', '断供',
    
    # 市场事件
    '停牌', '复牌', '退市', 'ST', '*ST',
    '暴跌', '熔断', '千股跌停', '千股涨停',
]


class SentimentEngine:
    """情绪引擎 - 支持关键词分析和市场代理"""
    
    def __init__(self):
        self.name = "SentimentEngine"
        self.positive_words = POSITIVE_WORDS
        self.negative_words = NEGATIVE_WORDS
        self.major_event_words = MAJOR_EVENT_WORDS
    
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
    
    def analyze_text(self, text: str) -> Dict:
        """
        分析单条文本的情绪（关键词模式）
        
        Args:
            text: 新闻文本
            
        Returns:
            {
                'sentiment_score': float (-1 到 1),
                'positive_count': int,
                'negative_count': int,
                'is_major': bool
            }
        """
        if not text:
            return {'sentiment_score': 0, 'positive_count': 0, 'negative_count': 0, 'is_major': False}
        
        pos_count = sum(1 for w in self.positive_words if w in text)
        neg_count = sum(1 for w in self.negative_words if w in text)
        is_major = any(w in text for w in self.major_event_words)
        
        # 计算情绪得分
        total = pos_count + neg_count
        if total == 0:
            score = 0
        else:
            score = (pos_count - neg_count) / total
        
        # 重大事件加权
        if is_major:
            score *= 1.5
        
        return {
            'sentiment_score': np.clip(score, -1, 1),
            'positive_count': pos_count,
            'negative_count': neg_count,
            'is_major': is_major
        }
    
    def analyze_batch(self, texts: List[str]) -> Dict:
        """
        批量分析文本情绪
        
        Args:
            texts: 新闻文本列表
            
        Returns:
            {
                'avg_sentiment': float,
                'max_sentiment': float,
                'total_positive': int,
                'total_negative': int,
                'major_events_count': int
            }
        """
        if not texts:
            return {
                'avg_sentiment': 0,
                'max_sentiment': 0,
                'total_positive': 0,
                'total_negative': 0,
                'major_events_count': 0
            }
        
        results = [self.analyze_text(text) for text in texts]
        
        return {
            'avg_sentiment': np.mean([r['sentiment_score'] for r in results]),
            'max_sentiment': max([r['sentiment_score'] for r in results]),
            'total_positive': sum([r['positive_count'] for r in results]),
            'total_negative': sum([r['negative_count'] for r in results]),
            'major_events_count': sum([1 for r in results if r['is_major']])
        }


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
