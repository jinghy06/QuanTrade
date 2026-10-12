"""
QuanTrade 2.0 - 热点板块轮动评分
================================
Layer 2: 板块轮动层

评分维度：
1. 动量得分：近期涨幅排名
2. 趋势得分：均线排列状态
3. 相对强度：相对基准的表现
4. 波动率调整：风险调整后收益

输出：每只ETF的轮动得分，用于选择最强板块
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple


class SectorRotation:
    """板块轮动评分系统"""
    
    def __init__(self, lookback_short: int = 5, lookback_mid: int = 20, lookback_long: int = 60):
        self.lookback_short = lookback_short
        self.lookback_mid = lookback_mid
        self.lookback_long = lookback_long
    
    def calculate_rotation_score(
        self, 
        all_prices: Dict[str, pd.DataFrame],
        benchmark_symbol: str = '510300'
    ) -> pd.DataFrame:
        """
        计算所有ETF的轮动得分
        
        Args:
            all_prices: {symbol: DataFrame} 所有ETF的价格数据
            benchmark_symbol: 基准ETF代码
            
        Returns:
            DataFrame: 日期 x ETF的轮动得分矩阵
        """
        # 获取所有交易日期
        all_dates = set()
        for df in all_prices.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)
        
        scores_dict = {}
        
        for symbol, df in all_prices.items():
            if len(df) < self.lookback_long + 10:
                continue
            
            scores = self._score_single_etf(df, all_prices.get(benchmark_symbol))
            scores_dict[symbol] = scores
        
        scores_df = pd.DataFrame(scores_dict)
        return scores_df
    
    def _score_single_etf(
        self, 
        df: pd.DataFrame, 
        benchmark_df: pd.DataFrame = None
    ) -> pd.Series:
        """单只ETF的轮动得分"""
        
        close = df['close']
        returns = close.pct_change()
        
        scores = pd.DataFrame(index=df.index)
        
        # 1. 动量得分（短期、中期、长期）
        mom_short = close.pct_change(self.lookback_short)
        mom_mid = close.pct_change(self.lookback_mid)
        mom_long = close.pct_change(self.lookback_long)
        
        # 归一化到 [-1, 1]
        scores['mom_short'] = self._normalize(mom_short)
        scores['mom_mid'] = self._normalize(mom_mid)
        scores['mom_long'] = self._normalize(mom_long)
        
        # 2. 趋势得分（均线排列）
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        
        trend_score = np.where((ma5 > ma20) & (ma20 > ma60), 1.0,   # 多头排列
                      np.where((ma5 < ma20) & (ma20 < ma60), -1.0,  # 空头排列
                      0))
        scores['trend'] = trend_score
        
        # 3. 相对强度（vs基准）
        if benchmark_df is not None and len(benchmark_df) > 0:
            bench_close = benchmark_df['close'].reindex(df.index, method='ffill')
            relative = close / bench_close
            rel_change = relative.pct_change(self.lookback_mid)
            scores['relative'] = self._normalize(rel_change)
        else:
            scores['relative'] = 0
        
        # 4. 波动率调整得分（夏普比率的简化版）
        rolling_return = returns.rolling(self.lookback_mid).mean()
        rolling_vol = returns.rolling(self.lookback_mid).std()
        sharpe_approx = rolling_return / (rolling_vol + 1e-8)
        scores['risk_adj'] = self._normalize(sharpe_approx)
        
        # 综合得分（加权平均）
        weights = {
            'mom_short': 0.2,
            'mom_mid': 0.25,
            'mom_long': 0.15,
            'trend': 0.2,
            'relative': 0.1,
            'risk_adj': 0.1
        }
        
        final_score = sum(scores[k] * v for k, v in weights.items())
        return final_score.clip(-1, 1)
    
    def _normalize(self, series: pd.Series, window: int = 60) -> pd.Series:
        """滚动归一化到 [-1, 1]"""
        rolling_mean = series.rolling(window, min_periods=20).mean()
        rolling_std = series.rolling(window, min_periods=20).std()
        z = (series - rolling_mean) / (rolling_std + 1e-8)
        return z.clip(-3, 3) / 3  # 归一化到 [-1, 1]
    
    def get_top_etfs(
        self, 
        scores_df: pd.DataFrame, 
        date: str = None,
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        获取某日得分最高的ETF
        
        Args:
            scores_df: 轮动得分矩阵
            date: 目标日期，默认最新
            top_n: 返回前N名
            
        Returns:
            [(symbol, score), ...]
        """
        if date is None:
            date = scores_df.index[-1]
        
        if date not in scores_df.index:
            return []
        
        scores_on_date = scores_df.loc[date].sort_values(ascending=False)
        return [(symbol, score) for symbol, score in scores_on_date.head(top_n).items()]
    
    def get_rotation_signal(
        self,
        scores_df: pd.DataFrame,
        current_holdings: List[str] = None,
        threshold: float = 0.1
    ) -> Dict:
        """
        生成轮动信号
        
        Args:
            scores_df: 轮动得分矩阵
            current_holdings: 当前持仓
            threshold: 切换阈值
            
        Returns:
            {'action': 'hold'/'switch', 'target': [symbols], 'reason': str}
        """
        if len(scores_df) == 0:
            return {'action': 'hold', 'target': [], 'reason': '无数据'}
        
        latest_scores = scores_df.iloc[-1].sort_values(ascending=False)
        top3 = latest_scores.head(3)
        
        if current_holdings is None or len(current_holdings) == 0:
            return {
                'action': 'buy',
                'target': [top3.index[0]],
                'reason': f'建仓: {top3.index[0]}得分最高({top3.iloc[0]:.3f})'
            }
        
        # 检查当前持仓是否在前3
        current_scores = {h: latest_scores.get(h, 0) for h in current_holdings}
        best_current = max(current_scores.values()) if current_scores else 0
        best_alternative = top3.iloc[0]
        best_alternative_symbol = top3.index[0]
        
        if best_alternative_symbol not in current_holdings:
            if best_alternative - best_current > threshold:
                return {
                    'action': 'switch',
                    'target': [best_alternative_symbol],
                    'reason': f'轮动: {best_alternative_symbol}({best_alternative:.3f}) > {list(current_scores.keys())[0]}({best_current:.3f})'
                }
        
        return {
            'action': 'hold',
            'target': current_holdings,
            'reason': f'持有: 当前持仓得分仍具竞争力'
        }


def test_rotation():
    """测试板块轮动"""
    print("=" * 60)
    print("QuanTrade 2.0 - 板块轮动测试")
    print("=" * 60)
    
    conn = sqlite3.connect("QuanTrade/quant_system/data/quant.db")
    
    # 加载所有ETF数据
    symbols = pd.read_sql(
        "SELECT DISTINCT symbol FROM etf_daily_prices", conn
    )['symbol'].tolist()
    
    all_prices = {}
    for symbol in symbols:
        df = pd.read_sql(
            f"SELECT * FROM etf_daily_prices WHERE symbol='{symbol}' ORDER BY trade_date", conn
        )
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        if len(df) >= 100:
            all_prices[symbol] = df
    
    conn.close()
    
    print(f"加载 {len(all_prices)} 只ETF")
    
    # 计算轮动得分
    rotation = SectorRotation()
    scores = rotation.calculate_rotation_score(all_prices, benchmark_symbol='510300')
    
    print(f"轮动得分计算完成: {scores.shape}")
    
    # 获取最新排名
    top_etfs = rotation.get_top_etfs(scores, top_n=5)
    print(f"\n最新排名:")
    for symbol, score in top_etfs:
        print(f"  {symbol}: {score:.3f}")
    
    # 获取轮动信号
    signal = rotation.get_rotation_signal(scores)
    print(f"\n轮动信号: {signal}")
    
    return scores


if __name__ == "__main__":
    test_rotation()
