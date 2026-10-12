"""
QuanTrade 2.0 - Layer 3: 黄金对冲模块
=====================================
目标：市场下跌时，用黄金ETF替代空仓，既能避险又能保值

黄金ETF：518880（华安黄金）

对冲逻辑：
- 强烈看跌：30%黄金 + 70%现金
- 看跌：20%黄金 + 30%股票 + 50%现金
- 中性：10%黄金 + 40%股票 + 50%现金
- 看涨：0%黄金 + 70%股票 + 30%现金
- 强烈看涨：0%黄金 + 90%股票 + 10%现金
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict


class GoldHedge:
    """黄金对冲模块"""
    
    def __init__(self, db_path: str = "QuanTrade/quant_system/data/quant.db"):
        self.db_path = db_path
        self.gold_data = None
    
    def load_gold_data(self):
        """加载黄金ETF数据"""
        conn = sqlite3.connect(self.db_path)
        self.gold_data = pd.read_sql(
            "SELECT * FROM gold_daily_prices ORDER BY date",
            conn
        )
        conn.close()
        
        if len(self.gold_data) > 0:
            self.gold_data['date'] = pd.to_datetime(self.gold_data['date'])
            self.gold_data = self.gold_data.set_index('date')
        
        return self.gold_data
    
    def calculate_gold_allocation(self, market_prob: float) -> Dict[str, float]:
        """
        根据市场概率计算黄金/股票/现金仓位
        
        Args:
            market_prob: 市场上涨概率 (0-1)
            
        Returns:
            {'gold': 0.0-0.3, 'stock': 0.0-0.9, 'cash': 0.1-0.7}
        """
        if market_prob < 0.35:
            # 强烈看跌：30%黄金 + 70%现金
            return {'gold': 0.30, 'stock': 0.00, 'cash': 0.70}
        
        elif market_prob < 0.45:
            # 看跌：20%黄金 + 30%股票 + 50%现金
            return {'gold': 0.20, 'stock': 0.30, 'cash': 0.50}
        
        elif market_prob < 0.55:
            # 中性：10%黄金 + 40%股票 + 50%现金
            return {'gold': 0.10, 'stock': 0.40, 'cash': 0.50}
        
        elif market_prob < 0.65:
            # 看涨：0%黄金 + 70%股票 + 30%现金
            return {'gold': 0.00, 'stock': 0.70, 'cash': 0.30}
        
        else:
            # 强烈看涨：0%黄金 + 90%股票 + 10%现金
            return {'gold': 0.00, 'stock': 0.90, 'cash': 0.10}
    
    def get_gold_signal(self, date: str = None) -> float:
        """
        获取黄金信号（基于黄金趋势）
        
        Returns:
            黄金趋势得分 (-1 到 1)
            正值=黄金上涨（避险情绪上升）
            负值=黄金下跌（风险偏好上升）
        """
        if self.gold_data is None:
            self.load_gold_data()
        
        if self.gold_data is None or len(self.gold_data) == 0:
            return 0
        
        # 获取到指定日期的数据
        if date:
            hist = self.gold_data.loc[:date]
        else:
            hist = self.gold_data
        
        if len(hist) < 60:
            return 0
        
        close = hist['close']
        
        # 短期动量（5天）
        mom_5d = close.iloc[-1] / close.iloc[-5] - 1
        
        # 中期动量（20天）
        mom_20d = close.iloc[-1] / close.iloc[-20] - 1
        
        # 趋势判断
        ma20 = close.rolling(20).mean().iloc[-1]
        in_uptrend = close.iloc[-1] > ma20
        
        # 综合信号
        signal = 0
        if in_uptrend and mom_5d > 0 and mom_20d > 0:
            signal = 0.8  # 黄金强势上涨
        elif in_uptrend and mom_5d > 0:
            signal = 0.4  # 黄金短期上涨
        elif not in_uptrend and mom_5d < 0 and mom_20d < 0:
            signal = -0.8  # 黄金强势下跌
        elif not in_uptrend and mom_5d < 0:
            signal = -0.4  # 黄金短期下跌
        
        return signal
    
    def adjust_allocation_by_gold(self, base_allocation: Dict[str, float], gold_signal: float) -> Dict[str, float]:
        """
        根据黄金信号调整仓位
        
        Args:
            base_allocation: 基础仓位 {'gold': x, 'stock': y, 'cash': z}
            gold_signal: 黄金信号 (-1 到 1)
            
        Returns:
            调整后的仓位
        """
        allocation = base_allocation.copy()
        
        # 黄金上涨信号强 → 增加黄金仓位，减少股票仓位
        if gold_signal > 0.5:
            gold_increase = min(0.15, allocation['stock'] * 0.3)
            allocation['gold'] += gold_increase
            allocation['stock'] -= gold_increase
        
        # 黄金下跌信号强 → 减少黄金仓位，增加股票仓位
        elif gold_signal < -0.5:
            gold_decrease = min(allocation['gold'], 0.15)
            allocation['gold'] -= gold_decrease
            allocation['stock'] += gold_decrease
        
        # 确保仓位合法
        allocation['gold'] = max(0, min(0.3, allocation['gold']))
        allocation['stock'] = max(0, min(0.9, allocation['stock']))
        allocation['cash'] = 1 - allocation['gold'] - allocation['stock']
        
        return allocation


def test_gold_hedge():
    """测试黄金对冲模块"""
    print("=" * 60)
    print("QuanTrade 2.0 - 黄金对冲模块测试")
    print("=" * 60)
    
    hedge = GoldHedge()
    
    # 加载数据
    gold_data = hedge.load_gold_data()
    print(f"\n黄金数据: {len(gold_data)} 条")
    
    # 测试仓位计算
    print("\n市场概率 → 仓位分配:")
    for prob in [0.2, 0.4, 0.5, 0.6, 0.8]:
        allocation = hedge.calculate_gold_allocation(prob)
        print(f"  概率={prob:.1f} → 黄金={allocation['gold']:.0%}, 股票={allocation['stock']:.0%}, 现金={allocation['cash']:.0%}")
    
    # 测试黄金信号
    print("\n黄金信号:")
    signal = hedge.get_gold_signal()
    print(f"  当前信号: {signal:.2f}")
    
    # 测试仓位调整
    print("\n仓位调整示例:")
    base = {'gold': 0.1, 'stock': 0.5, 'cash': 0.4}
    adjusted = hedge.adjust_allocation_by_gold(base, signal)
    print(f"  基础: {base}")
    print(f"  黄金信号={signal:.2f} → 调整后: {adjusted}")
    
    return hedge


if __name__ == "__main__":
    test_gold_hedge()
