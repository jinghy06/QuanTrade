"""
QuanTrade 2.0 - Layer 5: 建仓/减仓参数优化
==========================================
目标：用历史数据验证最优参数，而非凭感觉设定规则
方法：网格搜索 + 历史回测

需要优化的参数：
- 建仓阈值：跌1%/2%/3%/4%/5%
- RSI超卖阈值：20/25/30/35
- 回撤阈值：10%/15%/20%/25%
- 目标收益：10%/15%/20%/25%/30%
- 减仓比例：30%/40%/50%/60%
- RSI超买阈值：65/70/75/80
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple
from itertools import product


class TradingRuleOptimizer:
    """交易规则优化器"""
    
    def __init__(self, db_path: str = "QuanTrade/quant_system/data/quant.db"):
        self.db_path = db_path
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def should_open_position(
        self,
        etf_close: pd.Series,
        etf_rsi: pd.Series,
        market_close: pd.Series,
        params: Dict
    ) -> pd.Series:
        """
        判断是否应该建仓
        
        Args:
            etf_close: ETF收盘价
            etf_rsi: ETF的RSI
            market_close: 大盘收盘价
            params: 参数字典
            
        Returns:
            布尔序列，True表示应该建仓
        """
        # 条件1: 大盘大跌
        market_drop = market_close.pct_change() < -params['entry_threshold']
        
        # 条件2: ETF大跌
        etf_drop = etf_close.pct_change() < -params['entry_threshold']
        
        # 条件3: RSI超卖
        oversold = etf_rsi < params['rsi_oversold']
        
        # 条件4: 从高点回撤
        high_20d = etf_close.rolling(20).max()
        drawdown = (etf_close / high_20d - 1) < -params['drawdown_threshold']
        
        # 满足条件1+2，以及条件3或4
        should_open = market_drop & etf_drop & (oversold | drawdown)
        
        return should_open
    
    def should_reduce_position(
        self,
        etf_close: pd.Series,
        entry_price: float,
        etf_rsi: pd.Series,
        params: Dict
    ) -> pd.Series:
        """
        判断是否应该减仓
        
        Args:
            etf_close: ETF收盘价
            entry_price: 建仓价格
            etf_rsi: ETF的RSI
            params: 参数字典
            
        Returns:
            减仓比例序列 (0, 0.3, 0.5, etc.)
        """
        # 计算收益
        profit_pct = etf_close / entry_price - 1
        
        # 条件1: 达到目标收益
        hit_target = profit_pct > params['target_profit']
        
        # 条件2: RSI超买
        overbought = etf_rsi > params['rsi_overbought']
        
        # 计算减仓比例
        reduce_ratio = pd.Series(0, index=etf_close.index)
        reduce_ratio[hit_target] = params['reduce_ratio']
        reduce_ratio[overbought] = params['reduce_ratio'] * 0.5
        
        return reduce_ratio
    
    def backtest_with_params(
        self,
        etf_df: pd.DataFrame,
        market_df: pd.DataFrame,
        params: Dict,
        initial_capital: float = 50000
    ) -> Dict:
        """
        用指定参数回测
        
        Args:
            etf_df: ETF数据
            market_df: 大盘数据
            params: 参数字典
            initial_capital: 初始资金
            
        Returns:
            回测结果
        """
        # 对齐数据
        common_idx = etf_df.index.intersection(market_df.index)
        etf = etf_df.loc[common_idx]
        market = market_df.loc[common_idx]
        
        # 计算RSI
        etf_rsi = self.calculate_rsi(etf['close'])
        
        # 计算信号
        should_open = self.should_open_position(etf['close'], etf_rsi, market['close'], params)
        
        # 模拟交易
        capital = initial_capital
        holdings = 0
        entry_price = 0
        nav_history = []
        
        for i, date in enumerate(etf.index):
            price = etf['close'].iloc[i]
            
            # 建仓信号
            if should_open.iloc[i] and holdings == 0:
                shares = int(capital / price / 100) * 100
                if shares > 0:
                    holdings = shares
                    entry_price = price
                    capital -= shares * price
            
            # 减仓信号
            elif holdings > 0:
                reduce = self.should_reduce_position(
                    etf['close'].iloc[:i+1],
                    entry_price,
                    etf_rsi.iloc[:i+1],
                    params
                )
                
                if reduce.iloc[-1] > 0:
                    sell_shares = int(holdings * reduce.iloc[-1] / 100) * 100
                    if sell_shares > 0:
                        capital += sell_shares * price
                        holdings -= sell_shares
            
            # 计算NAV
            nav = capital + holdings * price
            nav_history.append(nav)
        
        # 计算指标
        nav_series = pd.Series(nav_history, index=etf.index)
        total_return = nav_series.iloc[-1] / initial_capital - 1
        
        # 最大回撤
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'final_nav': nav_series.iloc[-1],
            'params': params
        }
    
    def optimize(
        self,
        etf_df: pd.DataFrame,
        market_df: pd.DataFrame,
        param_grid: Dict = None
    ) -> Tuple[Dict, Dict]:
        """
        网格搜索优化参数
        
        Args:
            etf_df: ETF数据
            market_df: 大盘数据
            param_grid: 参数网格
            
        Returns:
            (最优参数, 最优结果)
        """
        if param_grid is None:
            param_grid = {
                'entry_threshold': [0.01, 0.02, 0.03],
                'rsi_oversold': [25, 30, 35],
                'drawdown_threshold': [0.10, 0.15, 0.20],
                'target_profit': [0.10, 0.15, 0.20],
                'reduce_ratio': [0.3, 0.5],
                'rsi_overbought': [65, 70, 75],
            }
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        best_return = -np.inf
        best_params = None
        best_result = None
        
        print(f"优化 {len(param_combinations)} 种参数组合...")
        
        for i, values in enumerate(param_combinations):
            params = dict(zip(param_names, values))
            
            result = self.backtest_with_params(etf_df, market_df, params)
            
            if result['total_return'] > best_return:
                best_return = result['total_return']
                best_params = params
                best_result = result
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(param_combinations)}")
        
        return best_params, best_result


def test_optimizer():
    """测试优化器"""
    print("=" * 60)
    print("QuanTrade 2.0 - 交易规则优化测试")
    print("=" * 60)
    
    conn = sqlite3.connect("QuanTrade/quant_system/data/quant.db")
    
    # 加载数据
    etf_df = pd.read_sql(
        "SELECT * FROM etf_daily_prices WHERE symbol='515070' ORDER BY trade_date",
        conn
    )
    market_df = pd.read_sql(
        "SELECT * FROM etf_daily_prices WHERE symbol='510300' ORDER BY trade_date",
        conn
    )
    conn.close()
    
    # 转换格式
    etf_df['trade_date'] = pd.to_datetime(etf_df['trade_date'])
    etf_df = etf_df.set_index('trade_date')
    
    market_df['trade_date'] = pd.to_datetime(market_df['trade_date'])
    market_df = market_df.set_index('trade_date')
    
    print(f"\nETF数据: {len(etf_df)} 条")
    print(f"大盘数据: {len(market_df)} 条")
    
    # 优化
    optimizer = TradingRuleOptimizer()
    
    # 小规模参数网格测试
    param_grid = {
        'entry_threshold': [0.02, 0.03],
        'rsi_oversold': [30, 35],
        'drawdown_threshold': [0.15],
        'target_profit': [0.15, 0.20],
        'reduce_ratio': [0.5],
        'rsi_overbought': [70],
    }
    
    best_params, best_result = optimizer.optimize(etf_df, market_df, param_grid)
    
    print(f"\n最优参数: {best_params}")
    print(f"最优收益: {best_result['total_return']:.2%}")
    print(f"最大回撤: {best_result['max_drawdown']:.2%}")
    
    return optimizer


if __name__ == "__main__":
    test_optimizer()
