"""
QuanTrade 2.0 监督Agent
======================
独立监督实现过程，防止偷工减料
检查每个组件是否按计划完成
"""

import ast
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CheckResult:
    """检查结果"""
    item: str
    status: str  # PASS/FAIL/WARNING
    details: str


class SupervisionAgent:
    """监督Agent - 检查是否偷工减料"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.checks: List[CheckResult] = []
    
    def check_all(self) -> List[CheckResult]:
        """执行所有检查"""
        self.checks = []
        
        print("=" * 70)
        print("        QuanTrade 2.0 监督Agent - 检查清单")
        print("=" * 70)
        
        # Layer 1 检查
        self._check_layer1()
        
        # Layer 2 检查
        self._check_layer2()
        
        # Layer 3 检查
        self._check_layer3()
        
        # Layer 4 检查
        self._check_layer4()
        
        # Layer 5 检查
        self._check_layer5()
        
        # 整合检查
        self._check_integration()
        
        # 打印结果
        self._print_results()
        
        return self.checks
    
    def _check_layer1(self):
        """检查Layer 1: 策略C核心（ML模型）"""
        print("\n[检查] Layer 1: 策略C核心（ML模型）")
        
        # 检查主脚本是否存在（优先检查v2.0.1）
        main_script = os.path.join(self.project_root, "run_etf_system_v2.0.1.py")
        if not os.path.exists(main_script):
            main_script = os.path.join(self.project_root, "run_etf_system_v2.py")
        if not os.path.exists(main_script):
            self.checks.append(CheckResult("Layer 1", "FAIL", "主脚本不存在"))
            return
        
        # 检查是否使用了ML模型
        with open(main_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ml_indicators = ['sklearn', 'lightgbm', 'xgboost', 'RandomForest', 'GradientBoosting', 'LGBMClassifier']
        has_ml = any(indicator in content for indicator in ml_indicators)
        
        if has_ml:
            self.checks.append(CheckResult("Layer 1", "PASS", "使用了ML模型"))
        else:
            self.checks.append(CheckResult("Layer 1", "FAIL", "没有使用ML模型，只有简单规则"))
        
        # 检查是否有特征工程
        feature_indicators = ['calculate_features', 'RSI', 'MA_', 'volatility', 'momentum']
        has_features = any(indicator in content for indicator in feature_indicators)
        
        if has_features:
            self.checks.append(CheckResult("Layer 1 特征", "PASS", "有特征工程"))
        else:
            self.checks.append(CheckResult("Layer 1 特征", "FAIL", "缺少特征工程"))
    
    def _check_layer2(self):
        """检查Layer 2: 热点赛道选股"""
        print("[检查] Layer 2: 热点赛道选股")
        
        # 检查文件是否存在
        sector_file = os.path.join(self.project_root, "sector_rotation.py")
        if not os.path.exists(sector_file):
            self.checks.append(CheckResult("Layer 2", "FAIL", "sector_rotation.py 不存在"))
            return
        
        with open(sector_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含4个因子
        factors = ['momentum', 'fund_flow', 'trend', 'relative']
        found_factors = [f for f in factors if f in content.lower()]
        
        if len(found_factors) >= 3:
            self.checks.append(CheckResult("Layer 2 因子", "PASS", f"包含{len(found_factors)}个因子: {found_factors}"))
        else:
            self.checks.append(CheckResult("Layer 2 因子", "WARNING", f"只找到{len(found_factors)}个因子: {found_factors}"))
        
        # 检查是否被主脚本使用
        main_script = os.path.join(self.project_root, "run_etf_system_v2.0.1.py")
        if not os.path.exists(main_script):
            main_script = os.path.join(self.project_root, "run_etf_system_v2.py")
        if os.path.exists(main_script):
            with open(main_script, 'r', encoding='utf-8') as f:
                main_content = f.read()
            
            if 'sector_rotation' in main_content or 'SectorRotation' in main_content:
                self.checks.append(CheckResult("Layer 2 集成", "PASS", "已集成到主脚本"))
            else:
                self.checks.append(CheckResult("Layer 2 集成", "FAIL", "未集成到主脚本"))
    
    def _check_layer3(self):
        """检查Layer 3: 黄金对冲"""
        print("[检查] Layer 3: 黄金对冲")
        
        # 检查文件是否存在
        gold_file = os.path.join(self.project_root, "gold_hedge.py")
        if not os.path.exists(gold_file):
            self.checks.append(CheckResult("Layer 3", "FAIL", "gold_hedge.py 不存在"))
            return
        
        with open(gold_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含对冲逻辑
        hedge_indicators = ['gold', '518880', 'hedge', 'allocation', '避险']
        found = [i for i in hedge_indicators if i in content.lower()]
        
        if len(found) >= 2:
            self.checks.append(CheckResult("Layer 3", "PASS", f"包含对冲逻辑: {found}"))
        else:
            self.checks.append(CheckResult("Layer 3", "FAIL", "缺少对冲逻辑"))
        
        # 检查是否被主脚本使用
        main_script = os.path.join(self.project_root, "run_etf_system_v2.0.1.py")
        if not os.path.exists(main_script):
            main_script = os.path.join(self.project_root, "run_etf_system_v2.py")
        if os.path.exists(main_script):
            with open(main_script, 'r', encoding='utf-8') as f:
                main_content = f.read()
            
            if 'gold_hedge' in main_content or 'GoldHedge' in main_content:
                self.checks.append(CheckResult("Layer 3 集成", "PASS", "已集成到主脚本"))
            else:
                self.checks.append(CheckResult("Layer 3 集成", "FAIL", "未集成到主脚本"))
    
    def _check_layer4(self):
        """检查Layer 4: 情绪/政策/地缘引擎"""
        print("[检查] Layer 4: 情绪/政策/地缘引擎")
        
        # 检查情绪引擎
        sentiment_file = os.path.join(self.project_root, "sentiment_engine.py")
        if os.path.exists(sentiment_file):
            with open(sentiment_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否有真实关键词库
            keyword_indicators = ['POSITIVE_WORDS', 'NEGATIVE_WORDS', '涨停', '跌停']
            found = [i for i in keyword_indicators if i in content]
            
            if len(found) >= 2:
                self.checks.append(CheckResult("Layer 4 关键词", "PASS", f"包含关键词库: {found}"))
            else:
                self.checks.append(CheckResult("Layer 4 关键词", "FAIL", "缺少真实关键词库"))
            
            # 检查是否只是市场代理
            proxy_indicators = ['pct_change', 'rolling', 'mean()']
            is_proxy = all(indicator in content for indicator in proxy_indicators)
            
            if is_proxy and '爬虫' not in content and 'crawler' not in content:
                self.checks.append(CheckResult("Layer 4 数据源", "WARNING", "使用市场代理，未实现爬虫"))
            else:
                self.checks.append(CheckResult("Layer 4 数据源", "PASS", "有爬虫或真实数据源"))
        else:
            self.checks.append(CheckResult("Layer 4", "FAIL", "sentiment_engine.py 不存在"))
        
        # 检查地缘政治引擎
        geo_file = os.path.join(self.project_root, "geopolitical_engine.py")
        if os.path.exists(geo_file):
            self.checks.append(CheckResult("Layer 4 地缘", "PASS", "geopolitical_engine.py 存在"))
        else:
            self.checks.append(CheckResult("Layer 4 地缘", "FAIL", "geopolitical_engine.py 不存在"))
        
        # 检查政策引擎
        policy_file = os.path.join(self.project_root, "policy_engine.py")
        if os.path.exists(policy_file):
            self.checks.append(CheckResult("Layer 4 政策", "PASS", "policy_engine.py 存在"))
        else:
            self.checks.append(CheckResult("Layer 4 政策", "FAIL", "policy_engine.py 不存在"))
    
    def _check_layer5(self):
        """检查Layer 5: 建仓/减仓参数优化"""
        print("[检查] Layer 5: 建仓/减仓参数优化")
        
        # 检查优化文件
        optimize_file = os.path.join(self.project_root, "optimize_trading_rules.py")
        if os.path.exists(optimize_file):
            with open(optimize_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否包含计划中的参数
            params = ['RSI', 'drawdown', 'target_profit', 'reduce_ratio', 'entry_threshold']
            found = [p for p in params if p in content]
            
            if len(found) >= 3:
                self.checks.append(CheckResult("Layer 5", "PASS", f"包含参数优化: {found}"))
            else:
                self.checks.append(CheckResult("Layer 5", "FAIL", f"参数不足: {found}"))
        else:
            self.checks.append(CheckResult("Layer 5", "FAIL", "optimize_trading_rules.py 不存在"))
    
    def _check_integration(self):
        """检查整合情况"""
        print("[检查] 整合检查")
        
        main_script = os.path.join(self.project_root, "run_etf_system_v2.py")
        if not os.path.exists(main_script):
            self.checks.append(CheckResult("整合", "FAIL", "主脚本不存在"))
            return
        
        with open(main_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否导入了所有模块
        imports = ['sentiment_engine', 'sector_rotation', 'gold_hedge', 'evaluator_agent']
        found = [i for i in imports if i in content]
        
        if len(found) >= 3:
            self.checks.append(CheckResult("整合 导入", "PASS", f"导入了{len(found)}个模块: {found}"))
        else:
            self.checks.append(CheckResult("整合 导入", "FAIL", f"只导入了{len(found)}个模块: {found}"))
        
        # 检查是否有评价Agent
        if 'EvaluationAgent' in content or 'evaluator_agent' in content:
            self.checks.append(CheckResult("整合 评价", "PASS", "集成了评价Agent"))
        else:
            self.checks.append(CheckResult("整合 评价", "FAIL", "未集成评价Agent"))
    
    def _print_results(self):
        """打印检查结果"""
        print("\n" + "=" * 70)
        print("                    检查结果汇总")
        print("=" * 70)
        
        pass_count = sum(1 for c in self.checks if c.status == "PASS")
        fail_count = sum(1 for c in self.checks if c.status == "FAIL")
        warn_count = sum(1 for c in self.checks if c.status == "WARNING")
        
        print(f"\n[OK] 通过: {pass_count}")
        print(f"[失败] 失败: {fail_count}")
        print(f"[警告] 警告: {warn_count}")
        
        if fail_count > 0:
            print(f"\n[失败] 失败项:")
            for c in self.checks:
                if c.status == "FAIL":
                    print(f"  - {c.item}: {c.details}")
        
        if warn_count > 0:
            print(f"\n[警告] 警告项:")
            for c in self.checks:
                if c.status == "WARNING":
                    print(f"  - {c.item}: {c.details}")
        
        # 总体评估
        print("\n" + "=" * 70)
        if fail_count == 0:
            print("[判断] 总体评估: [OK] 完全达标，无偷工减料")
        elif fail_count <= 2:
            print("[判断] 总体评估: [警告] 基本达标，有少量缺失")
        else:
            print("[判断] 总体评估: [失败] 未达标，存在严重偷工减料")
        print("=" * 70)


def run_supervision():
    """运行监督检查"""
    agent = SupervisionAgent()
    results = agent.check_all()
    return results


if __name__ == "__main__":
    run_supervision()
