"""
QuanTrade 2.0 - Layer 4: 国际政治分析引擎
==========================================
数据源：外交部新闻、商务部公告、美联储声明
关键词词库：地缘政治风险词库、缓和词库
"""

import numpy as np
import pandas as pd
from typing import List, Dict


# ============================================================
# 关键词词库
# ============================================================

# 地缘政治风险词库
GEOPOLITICAL_RISK_WORDS = [
    # 贸易摩擦
    '贸易战', '关税', '加征关税', '贸易摩擦', '贸易争端',
    '反倾销', '反补贴', '贸易壁垒', '出口管制',
    
    # 技术封锁
    '技术封锁', '芯片禁令', '实体清单', '出口管制',
    '技术脱钩', '供应链断裂', '断供', '卡脖子',
    
    # 地缘冲突
    '军事冲突', '战争', '军事行动', '导弹试射',
    '台海', '南海', '中东', '乌克兰', '俄罗斯',
    
    # 制裁
    '制裁', '金融制裁', '贸易制裁', '技术制裁',
    '冻结资产', '禁止交易', '黑名单',
    
    # 金融风险
    '金融危机', '银行倒闭', '债务危机', '货币危机',
    '资本外流', '汇率波动', '美元走强',
]

# 地缘政治缓和词库
GEOPOLITICAL_POSITIVE_WORDS = [
    '贸易谈判', '达成协议', '取消关税', '缓和',
    '对话', '合作', '互利', '共赢', '稳定',
    '和平', '停火', '谈判', '协商', '解决',
]

# 重大国际事件词库
MAJOR_INTERNATIONAL_EVENTS = [
    'G7', 'G20', 'APEC', '联合国', '北约',
    '美联储', '欧央行', '日本央行', '英国央行',
    'IMF', '世界银行', 'WTO',
]


class GeopoliticalEngine:
    """国际政治分析引擎"""
    
    def __init__(self):
        self.risk_words = GEOPOLITICAL_RISK_WORDS
        self.positive_words = GEOPOLITICAL_POSITIVE_WORDS
        self.major_events = MAJOR_INTERNATIONAL_EVENTS
    
    def analyze_text(self, text: str) -> Dict:
        """
        分析单条文本的地缘政治风险
        
        Args:
            text: 新闻文本
            
        Returns:
            {
                'risk_score': float (-1 到 1),
                'risk_count': int,
                'positive_count': int,
                'is_major': bool
            }
        """
        if not text:
            return {'risk_score': 0, 'risk_count': 0, 'positive_count': 0, 'is_major': False}
        
        risk_count = sum(1 for w in self.risk_words if w in text)
        positive_count = sum(1 for w in self.positive_words if w in text)
        is_major = any(w in text for w in self.major_events)
        
        # 计算风险得分
        total = risk_count + positive_count
        if total == 0:
            score = 0
        else:
            score = (risk_count - positive_count) / total
        
        # 重大事件加权
        if is_major:
            score *= 1.5
        
        return {
            'risk_score': np.clip(score, -1, 1),
            'risk_count': risk_count,
            'positive_count': positive_count,
            'is_major': is_major
        }
    
    def analyze_batch(self, texts: List[str]) -> Dict:
        """
        批量分析文本
        
        Args:
            texts: 新闻文本列表
            
        Returns:
            {
                'avg_risk_score': float,
                'max_risk_score': float,
                'total_risk_count': int,
                'total_positive_count': int,
                'major_events_count': int
            }
        """
        if not texts:
            return {
                'avg_risk_score': 0,
                'max_risk_score': 0,
                'total_risk_count': 0,
                'total_positive_count': 0,
                'major_events_count': 0
            }
        
        results = [self.analyze_text(text) for text in texts]
        
        return {
            'avg_risk_score': np.mean([r['risk_score'] for r in results]),
            'max_risk_score': max([r['risk_score'] for r in results]),
            'total_risk_count': sum([r['risk_count'] for r in results]),
            'total_positive_count': sum([r['positive_count'] for r in results]),
            'major_events_count': sum([1 for r in results if r['is_major']])
        }
    
    def get_geopolitical_factor(self, risk_score: float) -> float:
        """
        将风险得分转换为投资因子
        
        Args:
            risk_score: 风险得分 (-1 到 1)
            
        Returns:
            投资因子 (-1 到 1)
            负值=高风险应减仓
            正值=低风险可加仓
        """
        # 风险高时因子为负（应减仓）
        return -risk_score


def test_geopolitical_engine():
    """测试国际政治分析引擎"""
    print("=" * 60)
    print("QuanTrade 2.0 - 国际政治分析引擎测试")
    print("=" * 60)
    
    engine = GeopoliticalEngine()
    
    # 测试文本
    test_texts = [
        "中美贸易谈判取得积极进展，双方达成初步协议",
        "美国宣布对中国加征关税，贸易战升级",
        "台海局势紧张，军事冲突风险上升",
        "美联储宣布降息，市场反应积极",
        "G20峰会达成多项合作协议",
        "美国将中国企业列入实体清单，技术封锁加剧",
    ]
    
    print("\n单条文本分析:")
    for text in test_texts:
        result = engine.analyze_text(text)
        print(f"  文本: {text[:30]}...")
        print(f"    风险得分: {result['risk_score']:.2f}, 风险词: {result['risk_count']}, 缓和词: {result['positive_count']}")
    
    print("\n批量分析:")
    batch_result = engine.analyze_batch(test_texts)
    print(f"  平均风险: {batch_result['avg_risk_score']:.2f}")
    print(f"  最大风险: {batch_result['max_risk_score']:.2f}")
    print(f"  重大事件: {batch_result['major_events_count']}")
    
    return engine


if __name__ == "__main__":
    test_geopolitical_engine()
