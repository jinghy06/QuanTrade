"""
QuanTrade 2.0 - Layer 4: 国内政策分析引擎
==========================================
数据源：国务院政策文件、发改委公告、央行公告、证监会公告、工信部公告
关键词词库：利好政策词库、利空政策词库
"""

import numpy as np
import pandas as pd
from typing import List, Dict


# ============================================================
# 关键词词库
# ============================================================

# 利好政策词库
POSITIVE_POLICY_WORDS = [
    # 货币政策
    '降准', '降息', 'MLF下调', 'LPR下调', '逆回购',
    '宽松', '流动性充裕', '货币供应增加',
    
    # 财政政策
    '减税', '降费', '财政补贴', '专项债', '国债',
    '财政支出', '转移支付', '税收优惠',
    
    # 产业政策
    '产业扶持', '技术创新', '研发补贴', '产业基金',
    '专精特新', '智能制造', '数字化转型',
    
    # 行业政策（AI/芯片）
    '人工智能', '芯片', '半导体', '集成电路', '算力',
    '大模型', '数字经济', '新基建',
    
    # 行业政策（新能源）
    '新能源', '光伏', '储能', '碳中和', '碳达峰',
    '绿色金融', '清洁能源', '电动化',
    
    # 行业政策（军工）
    '国防现代化', '军事装备', '航天', '北斗',
    '军民融合', '国防科工',
]

# 利空政策词库
NEGATIVE_POLICY_WORDS = [
    # 货币政策
    '加息', '收紧', '去杠杆', '紧缩', '流动性收紧',
    
    # 监管政策
    '监管', '整顿', '规范', '限制', '禁止',
    '处罚', '罚款', '吊销', '关停',
    
    # 行业限制
    '产能过剩', '淘汰落后', '环保限产', '能耗限制',
    '房地产调控', '限购', '限贷', '限售',
    
    # 金融风险
    '金融风险', '债务风险', '影子银行', '非法集资',
    'P2P暴雷', '信托违约', '债券违约',
]

# 行业政策词库
INDUSTRY_POLICY_WORDS = {
    'ai': ['人工智能', 'AI', '大模型', '算力', '机器学习', '深度学习'],
    'chip': ['芯片', '半导体', '集成电路', 'EDA', '光刻机'],
    'military': ['军工', '国防', '航天', '导弹', '卫星', '北斗'],
    'new_energy': ['新能源', '光伏', '储能', '锂电池', '风电', '氢能'],
    'medical': ['医药', '创新药', '医疗器械', '生物医药', '疫苗'],
    'consumer': ['消费', '白酒', '食品', '饮料', '零售'],
}


class PolicyEngine:
    """国内政策分析引擎"""
    
    def __init__(self):
        self.positive_words = POSITIVE_POLICY_WORDS
        self.negative_words = NEGATIVE_POLICY_WORDS
        self.industry_words = INDUSTRY_POLICY_WORDS
    
    def analyze_text(self, text: str) -> Dict:
        """
        分析单条文本的政策影响
        
        Args:
            text: 政策文本
            
        Returns:
            {
                'policy_score': float (-1 到 1),
                'positive_count': int,
                'negative_count': int,
                'industry_hits': dict
            }
        """
        if not text:
            return {
                'policy_score': 0,
                'positive_count': 0,
                'negative_count': 0,
                'industry_hits': {}
            }
        
        positive_count = sum(1 for w in self.positive_words if w in text)
        negative_count = sum(1 for w in self.negative_words if w in text)
        
        # 计算政策得分
        total = positive_count + negative_count
        if total == 0:
            score = 0
        else:
            score = (positive_count - negative_count) / total
        
        # 检查行业相关性
        industry_hits = {}
        for industry, keywords in self.industry_words.items():
            hits = [k for k in keywords if k in text]
            if hits:
                industry_hits[industry] = hits
        
        return {
            'policy_score': np.clip(score, -1, 1),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'industry_hits': industry_hits
        }
    
    def analyze_batch(self, texts: List[str]) -> Dict:
        """
        批量分析文本
        
        Args:
            texts: 政策文本列表
            
        Returns:
            {
                'avg_policy_score': float,
                'max_policy_score': float,
                'total_positive': int,
                'total_negative': int,
                'industry_summary': dict
            }
        """
        if not texts:
            return {
                'avg_policy_score': 0,
                'max_policy_score': 0,
                'total_positive': 0,
                'total_negative': 0,
                'industry_summary': {}
            }
        
        results = [self.analyze_text(text) for text in texts]
        
        # 汇总行业命中
        industry_summary = {}
        for result in results:
            for industry, hits in result['industry_hits'].items():
                if industry not in industry_summary:
                    industry_summary[industry] = 0
                industry_summary[industry] += len(hits)
        
        return {
            'avg_policy_score': np.mean([r['policy_score'] for r in results]),
            'max_policy_score': max([r['policy_score'] for r in results]),
            'total_positive': sum([r['positive_count'] for r in results]),
            'total_negative': sum([r['negative_count'] for r in results]),
            'industry_summary': industry_summary
        }
    
    def get_policy_factor(self, policy_score: float) -> float:
        """
        将政策得分转换为投资因子
        
        Args:
            policy_score: 政策得分 (-1 到 1)
            
        Returns:
            投资因子 (-1 到 1)
            正值=政策利好可加仓
            负值=政策利空应减仓
        """
        return policy_score
    
    def get_industry_factor(self, industry_hits: Dict, target_industry: str) -> float:
        """
        获取特定行业的政策因子
        
        Args:
            industry_hits: 行业命中结果
            target_industry: 目标行业 ('ai', 'chip', 'military', etc.)
            
        Returns:
            行业政策因子 (-1 到 1)
        """
        if target_industry in industry_hits:
            hits = industry_hits[target_industry]
            return min(1.0, len(hits) * 0.2)  # 每个命中+0.2，最高1.0
        return 0


def test_policy_engine():
    """测试国内政策分析引擎"""
    print("=" * 60)
    print("QuanTrade 2.0 - 国内政策分析引擎测试")
    print("=" * 60)
    
    engine = PolicyEngine()
    
    # 测试文本
    test_texts = [
        "央行宣布降准0.5个百分点，释放流动性",
        "国务院发布人工智能产业发展规划",
        "证监会加强监管，整顿市场秩序",
        "工信部发布新能源汽车补贴政策",
        "发改委限制高耗能产业发展",
        "财政部推出减税降费措施",
    ]
    
    print("\n单条文本分析:")
    for text in test_texts:
        result = engine.analyze_text(text)
        print(f"  文本: {text[:30]}...")
        print(f"    政策得分: {result['policy_score']:.2f}, 利好: {result['positive_count']}, 利空: {result['negative_count']}")
        if result['industry_hits']:
            print(f"    行业相关: {result['industry_hits']}")
    
    print("\n批量分析:")
    batch_result = engine.analyze_batch(test_texts)
    print(f"  平均政策得分: {batch_result['avg_policy_score']:.2f}")
    print(f"  总利好: {batch_result['total_positive']}")
    print(f"  总利空: {batch_result['total_negative']}")
    print(f"  行业汇总: {batch_result['industry_summary']}")
    
    return engine


if __name__ == "__main__":
    test_policy_engine()
