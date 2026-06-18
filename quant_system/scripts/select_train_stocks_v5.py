#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展训练集股票池 - 筛选2005年前已上市的大盘股
"""

import akshare as ak
import pandas as pd
from datetime import datetime

# 候选大盘股列表（覆盖主要行业）
candidates = [
    # 金融
    ('600036.SH', '招商银行'), ('600030.SH', '中信证券'), ('600050.SH', '中国联通'),
    # 能源/原材料
    ('600028.SH', '中国石化'), ('600019.SH', '宝钢股份'), ('600188.SH', '兖矿能源'),
    ('600011.SH', '华能国际'), ('600900.SH', '长江电力'),
    # 消费
    ('600519.SH', '贵州茅台'), ('000858.SZ', '五粮液'), ('000651.SZ', '格力电器'),
    ('000333.SZ', '美的集团'), ('000568.SZ', '泸州老窖'), ('600600.SH', '青岛啤酒'),
    # 地产/基建
    ('000002.SZ', '万科A'), ('600031.SH', '三一重工'),
    # 医药
    ('600276.SH', '恒瑞医药'),
    # 科技/制造
    ('000725.SZ', '京东方A'), ('000063.SZ', '中兴通讯'), ('000625.SZ', '长安汽车'),
    ('600104.SH', '上汽集团'), ('600029.SH', '南方航空'),
    # 其他
    ('600048.SH', '保利发展'), ('601318.SH', '中国平安'), ('601012.SH', '隆基绿能'),
    ('002594.SZ', '比亚迪'), ('300750.SZ', '宁德时代'),
]

print("=" * 80)
print("查询候选股票上市日期")
print("=" * 80)

# 获取A股基本信息
stock_info = ak.stock_info_a_code_name()

# 构建代码映射
info_map = {}
for _, row in stock_info.iterrows():
    code = row['code']
    name = row['name']
    info_map[code] = name

# 获取上市日期
stock_yjbb = ak.stock_yjbb_em(date="20231231")  # 业绩快报，包含上市日期

results = []
for code, name in candidates:
    # 尝试从业绩快报查找
    stock_data = stock_yjbb[stock_yjbb['股票代码'] == code.replace('.SH', '').replace('.SZ', '')]
    if not stock_data.empty:
        list_date = stock_data.iloc[0].get('上市日期', '未知')
    else:
        list_date = '未知'
    
    # 尝试用akshare获取个股信息
    try:
        # 用stock_individual_info_em获取
        pure_code = code.replace('.SH', '').replace('.SZ', '')
        info = ak.stock_individual_info_em(symbol=pure_code)
        if not info.empty:
            list_date_row = info[info['item'] == '上市时间']
            if not list_date_row.empty:
                list_date = list_date_row.iloc[0]['value']
    except Exception as e:
        pass
    
    results.append({
        'code': code,
        'name': name,
        'list_date': str(list_date),
        'before_2005': '是' if str(list_date)[:4] < '2005' and str(list_date) != 'nan' else '否'
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

# 筛选2005年前上市的
selected = df[df['before_2005'] == '是']['code'].tolist()
print(f"\n2005年前上市的股票 ({len(selected)}只):")
for code in selected:
    row = df[df['code'] == code].iloc[0]
    print(f"  {code} {row['name']} (上市: {row['list_date']})")

# 保存到文件
with open('train_stocks_v5.txt', 'w', encoding='utf-8') as f:
    for code in selected:
        row = df[df['code'] == code].iloc[0]
        f.write(f"{code},{row['name']},{row['list_date']}\n")

print(f"\n已保存到 train_stocks_v5.txt")
