"""
Merge extended news data into news_sentiment_v4 and re-aggregate sentiment features
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

DB_PATH = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'

print("=" * 70)
print("【合并扩展新闻数据到 news_sentiment_v4】")
print("=" * 70)

conn = sqlite3.connect(DB_PATH)

# 1. 读取扩展新闻数据
try:
    df_ext = pd.read_sql_query("SELECT * FROM news_raw_v4_extended ORDER BY date, symbol", conn)
    print(f"[数据] news_raw_v4_extended: {len(df_ext)} 条")
except Exception as e:
    print(f"[错误] 无法读取 news_raw_v4_extended: {e}")
    conn.close()
    exit(1)

# 2. 读取现有 news_raw_v4
try:
    df_raw = pd.read_sql_query("SELECT * FROM news_raw_v4 ORDER BY date DESC", conn)
    print(f"[数据] news_raw_v4: {len(df_raw)} 条")
except:
    df_raw = pd.DataFrame()

# 3. 合并到 news_raw_v4（去重）
if len(df_raw) > 0 and len(df_ext) > 0:
    # 基于 title + symbol + date 去重，去掉id列避免冲突
    df_combined = pd.concat([df_raw, df_ext], ignore_index=True)
    df_combined = df_combined.drop(columns=['id'], errors='ignore')
    df_combined = df_combined.drop_duplicates(subset=['symbol', 'date', 'title'], keep='last')
    print(f"[合并] 去重后 news_raw_v4: {len(df_combined)} 条")
    
    # 清空并重新写入
    cursor = conn.cursor()
    cursor.execute("DELETE FROM news_raw_v4")
    conn.commit()
    df_combined.to_sql('news_raw_v4', conn, if_exists='append', index=False)
    print(f"[保存] news_raw_v4 已更新: {len(df_combined)} 条")
elif len(df_ext) > 0:
    df_combined = df_ext.copy()
    df_combined = df_combined.drop(columns=['id'], errors='ignore')
    print(f"[合并] 使用扩展数据 news_raw_v4: {len(df_combined)} 条")
    
    cursor = conn.cursor()
    cursor.execute("DELETE FROM news_raw_v4")
    conn.commit()
    df_combined.to_sql('news_raw_v4', conn, if_exists='append', index=False)
    print(f"[保存] news_raw_v4 已更新: {len(df_combined)} 条")
else:
    df_combined = pd.DataFrame()
    print("[警告] 无新闻数据可合并")

# 4. 重新聚合日级情绪
print("\n【重新聚合日级情绪特征...】")

# 使用合并后的数据重新计算
if len(df_combined) > 0:
    df = df_combined.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 按 symbol + date 聚合
    daily = df.groupby(['symbol', 'date']).agg({
        'sentiment_score': ['mean', 'count'],
        'is_major_event': 'sum'
    }).reset_index()
    daily.columns = ['symbol', 'date', 'sentiment_1d', 'news_count_1d', 'major_events_1d']
    
    # 计算滚动情绪 (3日/7日)
    all_sentiment = []
    for sym, group in daily.groupby('symbol'):
        group = group.sort_values('date')
        group['sentiment_3d'] = group['sentiment_1d'].rolling(3, min_periods=1).mean()
        group['sentiment_7d'] = group['sentiment_1d'].rolling(7, min_periods=1).mean()
        group['major_events_3d'] = group['major_events_1d'].rolling(3, min_periods=1).sum()
        group['news_count_3d'] = group['news_count_1d'].rolling(3, min_periods=1).sum()
        
        # 最新标题
        latest_title = df[df['symbol'] == sym].sort_values('date', ascending=False).iloc[0]['title'] if len(df[df['symbol'] == sym]) > 0 else ''
        latest_source = df[df['symbol'] == sym].sort_values('date', ascending=False).iloc[0]['source'] if len(df[df['symbol'] == sym]) > 0 else ''
        group['latest_title'] = latest_title
        group['latest_source'] = latest_source
        group['updated_at'] = datetime.now().isoformat()
        
        all_sentiment.append(group)
    
    df_sentiment = pd.concat(all_sentiment, ignore_index=True)
    
    # 清空并重新写入 news_sentiment_v4
    cursor = conn.cursor()
    cursor.execute("DELETE FROM news_sentiment_v4")
    conn.commit()
    df_sentiment.to_sql('news_sentiment_v4', conn, if_exists='append', index=False)
    print(f"[保存] news_sentiment_v4 已更新: {len(df_sentiment)} 条")
    print(f"  日期范围: {df_sentiment['date'].min()} ~ {df_sentiment['date'].max()}")
    print(f"  标的覆盖: {df_sentiment['symbol'].nunique()} 个")

# 5. 将情绪特征合并到 features_v4
print("\n【将情绪特征合并到 features_v4...】")

df_feat = pd.read_sql_query("SELECT * FROM features_v4 ORDER BY symbol, trade_date", conn)
df_feat['trade_date'] = pd.to_datetime(df_feat['trade_date'])

# 合并情绪特征（左连接，缺失填0）
df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
df_merged = pd.merge(
    df_feat,
    df_sentiment[['symbol', 'date', 'sentiment_1d', 'sentiment_3d', 'sentiment_7d', 
                   'major_events_1d', 'major_events_3d', 'news_count_1d', 'news_count_3d']],
    left_on=['symbol', 'trade_date'],
    right_on=['symbol', 'date'],
    how='left'
)

# 填充缺失值（无新闻的日子情绪为0，新闻计数为0）
sentiment_cols = ['sentiment_1d', 'sentiment_3d', 'sentiment_7d', 
                  'major_events_1d', 'major_events_3d', 'news_count_1d', 'news_count_3d']
for col in sentiment_cols:
    df_merged[col] = df_merged[col].fillna(0)

# 删除临时date列
df_merged = df_merged.drop(columns=['date'], errors='ignore')

print(f"[合并] features_v4 + sentiment: {len(df_merged)} 条")
print(f"  有情绪数据的天数: {(df_merged['sentiment_1d'] != 0).sum()}")
print(f"  情绪特征列: {sentiment_cols}")

# 保存回 features_v4（重建表）
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS features_v4")

# 构建CREATE TABLE（动态添加情绪列）
all_cols = [c for c in df_merged.columns if c != 'id']
def sql_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return 'INTEGER'
    elif pd.api.types.is_float_dtype(dtype):
        return 'REAL'
    else:
        return 'TEXT'

create_sql = "CREATE TABLE features_v4 (\n    id INTEGER PRIMARY KEY AUTOINCREMENT"
for col in all_cols:
    create_sql += f",\n    {col} {sql_type(df_merged[col].dtype)}"
create_sql += "\n)"
cursor.execute(create_sql)
conn.commit()

# 写入数据
df_merged[all_cols].to_sql('features_v4', conn, if_exists='append', index=False)
conn.commit()

print(f"[保存] features_v4 已更新（含情绪特征）: {len(df_merged)} 条, {len(all_cols)} 列")

conn.close()

print("\n" + "=" * 70)
print("【新闻情绪特征合并完成】")
print("=" * 70)
