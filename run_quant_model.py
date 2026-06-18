import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载 ====================
db_path = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'
conn = sqlite3.connect(db_path)
features_df = pd.read_sql_query("SELECT * FROM features ORDER BY trade_date, symbol", conn)
conn.close()

features_df['trade_date'] = pd.to_datetime(features_df['trade_date'])

print("=" * 60)
print("【量化预测模型与回测系统】")
print("=" * 60)
print(f"\n数据概览:")
print(f"  - 总记录数: {len(features_df)}")
print(f"  - 股票数量: {features_df['symbol'].nunique()}")
print(f"  - 股票列表: {features_df['symbol'].unique().tolist()}")
print(f"  - 日期范围: {features_df['trade_date'].min().strftime('%Y-%m-%d')} ~ {features_df['trade_date'].max().strftime('%Y-%m-%d')}")

# ==================== 2. 特征工程 ====================
# 选择特征列（排除目标变量、日期、股票代码等非特征列）
exclude_cols = ['trade_date', 'symbol', 'feature_version', 'created_at',
                'target_next_day_return', 'target_direction',
                'target_close_1d', 'target_close_3d', 'target_close_5d', 'target_close_10d',
                'target_return_1d', 'target_return_3d', 'target_return_5d', 'target_return_10d',
                'target_volatility_5d']

feature_cols = [c for c in features_df.columns if c not in exclude_cols]
print(f"\n特征列数量: {len(feature_cols)}")
print(f"特征列表: {feature_cols}")

# 处理缺失值
df_model = features_df.dropna(subset=['target_direction']).copy()
print(f"\n去除target_direction缺失后: {len(df_model)} 条")

# 填充其他特征的缺失值（用中位数）
for col in feature_cols:
    if df_model[col].isnull().sum() > 0:
        df_model[col] = df_model[col].fillna(df_model[col].median())

# ==================== 3. 时间序列划分训练集/测试集 ====================
# 按日期排序，前80%训练，后20%测试
df_model = df_model.sort_values('trade_date').reset_index(drop=True)

split_date = df_model['trade_date'].quantile(0.8)
train_df = df_model[df_model['trade_date'] < split_date].copy()
test_df = df_model[df_model['trade_date'] >= split_date].copy()

print(f"\n时间序列划分:")
print(f"  - 训练集: {len(train_df)} 条 ({train_df['trade_date'].min().strftime('%Y-%m-%d')} ~ {train_df['trade_date'].max().strftime('%Y-%m-%d')})")
print(f"  - 测试集: {len(test_df)} 条 ({test_df['trade_date'].min().strftime('%Y-%m-%d')} ~ {test_df['trade_date'].max().strftime('%Y-%m-%d')})")
print(f"  - 划分日期: {split_date.strftime('%Y-%m-%d')}")

X_train = train_df[feature_cols].values
y_train_clf = train_df['target_direction'].values
y_train_reg = train_df['target_next_day_return'].values

X_test = test_df[feature_cols].values
y_test_clf = test_df['target_direction'].values
y_test_reg = test_df['target_next_day_return'].values

# 标准化（仅用于逻辑回归）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 4. 模型训练 ====================
print("\n" + "=" * 60)
print("【模型训练】")
print("=" * 60)

# 4.1 随机森林分类器
print("\n1. 训练随机森林分类器...")
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train_clf)
rf_pred = rf_clf.predict(X_test)
rf_prob = rf_clf.predict_proba(X_test)[:, 1]

# 4.2 逻辑回归
print("2. 训练逻辑回归...")
lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train_scaled, y_train_clf)
lr_pred = lr_clf.predict(X_test_scaled)
lr_prob = lr_clf.predict_proba(X_test_scaled)[:, 1]

# 4.3 随机森林回归（预测收益率）
print("3. 训练随机森林回归（预测收益率）...")
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train_reg)
rf_reg_pred = rf_reg.predict(X_test)

# ==================== 5. 模型评估 ====================
print("\n" + "=" * 60)
print("【模型评估 - 分类任务（预测涨跌方向）】")
print("=" * 60)

def evaluate_classifier(name, y_true, y_pred, y_prob):
    print(f"\n>>> {name}")
    print(f"  准确率 (Accuracy):  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  精确率 (Precision): {precision_score(y_true, y_pred):.4f}")
    print(f"  召回率 (Recall):    {recall_score(y_true, y_pred):.4f}")
    print(f"  F1分数:             {f1_score(y_true, y_pred):.4f}")
    print(f"  正例预测概率均值:     {np.mean(y_prob):.4f}")
    print(f"  正例实际比例:         {np.mean(y_true):.4f}")

evaluate_classifier("随机森林", y_test_clf, rf_pred, rf_prob)
evaluate_classifier("逻辑回归", y_test_clf, lr_pred, lr_prob)

# 特征重要性
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n>>> 随机森林 Top 10 重要特征:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:20s} {row['importance']:.4f}")

# 回归评估
print("\n" + "=" * 60)
print("【模型评估 - 回归任务（预测次日收益率）】")
print("=" * 60)
mse = mean_squared_error(y_test_reg, rf_reg_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, rf_reg_pred)
print(f"\n>>> 随机森林回归")
print(f"  MSE:  {mse:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  R²:   {r2:.4f}")
print(f"  实际收益率均值: {np.mean(y_test_reg):.6f}")
print(f"  预测收益率均值: {np.mean(rf_reg_pred):.6f}")

# ==================== 6. 量化回测框架 ====================
print("\n" + "=" * 60)
print("【量化回测框架】")
print("=" * 60)

# 合并测试集数据与预测结果
test_results = test_df.copy()
test_results['rf_pred_direction'] = rf_pred
test_results['rf_prob_up'] = rf_prob
test_results['lr_pred_direction'] = lr_pred
test_results['lr_prob_up'] = lr_prob
test_results['rf_pred_return'] = rf_reg_pred

# 策略1: 随机森林分类信号 - 预测为1（涨）时买入，持有1天
# 策略2: 概率阈值策略 - 上涨概率>0.6时买入
# 策略3: 回归策略 - 预测收益率>0时买入
# 基准: 买入并持有各股票

strategies = {}

for symbol in test_results['symbol'].unique():
    sym_data = test_results[test_results['symbol'] == symbol].sort_values('trade_date').reset_index(drop=True)
    if len(sym_data) < 5:
        continue

    actual_returns = sym_data['target_next_day_return'].values

    # 基准策略: 每天满仓持有
    benchmark_cum = np.cumprod(1 + actual_returns)

    # 策略1: RF分类信号
    s1_signals = sym_data['rf_pred_direction'].values
    s1_returns = actual_returns * s1_signals  # 预测涨则持有，预测跌则空仓（收益为0）
    s1_cum = np.cumprod(1 + s1_returns)

    # 策略2: RF概率阈值 (>0.55)
    s2_signals = (sym_data['rf_prob_up'].values > 0.55).astype(int)
    s2_returns = actual_returns * s2_signals
    s2_cum = np.cumprod(1 + s2_returns)

    # 策略3: 回归预测 > 0
    s3_signals = (sym_data['rf_pred_return'].values > 0).astype(int)
    s3_returns = actual_returns * s3_signals
    s3_cum = np.cumprod(1 + s3_returns)

    strategies[symbol] = {
        'dates': sym_data['trade_date'].values,
        'actual': actual_returns,
        'benchmark': benchmark_cum,
        's1_rf_class': s1_cum,
        's2_rf_prob': s2_cum,
        's3_rf_reg': s3_cum,
        's1_signals': s1_signals,
        's2_signals': s2_signals,
        's3_signals': s3_signals
    }

# ==================== 7. 回测结果汇总 ====================
print("\n>>> 各策略回测结果汇总:\n")
print(f"{'股票代码':<12} {'策略':<20} {'总收益率':<12} {'年化收益率':<12} {'最大回撤':<12} {'夏普比率':<12} {'交易次数':<10}")
print("-" * 90)

all_returns = []

for symbol, data in strategies.items():
    for s_name, s_key, sig_key in [
        ('基准(持有)', 'benchmark', None),
        ('RF分类信号', 's1_rf_class', 's1_signals'),
        ('RF概率>0.55', 's2_rf_prob', 's2_signals'),
        ('RF回归>0', 's3_rf_reg', 's3_signals')
    ]:
        cum = data[s_key]
        total_ret = cum[-1] - 1
        # 年化（假设约252交易日）
        n_days = len(cum)
        annual_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 else 0
        # 最大回撤
        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_dd = np.min(drawdown)
        # 夏普（简化，假设无风险利率0）
        daily_rets = np.diff(cum) / cum[:-1]
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
        # 交易次数
        trades = np.sum(data[sig_key]) if sig_key else n_days

        print(f"{symbol:<12} {s_name:<20} {total_ret*100:>10.2f}% {annual_ret*100:>10.2f}% {max_dd*100:>10.2f}% {sharpe:>10.2f} {trades:>8}")

        if s_name != '基准(持有)':
            all_returns.append({
                'symbol': symbol,
                'strategy': s_name,
                'total_return': total_ret,
                'annual_return': annual_ret,
                'max_drawdown': max_dd,
                'sharpe': sharpe,
                'trades': trades
            })

# ==================== 8. 可视化 ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('量化预测模型 - 回测结果', fontsize=16, fontweight='bold')

# 子图1: 各股票策略累计收益对比
ax1 = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, (symbol, data) in enumerate(strategies.items()):
    ax1.plot(data['dates'], data['benchmark'] - 1, '--', alpha=0.5, label=f'{symbol} 基准')
    ax1.plot(data['dates'], data['s2_rf_prob'] - 1, '-', color=colors[i % len(colors)], linewidth=2, label=f'{symbol} RF概率')
ax1.set_title('累计收益率对比 (RF概率策略 vs 基准)')
ax1.set_xlabel('日期')
ax1.set_ylabel('累计收益率')
ax1.legend(fontsize=7, loc='upper left')
ax1.grid(True, alpha=0.3)

# 子图2: 特征重要性
ax2 = axes[0, 1]
top_features = importance_df.head(12)
ax2.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'].values, fontsize=8)
ax2.set_title('Top 12 特征重要性 (随机森林)')
ax2.set_xlabel('重要性')
ax2.invert_yaxis()

# 子图3: 混淆矩阵
ax3 = axes[0, 2]
cm = confusion_matrix(y_test_clf, rf_pred)
im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax3.set_title('混淆矩阵 (随机森林)')
tick_marks = np.arange(2)
ax3.set_xticks(tick_marks)
ax3.set_yticks(tick_marks)
ax3.set_xticklabels(['跌(0)', '涨(1)'])
ax3.set_yticklabels(['跌(0)', '涨(1)'])
ax3.set_ylabel('真实标签')
ax3.set_xlabel('预测标签')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

# 子图4: 预测概率分布
ax4 = axes[1, 0]
ax4.hist(rf_prob[y_test_clf == 0], bins=20, alpha=0.6, label='实际跌', color='red')
ax4.hist(rf_prob[y_test_clf == 1], bins=20, alpha=0.6, label='实际涨', color='green')
ax4.set_title('预测上涨概率分布')
ax4.set_xlabel('预测上涨概率')
ax4.set_ylabel('频数')
ax4.legend()
ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

# 子图5: 预测收益率 vs 实际收益率散点图
ax5 = axes[1, 1]
ax5.scatter(y_test_reg, rf_reg_pred, alpha=0.5, s=20)
ax5.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], 'r--', lw=2)
ax5.set_title('预测收益率 vs 实际收益率')
ax5.set_xlabel('实际次日收益率')
ax5.set_ylabel('预测次日收益率')
ax5.grid(True, alpha=0.3)

# 子图6: 各策略平均表现
ax6 = axes[1, 2]
if all_returns:
    summary = pd.DataFrame(all_returns).groupby('strategy').mean()
    x_pos = np.arange(len(summary))
    ax6.bar(x_pos - 0.2, summary['total_return'] * 100, 0.4, label='总收益率(%)', color='steelblue')
    ax6.bar(x_pos + 0.2, summary['max_drawdown'] * 100, 0.4, label='最大回撤(%)', color='coral')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(summary.index, rotation=15, ha='right')
    ax6.set_title('策略平均表现')
    ax6.set_ylabel('百分比 (%)')
    ax6.legend()
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\HY\PycharmProjects\QuanTrade\quant_model_backtest.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存: C:\Users\HY\PycharmProjects\QuanTrade\quant_model_backtest.png")

# ==================== 9. 保存预测结果到数据库 ====================
conn = sqlite3.connect(db_path)
test_results[['trade_date', 'symbol', 'target_direction', 'target_next_day_return',
              'rf_pred_direction', 'rf_prob_up', 'lr_pred_direction', 'lr_prob_up',
              'rf_pred_return']].to_sql('model_predictions', conn, if_exists='replace', index=False)
conn.close()
print(f"预测结果已保存到数据库表: model_predictions")

print("\n" + "=" * 60)
print("【分析完成】")
print("=" * 60)
