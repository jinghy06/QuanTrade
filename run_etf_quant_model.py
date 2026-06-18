"""
ETF量化预测模型与回测系统
针对ETF: 562500.SH, 159382.SZ, 588790.SH, 159241.SZ, 588200.SH
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载 ====================
db_path = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
conn = sqlite3.connect(db_path)

etf_list = ['562500.SH', '159382.SZ', '588790.SH', '159241.SZ', '588200.SH']

# 读取所有ETF日K数据
all_data = []
for etf in etf_list:
    df = pd.read_sql_query(f"SELECT * FROM daily_prices WHERE symbol='{etf}' ORDER BY trade_date", conn)
    all_data.append(df)
    print(f"{etf}: {len(df)} 条记录, 日期 {df['trade_date'].min()} ~ {df['trade_date'].max()}")

conn.close()

# 合并
df_all = pd.concat(all_data, ignore_index=True)
df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
df_all = df_all.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

print(f"\n总数据量: {len(df_all)} 条")
print(f"ETF数量: {df_all['symbol'].nunique()}")

# ==================== 2. 特征工程 ====================
def calc_features(group):
    """为单只股票计算技术指标特征"""
    g = group.sort_values('trade_date').copy()
    
    # 基础价格特征
    g['return_1d'] = g['pct_change'] / 100  # 当日收益率（小数形式）
    g['return_5d'] = g['close'].pct_change(5)
    g['return_10d'] = g['close'].pct_change(10)
    g['return_20d'] = g['close'].pct_change(20)
    
    # 移动平均线
    for window in [5, 10, 20, 60]:
        g[f'ma_{window}'] = g['close'].rolling(window=window).mean()
        g[f'ma_dist_{window}'] = (g['close'] - g[f'ma_{window}']) / g[f'ma_{window}']
    
    # MA排列
    g['ma_alignment'] = np.where(
        (g['ma_5'] > g['ma_10']) & (g['ma_10'] > g['ma_20']), 1,
        np.where((g['ma_5'] < g['ma_10']) & (g['ma_10'] < g['ma_20']), -1, 0)
    )
    
    # 价格位置（在N日高低点中的位置）
    for window in [20, 60]:
        high = g['high'].rolling(window=window).max()
        low = g['low'].rolling(window=window).min()
        g[f'price_position_{window}'] = (g['close'] - low) / (high - low)
    
    # RSI
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    g['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = g['close'].ewm(span=12, adjust=False).mean()
    ema_26 = g['close'].ewm(span=26, adjust=False).mean()
    g['macd_dif'] = ema_12 - ema_26
    g['macd_dea'] = g['macd_dif'].ewm(span=9, adjust=False).mean()
    g['macd_hist'] = g['macd_dif'] - g['macd_dea']
    
    # 波动率
    g['std_5d'] = g['return_1d'].rolling(5).std()
    g['std_20d'] = g['return_1d'].rolling(20).std()
    g['atr_14'] = (g['high'] - g['low']).rolling(14).mean() / g['close']
    
    # 成交量特征
    g['volume_ma5'] = g['volume'].rolling(5).mean()
    g['volume_ma20'] = g['volume'].rolling(20).mean()
    g['volume_ratio'] = g['volume'] / g['volume_ma5']
    g['vol_percentile'] = g['volume'].rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 60 else np.nan, raw=False)
    
    # 振幅
    g['amplitude_pct'] = g['amplitude'] / 100
    
    # 换手率
    g['turnover_ma5'] = g['turnover'].rolling(5).mean()
    
    # 趋势斜率（20日收盘价线性回归斜率）
    def linear_slope(x):
        if len(x) < 10 or np.all(np.isnan(x)):
            return np.nan
        x_idx = np.arange(len(x))
        mask = ~np.isnan(x)
        if mask.sum() < 10:
            return np.nan
        return np.polyfit(x_idx[mask], x[mask], 1)[0] / np.mean(x[mask])
    
    g['trend_slope'] = g['close'].rolling(20).apply(linear_slope, raw=True)
    
    # 支撑/阻力距离
    g['dist_to_support'] = (g['close'] - g['low'].rolling(20).min()) / g['close']
    g['dist_to_resistance'] = (g['high'].rolling(20).max() - g['close']) / g['close']
    
    # OBV (On Balance Volume)
    obv = [0]
    for i in range(1, len(g)):
        if g['close'].iloc[i] > g['close'].iloc[i-1]:
            obv.append(obv[-1] + g['volume'].iloc[i])
        elif g['close'].iloc[i] < g['close'].iloc[i-1]:
            obv.append(obv[-1] - g['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    g['obv'] = obv
    g['obv_ma20'] = g['obv'].rolling(20).mean()
    g['obv_ratio'] = g['obv'] / g['obv_ma20']
    
    # 目标变量: 次日涨跌方向和收益率
    g['target_next_day_return'] = g['close'].shift(-1) / g['close'] - 1
    g['target_direction'] = (g['target_next_day_return'] > 0).astype(int)
    
    return g

# 按股票分组计算特征
print("\n正在计算技术指标特征...")
df_features_list = []
for symbol, group in df_all.groupby('symbol'):
    df_features_list.append(calc_features(group))
df_features = pd.concat(df_features_list, ignore_index=True)

print(f"\n总数据量: {len(df_features)} 条")
print(f"ETF数量: {df_features['symbol'].nunique()}")

# 选择特征列
feature_cols = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'ma_dist_5', 'ma_dist_10', 'ma_dist_20', 'ma_dist_60',
    'ma_alignment', 'price_position_20', 'price_position_60',
    'rsi_14', 'macd_dif', 'macd_dea', 'macd_hist',
    'std_5d', 'std_20d', 'atr_14',
    'volume_ratio', 'vol_percentile',
    'amplitude_pct', 'turnover_ma5',
    'trend_slope', 'dist_to_support', 'dist_to_resistance',
    'obv_ratio'
]

# 去除有缺失值的行（主要是前期MA计算需要的历史数据）
df_model = df_features.dropna(subset=feature_cols + ['target_direction']).copy()
print(f"\n去除缺失值后可用数据: {len(df_model)} 条")
print(f"各ETF数据量:")
print(df_model['symbol'].value_counts())

# ==================== 3. 时间序列划分训练集/测试集 ====================
# 全局按日期排序，前75%训练，后25%测试（时间序列划分）
df_model = df_model.sort_values('trade_date').reset_index(drop=True)

split_idx = int(len(df_model) * 0.75)
train_df = df_model.iloc[:split_idx].copy()
test_df = df_model.iloc[split_idx:].copy()

print(f"\n时间序列划分:")
print(f"  - 训练集: {len(train_df)} 条 ({train_df['trade_date'].min().strftime('%Y-%m-%d')} ~ {train_df['trade_date'].max().strftime('%Y-%m-%d')})")
print(f"  - 测试集: {len(test_df)} 条 ({test_df['trade_date'].min().strftime('%Y-%m-%d')} ~ {test_df['trade_date'].max().strftime('%Y-%m-%d')})")

X_train = train_df[feature_cols].values
y_train_clf = train_df['target_direction'].values
y_train_reg = train_df['target_next_day_return'].values

X_test = test_df[feature_cols].values
y_test_clf = test_df['target_direction'].values
y_test_reg = test_df['target_next_day_return'].values

# 标准化
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
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train_clf)
rf_pred = rf_clf.predict(X_test)
rf_prob = rf_clf.predict_proba(X_test)[:, 1]

# 回归模型需要额外去除 target_next_day_return 的 NaN
mask_reg = ~np.isnan(y_train_reg)
X_train_reg = X_train[mask_reg]
y_train_reg_clean = y_train_reg[mask_reg]

mask_reg_test = ~np.isnan(y_test_reg)
X_test_reg = X_test[mask_reg_test]
y_test_reg_clean = y_test_reg[mask_reg_test]

# 4.2 随机森林回归（预测收益率）
print("2. 训练随机森林回归（预测收益率）...")
rf_reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train_reg, y_train_reg_clean)
rf_reg_pred = rf_reg.predict(X_test_reg)

# ==================== 5. 模型评估 ====================
print("\n" + "=" * 60)
print("【模型评估 - 分类任务（预测涨跌方向）】")
print("=" * 60)

print(f"\n测试集样本数: {len(y_test_clf)}")
print(f"实际上涨比例: {np.mean(y_test_clf):.4f}")
print(f"预测上涨比例: {np.mean(rf_pred):.4f}")

print(f"\n>>> 随机森林分类器")
print(f"  准确率 (Accuracy):  {accuracy_score(y_test_clf, rf_pred):.4f}")
print(f"  精确率 (Precision): {precision_score(y_test_clf, rf_pred):.4f}")
print(f"  召回率 (Recall):    {recall_score(y_test_clf, rf_pred):.4f}")
print(f"  F1分数:             {f1_score(y_test_clf, rf_pred):.4f}")

# 特征重要性
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n>>> Top 15 重要特征:")
for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:20s} {row['importance']:.4f}")

# 回归评估
print("\n" + "=" * 60)
print("【模型评估 - 回归任务（预测次日收益率）】")
print("=" * 60)
mse = mean_squared_error(y_test_reg_clean, rf_reg_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg_clean, rf_reg_pred)
print(f"\n>>> 随机森林回归")
print(f"  MSE:  {mse:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  R2:   {r2:.4f}")
print(f"  实际收益率均值: {np.mean(y_test_reg_clean):.6f}")
print(f"  预测收益率均值: {np.mean(rf_reg_pred):.6f}")

# 预测与实际相关性
corr = np.corrcoef(y_test_reg_clean, rf_reg_pred)[0, 1]
print(f"  预测与实际相关系数: {corr:.4f}")

# ==================== 6. 量化回测框架 ====================
print("\n" + "=" * 60)
print("【量化回测框架】")
print("=" * 60)

# 合并测试集数据与预测结果
test_results = test_df.copy()
test_results['rf_pred_direction'] = rf_pred
test_results['rf_prob_up'] = rf_prob

# 回归预测只针对有次日收益率的样本
reg_pred_full = np.full(len(test_results), np.nan)
mask_reg_test_full = ~np.isnan(test_results['target_next_day_return'].values)
reg_pred_full[mask_reg_test_full] = rf_reg_pred
test_results['rf_pred_return'] = reg_pred_full

# 策略定义:
# 策略1: RF分类信号 - 预测涨则满仓，预测跌则空仓
# 策略2: RF概率阈值 - 上涨概率>0.55时买入
# 策略3: 回归策略 - 预测收益率>0时买入
# 策略4: 概率加权 - 按上涨概率分配仓位
# 基准: 每天满仓持有

strategies = {}

for symbol in test_results['symbol'].unique():
    sym_data = test_results[test_results['symbol'] == symbol].sort_values('trade_date').reset_index(drop=True)
    if len(sym_data) < 10:
        continue

    # 过滤掉没有次日收益率的数据（最后一天）
    valid_mask = ~sym_data['target_next_day_return'].isna()
    sym_data_valid = sym_data[valid_mask].reset_index(drop=True)
    if len(sym_data_valid) < 5:
        continue

    actual_returns = sym_data_valid['target_next_day_return'].values
    dates = sym_data_valid['trade_date'].values

    # 基准: 每天满仓持有
    benchmark_cum = np.cumprod(1 + actual_returns)

    # 策略1: RF分类信号 (满仓/空仓)
    s1_signals = sym_data_valid['rf_pred_direction'].values
    s1_returns = actual_returns * s1_signals
    s1_cum = np.cumprod(1 + s1_returns)

    # 策略2: RF概率阈值 > 0.55
    s2_signals = (sym_data_valid['rf_prob_up'].values > 0.55).astype(int)
    s2_returns = actual_returns * s2_signals
    s2_cum = np.cumprod(1 + s2_returns)

    # 策略3: 回归预测 > 0
    s3_signals = (sym_data_valid['rf_pred_return'].values > 0).astype(int)
    s3_returns = actual_returns * s3_signals
    s3_cum = np.cumprod(1 + s3_returns)

    # 策略4: 概率加权仓位 (0~1之间)
    s4_weights = sym_data_valid['rf_prob_up'].values
    s4_returns = actual_returns * s4_weights
    s4_cum = np.cumprod(1 + s4_returns)

    strategies[symbol] = {
        'dates': dates,
        'actual': actual_returns,
        'benchmark': benchmark_cum,
        's1_rf_class': s1_cum,
        's2_rf_prob': s2_cum,
        's3_rf_reg': s3_cum,
        's4_prob_weight': s4_cum,
        's1_signals': s1_signals,
        's2_signals': s2_signals,
        's3_signals': s3_signals,
        's4_weights': s4_weights
    }

# ==================== 7. 回测结果汇总 ====================
print("\n>>> 各ETF策略回测结果:\n")
print(f"{'ETF代码':<12} {'策略':<18} {'总收益率':<12} {'年化收益率':<12} {'最大回撤':<12} {'夏普比率':<12} {'胜率':<10} {'交易次数':<10}")
print("-" * 100)

all_returns = []

for symbol, data in strategies.items():
    for s_name, s_key, sig_key in [
        ('基准(持有)', 'benchmark', None),
        ('RF分类信号', 's1_rf_class', 's1_signals'),
        ('RF概率>0.55', 's2_rf_prob', 's2_signals'),
        ('RF回归>0', 's3_rf_reg', 's3_signals'),
        ('概率加权', 's4_prob_weight', 's4_weights')
    ]:
        cum = data[s_key]
        total_ret = cum[-1] - 1
        n_days = len(cum)
        annual_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 and total_ret > -1 else -1
        
        # 最大回撤
        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_dd = np.min(drawdown)
        
        # 夏普
        daily_rets = np.diff(cum) / cum[:-1]
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
        
        # 胜率
        if sig_key:
            if s_name == '概率加权':
                # 加权策略的胜率：正收益天数 / 总天数
                win_rate = np.sum(daily_rets > 0) / len(daily_rets) if len(daily_rets) > 0 else 0
                trades = n_days
            else:
                signals = data[sig_key]
                traded_rets = data['actual'] * signals
                traded_days = signals > 0
                win_rate = np.sum(traded_rets[traded_days] > 0) / np.sum(traded_days) if np.sum(traded_days) > 0 else 0
                trades = np.sum(signals)
        else:
            win_rate = np.sum(data['actual'] > 0) / len(data['actual'])
            trades = n_days

        print(f"{symbol:<12} {s_name:<18} {total_ret*100:>10.2f}% {annual_ret*100:>10.2f}% {max_dd*100:>10.2f}% {sharpe:>10.2f} {win_rate*100:>8.1f}% {trades:>8}")

        if s_name != '基准(持有)':
            all_returns.append({
                'symbol': symbol,
                'strategy': s_name,
                'total_return': total_ret,
                'annual_return': annual_ret,
                'max_drawdown': max_dd,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'trades': trades
            })

# ==================== 8. 可视化 ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('ETF量化预测模型 - 回测结果', fontsize=16, fontweight='bold')

# 子图1: 各ETF累计收益对比 (RF概率策略 vs 基准)
ax1 = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (symbol, data) in enumerate(strategies.items()):
    ax1.plot(data['dates'], data['benchmark'] - 1, '--', alpha=0.4, color=colors[i % len(colors)])
    ax1.plot(data['dates'], data['s2_rf_prob'] - 1, '-', color=colors[i % len(colors)], linewidth=2, label=symbol)
ax1.set_title('累计收益率: RF概率>0.55策略 vs 基准')
ax1.set_xlabel('日期')
ax1.set_ylabel('累计收益率')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=30)

# 子图2: 特征重要性
ax2 = axes[0, 1]
top_features = importance_df.head(15)
ax2.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'].values, fontsize=8)
ax2.set_title('Top 15 特征重要性')
ax2.set_xlabel('重要性')
ax2.invert_yaxis()

# 子图3: 混淆矩阵
ax3 = axes[0, 2]
cm = confusion_matrix(y_test_clf, rf_pred)
im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax3.set_title('混淆矩阵')
tick_marks = np.arange(2)
ax3.set_xticks(tick_marks)
ax3.set_yticks(tick_marks)
ax3.set_xticklabels(['跌(0)', '涨(1)'])
ax3.set_yticklabels(['跌(0)', '涨(1)'])
ax3.set_ylabel('真实标签')
ax3.set_xlabel('预测标签')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", 
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)

# 子图4: 预测概率分布
ax4 = axes[1, 0]
ax4.hist(rf_prob[y_test_clf == 0], bins=20, alpha=0.6, label='实际跌', color='red', density=True)
ax4.hist(rf_prob[y_test_clf == 1], bins=20, alpha=0.6, label='实际涨', color='green', density=True)
ax4.set_title('预测上涨概率分布')
ax4.set_xlabel('预测上涨概率')
ax4.set_ylabel('密度')
ax4.legend()
ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

# 子图5: 预测收益率 vs 实际收益率
ax5 = axes[1, 1]
ax5.scatter(y_test_reg_clean, rf_reg_pred, alpha=0.5, s=25, c='steelblue', edgecolors='none')
ax5.plot([min(y_test_reg_clean), max(y_test_reg_clean)], [min(y_test_reg_clean), max(y_test_reg_clean)], 'r--', lw=2, alpha=0.7)
ax5.set_title(f'预测 vs 实际收益率 (R2={r2:.3f})')
ax5.set_xlabel('实际次日收益率')
ax5.set_ylabel('预测次日收益率')
ax5.grid(True, alpha=0.3)

# 子图6: 各策略平均表现对比
ax6 = axes[1, 2]
if all_returns:
    summary = pd.DataFrame(all_returns).groupby('strategy')[['total_return', 'annual_return', 'max_drawdown', 'sharpe', 'win_rate']].mean()
    x_pos = np.arange(len(summary))
    width = 0.25
    ax6.bar(x_pos - width, summary['total_return'] * 100, width, label='总收益率(%)', color='steelblue')
    ax6.bar(x_pos, summary['annual_return'] * 100, width, label='年化收益率(%)', color='seagreen')
    ax6.bar(x_pos + width, summary['max_drawdown'] * 100, width, label='最大回撤(%)', color='coral')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(summary.index, rotation=15, ha='right', fontsize=9)
    ax6.set_title('策略平均表现对比')
    ax6.set_ylabel('百分比 (%)')
    ax6.legend(fontsize=8)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:/Users/HY/PycharmProjects/QuanTrade/etf_quant_backtest.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存: C:/Users/HY/PycharmProjects/QuanTrade/etf_quant_backtest.png")

# ==================== 9. 保存结果到数据库 ====================
conn = sqlite3.connect(db_path)
test_results[['trade_date', 'symbol', 'close', 'target_direction', 'target_next_day_return',
              'rf_pred_direction', 'rf_prob_up', 'rf_pred_return']].to_sql(
    'etf_model_predictions', conn, if_exists='replace', index=False)
conn.close()
print(f"预测结果已保存到数据库表: etf_model_predictions")

# ==================== 10. 按ETF详细分析 ====================
print("\n" + "=" * 60)
print("【各ETF详细回测分析】")
print("=" * 60)

for symbol, data in strategies.items():
    print(f"\n>>> {symbol}")
    print(f"  测试期交易天数: {len(data['actual'])}")
    print(f"  基准总收益:     {(data['benchmark'][-1]-1)*100:.2f}%")
    print(f"  RF分类策略收益: {(data['s1_rf_class'][-1]-1)*100:.2f}%")
    print(f"  RF概率策略收益: {(data['s2_rf_prob'][-1]-1)*100:.2f}%")
    print(f"  RF回归策略收益: {(data['s3_rf_reg'][-1]-1)*100:.2f}%")
    print(f"  概率加权收益:   {(data['s4_prob_weight'][-1]-1)*100:.2f}%")
    
    # 计算超额收益
    excess_s1 = (data['s1_rf_class'][-1] - data['benchmark'][-1]) * 100
    excess_s2 = (data['s2_rf_prob'][-1] - data['benchmark'][-1]) * 100
    print(f"  分类策略超额:   {excess_s1:+.2f}%")
    print(f"  概率策略超额:   {excess_s2:+.2f}%")

print("\n" + "=" * 60)
print("【分析完成】")
print("=" * 60)
