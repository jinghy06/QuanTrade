"""
Phase 2+3 精简版: 策略体系升级
- 减少长历史依赖的特征，适应有限数据
- 趋势过滤 + Kelly仓位 + 动态阈值
- 多策略融合回测
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("【Phase 2+3: 策略体系升级 (精简版)】")
print("=" * 70)

# ==================== 1. 数据加载 ====================
db_path = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
conn = sqlite3.connect(db_path)

etf_list = ['562500.SH', '159382.SZ', '588790.SH', '159241.SZ', '588200.SH']

all_data = []
for etf in etf_list:
    df = pd.read_sql_query(f"SELECT * FROM daily_prices WHERE symbol='{etf}' ORDER BY trade_date", conn)
    all_data.append(df)
    print(f"{etf}: {len(df)} 条")

conn.close()

df_all = pd.concat(all_data, ignore_index=True)
df_all['trade_date'] = pd.to_datetime(df_all['trade_date'], format='mixed')
df_all = df_all.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

print(f"\n总数据量: {len(df_all)} 条")

# ==================== 2. 精简特征工程 (40+ 特征) ====================
print("\n" + "=" * 70)
print("【2. 精简特征工程 (40+ Alpha因子)】")
print("=" * 70)

def calc_features_compact(group):
    """精简版特征工程 - 减少长历史依赖"""
    g = group.sort_values('trade_date').copy()
    n = len(g)
    if n < 20:
        return g

    # 基础价格特征
    g['return_1d'] = g['close'].pct_change()
    for w in [2, 3, 5, 10]:
        g[f'return_{w}d'] = g['close'].pct_change(w)

    # 移动平均线 (最多20日)
    for w in [3, 5, 10, 20]:
        g[f'ma_{w}'] = g['close'].rolling(w).mean()
        g[f'ma_dist_{w}'] = (g['close'] - g[f'ma_{w}']) / g[f'ma_{w}']
        g[f'ema_{w}'] = g['close'].ewm(span=w, adjust=False).mean()

    g['ma_bullish'] = (g['close'] > g['ma_5']) & (g['ma_5'] > g['ma_10']) & (g['ma_10'] > g['ma_20'])
    g['ma_bearish'] = (g['close'] < g['ma_5']) & (g['ma_5'] < g['ma_10']) & (g['ma_10'] < g['ma_20'])

    # 布林带 (10日)
    ma10 = g['close'].rolling(10).mean()
    std10 = g['close'].rolling(10).std()
    g['boll_upper'] = ma10 + 2 * std10
    g['boll_lower'] = ma10 - 2 * std10
    g['boll_position'] = (g['close'] - g['boll_lower']) / (g['boll_upper'] - g['boll_lower'])

    # RSI (14日)
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    g['rsi_14'] = 100 - (100 / (1 + rs))
    g['rsi_oversold'] = (g['rsi_14'] < 30).astype(int)
    g['rsi_overbought'] = (g['rsi_14'] > 70).astype(int)

    # MACD
    ema_12 = g['close'].ewm(span=12, adjust=False).mean()
    ema_26 = g['close'].ewm(span=26, adjust=False).mean()
    g['macd_dif'] = ema_12 - ema_26
    g['macd_dea'] = g['macd_dif'].ewm(span=9, adjust=False).mean()
    g['macd_hist'] = g['macd_dif'] - g['macd_dea']

    # KDJ
    low_min = g['low'].rolling(9).min()
    high_max = g['high'].rolling(9).max()
    rsv = (g['close'] - low_min) / (high_max - low_min) * 100
    g['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    g['kdj_d'] = g['kdj_k'].ewm(com=2, adjust=False).mean()
    g['kdj_j'] = 3 * g['kdj_k'] - 2 * g['kdj_d']

    # 波动率
    for w in [5, 10]:
        g[f'std_{w}d'] = g['return_1d'].rolling(w).std()
    g['atr_14'] = (g['high'] - g['low']).rolling(14).mean() / g['close']

    # 成交量
    g['volume_ma_5'] = g['volume'].rolling(5).mean()
    g['volume_ratio'] = g['volume'] / g['volume_ma_5']
    g['amount_ma5'] = g['amount'].rolling(5).mean()
    g['amount_ratio'] = g['amount'] / g['amount_ma5']

    # 价格位置
    high20 = g['high'].rolling(20).max()
    low20 = g['low'].rolling(20).min()
    g['price_position_20'] = (g['close'] - low20) / (high20 - low20)

    # 趋势斜率 (10日)
    def linear_slope(x):
        if len(x) < 5 or np.all(np.isnan(x)):
            return np.nan
        xi = np.arange(len(x))
        mask = ~np.isnan(x)
        if mask.sum() < 5:
            return np.nan
        return np.polyfit(xi[mask], x[mask], 1)[0] / np.mean(x[mask])
    g['trend_slope_10'] = g['close'].rolling(10).apply(linear_slope, raw=True)

    # OBV
    obv = [0]
    for i in range(1, len(g)):
        if g['close'].iloc[i] > g['close'].iloc[i-1]:
            obv.append(obv[-1] + g['volume'].iloc[i])
        elif g['close'].iloc[i] < g['close'].iloc[i-1]:
            obv.append(obv[-1] - g['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    g['obv'] = obv
    g['obv_ma10'] = g['obv'].rolling(10).mean()
    g['obv_ratio'] = g['obv'] / g['obv_ma10']

    # 价格行为
    g['body_pct'] = (g['close'] - g['open']) / g['open']
    g['upper_shadow'] = (g['high'] - g[['close', 'open']].max(axis=1)) / g['close']
    g['lower_shadow'] = (g[['close', 'open']].min(axis=1) - g['low']) / g['close']

    # 时间特征
    g['dayofweek'] = g['trade_date'].dt.dayofweek
    g['month'] = g['trade_date'].dt.month
    g['is_week_end'] = (g['dayofweek'] == 4).astype(int)

    # 目标变量
    g['target_next_day_return'] = g['close'].shift(-1) / g['close'] - 1
    g['target_direction'] = (g['target_next_day_return'] > 0).astype(int)

    return g

print("\n计算精简特征...")
df_features = []
for symbol, group in df_all.groupby('symbol'):
    df_features.append(calc_features_compact(group))
df_features = pd.concat(df_features, ignore_index=True)

exclude_cols = ['trade_date', 'symbol', 'created_at',
                'target_next_day_return', 'target_direction',
                'ma_3', 'ma_5', 'ma_10', 'ma_20',
                'ema_3', 'ema_5', 'ema_10', 'ema_20',
                'boll_upper', 'boll_lower',
                'volume_ma_5', 'amount_ma5',
                'obv', 'obv_ma10',
                'low_min', 'high_max', 'rsv']

feature_cols = [c for c in df_features.columns if c not in exclude_cols]
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_features[c])]

df_model = df_features.dropna(subset=feature_cols + ['target_direction']).copy()
print(f"\n去除缺失值后可用数据: {len(df_model)} 条")
print(f"特征数量: {len(feature_cols)}")
print(f"各ETF数据量:")
print(df_model['symbol'].value_counts())

# ==================== 3. 训练与预测 ====================
print("\n" + "=" * 70)
print("【3. 模型训练与预测】")
print("=" * 70)

df_model = df_model.sort_values('trade_date').reset_index(drop=True)

# 时间序列划分: 75%训练 / 25%测试
split_idx = int(len(df_model) * 0.75)
train_df = df_model.iloc[:split_idx].copy()
test_df = df_model.iloc[split_idx:].copy()

print(f"训练集: {len(train_df)} 条 ({train_df['trade_date'].min().strftime('%Y-%m-%d')} ~ {train_df['trade_date'].max().strftime('%Y-%m-%d')})")
print(f"测试集: {len(test_df)} 条 ({test_df['trade_date'].min().strftime('%Y-%m-%d')} ~ {test_df['trade_date'].max().strftime('%Y-%m-%d')})")

X_train = train_df[feature_cols].values
y_train_clf = train_df['target_direction'].values
y_train_reg = train_df['target_next_day_return'].values

X_test = test_df[feature_cols].values
y_test_clf = test_df['target_direction'].values
y_test_reg = test_df['target_next_day_return'].values

valid_train = ~np.isnan(y_train_reg)
valid_test = ~np.isnan(y_test_reg)
X_train_reg = X_train[valid_train]
y_train_reg_clean = y_train_reg[valid_train]
X_test_reg = X_test[valid_test]
y_test_reg_clean = y_test_reg[valid_test]

# 训练模型
models = {}
if LGB_AVAILABLE:
    models['LightGBM'] = {
        'clf': lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbose=-1),
        'reg': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbose=-1)
    }

models['RandomForest'] = {
    'clf': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1),
    'reg': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)
}

predictions = {}
for name, m in models.items():
    print(f"\n训练 {name}...")
    m['clf'].fit(X_train, y_train_clf)
    pred_clf = m['clf'].predict(X_test)
    prob_clf = m['clf'].predict_proba(X_test)[:, 1]

    m['reg'].fit(X_train_reg, y_train_reg_clean)
    pred_reg = m['reg'].predict(X_test_reg)

    predictions[name] = {
        'pred_clf': pred_clf, 'prob_clf': prob_clf,
        'pred_reg': pred_reg,
        'acc': accuracy_score(y_test_clf, pred_clf),
        'prec': precision_score(y_test_clf, pred_clf, zero_division=0),
        'rec': recall_score(y_test_clf, pred_clf, zero_division=0),
        'f1': f1_score(y_test_clf, pred_clf, zero_division=0),
    }
    try:
        predictions[name]['auc'] = roc_auc_score(y_test_clf, prob_clf)
    except:
        predictions[name]['auc'] = 0.5

    print(f"  准确率: {predictions[name]['acc']:.4f} | 精确率: {predictions[name]['prec']:.4f} | 召回率: {predictions[name]['rec']:.4f} | F1: {predictions[name]['f1']:.4f} | AUC: {predictions[name]['auc']:.4f}")

# 融合模型概率
probs = [predictions[name]['prob_clf'] for name in predictions.keys()]
fusion_prob = np.mean(probs, axis=0)
fusion_pred = (fusion_prob > 0.5).astype(int)

print(f"\n>>> 融合模型")
print(f"  准确率: {accuracy_score(y_test_clf, fusion_pred):.4f}")
print(f"  AUC: {roc_auc_score(y_test_clf, fusion_prob):.4f}")

# 特征重要性
if LGB_AVAILABLE:
    imp_model = models['LightGBM']['clf']
else:
    imp_model = models['RandomForest']['clf']

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': imp_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 重要特征:")
for _, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:25s} {row['importance']:.1f}")

# ==================== 4. 策略体系 ====================
print("\n" + "=" * 70)
print("【4. 多策略融合: 趋势过滤 + Kelly仓位 + 动态阈值】")
print("=" * 70)

test_df['fusion_prob'] = fusion_prob
test_df['fusion_pred'] = fusion_pred

# 对齐回归预测
reg_pred_full = np.full(len(test_df), np.nan)
reg_pred_full[valid_test] = np.mean([predictions[name]['pred_reg'] for name in predictions.keys()], axis=0)
test_df['fusion_reg'] = reg_pred_full

# 策略1: 基础融合
test_df['signal_base'] = test_df['fusion_pred']

# 策略2: 趋势过滤 (MA多头排列降低阈值到0.45)
def trend_signal(row):
    if row['ma_bullish']:
        return 1 if row['fusion_prob'] > 0.45 else 0
    elif row['ma_bearish']:
        return 1 if row['fusion_prob'] > 0.60 else 0
    return 1 if row['fusion_prob'] > 0.50 else 0

test_df['signal_trend'] = test_df.apply(trend_signal, axis=1)

# 策略3: 动态阈值 (基于近期胜率)
test_df = test_df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
for symbol in test_df['symbol'].unique():
    mask = test_df['symbol'] == symbol
    sym = test_df.loc[mask].copy()
    correct = (sym['fusion_pred'] == sym['target_direction']).astype(float)
    sym['rolling_acc'] = correct.rolling(10, min_periods=3).mean()
    sym['dynamic_threshold'] = 0.50 - (sym['rolling_acc'] - 0.5) * 0.2
    sym['dynamic_threshold'] = sym['dynamic_threshold'].fillna(0.50)
    sym['signal_dynamic'] = (sym['fusion_prob'] > sym['dynamic_threshold']).astype(int)
    test_df.loc[mask, 'rolling_acc'] = sym['rolling_acc'].values
    test_df.loc[mask, 'dynamic_threshold'] = sym['dynamic_threshold'].values
    test_df.loc[mask, 'signal_dynamic'] = sym['signal_dynamic'].values

# 策略4: Kelly仓位
for symbol in test_df['symbol'].unique():
    mask = test_df['symbol'] == symbol
    sym = test_df.loc[mask].copy().sort_values('trade_date').reset_index(drop=True)
    positions = []
    for i in range(len(sym)):
        if i < 10:
            positions.append(0.5)
            continue
        recent = sym.iloc[max(0, i-10):i]
        wins = recent[recent['fusion_pred'] == recent['target_direction']]
        losses = recent[recent['fusion_pred'] != recent['target_direction']]
        p = len(wins) / len(recent) if len(recent) > 0 else 0.5
        win_ret = wins['target_next_day_return'].values
        loss_ret = losses['target_next_day_return'].values
        avg_win = np.mean(win_ret[win_ret > 0]) if len(win_ret[win_ret > 0]) > 0 else 0.01
        avg_loss = abs(np.mean(loss_ret[loss_ret < 0])) if len(loss_ret[loss_ret < 0]) > 0 else 0.01
        b = avg_win / avg_loss if avg_loss > 0 else 1
        q = 1 - p
        kelly = (p * b - q) / b if b > 0 else 0
        positions.append(max(0, min(1, kelly * 0.5)))
    test_df.loc[mask, 'kelly_position'] = positions

test_df['signal_kelly'] = (test_df['kelly_position'] > 0).astype(int)

# 策略5: 综合 (趋势信号 * Kelly仓位)
test_df['combined_position'] = test_df['signal_trend'] * test_df['kelly_position']
test_df['signal_combined'] = (test_df['combined_position'] > 0).astype(int)

print(f"\n各策略信号统计:")
for s in ['signal_base', 'signal_trend', 'signal_dynamic', 'signal_kelly', 'signal_combined']:
    print(f"  {s}: 买入比例 {test_df[s].mean()*100:.1f}%")

# ==================== 5. 回测 ====================
print("\n" + "=" * 70)
print("【5. 增强回测引擎】")
print("=" * 70)

FEE_RATE = 0.0001
SLIPPAGE = 0.0001
STOP_LOSS_ATR = 2.0

strategies_final = {}

for symbol in test_df['symbol'].unique():
    sym = test_df[test_df['symbol'] == symbol].sort_values('trade_date').reset_index(drop=True)
    if len(sym) < 5:
        continue

    actual = sym['target_next_day_return'].values
    dates = sym['trade_date'].values
    atrs = sym['atr_14'].fillna(0.02).values
    n = len(actual)

    benchmark = np.cumprod(1 + actual)

    results = {'dates': dates, 'actual': actual, 'benchmark': benchmark}

    for sig_col, pos_col, name in [
        ('signal_base', None, '基础融合'),
        ('signal_trend', None, '趋势过滤'),
        ('signal_dynamic', None, '动态阈值'),
        ('signal_kelly', 'kelly_position', 'Kelly仓位'),
        ('signal_combined', 'combined_position', '综合策略'),
    ]:
        signals = sym[sig_col].values.astype(float)
        positions = sym[pos_col].values if pos_col else signals

        cum = [1.0]
        in_pos = False

        for i in range(n):
            signal = signals[i]
            position = positions[i]
            daily_ret = actual[i]
            atr = atrs[i]

            if signal == 1 and not in_pos:
                in_pos = True
                cost = FEE_RATE + SLIPPAGE
                cum[-1] *= (1 - cost)
            elif signal == 0 and in_pos:
                in_pos = False
                cost = FEE_RATE + SLIPPAGE
                cum[-1] *= (1 - cost)

            if in_pos:
                if daily_ret < -STOP_LOSS_ATR * atr:
                    daily_ret = -STOP_LOSS_ATR * atr
                    in_pos = False
                ret = daily_ret * position
                cum.append(cum[-1] * (1 + ret))
            else:
                cum.append(cum[-1])

        results[f's_{name}'] = np.array(cum[1:])

    strategies_final[symbol] = results

# 汇总
print(f"\n{'ETF':<12} {'策略':<12} {'总收益':<10} {'年化':<10} {'最大回撤':<10} {'夏普':<8} {'胜率':<8}")
print("-" * 85)

all_results = []
for symbol, data in strategies_final.items():
    for s_name, s_key in [
        ('基准', 'benchmark'), ('基础融合', 's_基础融合'), ('趋势过滤', 's_趋势过滤'),
        ('动态阈值', 's_动态阈值'), ('Kelly仓位', 's_Kelly仓位'), ('综合策略', 's_综合策略')
    ]:
        cum = data[s_key]
        total = cum[-1] - 1
        n_days = len(cum)
        annual = (1 + total) ** (252 / n_days) - 1 if n_days > 0 and total > -1 else -1
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = np.min(dd)
        daily_rets = np.diff(cum) / cum[:-1]
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
        win_rate = np.sum(daily_rets > 0) / len(daily_rets) if len(daily_rets) > 0 else 0

        print(f"{symbol:<12} {s_name:<12} {total*100:>8.2f}% {annual*100:>8.2f}% {max_dd*100:>8.2f}% {sharpe:>6.2f} {win_rate*100:>6.1f}%")

        if s_name != '基准':
            all_results.append({
                'symbol': symbol, 'strategy': s_name,
                'total_return': total, 'annual_return': annual,
                'max_drawdown': max_dd, 'sharpe': sharpe, 'win_rate': win_rate
            })

# ==================== 6. 可视化 ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Phase 2+3: 多策略融合 + Kelly仓位 + 趋势过滤', fontsize=14, fontweight='bold')

ax1 = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (symbol, data) in enumerate(strategies_final.items()):
    ax1.plot(data['dates'], data['benchmark'] - 1, '--', alpha=0.4, color=colors[i % len(colors)])
    ax1.plot(data['dates'], data['s_综合策略'] - 1, '-', color=colors[i % len(colors)], linewidth=2, label=symbol)
ax1.set_title('累计收益率: 综合策略 vs 基准')
ax1.set_xlabel('日期')
ax1.set_ylabel('累计收益率')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
top_features = importance_df.head(15)
ax2.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'].values, fontsize=8)
ax2.set_title('Top 15 特征重要性')
ax2.set_xlabel('重要性')
ax2.invert_yaxis()

ax3 = axes[0, 2]
cm = confusion_matrix(y_test_clf, fusion_pred)
im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax3.set_title('混淆矩阵 (融合模型)')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['跌(0)', '涨(1)'])
ax3.set_yticklabels(['跌(0)', '涨(1)'])
for i in range(2):
    for j in range(2):
        ax3.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)

ax4 = axes[1, 0]
ax4.hist(fusion_prob[y_test_clf == 0], bins=20, alpha=0.6, label='实际跌', color='red', density=True)
ax4.hist(fusion_prob[y_test_clf == 1], bins=20, alpha=0.6, label='实际涨', color='green', density=True)
ax4.set_title('融合模型预测概率分布')
ax4.set_xlabel('预测上涨概率')
ax4.set_ylabel('密度')
ax4.legend()
ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

ax5 = axes[1, 1]
if all_results:
    summary = pd.DataFrame(all_results).groupby('strategy')[['total_return', 'annual_return', 'max_drawdown', 'sharpe']].mean()
    x_pos = np.arange(len(summary))
    width = 0.2
    ax5.bar(x_pos - width, summary['total_return'] * 100, width, label='总收益率(%)', color='steelblue')
    ax5.bar(x_pos, summary['annual_return'] * 100, width, label='年化收益率(%)', color='seagreen')
    ax5.bar(x_pos + width, summary['max_drawdown'] * 100, width, label='最大回撤(%)', color='coral')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(summary.index, rotation=15, ha='right', fontsize=9)
    ax5.set_title('策略平均表现对比')
    ax5.set_ylabel('百分比 (%)')
    ax5.legend(fontsize=8)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)

ax6 = axes[1, 2]
sample_sym = list(strategies_final.keys())[0]
sample_data = test_df[test_df['symbol'] == sample_sym].sort_values('trade_date')
ax6.plot(sample_data['trade_date'], sample_data['kelly_position'], color='steelblue', linewidth=1.5)
ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='半Kelly基准')
ax6.set_title(f'{sample_sym} Kelly仓位变化')
ax6.set_xlabel('日期')
ax6.set_ylabel('仓位比例')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:/Users/HY/PycharmProjects/QuanTrade/phase23_strategy_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存")

# 保存到数据库
conn = sqlite3.connect(db_path)
test_df[['trade_date', 'symbol', 'close', 'target_direction', 'target_next_day_return',
         'fusion_prob', 'fusion_pred', 'fusion_reg',
         'signal_base', 'signal_trend', 'signal_dynamic', 'signal_kelly', 'signal_combined',
         'kelly_position', 'combined_position', 'dynamic_threshold',
         'ma_bullish', 'ma_bearish']].to_sql(
    'strategy_signals_v3', conn, if_exists='replace', index=False)

if all_results:
    pd.DataFrame(all_results).to_sql('backtest_results_v3', conn, if_exists='replace', index=False)

conn.close()
print(f"策略信号已保存到数据库")

# 详细分析
print("\n" + "=" * 70)
print("【6. 各ETF详细分析】")
print("=" * 70)

for symbol, data in strategies_final.items():
    print(f"\n>>> {symbol}")
    print(f"  基准:       {(data['benchmark'][-1]-1)*100:+.2f}%")
    print(f"  基础融合:   {(data['s_基础融合'][-1]-1)*100:+.2f}%")
    print(f"  趋势过滤:   {(data['s_趋势过滤'][-1]-1)*100:+.2f}%")
    print(f"  动态阈值:   {(data['s_动态阈值'][-1]-1)*100:+.2f}%")
    print(f"  Kelly仓位:  {(data['s_Kelly仓位'][-1]-1)*100:+.2f}%")
    print(f"  综合策略:   {(data['s_综合策略'][-1]-1)*100:+.2f}%")
    excess = (data['s_综合策略'][-1] - data['benchmark'][-1]) * 100
    print(f"  综合超额:   {excess:+.2f}%")

print("\n" + "=" * 70)
print("【Phase 2+3 完成】")
print("=" * 70)
