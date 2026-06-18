"""
Phase 1 改进: 扩展特征工程 + LightGBM/XGBoost + Walk-Forward滚动训练 + 增强回测
基于现有ETF数据 (300-400条/只)
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 尝试导入LightGBM和XGBoost
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[WARN] LightGBM未安装，将使用XGBoost替代")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN] XGBoost未安装")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, mean_squared_error, r2_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("【Phase 1 改进: 扩展特征工程 + 高级模型 + Walk-Forward + 增强回测】")
print("=" * 70)

# ==================== 1. 数据加载 ====================
db_path = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
conn = sqlite3.connect(db_path)

etf_list = ['562500.SH', '159382.SZ', '588790.SH', '159241.SZ', '588200.SH']

all_data = []
for etf in etf_list:
    df = pd.read_sql_query(f"SELECT * FROM daily_prices WHERE symbol='{etf}' ORDER BY trade_date", conn)
    all_data.append(df)
    print(f"{etf}: {len(df)} 条, {df['trade_date'].min()} ~ {df['trade_date'].max()}")

conn.close()

df_all = pd.concat(all_data, ignore_index=True)
df_all['trade_date'] = pd.to_datetime(df_all['trade_date'], format='mixed')
df_all = df_all.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

print(f"\n总数据量: {len(df_all)} 条, ETF数量: {df_all['symbol'].nunique()}")

# ==================== 2. 扩展特征工程 (50+ 特征) ====================
print("\n" + "=" * 70)
print("【2. 扩展特征工程】")
print("=" * 70)

def calc_features_v2(group):
    """扩展版特征工程 - 50+ Alpha因子"""
    g = group.sort_values('trade_date').copy()
    n = len(g)
    if n < 30:
        return g

    # --- 基础价格特征 ---
    g['return_1d'] = g['close'].pct_change()
    for w in [3, 5, 10, 20, 60]:
        g[f'return_{w}d'] = g['close'].pct_change(w)

    # --- 移动平均线体系 ---
    for w in [5, 10, 20, 60]:
        g[f'ma_{w}'] = g['close'].rolling(w).mean()
        g[f'ma_dist_{w}'] = (g['close'] - g[f'ma_{w}']) / g[f'ma_{w}']

    # MA排列信号
    g['ma_alignment'] = np.where(
        (g['ma_5'] > g['ma_10']) & (g['ma_10'] > g['ma_20']), 1,
        np.where((g['ma_5'] < g['ma_10']) & (g['ma_10'] < g['ma_20']), -1, 0)
    )
    g['ma_golden_cross'] = ((g['ma_5'] > g['ma_10']) & (g['ma_5'].shift(1) <= g['ma_10'].shift(1))).astype(int)
    g['ma_death_cross'] = ((g['ma_5'] < g['ma_10']) & (g['ma_5'].shift(1) >= g['ma_10'].shift(1))).astype(int)

    # --- 布林带 (BOLL) ---
    for w in [20]:
        ma = g['close'].rolling(w).mean()
        std = g['close'].rolling(w).std()
        g[f'boll_upper_{w}'] = ma + 2 * std
        g[f'boll_lower_{w}'] = ma - 2 * std
        g[f'boll_width_{w}'] = (g[f'boll_upper_{w}'] - g[f'boll_lower_{w}']) / ma
        g[f'boll_position_{w}'] = (g['close'] - g[f'boll_lower_{w}']) / (g[f'boll_upper_{w}'] - g[f'boll_lower_{w}'])

    # --- RSI体系 ---
    delta = g['close'].diff()
    for w in [6, 14, 28]:
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / loss
        g[f'rsi_{w}'] = 100 - (100 / (1 + rs))
    g['rsi_diff'] = g['rsi_6'] - g['rsi_14']

    # --- MACD体系 ---
    ema_12 = g['close'].ewm(span=12, adjust=False).mean()
    ema_26 = g['close'].ewm(span=26, adjust=False).mean()
    g['macd_dif'] = ema_12 - ema_26
    g['macd_dea'] = g['macd_dif'].ewm(span=9, adjust=False).mean()
    g['macd_hist'] = g['macd_dif'] - g['macd_dea']
    g['macd_golden'] = ((g['macd_dif'] > g['macd_dea']) & (g['macd_dif'].shift(1) <= g['macd_dea'].shift(1))).astype(int)
    g['macd_death'] = ((g['macd_dif'] < g['macd_dea']) & (g['macd_dif'].shift(1) >= g['macd_dea'].shift(1))).astype(int)

    # --- KDJ ---
    low_min = g['low'].rolling(9).min()
    high_max = g['high'].rolling(9).max()
    rsv = (g['close'] - low_min) / (high_max - low_min) * 100
    g['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    g['kdj_d'] = g['kdj_k'].ewm(com=2, adjust=False).mean()
    g['kdj_j'] = 3 * g['kdj_k'] - 2 * g['kdj_d']

    # --- 波动率体系 ---
    for w in [5, 10, 20]:
        g[f'std_{w}d'] = g['return_1d'].rolling(w).std()
    g['atr_14'] = (g['high'] - g['low']).rolling(14).mean() / g['close']
    g['volatility_regime'] = (g['std_20d'] > g['std_20d'].rolling(60).mean()).astype(int)

    # --- 成交量体系 ---
    for w in [5, 20]:
        g[f'volume_ma_{w}'] = g['volume'].rolling(w).mean()
    g['volume_ratio'] = g['volume'] / g['volume_ma_5']
    g['volume_zscore'] = (g['volume'] - g['volume'].rolling(20).mean()) / g['volume'].rolling(20).std()
    g['amount_ma5'] = g['amount'].rolling(5).mean()
    g['amount_ratio'] = g['amount'] / g['amount_ma5']

    # --- 价格位置与支撑阻力 ---
    for w in [20, 60]:
        high = g['high'].rolling(w).max()
        low = g['low'].rolling(w).min()
        g[f'price_position_{w}'] = (g['close'] - low) / (high - low)
        g[f'dist_to_support_{w}'] = (g['close'] - low) / g['close']
        g[f'dist_to_resistance_{w}'] = (high - g['close']) / g['close']

    # --- 趋势斜率 ---
    def linear_slope(x):
        if len(x) < 5 or np.all(np.isnan(x)):
            return np.nan
        xi = np.arange(len(x))
        mask = ~np.isnan(x)
        if mask.sum() < 5:
            return np.nan
        return np.polyfit(xi[mask], x[mask], 1)[0] / np.mean(x[mask])

    for w in [10, 20]:
        g[f'trend_slope_{w}'] = g['close'].rolling(w).apply(linear_slope, raw=True)

    # --- OBV ---
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

    # --- 振幅与价格行为 ---
    g['amplitude_pct'] = g['amplitude'] / 100
    g['body_pct'] = (g['close'] - g['open']) / g['open']
    g['upper_shadow'] = (g['high'] - g[['close', 'open']].max(axis=1)) / g['close']
    g['lower_shadow'] = (g[['close', 'open']].min(axis=1) - g['low']) / g['close']

    # --- 动量与反转 ---
    g['momentum_10'] = g['close'] / g['close'].shift(10) - 1
    g['momentum_20'] = g['close'] / g['close'].shift(20) - 1
    g['roc_12'] = (g['close'] - g['close'].shift(12)) / g['close'].shift(12)

    # --- 威廉指标 (WR) ---
    for w in [10, 20]:
        high_w = g['high'].rolling(w).max()
        low_w = g['low'].rolling(w).min()
        g[f'wr_{w}'] = (high_w - g['close']) / (high_w - low_w) * 100

    # --- CCI (顺势指标) ---
    tp = (g['high'] + g['low'] + g['close']) / 3
    for w in [14, 20]:
        ma_tp = tp.rolling(w).mean()
        md = tp.rolling(w).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        g[f'cci_{w}'] = (tp - ma_tp) / (0.015 * md)

    # --- 量价背离 ---
    g['price_volume_corr_20'] = g['close'].rolling(20).corr(g['volume'])

    # --- 目标变量 ---
    g['target_next_day_return'] = g['close'].shift(-1) / g['close'] - 1
    g['target_direction'] = (g['target_next_day_return'] > 0).astype(int)

    return g

# 按股票分组计算特征
print("\n计算扩展特征 (50+ Alpha因子)...")
df_features = []
for symbol, group in df_all.groupby('symbol'):
    df_features.append(calc_features_v2(group))
df_features = pd.concat(df_features, ignore_index=True)

# 选择特征列
feature_cols = [c for c in df_features.columns if c not in [
    'trade_date', 'symbol', 'created_at',
    'target_next_day_return', 'target_direction',
    'ma_5', 'ma_10', 'ma_20', 'ma_60',
    'boll_upper_20', 'boll_lower_20',
    'volume_ma_5', 'volume_ma_20', 'amount_ma5',
    'obv', 'obv_ma20',
    'high_max', 'low_min', 'high_w', 'low_w', 'ma_tp', 'md', 'tp'
]]

# 确保都是数值列
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_features[c])]

print(f"特征列数量: {len(feature_cols)}")
print(f"Top 20特征: {feature_cols[:20]}")

# 去除缺失值
df_model = df_features.dropna(subset=feature_cols + ['target_direction']).copy()
print(f"\n去除缺失值后可用数据: {len(df_model)} 条")
print(f"各ETF数据量:")
print(df_model['symbol'].value_counts())

# ==================== 3. Walk-Forward 滚动训练 ====================
print("\n" + "=" * 70)
print("【3. Walk-Forward 滚动训练】")
print("=" * 70)

# 全局按日期排序
df_model = df_model.sort_values('trade_date').reset_index(drop=True)

# Walk-Forward 参数
TRAIN_WINDOW = 120   # 训练窗口: 120天 (~6个月)
TEST_WINDOW = 20    # 测试窗口: 20天 (~1个月)
STEP = 20            # 滚动步长: 20天

# 生成滚动窗口
min_date = df_model['trade_date'].min()
max_date = df_model['trade_date'].max()
date_range = (max_date - min_date).days

windows = []
start = min_date + pd.Timedelta(days=TRAIN_WINDOW)
while start + pd.Timedelta(days=TEST_WINDOW) <= max_date:
    train_end = start - pd.Timedelta(days=1)
    test_end = start + pd.Timedelta(days=TEST_WINDOW) - pd.Timedelta(days=1)
    windows.append({
        'train_start': min_date,
        'train_end': train_end,
        'test_start': start,
        'test_end': test_end,
    })
    start += pd.Timedelta(days=STEP)

print(f"生成 {len(windows)} 个滚动窗口")
for i, w in enumerate(windows[:3]):
    print(f"  窗口{i+1}: 训练 {w['train_start'].strftime('%Y-%m-%d')}~{w['train_end'].strftime('%Y-%m-%d')} | 测试 {w['test_start'].strftime('%Y-%m-%d')}~{w['test_end'].strftime('%Y-%m-%d')}")
if len(windows) > 3:
    print(f"  ... 共{len(windows)}个窗口")

# ==================== 4. 模型训练与预测 ====================
print("\n" + "=" * 70)
print("【4. 模型训练与预测】")
print("=" * 70)

# 存储所有窗口的预测结果
all_predictions = []
all_actuals = []
all_dates = []
all_symbols = []

# 模型配置
model_configs = {}

if LGB_AVAILABLE:
    model_configs['LightGBM'] = {
        'clf': lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        ),
        'reg': lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
    }

if XGB_AVAILABLE:
    model_configs['XGBoost'] = {
        'clf': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        ),
        'reg': xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42
        )
    }

model_configs['RandomForest'] = {
    'clf': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1),
    'reg': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)
}

# 标准化器
scaler = StandardScaler()

# 逐窗口训练
window_results = []

for wi, w in enumerate(windows):
    # 划分数据
    train_mask = (df_model['trade_date'] >= w['train_start']) & (df_model['trade_date'] <= w['train_end'])
    test_mask = (df_model['trade_date'] >= w['test_start']) & (df_model['trade_date'] <= w['test_end'])

    train_df = df_model[train_mask].copy()
    test_df = df_model[test_mask].copy()

    if len(train_df) < 50 or len(test_df) < 5:
        continue

    X_train = train_df[feature_cols].values
    y_train_clf = train_df['target_direction'].values
    y_train_reg = train_df['target_next_day_return'].values

    X_test = test_df[feature_cols].values
    y_test_clf = test_df['target_direction'].values
    y_test_reg = test_df['target_next_day_return'].values

    # 去除NaN
    valid_train = ~np.isnan(y_train_reg)
    valid_test = ~np.isnan(y_test_reg)

    X_train_reg = X_train[valid_train]
    y_train_reg_clean = y_train_reg[valid_train]
    X_test_reg = X_test[valid_test]
    y_test_reg_clean = y_test_reg[valid_test]

    # 标准化
    scaler_w = StandardScaler()
    X_train_scaled = scaler_w.fit_transform(X_train)
    X_test_scaled = scaler_w.transform(X_test)

    window_pred = {
        'window': wi + 1,
        'test_dates': test_df['trade_date'].values,
        'test_symbols': test_df['symbol'].values,
        'y_test_clf': y_test_clf,
        'y_test_reg': y_test_reg,
    }

    # 训练各模型
    for model_name, models in model_configs.items():
        # 分类
        try:
            models['clf'].fit(X_train, y_train_clf)
            pred_clf = models['clf'].predict(X_test)
            prob_clf = models['clf'].predict_proba(X_test)[:, 1]
            window_pred[f'{model_name}_pred'] = pred_clf
            window_pred[f'{model_name}_prob'] = prob_clf
        except Exception as e:
            print(f"  窗口{wi+1} {model_name}分类失败: {e}")

        # 回归
        if len(X_train_reg) > 10 and len(X_test_reg) > 0:
            try:
                models['reg'].fit(X_train_reg, y_train_reg_clean)
                pred_reg = models['reg'].predict(X_test_reg)
                window_pred[f'{model_name}_reg'] = pred_reg
            except Exception as e:
                print(f"  窗口{wi+1} {model_name}回归失败: {e}")

    window_results.append(window_pred)

    if (wi + 1) % 5 == 0 or wi == 0:
        print(f"  窗口 {wi+1}/{len(windows)} 完成 | 训练{len(train_df)}条 测试{len(test_df)}条")

print(f"\n完成 {len(window_results)} 个窗口的训练与预测")

# ==================== 5. 模型评估 ====================
print("\n" + "=" * 70)
print("【5. 模型评估 - Walk-Forward汇总】")
print("=" * 70)

# 汇总所有窗口的预测
for model_name in model_configs.keys():
    all_pred = []
    all_prob = []
    all_reg = []
    all_y_clf = []
    all_y_reg = []

    for wr in window_results:
        if f'{model_name}_pred' in wr:
            all_pred.extend(wr[f'{model_name}_pred'])
            all_prob.extend(wr[f'{model_name}_prob'])
            all_y_clf.extend(wr['y_test_clf'])
        if f'{model_name}_reg' in wr:
            # 对齐回归预测和实际值（去除NaN）
            valid_mask = ~np.isnan(wr['y_test_reg'])
            if valid_mask.sum() > 0:
                all_reg.extend(wr[f'{model_name}_reg'])
                all_y_reg.extend(wr['y_test_reg'][valid_mask])

    if len(all_pred) == 0:
        continue

    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)
    all_y_clf = np.array(all_y_clf)

    print(f"\n>>> {model_name} 分类 (样本数: {len(all_pred)})")
    print(f"  准确率:  {accuracy_score(all_y_clf, all_pred):.4f}")
    print(f"  精确率:  {precision_score(all_y_clf, all_pred, zero_division=0):.4f}")
    print(f"  召回率:  {recall_score(all_y_clf, all_pred, zero_division=0):.4f}")
    print(f"  F1分数:  {f1_score(all_y_clf, all_pred, zero_division=0):.4f}")
    try:
        print(f"  AUC:     {roc_auc_score(all_y_clf, all_prob):.4f}")
    except:
        pass
    print(f"  实际涨比例: {np.mean(all_y_clf):.4f} | 预测涨比例: {np.mean(all_pred):.4f}")

    if len(all_reg) > 0:
        all_reg = np.array(all_reg)
        all_y_reg = np.array(all_y_reg)
        print(f"\n>>> {model_name} 回归 (样本数: {len(all_reg)})")
        print(f"  RMSE: {np.sqrt(mean_squared_error(all_y_reg, all_reg)):.6f}")
        print(f"  R2:   {r2_score(all_y_reg, all_reg):.4f}")
        corr = np.corrcoef(all_y_reg, all_reg)[0, 1] if len(all_reg) > 1 else 0
        print(f"  相关系数: {corr:.4f}")

# 特征重要性 (取最后一个窗口的LightGBM或RandomForest)
if LGB_AVAILABLE and 'LightGBM' in model_configs:
    last_model = model_configs['LightGBM']['clf']
elif XGB_AVAILABLE and 'XGBoost' in model_configs:
    last_model = model_configs['XGBoost']['clf']
else:
    last_model = model_configs['RandomForest']['clf']

if hasattr(last_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': last_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n>>> Top 20 重要特征 ({type(last_model).__name__}):")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:25s} {row['importance']:.4f}")

# ==================== 6. 多模型融合 ====================
print("\n" + "=" * 70)
print("【6. 多模型融合 (Stacking)】")
print("=" * 70)

# 简单融合: 各模型概率平均
fusion_pred = []
fusion_prob = []
fusion_dates = []
fusion_symbols = []
fusion_actual = []

for wr in window_results:
    probs = []
    for model_name in model_configs.keys():
        key = f'{model_name}_prob'
        if key in wr:
            probs.append(wr[key])

    if len(probs) >= 2:
        avg_prob = np.mean(probs, axis=0)
        fusion_pred.extend((avg_prob > 0.5).astype(int))
        fusion_prob.extend(avg_prob)
        fusion_dates.extend(wr['test_dates'])
        fusion_symbols.extend(wr['test_symbols'])
        fusion_actual.extend(wr['y_test_clf'])

if len(fusion_pred) > 0:
    fusion_pred = np.array(fusion_pred)
    fusion_prob = np.array(fusion_prob)
    fusion_actual = np.array(fusion_actual)

    print(f"\n>>> 融合模型 (样本数: {len(fusion_pred)})")
    print(f"  准确率:  {accuracy_score(fusion_actual, fusion_pred):.4f}")
    print(f"  精确率:  {precision_score(fusion_actual, fusion_pred, zero_division=0):.4f}")
    print(f"  召回率:  {recall_score(fusion_actual, fusion_pred, zero_division=0):.4f}")
    print(f"  F1分数:  {f1_score(fusion_actual, fusion_pred, zero_division=0):.4f}")
    try:
        print(f"  AUC:     {roc_auc_score(fusion_actual, fusion_prob):.4f}")
    except:
        pass

# ==================== 7. 增强回测引擎 ====================
print("\n" + "=" * 70)
print("【7. 增强回测引擎 (含成本+风控)】")
print("=" * 70)

# 回测参数
FEE_RATE = 0.0001       # ETF手续费: 万1 (0.01%)
SLIPPAGE = 0.0001       # 滑点: 0.01%
STOP_LOSS_ATR = 2.0     # ATR倍数止损
MAX_POSITION = 1.0      # 最大仓位100%
MIN_POSITION = 0.0      # 最小仓位0%

# 构建回测数据框
backtest_df = pd.DataFrame({
    'trade_date': fusion_dates,
    'symbol': fusion_symbols,
    'actual_direction': fusion_actual,
    'fusion_pred': fusion_pred,
    'fusion_prob': fusion_prob,
})

# 合并实际收益率
bt_merged = []
for _, row in backtest_df.iterrows():
    match = df_model[(df_model['trade_date'] == row['trade_date']) & (df_model['symbol'] == row['symbol'])]
    if len(match) > 0:
        r = match.iloc[0]
        bt_merged.append({
            'trade_date': row['trade_date'],
            'symbol': row['symbol'],
            'actual_return': r['target_next_day_return'],
            'pred_direction': row['fusion_pred'],
            'pred_prob': row['fusion_prob'],
            'atr_14': r['atr_14'] if 'atr_14' in r else 0.02,
            'close': r['close'],
        })

bt_df = pd.DataFrame(bt_merged)
bt_df = bt_df.dropna(subset=['actual_return'])

print(f"回测样本数: {len(bt_df)} 条")

# 策略回测
strategies_v2 = {}

for symbol in bt_df['symbol'].unique():
    sym_data = bt_df[bt_df['symbol'] == symbol].sort_values('trade_date').reset_index(drop=True)
    if len(sym_data) < 10:
        continue

    actual_returns = sym_data['actual_return'].values
    pred_probs = sym_data['pred_prob'].values
    pred_directions = sym_data['pred_direction'].values
    atrs = sym_data['atr_14'].values if 'atr_14' in sym_data.columns else np.full(len(sym_data), 0.02)

    n = len(actual_returns)

    # 基准: 满仓持有
    benchmark_cum = np.cumprod(1 + actual_returns)

    # 策略1: 融合模型分类信号 (满仓/空仓)
    s1_signals = pred_directions.astype(float)
    s1_returns = actual_returns * s1_signals
    s1_cum = np.cumprod(1 + s1_returns)

    # 策略2: 概率阈值 > 0.55
    s2_signals = (pred_probs > 0.55).astype(float)
    s2_returns = actual_returns * s2_signals
    s2_cum = np.cumprod(1 + s2_returns)

    # 策略3: 概率加权仓位 (0~1)
    s3_weights = pred_probs
    s3_returns = actual_returns * s3_weights
    s3_cum = np.cumprod(1 + s3_returns)

    # 策略4: 融合模型 + ATR止损 + 手续费
    s4_returns = []
    s4_cum = [1.0]
    in_position = False
    entry_price = 0

    for i in range(n):
        signal = pred_directions[i]
        daily_ret = actual_returns[i]
        atr = atrs[i] if i < len(atrs) else 0.02

        if signal == 1 and not in_position:
            # 买入
            in_position = True
            entry_price = 1.0  # 归一化价格
            # 扣除手续费和滑点
            cost = FEE_RATE + SLIPPAGE
            s4_cum[-1] *= (1 - cost)
        elif signal == 0 and in_position:
            # 卖出
            in_position = False
            cost = FEE_RATE + SLIPPAGE
            s4_cum[-1] *= (1 - cost)

        if in_position:
            # 检查止损
            price_change = daily_ret
            if price_change < -STOP_LOSS_ATR * atr:
                # 触发止损，当日收益按止损计算
                daily_ret = -STOP_LOSS_ATR * atr
                in_position = False

            s4_returns.append(daily_ret)
            s4_cum.append(s4_cum[-1] * (1 + daily_ret))
        else:
            s4_returns.append(0)
            s4_cum.append(s4_cum[-1])

    s4_cum = np.array(s4_cum[1:])  # 去掉初始值

    strategies_v2[symbol] = {
        'dates': sym_data['trade_date'].values,
        'actual': actual_returns,
        'benchmark': benchmark_cum,
        's1_fusion_class': s1_cum,
        's2_fusion_prob': s2_cum,
        's3_prob_weight': s3_cum,
        's4_with_cost_stop': s4_cum,
    }

# ==================== 8. 回测结果汇总 ====================
print("\n>>> 增强回测结果:\n")
print(f"{'ETF代码':<12} {'策略':<22} {'总收益率':<10} {'年化收益率':<10} {'最大回撤':<10} {'夏普比率':<8} {'胜率':<8}")
print("-" * 90)

all_returns_v2 = []

for symbol, data in strategies_v2.items():
    for s_name, s_key in [
        ('基准(持有)', 'benchmark'),
        ('融合分类信号', 's1_fusion_class'),
        ('融合概率>0.55', 's2_fusion_prob'),
        ('概率加权', 's3_prob_weight'),
        ('含成本+止损', 's4_with_cost_stop'),
    ]:
        cum = data[s_key]
        total_ret = cum[-1] - 1
        n_days = len(cum)
        annual_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 and total_ret > -1 else -1

        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_dd = np.min(drawdown)

        daily_rets = np.diff(cum) / cum[:-1]
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0

        win_rate = np.sum(daily_rets > 0) / len(daily_rets) if len(daily_rets) > 0 else 0

        print(f"{symbol:<12} {s_name:<22} {total_ret*100:>8.2f}% {annual_ret*100:>8.2f}% {max_dd*100:>8.2f}% {sharpe:>6.2f} {win_rate*100:>6.1f}%")

        if s_name != '基准(持有)':
            all_returns_v2.append({
                'symbol': symbol,
                'strategy': s_name,
                'total_return': total_ret,
                'annual_return': annual_ret,
                'max_drawdown': max_dd,
                'sharpe': sharpe,
                'win_rate': win_rate,
            })

# ==================== 9. 可视化 ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Phase 1 改进: Walk-Forward + 多模型融合 + 增强回测', fontsize=14, fontweight='bold')

# 子图1: 累计收益对比
ax1 = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (symbol, data) in enumerate(strategies_v2.items()):
    ax1.plot(data['dates'], data['benchmark'] - 1, '--', alpha=0.4, color=colors[i % len(colors)])
    ax1.plot(data['dates'], data['s4_with_cost_stop'] - 1, '-', color=colors[i % len(colors)], linewidth=2, label=symbol)
ax1.set_title('累计收益率: 含成本+止损策略 vs 基准')
ax1.set_xlabel('日期')
ax1.set_ylabel('累计收益率')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)

# 子图2: 特征重要性
ax2 = axes[0, 1]
if 'importance_df' in dir() or 'importance_df' in globals():
    top_features = importance_df.head(15)
    ax2.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'].values, fontsize=8)
    ax2.set_title('Top 15 特征重要性')
    ax2.set_xlabel('重要性')
    ax2.invert_yaxis()

# 子图3: 混淆矩阵 (融合模型)
ax3 = axes[0, 2]
if len(fusion_pred) > 0:
    cm = confusion_matrix(fusion_actual, fusion_pred)
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title('混淆矩阵 (融合模型)')
    tick_marks = np.arange(2)
    ax3.set_xticks(tick_marks)
    ax3.set_yticks(tick_marks)
    ax3.set_xticklabels(['跌(0)', '涨(1)'])
    ax3.set_yticklabels(['跌(0)', '涨(1)'])
    ax3.set_ylabel('真实')
    ax3.set_xlabel('预测')
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)

# 子图4: 预测概率分布
ax4 = axes[1, 0]
if len(fusion_prob) > 0:
    ax4.hist(fusion_prob[fusion_actual == 0], bins=20, alpha=0.6, label='实际跌', color='red', density=True)
    ax4.hist(fusion_prob[fusion_actual == 1], bins=20, alpha=0.6, label='实际涨', color='green', density=True)
    ax4.set_title('融合模型预测概率分布')
    ax4.set_xlabel('预测上涨概率')
    ax4.set_ylabel('密度')
    ax4.legend()
    ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

# 子图5: 各模型准确率对比
ax5 = axes[1, 1]
model_accs = {}
for model_name in model_configs.keys():
    all_pred_m = []
    all_y_m = []
    for wr in window_results:
        key = f'{model_name}_pred'
        if key in wr:
            all_pred_m.extend(wr[key])
            all_y_m.extend(wr['y_test_clf'])
    if len(all_pred_m) > 0:
        model_accs[model_name] = accuracy_score(np.array(all_y_m), np.array(all_pred_m))

if len(fusion_pred) > 0:
    model_accs['融合模型'] = accuracy_score(fusion_actual, fusion_pred)

if model_accs:
    names = list(model_accs.keys())
    accs = list(model_accs.values())
    colors_bar = ['steelblue'] * len(names)
    if '融合模型' in names:
        colors_bar[names.index('融合模型')] = 'coral'
    ax5.bar(names, accs, color=colors_bar)
    ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='随机水平')
    ax5.set_title('各模型准确率对比')
    ax5.set_ylabel('准确率')
    ax5.set_ylim(0.4, 0.7)
    ax5.legend()

# 子图6: 策略平均表现
ax6 = axes[1, 2]
if all_returns_v2:
    summary = pd.DataFrame(all_returns_v2).groupby('strategy')[['total_return', 'annual_return', 'max_drawdown', 'sharpe']].mean()
    x_pos = np.arange(len(summary))
    width = 0.2
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
plt.savefig(r'C:/Users/HY/PycharmProjects/QuanTrade/phase1_enhanced_backtest.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存: C:/Users/HY/PycharmProjects/QuanTrade/phase1_enhanced_backtest.png")

# ==================== 10. 保存结果到数据库 ====================
conn = sqlite3.connect(db_path)

# 保存融合预测结果
if len(fusion_pred) > 0:
    pred_df = pd.DataFrame({
        'trade_date': fusion_dates,
        'symbol': fusion_symbols,
        'actual_direction': fusion_actual,
        'fusion_pred': fusion_pred,
        'fusion_prob': fusion_prob,
    })
    pred_df.to_sql('fusion_predictions', conn, if_exists='replace', index=False)
    print(f"融合预测结果已保存: fusion_predictions ({len(pred_df)}条)")

# 保存回测结果
if all_returns_v2:
    bt_result_df = pd.DataFrame(all_returns_v2)
    bt_result_df.to_sql('backtest_results_v2', conn, if_exists='replace', index=False)
    print(f"回测结果已保存: backtest_results_v2 ({len(bt_result_df)}条)")

conn.close()

# ==================== 11. 各ETF详细分析 ====================
print("\n" + "=" * 70)
print("【8. 各ETF详细回测分析】")
print("=" * 70)

for symbol, data in strategies_v2.items():
    print(f"\n>>> {symbol}")
    print(f"  测试期交易天数: {len(data['actual'])}")
    print(f"  基准总收益:       {(data['benchmark'][-1]-1)*100:+.2f}%")
    print(f"  融合分类策略:     {(data['s1_fusion_class'][-1]-1)*100:+.2f}%")
    print(f"  融合概率>0.55:    {(data['s2_fusion_prob'][-1]-1)*100:+.2f}%")
    print(f"  概率加权:         {(data['s3_prob_weight'][-1]-1)*100:+.2f}%")
    print(f"  含成本+止损:      {(data['s4_with_cost_stop'][-1]-1)*100:+.2f}%")

    excess_s1 = (data['s1_fusion_class'][-1] - data['benchmark'][-1]) * 100
    excess_s4 = (data['s4_with_cost_stop'][-1] - data['benchmark'][-1]) * 100
    print(f"  分类策略超额:     {excess_s1:+.2f}%")
    print(f"  成本止损超额:     {excess_s4:+.2f}%")

print("\n" + "=" * 70)
print("【Phase 1 改进完成】")
print("=" * 70)
print(f"\n改进总结:")
print(f"  - 特征数量: 26 → {len(feature_cols)} 个")
print(f"  - 模型: RandomForest → LightGBM/XGBoost/RF融合")
print(f"  - 训练方式: 单次划分 → Walk-Forward滚动 ({len(window_results)}个窗口)")
print(f"  - 回测: 无成本 → 含手续费({FEE_RATE*100:.2f}%) + 滑点({SLIPPAGE*100:.2f}%) + ATR止损")
print(f"  - 评估: Accuracy → AUC + 多模型对比")
