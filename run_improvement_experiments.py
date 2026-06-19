"""
4大改进方案对比实验
方案1: 缩短预测窗口 (10日→5日→3日)
方案2: 使用回归任务 (直接预测收益率)
方案3: 增加特征 (参考Qlib Alpha158)
方案4: 优化回测策略 (基于概率的动态仓位)

每个方案输出: 独立的 predictions/backtest 表
最后汇总对比所有方案
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'
BASE_MODEL_DIR = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\models'


# ============================================================
# 1. 增强特征工程 (方案3)
# ============================================================

def add_enhanced_features(df):
    """增加更多特征，参考Qlib Alpha158"""
    print("  增加增强特征...")
    result = []

    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        s = df.loc[mask].copy()

        # 原始特征已经存在，增加新的特征

        # 1. 动量特征
        for window in [3, 5, 10, 20]:
            s[f'return_{window}d'] = s['close'].pct_change(window)
            s[f'volatility_{window}d'] = s['close'].pct_change().rolling(window).std()

        # 2. 价格位置特征
        for window in [10, 20, 60]:
            s[f'high_{window}d'] = s['high'].rolling(window).max()
            s[f'low_{window}d'] = s['low'].rolling(window).min()
            s[f'price_position_{window}d'] = (s['close'] - s[f'low_{window}d']) / (s[f'high_{window}d'] - s[f'low_{window}d'] + 1e-10)

        # 3. 成交量特征
        for window in [5, 10, 20]:
            s[f'volume_ma_{window}'] = s['volume'].rolling(window).mean()
            s[f'volume_ratio_{window}'] = s['volume'] / (s[f'volume_ma_{window}'] + 1e-10)

        # 4. 技术指标
        # RSI
        for window in [6, 14]:
            delta = s['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + 1e-10)
            s[f'rsi_{window}'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = s['close'].ewm(span=12, adjust=False).mean()
        exp2 = s['close'].ewm(span=26, adjust=False).mean()
        s['macd'] = exp1 - exp2
        s['macd_signal'] = s['macd'].ewm(span=9, adjust=False).mean()
        s['macd_hist'] = s['macd'] - s['macd_signal']

        # 5. ATR (Average True Range)
        high_low = s['high'] - s['low']
        high_close = np.abs(s['high'] - s['close'].shift())
        low_close = np.abs(s['low'] - s['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        s['atr_14'] = true_range.rolling(14).mean()
        s['atr_ratio'] = true_range / (s['atr_14'] + 1e-10)

        # 6. 布林带
        s['bb_middle'] = s['close'].rolling(20).mean()
        s['bb_std'] = s['close'].rolling(20).std()
        s['bb_upper'] = s['bb_middle'] + 2 * s['bb_std']
        s['bb_lower'] = s['bb_middle'] - 2 * s['bb_std']
        s['bb_width'] = (s['bb_upper'] - s['bb_lower']) / (s['bb_middle'] + 1e-10)
        s['bb_position'] = (s['close'] - s['bb_lower']) / (s['bb_upper'] - s['bb_lower'] + 1e-10)

        # 7. 趋势强度
        s['trend_5d'] = s['close'].pct_change(5)
        s['trend_10d'] = s['close'].pct_change(10)
        s['trend_20d'] = s['close'].pct_change(20)

        # 8. 波动率比率
        s['vol_ratio_5_20'] = s['volatility_5d'] / (s['volatility_20d'] + 1e-10)

        result.append(s)

    return pd.concat(result, ignore_index=True)


# ============================================================
# 2. 标签生成函数
# ============================================================

def make_scenario_label(r, fav_th, adv_th):
    """根据阈值生成三情景标签"""
    labels = pd.Series('base', index=r.index)
    labels[r < adv_th] = 'adverse'
    labels[r > fav_th] = 'favorable'
    return labels


def make_binary_label(r, mode='simple', up_th=0.0, down_th=0.0):
    """生成二分类标签"""
    if mode == 'simple':
        return (r > up_th).astype(int)
    elif mode == 'filtered':
        labels = pd.Series(np.nan, index=r.index)
        labels[r > up_th] = 1
        labels[r < down_th] = 0
        return labels
    return (r > 0).astype(int)


# ============================================================
# 3. 分类模型训练
# ============================================================

def train_classifier(df, feature_cols, label_col, exp_name, model_dir):
    """训练分类模型"""
    y = df[label_col].astype(int)
    X = df[feature_cols].copy()

    # 时间划分
    train_mask = df['trade_date'] < '2023-01-01'
    val_mask = (df['trade_date'] >= '2023-01-01') & (df['trade_date'] < '2024-01-01')
    test_mask = df['trade_date'] >= '2024-01-01'

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # 手动过采样
    max_count = y_train.value_counts().max()
    oversampled_indices = []
    for cls in y_train.unique():
        cls_idx = y_train[y_train == cls].index
        if len(cls_idx) < max_count:
            oversampled_idx = np.random.choice(cls_idx, size=max_count, replace=True)
            oversampled_indices.extend(oversampled_idx)
        else:
            oversampled_indices.extend(cls_idx)
    X_train = X_train.loc[oversampled_indices]
    y_train = y_train.loc[oversampled_indices]

    # RF
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    # LGBM
    n_classes = len(y.unique())
    lgbm_params = {
        'objective': 'multiclass' if n_classes > 2 else 'binary',
        'num_class': n_classes if n_classes > 2 else 1,
        'n_estimators': 1000, 'learning_rate': 0.05,
        'max_depth': 8, 'num_leaves': 63, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'random_state': 42, 'verbosity': -1,
        'n_jobs': -1,
    }
    lgbm = lgb.LGBMClassifier(**lgbm_params)
    lgbm.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))

    # Fusion
    rf_proba = rf.predict_proba(X_test)
    lgbm_proba = lgbm.predict_proba(X_test)
    fusion_proba = (rf_proba + lgbm_proba) / 2
    fusion_pred = np.argmax(fusion_proba, axis=1)
    fusion_acc = accuracy_score(y_test, fusion_pred)

    # 保存模型
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    with open(os.path.join(model_dir, 'lgbm_model.pkl'), 'wb') as f:
        pickle.dump(lgbm, f)

    return {
        'rf_acc': rf_acc, 'lgbm_acc': lgbm_acc, 'fusion_acc': fusion_acc,
        'rf': rf, 'lgbm': lgbm,
    }


# ============================================================
# 4. 回归模型训练 (方案2)
# ============================================================

def train_regressor(df, feature_cols, target_col, exp_name, model_dir):
    """训练回归模型"""
    y = df[target_col]
    X = df[feature_cols].copy()

    # 时间划分
    train_mask = df['trade_date'] < '2023-01-01'
    val_mask = (df['trade_date'] >= '2023-01-01') & (df['trade_date'] < '2024-01-01')
    test_mask = df['trade_date'] >= '2024-01-01'

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # RF
    rf = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)

    # LGBM
    lgbm_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000, 'learning_rate': 0.05,
        'max_depth': 8, 'num_leaves': 63, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'random_state': 42, 'verbosity': -1,
        'n_jobs': -1,
    }
    lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    lgbm_pred = lgbm.predict(X_test)
    lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))
    lgbm_r2 = r2_score(y_test, lgbm_pred)

    # Fusion
    fusion_pred = (rf_pred + lgbm_pred) / 2
    fusion_rmse = np.sqrt(mean_squared_error(y_test, fusion_pred))
    fusion_r2 = r2_score(y_test, fusion_pred)

    # 保存模型
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    with open(os.path.join(model_dir, 'lgbm_model.pkl'), 'wb') as f:
        pickle.dump(lgbm, f)

    return {
        'rf_rmse': rf_rmse, 'rf_r2': rf_r2,
        'lgbm_rmse': lgbm_rmse, 'lgbm_r2': lgbm_r2,
        'fusion_rmse': fusion_rmse, 'fusion_r2': fusion_r2,
        'rf': rf, 'lgbm': lgbm,
    }


# ============================================================
# 5. 回测策略
# ============================================================

CRASH_CONFIG = {
    'vol_spike_ratio': 1.5, 'drawdown_threshold': 0.15,
    'liquidity_dryup_ratio': 0.5, 'red_alert_reduction': 1.0,
    'orange_alert_reduction': 0.5, 'yellow_alert_reduction': 0.2,
    'recovery_required_days': 5,
}
TREND_CONFIG = {
    'trend_threshold_strong': 0.20, 'trend_threshold_moderate': 0.10,
    'add_position_step': 0.10, 'add_position_max': 0.90,
}
SCENARIO_POSITIONS = {'adverse': 0.0, 'base': 0.5, 'favorable': 1.0}


def run_backtest(pred_df, exp_name, results_table, daily_table, strategy='standard'):
    """运行回测"""
    price_df = pd.read_sql_query(
        "SELECT trade_date, symbol, open, high, low, close, volume FROM daily_prices_v5",
        sqlite3.connect(DB_PATH)
    )
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    pred_df = pred_df.copy()
    pred_df['trade_date'] = pd.to_datetime(pred_df['trade_date'])
    pred_df = pred_df.rename(columns={'close': 'pred_close'})

    # 计算价格指标
    price_df = price_df.sort_values(['symbol', 'trade_date'])
    for symbol in price_df['symbol'].unique():
        mask = price_df['symbol'] == symbol
        s = price_df.loc[mask].copy()
        s['return_120d'] = s['close'].pct_change(120)
        s['ma_20'] = s['close'].rolling(20).mean()
        s['ma_60'] = s['close'].rolling(60).mean()
        s['ma_120'] = s['close'].rolling(120).mean()
        s['ma_bullish'] = (s['ma_20'] > s['ma_60']) & (s['ma_60'] > s['ma_120'])
        s['high_120d'] = s['high'].rolling(120).max()
        s['drawdown_from_high'] = (s['close'] - s['high_120d']) / s['high_120d']
        s['tr'] = np.maximum(s['high'] - s['low'],
                              np.maximum(abs(s['high'] - s['close'].shift(1)),
                                         abs(s['low'] - s['close'].shift(1))))
        s['atr_14'] = s['tr'].rolling(14).mean()
        s['atr_20'] = s['tr'].rolling(20).mean()
        s['vol_spike'] = s['atr_20'] / s['atr_14']
        s['volume_ma_20'] = s['volume'].rolling(20).mean()
        s['liquidity_ratio'] = s['volume'] / s['volume_ma_20']
        s['momentum_20d'] = s['close'].pct_change(20)
        s['momentum_60d'] = s['close'].pct_change(60)
        for col in ['return_120d', 'ma_bullish', 'drawdown_from_high', 'vol_spike',
                     'liquidity_ratio', 'momentum_20d', 'momentum_60d']:
            price_df.loc[mask, col] = s[col].values

    merged = pred_df.merge(price_df, on=['trade_date', 'symbol'], how='inner')
    merged = merged.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

    results = []
    daily_records = []

    for symbol in merged['symbol'].unique():
        sym_data = merged[merged['symbol'] == symbol].copy()
        if len(sym_data) < 20:
            continue

        cash = 1.0
        position = 0.0
        nav = 1.0
        peak_nav = 1.0
        max_dd = 0.0
        alert_active = False
        consecutive_safe_days = 0
        sym_daily_navs = []

        for idx in range(len(sym_data)):
            row = sym_data.iloc[idx]
            date = row['trade_date']
            close = row['close']

            if idx > 0:
                prev_close = sym_data.iloc[idx - 1]['close']
                position_value = position * (close / prev_close)
            else:
                position_value = 0.0

            # 股灾预警
            alerts = 0
            if not pd.isna(row.get('vol_spike')) and row['vol_spike'] > CRASH_CONFIG['vol_spike_ratio']:
                alerts += 1
            if not pd.isna(row.get('drawdown_from_high')) and row['drawdown_from_high'] < -CRASH_CONFIG['drawdown_threshold']:
                alerts += 1
            if not pd.isna(row.get('liquidity_ratio')) and row['liquidity_ratio'] < CRASH_CONFIG['liquidity_dryup_ratio']:
                alerts += 1
            if not pd.isna(row.get('momentum_20d')) and row['momentum_20d'] < -0.20:
                alerts += 1
            if not pd.isna(row.get('momentum_60d')) and row['momentum_60d'] < -0.30:
                alerts += 1

            if alerts > 0:
                if not alert_active:
                    alert_active = True
                consecutive_safe_days = 0
            else:
                consecutive_safe_days += 1
                if alert_active and consecutive_safe_days >= CRASH_CONFIG['recovery_required_days']:
                    alert_active = False

            if alerts >= 3:
                crash_reduction = CRASH_CONFIG['red_alert_reduction']
            elif alerts == 2:
                crash_reduction = CRASH_CONFIG['orange_alert_reduction']
            elif alerts == 1:
                crash_reduction = CRASH_CONFIG['yellow_alert_reduction']
            else:
                crash_reduction = 0.0

            # 计算仓位
            if strategy == 'probabilistic':
                # 方案4: 基于概率的动态仓位
                p_adverse = row.get('adverse', 0)
                p_base = row.get('base', 0)
                p_favorable = row.get('favorable', 0)
                base_position = p_favorable * 1.0 + p_base * 0.5 + p_adverse * 0.0
            else:
                # 标准策略
                probs = {'adverse': row.get('adverse', 0), 'base': row.get('base', 0), 'favorable': row.get('favorable', 0)}
                scenario = max(probs, key=probs.get)
                base_position = SCENARIO_POSITIONS.get(scenario, 0.0)

            # 趋势加仓
            trend_position = base_position
            if not alert_active:
                return_120d = row.get('return_120d')
                ma_bullish = row.get('ma_bullish', False)
                if not pd.isna(return_120d):
                    if return_120d > TREND_CONFIG['trend_threshold_strong']:
                        new_pos = min(base_position + TREND_CONFIG['add_position_step'], TREND_CONFIG['add_position_max'])
                        if new_pos > base_position:
                            trend_position = new_pos
                    elif return_120d > TREND_CONFIG['trend_threshold_moderate']:
                        new_pos = min(base_position + TREND_CONFIG['add_position_step'] * 0.5, TREND_CONFIG['add_position_max'])
                        if new_pos > base_position:
                            trend_position = new_pos

            # 股灾减仓
            final_position = trend_position
            if alert_active:
                final_position = trend_position * (1 - crash_reduction)
            final_position = max(0.0, min(1.0, final_position))

            # 执行调仓
            current_total = cash + position_value
            target_value = current_total * final_position
            if idx > 0:
                trade_value = target_value - position_value
                cash -= trade_value
                position_value = target_value
            else:
                position_value = current_total * final_position
                cash = current_total - position_value

            position = final_position
            nav = cash + position_value
            if nav > peak_nav:
                peak_nav = nav
            dd = (nav - peak_nav) / peak_nav
            if dd < max_dd:
                max_dd = dd

            sym_daily_navs.append(nav)

            daily_records.append({
                'trade_date': date, 'symbol': symbol, 'strategy': exp_name,
                'nav': nav, 'position': position, 'close': close
            })

        if len(sym_data) > 1:
            total_return = nav - 1.0
            start_date = sym_data['trade_date'].iloc[0]
            end_date = sym_data['trade_date'].iloc[-1]
            years = (end_date - start_date).days / 365.25
            annual_return = (nav ** (1 / years) - 1) if years > 0 and nav > 0 else 0
            benchmark_return = (sym_data['close'].iloc[-1] / sym_data['close'].iloc[0]) - 1

            daily_returns = []
            for j in range(1, len(sym_daily_navs)):
                if sym_daily_navs[j - 1] > 0:
                    daily_returns.append((sym_daily_navs[j] - sym_daily_navs[j - 1]) / sym_daily_navs[j - 1])

            if daily_returns:
                sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
                win_rate = sum(1 for r in daily_returns if r > 0) / len(daily_returns)
            else:
                sharpe = 0
                win_rate = 0

            results.append({
                'symbol': symbol, 'strategy': exp_name,
                'total_return': total_return, 'annual_return': annual_return,
                'max_drawdown': max_dd, 'sharpe': sharpe, 'win_rate': win_rate,
                'benchmark_return': benchmark_return,
                'excess_return': total_return - benchmark_return,
                'avg_position': np.mean([d['position'] for d in daily_records if d['symbol'] == symbol]),
            })

    results_df = pd.DataFrame(results)
    daily_df = pd.DataFrame(daily_records)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"DROP TABLE IF EXISTS {results_table}")
    conn.execute(f"DROP TABLE IF EXISTS {daily_table}")
    results_df.to_sql(results_table, conn, if_exists='replace', index=False)
    daily_df.to_sql(daily_table, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return results_df


# ============================================================
# 6. 预测生成
# ============================================================

def generate_predictions(df, feature_cols, model_dir, exp_name, table_name, model_type='classifier'):
    """生成预测"""
    with open(os.path.join(model_dir, 'rf_model.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(model_dir, 'lgbm_model.pkl'), 'rb') as f:
        lgbm = pickle.load(f)

    test_mask = df['trade_date'] >= '2024-01-01'
    test_df = df[test_mask].copy()
    X_test = test_df[feature_cols].copy()

    if model_type == 'classifier':
        rf_proba = rf.predict_proba(X_test)
        lgbm_proba = lgbm.predict_proba(X_test)
        fusion_proba = (rf_proba + lgbm_proba) / 2
        fusion_pred = np.argmax(fusion_proba, axis=1)

        n_classes = rf_proba.shape[1]
        label_map = {0: 'adverse', 1: 'base', 2: 'favorable'} if n_classes == 3 else {0: 'adverse', 1: 'favorable'}

        pred_records = []
        for i, idx in enumerate(test_df.index):
            row = {
                'trade_date': str(test_df.loc[idx, 'trade_date']),
                'symbol': test_df.loc[idx, 'symbol'],
                'close': float(test_df.loc[idx, 'close']),
                'prediction': label_map.get(fusion_pred[i], str(fusion_pred[i])),
                'adverse': float(fusion_proba[i][0]),
                'base': float(fusion_proba[i][1]) if n_classes == 3 else 0.0,
                'favorable': float(fusion_proba[i][2]) if n_classes == 3 else float(fusion_proba[i][1]),
            }
            pred_records.append(row)
    else:
        # 回归模型
        rf_pred = rf.predict(X_test)
        lgbm_pred = lgbm.predict(X_test)
        fusion_pred = (rf_pred + lgbm_pred) / 2

        pred_records = []
        for i, idx in enumerate(test_df.index):
            pred_value = fusion_pred[i]
            # 将回归预测转换为三情景
            if pred_value > 0.05:
                scenario = 'favorable'
                p_favorable = 0.7
                p_base = 0.2
                p_adverse = 0.1
            elif pred_value < -0.05:
                scenario = 'adverse'
                p_favorable = 0.1
                p_base = 0.2
                p_adverse = 0.7
            else:
                scenario = 'base'
                p_favorable = 0.2
                p_base = 0.6
                p_adverse = 0.2

            row = {
                'trade_date': str(test_df.loc[idx, 'trade_date']),
                'symbol': test_df.loc[idx, 'symbol'],
                'close': float(test_df.loc[idx, 'close']),
                'prediction': scenario,
                'adverse': p_adverse,
                'base': p_base,
                'favorable': p_favorable,
            }
            pred_records.append(row)

    pred_df = pd.DataFrame(pred_records)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    pred_df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    print(f"  预测写入 {table_name}: {len(pred_df)}条")
    print(f"  预测分布: {pred_df['prediction'].value_counts().to_dict()}")
    return pred_df


# ============================================================
# 7. 主流程
# ============================================================

def main():
    print("=" * 80)
    print("4大改进方案对比实验")
    print("=" * 80)

    # 加载基线数据
    conn = sqlite3.connect(DB_PATH)
    base_df = pd.read_sql_query("SELECT * FROM features_v5", conn)
    conn.close()
    base_df['trade_date'] = pd.to_datetime(base_df['trade_date'])

    print(f"\n基线数据: {len(base_df)}条, {base_df['symbol'].nunique()}只股票")

    # 基础特征列
    EXCLUDE = {'trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount',
               'target_return_10d', 'target_direction_10d', 'scenario_label_10d',
               'scenario_label', 'binary_label', 'scenario_int'}
    base_feature_cols = [c for c in base_df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(base_df[c])]
    print(f"基础特征数: {len(base_feature_cols)}")

    all_results = []

    # ============================================================
    # Baseline: 原始方案 (10日收益率, 标准回测)
    # ============================================================
    print(f"\n{'=' * 80}")
    print("Baseline: 10日收益率 + 标准回测")
    print(f"{'=' * 80}")

    df = base_df.copy()
    r = df['target_return_10d']
    df['scenario_label'] = make_scenario_label(r, 0.05, -0.05)
    df['scenario_int'] = df['scenario_label'].map({'adverse': 0, 'base': 1, 'favorable': 2})

    model_dir = os.path.join(BASE_MODEL_DIR, 'exp_baseline')
    metrics = train_classifier(df, base_feature_cols, 'scenario_int', 'baseline', model_dir)

    pred_table = 'predictions_exp_baseline'
    pred_df = generate_predictions(df, base_feature_cols, model_dir, 'baseline', pred_table, 'classifier')

    results_table = 'backtest_results_exp_baseline'
    daily_table = 'backtest_daily_exp_baseline'
    bt_df = run_backtest(pred_df, 'baseline', results_table, daily_table, 'standard')

    all_results.append({
        'name': 'Baseline(10d)',
        'metrics': metrics,
        'backtest': bt_df,
    })

    # ============================================================
    # 方案1: 缩短预测窗口 (5日)
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案1A: 缩短预测窗口 (5日)")
    print(f"{'=' * 80}")

    df = base_df.copy()
    # 生成5日收益率
    df['target_return_5d'] = df['close'].shift(-5) / df['close'] - 1
    r = df['target_return_5d']
    df['scenario_label'] = make_scenario_label(r, 0.03, -0.03)
    df['scenario_int'] = df['scenario_label'].map({'adverse': 0, 'base': 1, 'favorable': 2})
    df = df.dropna(subset=['target_return_5d'])

    model_dir = os.path.join(BASE_MODEL_DIR, 'exp_5d')
    metrics = train_classifier(df, base_feature_cols, 'scenario_int', 'exp_5d', model_dir)

    pred_table = 'predictions_exp_5d'
    pred_df = generate_predictions(df, base_feature_cols, model_dir, 'exp_5d', pred_table, 'classifier')

    results_table = 'backtest_results_exp_5d'
    daily_table = 'backtest_daily_exp_5d'
    bt_df = run_backtest(pred_df, 'exp_5d', results_table, daily_table, 'standard')

    all_results.append({
        'name': '1A: 5日窗口',
        'metrics': metrics,
        'backtest': bt_df,
    })

    # ============================================================
    # 方案1B: 缩短预测窗口 (3日)
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案1B: 缩短预测窗口 (3日)")
    print(f"{'=' * 80}")

    df = base_df.copy()
    # 生成3日收益率
    df['target_return_3d'] = df['close'].shift(-3) / df['close'] - 1
    r = df['target_return_3d']
    df['scenario_label'] = make_scenario_label(r, 0.02, -0.02)
    df['scenario_int'] = df['scenario_label'].map({'adverse': 0, 'base': 1, 'favorable': 2})
    df = df.dropna(subset=['target_return_3d'])

    model_dir = os.path.join(BASE_MODEL_DIR, 'exp_3d')
    metrics = train_classifier(df, base_feature_cols, 'scenario_int', 'exp_3d', model_dir)

    pred_table = 'predictions_exp_3d'
    pred_df = generate_predictions(df, base_feature_cols, model_dir, 'exp_3d', pred_table, 'classifier')

    results_table = 'backtest_results_exp_3d'
    daily_table = 'backtest_daily_exp_3d'
    bt_df = run_backtest(pred_df, 'exp_3d', results_table, daily_table, 'standard')

    all_results.append({
        'name': '1B: 3日窗口',
        'metrics': metrics,
        'backtest': bt_df,
    })

    # ============================================================
    # 方案2: 使用回归任务
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案2: 使用回归任务 (直接预测收益率)")
    print(f"{'=' * 80}")

    df = base_df.copy()
    r = df['target_return_10d']

    model_dir = os.path.join(BASE_MODEL_DIR, 'exp_regressor')
    reg_metrics = train_regressor(df, base_feature_cols, 'target_return_10d', 'exp_regressor', model_dir)
    print(f"  RF RMSE: {reg_metrics['rf_rmse']:.4f}  R²: {reg_metrics['rf_r2']:.4f}")
    print(f"  LGBM RMSE: {reg_metrics['lgbm_rmse']:.4f}  R²: {reg_metrics['lgbm_r2']:.4f}")
    print(f"  Fusion RMSE: {reg_metrics['fusion_rmse']:.4f}  R²: {reg_metrics['fusion_r2']:.4f}")

    pred_table = 'predictions_exp_regressor'
    pred_df = generate_predictions(df, base_feature_cols, model_dir, 'exp_regressor', pred_table, 'regressor')

    results_table = 'backtest_results_exp_regressor'
    daily_table = 'backtest_daily_exp_regressor'
    bt_df = run_backtest(pred_df, 'exp_regressor', results_table, daily_table, 'standard')

    all_results.append({
        'name': '2: 回归任务',
        'metrics': reg_metrics,
        'backtest': bt_df,
    })

    # ============================================================
    # 方案3: 增加特征
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案3: 增加特征 (参考Qlib Alpha158)")
    print(f"{'=' * 80}")

    df = add_enhanced_features(base_df)
    enhanced_feature_cols = [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  增强特征数: {len(enhanced_feature_cols)}")

    r = df['target_return_10d']
    df['scenario_label'] = make_scenario_label(r, 0.05, -0.05)
    df['scenario_int'] = df['scenario_label'].map({'adverse': 0, 'base': 1, 'favorable': 2})
    df = df.dropna(subset=enhanced_feature_cols + ['scenario_int'])

    model_dir = os.path.join(BASE_MODEL_DIR, 'exp_enhanced')
    metrics = train_classifier(df, enhanced_feature_cols, 'scenario_int', 'exp_enhanced', model_dir)

    pred_table = 'predictions_exp_enhanced'
    pred_df = generate_predictions(df, enhanced_feature_cols, model_dir, 'exp_enhanced', pred_table, 'classifier')

    results_table = 'backtest_results_exp_enhanced'
    daily_table = 'backtest_daily_exp_enhanced'
    bt_df = run_backtest(pred_df, 'exp_enhanced', results_table, daily_table, 'standard')

    all_results.append({
        'name': '3: 增强特征',
        'metrics': metrics,
        'backtest': bt_df,
    })

    # ============================================================
    # 方案4: 优化回测策略 (基于概率的动态仓位)
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案4: 优化回测策略 (基于概率的动态仓位)")
    print(f"{'=' * 80}")

    # 使用 Baseline 的预测，但用新的回测策略
    pred_table = 'predictions_exp_baseline'
    pred_df = pd.read_sql_query(f"SELECT * FROM {pred_table}", sqlite3.connect(DB_PATH))

    results_table = 'backtest_results_exp_probabilistic'
    daily_table = 'backtest_daily_exp_probabilistic'
    bt_df = run_backtest(pred_df, 'exp_probabilistic', results_table, daily_table, 'probabilistic')

    all_results.append({
        'name': '4: 概率仓位',
        'metrics': all_results[0]['metrics'],  # 使用 Baseline 的指标
        'backtest': bt_df,
    })

    # ============================================================
    # 汇总对比
    # ============================================================
    print(f"\n\n{'=' * 100}")
    print("汇总对比: 所有方案")
    print(f"{'=' * 100}")

    # 模型指标对比
    print(f"\n{'方案':<20s} {'RF_acc':>10s} {'LGBM_acc':>12s} {'Fusion_acc':>14s}")
    print("-" * 60)
    for r in all_results:
        m = r['metrics']
        if 'fusion_acc' in m:
            print(f"{r['name']:<20s} {m.get('rf_acc', 0):>10.4f} {m.get('lgbm_acc', 0):>12.4f} {m.get('fusion_acc', 0):>14.4f}")
        elif 'fusion_r2' in m:
            print(f"{r['name']:<20s} {'RMSE':>10s} {'R²':>12s} {'':>14s}")
            print(f"{'':20s} {m.get('fusion_rmse', 0):>10.4f} {m.get('fusion_r2', 0):>12.4f}")

    # 回测对比
    print(f"\n{'方案':<20s} {'总收益':>8s} {'年化':>8s} {'最大回撤':>8s} {'夏普':>6s} {'超额':>8s} {'平均仓位':>8s}")
    print("-" * 70)
    for r in all_results:
        bt = r['backtest']
        if len(bt) > 0:
            print(f"{r['name']:<20s} {bt['total_return'].mean()*100:>7.2f}% {bt['annual_return'].mean()*100:>7.2f}% "
                  f"{bt['max_drawdown'].mean()*100:>7.2f}% {bt['sharpe'].mean():>6.2f} "
                  f"{bt['excess_return'].mean()*100:>7.2f}% {bt['avg_position'].mean()*100:>7.1f}%")

    # 确定最优方案
    print(f"\n{'=' * 100}")
    print("结论")
    print(f"{'=' * 100}")

    # 按回测超额收益排序
    bt_results_with_data = [r for r in all_results if len(r['backtest']) > 0]
    if bt_results_with_data:
        best_bt = max(bt_results_with_data, key=lambda x: x['backtest']['excess_return'].mean())
        print(f"  最高超额收益: {best_bt['name']} ({best_bt['backtest']['excess_return'].mean()*100:.2f}%)")

    # 按夏普比率排序
    if bt_results_with_data:
        best_sharpe = max(bt_results_with_data, key=lambda x: x['backtest']['sharpe'].mean())
        print(f"  最高夏普比率: {best_sharpe['name']} ({best_sharpe['backtest']['sharpe'].mean():.2f})")

    # 按准确率排序（仅分类模型）
    cls_results = [r for r in all_results if 'fusion_acc' in r['metrics']]
    if cls_results:
        best_acc = max(cls_results, key=lambda x: x['metrics']['fusion_acc'])
        print(f"  最高准确率: {best_acc['name']} ({best_acc['metrics']['fusion_acc']:.4f})")

    # 保存结果
    results_summary = {
        'experiments': [],
    }
    for r in all_results:
        exp_data = {'name': r['name']}
        if 'fusion_acc' in r['metrics']:
            exp_data['accuracy'] = r['metrics']['fusion_acc']
        if 'fusion_r2' in r['metrics']:
            exp_data['r2'] = r['metrics']['fusion_r2']
        if len(r['backtest']) > 0:
            bt = r['backtest']
            exp_data['backtest'] = {
                'total_return': float(bt['total_return'].mean()),
                'annual_return': float(bt['annual_return'].mean()),
                'max_drawdown': float(bt['max_drawdown'].mean()),
                'sharpe': float(bt['sharpe'].mean()),
                'excess_return': float(bt['excess_return'].mean()),
            }
        results_summary['experiments'].append(exp_data)

    with open(os.path.join(BASE_MODEL_DIR, 'improvement_experiment_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {os.path.join(BASE_MODEL_DIR, 'improvement_experiment_results.json')}")


if __name__ == '__main__':
    main()
