"""
实验对比脚本 - 三个优化方案 vs 原始baseline
方案A: 阈值 +3%/-2% (更平衡的三情景)
方案B: +A基础上改二分类 (>+1%=up, <-1%=down, 中间丢弃)
方案C: +B基础上加 class_weight='balanced'

每个方案输出: 独立的 features/predictions/backtest 表 + 模型目录
最后汇总对比所有方案
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'
BASE_MODEL_DIR = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\models'

# ============================================================
# 1. 标签生成函数
# ============================================================

def make_scenario_label(r, fav_th, adv_th):
    """根据阈值生成三情景标签"""
    labels = pd.Series('base', index=r.index)
    labels[r < adv_th] = 'adverse'
    labels[r > fav_th] = 'favorable'
    return labels


def make_binary_label(r, mode='simple', up_th=0.0, down_th=0.0):
    """
    mode='simple': r > 0 => 1, else => 0
    mode='filtered': r > up_th => 1, r < down_th => 0, middle => NaN
    """
    if mode == 'simple':
        return (r > up_th).astype(int)
    elif mode == 'filtered':
        labels = pd.Series(np.nan, index=r.index)
        labels[r > up_th] = 1
        labels[r < down_th] = 0
        return labels
    return (r > 0).astype(int)


# ============================================================
# 2. 训练 + 评估 (复用 v5 训练逻辑)
# ============================================================

def train_and_evaluate(df, feature_cols, label_col, exp_name, model_dir, class_weight_mode=False):
    """训练 RF + LGBM (scenario + binary), 返回指标dict"""

    y = df[label_col].astype(int)
    X = df[feature_cols].copy()

    # 时间划分
    train_mask = df['trade_date'] < '2023-01-01'
    val_mask = (df['trade_date'] >= '2023-01-01') & (df['trade_date'] < '2024-01-01')
    test_mask = df['trade_date'] >= '2024-01-01'

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # 手动过采样 (仅训练集, 仅当不使用 class_weight 时)
    if not class_weight_mode:
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
    if class_weight_mode:
        rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
    else:
        rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                     random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    # LGBM
    n_classes = len(y.unique())
    if class_weight_mode:
        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        cw_dict = dict(zip(classes, cw))
        sample_weights = y_train.map(cw_dict).values
        lgbm_params = {
            'objective': 'multiclass' if n_classes > 2 else 'binary',
            'num_class': n_classes if n_classes > 2 else 1,
            'n_estimators': 1000, 'learning_rate': 0.05,
            'max_depth': 8, 'num_leaves': 63, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'random_state': 42, 'verbosity': -1,
            'n_jobs': -1,
        }
        lgbm = lgb.LGBMClassifier(**lgbm_params)
        lgbm.fit(X_train, y_train, sample_weight=sample_weights,
                 eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    else:
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
    with open(os.path.join(model_dir, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)

    # 测试集详细报告
    classes_sorted = sorted(y_test.unique())
    report = classification_report(y_test, fusion_pred, target_names=[str(c) for c in classes_sorted],
                                    output_dict=True, zero_division=0)

    return {
        'exp_name': exp_name,
        'n_train': len(X_train), 'n_val': len(X_val), 'n_test': len(X_test),
        'rf_acc': rf_acc, 'lgbm_acc': lgbm_acc, 'fusion_acc': fusion_acc,
        'n_classes': n_classes,
        'class_dist': y.value_counts().to_dict(),
        'test_class_dist': y_test.value_counts().to_dict(),
        'report': report,
        'confusion': confusion_matrix(y_test, fusion_pred).tolist(),
        'model_dir': model_dir,
    }


# ============================================================
# 3. 生成预测 (复用 v5 predict 逻辑)
# ============================================================

def generate_predictions(df, feature_cols, model_dir, exp_name, table_name):
    """对测试集生成融合预测, 写入DB"""
    with open(os.path.join(model_dir, 'rf_model.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(model_dir, 'lgbm_model.pkl'), 'rb') as f:
        lgbm = pickle.load(f)

    test_mask = df['trade_date'] >= '2024-01-01'
    test_df = df[test_mask].copy()
    X_test = test_df[feature_cols].copy()

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
# 4. 回测 (复用 backtest_v5.py 策略逻辑)
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
    'ma_bullish_required': False,
}
SCENARIO_POSITIONS = {'adverse': 0.0, 'base': 0.5, 'favorable': 1.0}


def run_backtest(pred_df, exp_name, results_table, daily_table):
    """对预测结果运行回测策略"""
    price_df = pd.read_sql_query(
        "SELECT trade_date, symbol, open, high, low, close, volume FROM daily_prices_v5",
        sqlite3.connect(DB_PATH)
    )
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    pred_df = pred_df.copy()
    pred_df['trade_date'] = pd.to_datetime(pred_df['trade_date'])
    # 修复 3: 合并前重命名 pred_df 的 close 列，避免 close_x/close_y 问题
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
        # 修复 2: 为每个 symbol 单独维护 daily_returns
        sym_daily_navs = []

        for idx in range(len(sym_data)):
            row = sym_data.iloc[idx]
            date = row['trade_date']
            # 修复 3: 使用 price_df 的 close 列（合并后就是 'close'）
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

            # 三情景基础仓位
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
                        if not TREND_CONFIG['ma_bullish_required'] or ma_bullish:
                            new_pos = min(base_position + TREND_CONFIG['add_position_step'], TREND_CONFIG['add_position_max'])
                            if new_pos > base_position:
                                trend_position = new_pos
                    elif return_120d > TREND_CONFIG['trend_threshold_moderate']:
                        if not TREND_CONFIG['ma_bullish_required'] or ma_bullish:
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

            # 修复 2: 记录每日 nav 用于计算 daily_returns
            sym_daily_navs.append(nav)

            daily_records.append({
                'trade_date': date, 'symbol': symbol, 'strategy': exp_name,
                'nav': nav, 'position': position, 'scenario': scenario,
                'close': close
            })

        if len(sym_data) > 1:
            total_return = nav - 1.0
            start_date = sym_data['trade_date'].iloc[0]
            end_date = sym_data['trade_date'].iloc[-1]
            years = (end_date - start_date).days / 365.25
            annual_return = (nav ** (1 / years) - 1) if years > 0 and nav > 0 else 0
            # 修复 3: 使用 price_df 的 close 列计算 benchmark
            benchmark_return = (sym_data['close'].iloc[-1] / sym_data['close'].iloc[0]) - 1

            # 修复 2: 使用 sym_daily_navs 计算 daily_returns
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
# 5. 实验执行器
# ============================================================

def run_experiment(exp_name, fav_th, adv_th, binary_mode, binary_up_th, binary_down_th,
                   class_weight, feature_cols, base_df):
    """运行单个实验: 标签生成 → 训练 → 预测 → 回测"""
    print(f"\n{'=' * 70}")
    print(f"实验: {exp_name}")
    print(f"  scenario阈值: favorable > {fav_th*100:+.0f}%, adverse < {adv_th*100:+.0f}%")
    print(f"  binary模式: {binary_mode}, up_th={binary_up_th}, down_th={binary_down_th}")
    print(f"  class_weight: {class_weight}")
    print(f"{'=' * 70}")

    df = base_df.copy()

    # 生成三情景标签（始终使用全部样本）
    r = df['target_return_10d']
    df['scenario_label'] = make_scenario_label(r, fav_th, adv_th)
    df['scenario_int'] = df['scenario_label'].map({'adverse': 0, 'base': 1, 'favorable': 2})

    # 打印三情景分布
    print(f"\n  三情景分布:")
    for lbl in ['adverse', 'base', 'favorable']:
        n = (df['scenario_label'] == lbl).sum()
        print(f"    {lbl}: {n} ({n / len(df) * 100:.1f}%)")

    # ---- Scenario 模型（使用全部样本） ----
    model_dir_s = os.path.join(BASE_MODEL_DIR, exp_name, 'scenario')
    print(f"\n  训练三情景模型（全部 {len(df)} 条样本）...")
    scenario_metrics = train_and_evaluate(df, feature_cols, 'scenario_int', exp_name + '_scenario',
                                          model_dir_s, class_weight_mode=class_weight)
    print(f"    RF: {scenario_metrics['rf_acc']:.4f}  LGBM: {scenario_metrics['lgbm_acc']:.4f}  Fusion: {scenario_metrics['fusion_acc']:.4f}")

    # ---- 生成 binary 标签 ----
    if binary_mode == 'simple':
        df['binary_label'] = make_binary_label(r, mode='simple')
        df_binary = df
    else:
        df['binary_label'] = make_binary_label(r, mode='filtered', up_th=binary_up_th, down_th=binary_down_th)
        df_binary = df.dropna(subset=['binary_label'])
        print(f"\n  丢弃中间区间样本（仅影响 binary）: {len(df)} -> {len(df_binary)} ({len(df) - len(df_binary)}条被丢弃)")

    # 打印二分类分布
    print(f"\n  二分类分布:")
    for v in sorted(df_binary['binary_label'].unique()):
        n = (df_binary['binary_label'] == v).sum()
        print(f"    {int(v)}: {n} ({n / len(df_binary) * 100:.1f}%)")

    # ---- Binary 模型（使用过滤后的样本） ----
    model_dir_b = os.path.join(BASE_MODEL_DIR, exp_name, 'binary')
    print(f"\n  训练二分类模型（{len(df_binary)} 条样本）...")
    binary_metrics = train_and_evaluate(df_binary, feature_cols, 'binary_label', exp_name + '_binary',
                                        model_dir_b, class_weight_mode=class_weight)
    print(f"    RF: {binary_metrics['rf_acc']:.4f}  LGBM: {binary_metrics['lgbm_acc']:.4f}  Fusion: {binary_metrics['fusion_acc']:.4f}")

    # ---- 生成预测 (用 scenario 模型，全部样本) ----
    pred_table = f"predictions_{exp_name}"
    print(f"\n  生成预测...")
    pred_df = generate_predictions(df, feature_cols, model_dir_s, exp_name, pred_table)

    # ---- 回测 ----
    results_table = f"backtest_results_{exp_name}"
    daily_table = f"backtest_daily_{exp_name}"
    print(f"\n  运行回测...")
    bt_df = run_backtest(pred_df, exp_name, results_table, daily_table)
    if len(bt_df) > 0:
        print(f"  平均总收益: {bt_df['total_return'].mean()*100:.2f}%  "
              f"平均超额: {bt_df['excess_return'].mean()*100:.2f}%  "
              f"平均夏普: {bt_df['sharpe'].mean():.2f}")

    return {
        'exp_name': exp_name,
        'scenario_metrics': scenario_metrics,
        'binary_metrics': binary_metrics,
        'backtest': bt_df,
    }


# ============================================================
# 6. 主流程
# ============================================================

def main():
    print("=" * 70)
    print("优化实验: A(阈值) / B(二分类) / C(加权)")
    print("=" * 70)

    # 加载基线数据
    conn = sqlite3.connect(DB_PATH)
    base_df = pd.read_sql_query("SELECT * FROM features_v5", conn)
    conn.close()
    base_df['trade_date'] = pd.to_datetime(base_df['trade_date'])

    # 数值列自动识别为特征
    EXCLUDE = {'trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount',
               'target_return_10d', 'target_direction_10d', 'scenario_label_10d',
               'scenario_label', 'binary_label', 'scenario_int'}
    feature_cols = [c for c in base_df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(base_df[c])]
    print(f"\n特征数: {len(feature_cols)}")
    print(f"基线数据: {len(base_df)}条, {base_df['symbol'].nunique()}只股票")

    # 加载原始 baseline 指标（仅用于参考）
    metrics_path = os.path.join(BASE_MODEL_DIR, 'v5', 'metrics_v5.json')
    baseline_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            baseline_metrics = json.load(f)
        print(f"\n原始Baseline (v5) 指标（来自之前的训练）:")
        print(f"  RF 3-scenario: {baseline_metrics.get('rf_scenario_acc', 'N/A')}")
        print(f"  LGBM 3-scenario: {baseline_metrics.get('lgb_scenario_acc', 'N/A')}")
        print(f"  Fusion 3-scenario: {baseline_metrics.get('fusion_scenario_acc', 'N/A')}")
        print(f"  RF Binary: {baseline_metrics.get('rf_binary_acc', 'N/A')}")
        print(f"  LGBM Binary: {baseline_metrics.get('lgb_binary_acc', 'N/A')}")

    # ---- Baseline: 原始阈值 +5%/-5% ----
    print(f"\n{'=' * 70}")
    print(f"运行 Baseline 回测（使用 v5 原始阈值 +5%/-5%）...")
    print(f"{'=' * 70}")
    result_baseline = run_experiment(
        exp_name='baseline',
        fav_th=0.05, adv_th=-0.05,
        binary_mode='simple',
        binary_up_th=0.0, binary_down_th=0.0,
        class_weight=False,
        feature_cols=feature_cols, base_df=base_df
    )

    # ---- 方案A: 阈值 +3%/-2% ----
    result_a = run_experiment(
        exp_name='exp_a',
        fav_th=0.03, adv_th=-0.02,
        binary_mode='simple',
        binary_up_th=0.0, binary_down_th=0.0,
        class_weight=False,
        feature_cols=feature_cols, base_df=base_df
    )

    # ---- 方案B: +A基础上, 二分类改为 >+1%/<-1% (丢弃中间) ----
    result_b = run_experiment(
        exp_name='exp_b',
        fav_th=0.03, adv_th=-0.02,
        binary_mode='filtered',
        binary_up_th=0.01, binary_down_th=-0.01,
        class_weight=False,
        feature_cols=feature_cols, base_df=base_df
    )

    # ---- 方案C: +B基础上, class_weight=balanced ----
    result_c = run_experiment(
        exp_name='exp_c',
        fav_th=0.03, adv_th=-0.02,
        binary_mode='filtered',
        binary_up_th=0.01, binary_down_th=-0.01,
        class_weight=True,
        feature_cols=feature_cols, base_df=base_df
    )

    # ============================================================
    # 汇总对比
    # ============================================================
    print(f"\n\n{'=' * 90}")
    print("汇总对比: Baseline vs A vs B vs C")
    print(f"{'=' * 90}")

    all_results = [
        {'name': 'Baseline(+5%/-5%)', 's': result_baseline['scenario_metrics'], 'b': result_baseline['binary_metrics'], 'bt': result_baseline['backtest']},
        {'name': 'A: +3%/-2%', 's': result_a['scenario_metrics'], 'b': result_a['binary_metrics'], 'bt': result_a['backtest']},
        {'name': 'B: +二分类', 's': result_b['scenario_metrics'], 'b': result_b['binary_metrics'], 'bt': result_b['backtest']},
        {'name': 'C: +class_wt', 's': result_c['scenario_metrics'], 'b': result_c['binary_metrics'], 'bt': result_c['backtest']},
    ]

    # 模型准确率对比
    print(f"\n{'方案':<20s} {'RF_3class':>10s} {'LGBM_3class':>12s} {'Fusion_3class':>14s} {'RF_bin':>8s} {'LGBM_bin':>10s}")
    print("-" * 80)
    for r in all_results:
        s = r['s']
        b = r['b']
        rf_s = s.get('rf_acc', 0)
        lgb_s = s.get('lgbm_acc', 0)
        fu_s = s.get('fusion_acc', 0)
        rf_b = b.get('rf_acc', 0)
        lgb_b = b.get('lgbm_acc', 0)
        print(f"{r['name']:<20s} {rf_s:>10.4f} {lgb_s:>12.4f} {fu_s:>14.4f} {rf_b:>8.4f} {lgb_b:>10.4f}")

    # 回测对比
    bt_results = [r for r in all_results if r['bt'] is not None and len(r['bt']) > 0]
    if bt_results:
        print(f"\n{'方案':<20s} {'总收益':>8s} {'年化':>8s} {'最大回撤':>8s} {'夏普':>6s} {'超额':>8s} {'平均仓位':>8s}")
        print("-" * 70)
        for r in bt_results:
            bt = r['bt']
            print(f"{r['name']:<20s} {bt['total_return'].mean()*100:>7.2f}% {bt['annual_return'].mean()*100:>7.2f}% "
                  f"{bt['max_drawdown'].mean()*100:>7.2f}% {bt['sharpe'].mean():>6.2f} "
                  f"{bt['excess_return'].mean()*100:>7.2f}% {bt['avg_position'].mean()*100:>7.1f}%")

    print(f"\n{'=' * 90}")
    print("结论")
    print(f"{'=' * 90}")
    best_s = max(all_results, key=lambda x: x['s'].get('fusion_acc', 0))
    bt_best = max(bt_results, key=lambda x: x['bt']['excess_return'].mean()) if bt_results else None
    print(f"  最高三情景准确率: {best_s['name']} ({best_s['s'].get('fusion_acc', 0):.4f})")
    if bt_best:
        print(f"  最高超额收益: {bt_best['name']} ({bt_best['bt']['excess_return'].mean()*100:.2f}%)")

    # 保存结果到 JSON
    results_summary = {
        'experiments': [],
        'best_scenario_acc': {'name': best_s['name'], 'value': best_s['s'].get('fusion_acc', 0)},
    }
    for r in all_results:
        exp_data = {
            'name': r['name'],
            'scenario_acc': {
                'rf': r['s'].get('rf_acc', 0),
                'lgbm': r['s'].get('lgbm_acc', 0),
                'fusion': r['s'].get('fusion_acc', 0),
            },
            'binary_acc': {
                'rf': r['b'].get('rf_acc', 0),
                'lgbm': r['b'].get('lgbm_acc', 0),
            },
        }
        if r['bt'] is not None and len(r['bt']) > 0:
            bt = r['bt']
            exp_data['backtest'] = {
                'total_return': float(bt['total_return'].mean()),
                'annual_return': float(bt['annual_return'].mean()),
                'max_drawdown': float(bt['max_drawdown'].mean()),
                'sharpe': float(bt['sharpe'].mean()),
                'excess_return': float(bt['excess_return'].mean()),
            }
        results_summary['experiments'].append(exp_data)

    if bt_best:
        results_summary['best_excess_return'] = {'name': bt_best['name'], 'value': float(bt_best['bt']['excess_return'].mean())}

    with open(os.path.join(BASE_MODEL_DIR, 'experiment_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {os.path.join(BASE_MODEL_DIR, 'experiment_results.json')}")


if __name__ == '__main__':
    main()
