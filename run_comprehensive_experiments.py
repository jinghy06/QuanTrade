"""
全面改进方案测试框架
包含：
1. 概率加权仓位策略
2. 降低预警敏感度策略
3. 增加有效特征方案
4. 改用回归任务方案
5. 缩短预测窗口方案
6. 深度学习算法（LSTM/GRU/Transformer）
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'
BASE_MODEL_DIR = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\models'

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ============================================================
# 1. 增强特征工程
# ============================================================

def add_enhanced_features(df):
    """增加更多特征，参考Qlib Alpha158"""
    print("  增加增强特征...")
    result = []

    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        s = df.loc[mask].copy()

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

        # 9. 更多统计特征
        for window in [10, 20]:
            s[f'skew_{window}d'] = s['close'].pct_change().rolling(window).skew()
            s[f'kurt_{window}d'] = s['close'].pct_change().rolling(window).kurt()

        # 10. 价格变化率
        for window in [1, 2, 3]:
            s[f'price_change_{window}d'] = s['close'].pct_change(window)

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
# 3. 深度学习模型
# ============================================================

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, X, y, seq_len=10):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len]


class LSTMModel(nn.Module):
    """LSTM模型"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class GRUModel(nn.Module):
    """GRU模型"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        return output


class TransformerModel(nn.Module):
    """Transformer模型"""
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=1, output_size=1, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        x = self.input_projection(x)
        transformer_out = self.transformer_encoder(x)
        last_output = transformer_out[:, -1, :]
        output = self.fc(last_output)
        return output


# ============================================================
# 4. 训练函数
# ============================================================

def train_dl_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    """训练深度学习模型"""
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output.squeeze(), y_batch)
                val_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    早停于 epoch {epoch+1}")
                break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


# ============================================================
# 5. 回测策略
# ============================================================

CRASH_CONFIG_STANDARD = {
    'vol_spike_ratio': 1.5, 'drawdown_threshold': 0.15,
    'liquidity_dryup_ratio': 0.5, 'red_alert_reduction': 1.0,
    'orange_alert_reduction': 0.5, 'yellow_alert_reduction': 0.2,
    'recovery_required_days': 5,
}

CRASH_CONFIG_RELAXED = {
    'vol_spike_ratio': 2.0, 'drawdown_threshold': 0.25,
    'liquidity_dryup_ratio': 0.3, 'red_alert_reduction': 0.8,
    'orange_alert_reduction': 0.3, 'yellow_alert_reduction': 0.1,
    'recovery_required_days': 3,
}

TREND_CONFIG = {
    'trend_threshold_strong': 0.15,
    'trend_threshold_moderate': 0.08,
    'add_position_step': 0.10,
    'add_position_max': 0.90,
}


def run_backtest(pred_df, exp_name, results_table, daily_table, crash_config, strategy='standard'):
    """运行回测"""
    conn = sqlite3.connect(DB_PATH)
    price_df = pd.read_sql_query(
        "SELECT trade_date, symbol, open, high, low, close, volume FROM daily_prices_v5",
        conn
    )
    conn.close()

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
            if not pd.isna(row.get('vol_spike')) and row['vol_spike'] > crash_config['vol_spike_ratio']:
                alerts += 1
            if not pd.isna(row.get('drawdown_from_high')) and row['drawdown_from_high'] < -crash_config['drawdown_threshold']:
                alerts += 1
            if not pd.isna(row.get('liquidity_ratio')) and row['liquidity_ratio'] < crash_config['liquidity_dryup_ratio']:
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
                if alert_active and consecutive_safe_days >= crash_config['recovery_required_days']:
                    alert_active = False

            if alerts >= 3:
                crash_reduction = crash_config['red_alert_reduction']
            elif alerts == 2:
                crash_reduction = crash_config['orange_alert_reduction']
            elif alerts == 1:
                crash_reduction = crash_config['yellow_alert_reduction']
            else:
                crash_reduction = 0.0

            # 计算仓位
            if strategy == 'probabilistic':
                # 概率加权仓位
                p_adverse = row.get('adverse', 0)
                p_base = row.get('base', 0)
                p_favorable = row.get('favorable', 0)
                base_position = p_favorable * 1.0 + p_base * 0.5 + p_adverse * 0.0
            elif strategy == 'aggressive':
                # 激进策略
                p_adverse = row.get('adverse', 0)
                p_base = row.get('base', 0)
                p_favorable = row.get('favorable', 0)
                base_position = p_favorable * 1.2 + p_base * 0.7 + p_adverse * 0.2
                base_position = min(1.0, base_position)
            else:
                # 标准策略
                probs = {'adverse': row.get('adverse', 0), 'base': row.get('base', 0), 'favorable': row.get('favorable', 0)}
                scenario = max(probs, key=probs.get)
                if scenario == 'favorable':
                    base_position = 1.0
                elif scenario == 'base':
                    base_position = 0.5
                else:
                    base_position = 0.0

            # 趋势加仓
            trend_position = base_position
            if not alert_active:
                return_120d = row.get('return_120d')
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
# 6. 主流程
# ============================================================

def main():
    print("=" * 80)
    print("全面改进方案测试框架")
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
    # 方案1: 概率加权仓位策略
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案1: 概率加权仓位策略")
    print(f"{'=' * 80}")

    # 使用 Baseline 的预测
    conn = sqlite3.connect(DB_PATH)
    pred_df = pd.read_sql_query("SELECT * FROM predictions_exp_baseline", conn)
    conn.close()

    results_table = 'backtest_results_probabilistic'
    daily_table = 'backtest_daily_probabilistic'
    bt_df = run_backtest(pred_df, 'probabilistic', results_table, daily_table,
                         CRASH_CONFIG_STANDARD, 'probabilistic')

    all_results.append({
        'name': '1: 概率仓位',
        'backtest': bt_df,
    })

    # ============================================================
    # 方案2: 降低预警敏感度策略
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案2: 降低预警敏感度策略")
    print(f"{'=' * 80}")

    conn = sqlite3.connect(DB_PATH)
    pred_df = pd.read_sql_query("SELECT * FROM predictions_exp_baseline", conn)
    conn.close()

    results_table = 'backtest_results_relaxed'
    daily_table = 'backtest_daily_relaxed'
    bt_df = run_backtest(pred_df, 'relaxed', results_table, daily_table,
                         CRASH_CONFIG_RELAXED, 'standard')

    all_results.append({
        'name': '2: 降低预警',
        'backtest': bt_df,
    })

    # ============================================================
    # 方案3: 增加有效特征 + 概率加权仓位
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案3: 增加有效特征 + 概率加权仓位")
    print(f"{'=' * 80}")

    df = add_enhanced_features(base_df)
    enhanced_feature_cols = [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  增强特征数: {len(enhanced_feature_cols)}")

    r = df['target_return_10d']
    df['scenario_label'] = make_scenario_label(r, 0.05, -0.05)
    df['scenario_int'] = df['scenario_label'].map({'adverse': 0, 'base': 1, 'favorable': 2})
    df = df.dropna(subset=enhanced_feature_cols + ['scenario_int'])

    # 训练模型
    y = df['scenario_int'].astype(int)
    X = df[enhanced_feature_cols].copy()

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
    X_train_bal = X_train.loc[oversampled_indices]
    y_train_bal = y_train.loc[oversampled_indices]

    # RF
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)

    # LGBM
    lgbm = lgb.LGBMClassifier(
        objective='multiclass', num_class=3,
        n_estimators=1000, learning_rate=0.05,
        max_depth=8, num_leaves=63, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=-1
    )
    lgbm.fit(X_train_bal, y_train_bal,
             eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    # 生成预测
    rf_proba = rf.predict_proba(X_test)
    lgbm_proba = lgbm.predict_proba(X_test)
    fusion_proba = (rf_proba + lgbm_proba) / 2
    fusion_pred = np.argmax(fusion_proba, axis=1)

    test_df = df[test_mask].copy()
    pred_records = []
    for i, idx in enumerate(test_df.index):
        row = {
            'trade_date': str(test_df.loc[idx, 'trade_date']),
            'symbol': test_df.loc[idx, 'symbol'],
            'close': float(test_df.loc[idx, 'close']),
            'prediction': ['adverse', 'base', 'favorable'][fusion_pred[i]],
            'adverse': float(fusion_proba[i][0]),
            'base': float(fusion_proba[i][1]),
            'favorable': float(fusion_proba[i][2]),
        }
        pred_records.append(row)

    pred_df = pd.DataFrame(pred_records)

    results_table = 'backtest_results_enhanced_prob'
    daily_table = 'backtest_daily_enhanced_prob'
    bt_df = run_backtest(pred_df, 'enhanced_prob', results_table, daily_table,
                         CRASH_CONFIG_STANDARD, 'probabilistic')

    all_results.append({
        'name': '3: 增强特征+概率',
        'backtest': bt_df,
    })

    # ============================================================
    # 方案4: 改用回归任务
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案4: 改用回归任务")
    print(f"{'=' * 80}")

    df = base_df.copy()
    y = df['target_return_10d']
    X = df[base_feature_cols].copy()

    train_mask = df['trade_date'] < '2023-01-01'
    val_mask = (df['trade_date'] >= '2023-01-01') & (df['trade_date'] < '2024-01-01')
    test_mask = df['trade_date'] >= '2024-01-01'

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # RF
    rf_reg = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                    random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train)

    # LGBM
    lgbm_reg = lgb.LGBMRegressor(
        objective='regression', metric='rmse',
        n_estimators=1000, learning_rate=0.05,
        max_depth=8, num_leaves=63, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=-1
    )
    lgbm_reg.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    # 生成预测
    rf_pred = rf_reg.predict(X_test)
    lgbm_pred = lgbm_reg.predict(X_test)
    fusion_pred = (rf_pred + lgbm_pred) / 2

    test_df = df[test_mask].copy()
    pred_records = []
    for i, idx in enumerate(test_df.index):
        pred_value = fusion_pred[i]
        # 将回归预测转换为概率
        if pred_value > 0.05:
            p_favorable = 0.7
            p_base = 0.2
            p_adverse = 0.1
        elif pred_value > 0.02:
            p_favorable = 0.5
            p_base = 0.3
            p_adverse = 0.2
        elif pred_value > 0:
            p_favorable = 0.3
            p_base = 0.5
            p_adverse = 0.2
        elif pred_value > -0.02:
            p_favorable = 0.2
            p_base = 0.5
            p_adverse = 0.3
        elif pred_value > -0.05:
            p_favorable = 0.1
            p_base = 0.3
            p_adverse = 0.6
        else:
            p_favorable = 0.1
            p_base = 0.2
            p_adverse = 0.7

        row = {
            'trade_date': str(test_df.loc[idx, 'trade_date']),
            'symbol': test_df.loc[idx, 'symbol'],
            'close': float(test_df.loc[idx, 'close']),
            'prediction': 'base',
            'adverse': p_adverse,
            'base': p_base,
            'favorable': p_favorable,
        }
        pred_records.append(row)

    pred_df = pd.DataFrame(pred_records)

    results_table = 'backtest_results_regressor_prob'
    daily_table = 'backtest_daily_regressor_prob'
    bt_df = run_backtest(pred_df, 'regressor_prob', results_table, daily_table,
                         CRASH_CONFIG_STANDARD, 'probabilistic')

    all_results.append({
        'name': '4: 回归+概率',
        'backtest': bt_df,
    })

    # ============================================================
    # 方案5: 缩短预测窗口 (5日)
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案5: 缩短预测窗口 (5日)")
    print(f"{'=' * 80}")

    df = base_df.copy()
    df['target_return_5d'] = df['close'].shift(-5) / df['close'] - 1
    r = df['target_return_5d']
    df['scenario_label'] = make_scenario_label(r, 0.03, -0.03)
    df['scenario_int'] = df['scenario_label'].map({'adverse': 0, 'base': 1, 'favorable': 2})
    df = df.dropna(subset=['target_return_5d'])

    y = df['scenario_int'].astype(int)
    X = df[base_feature_cols].copy()

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
    X_train_bal = X_train.loc[oversampled_indices]
    y_train_bal = y_train.loc[oversampled_indices]

    # RF
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)

    # LGBM
    lgbm = lgb.LGBMClassifier(
        objective='multiclass', num_class=3,
        n_estimators=1000, learning_rate=0.05,
        max_depth=8, num_leaves=63, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=-1
    )
    lgbm.fit(X_train_bal, y_train_bal,
             eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    # 生成预测
    rf_proba = rf.predict_proba(X_test)
    lgbm_proba = lgbm.predict_proba(X_test)
    fusion_proba = (rf_proba + lgbm_proba) / 2
    fusion_pred = np.argmax(fusion_proba, axis=1)

    test_df = df[test_mask].copy()
    pred_records = []
    for i, idx in enumerate(test_df.index):
        row = {
            'trade_date': str(test_df.loc[idx, 'trade_date']),
            'symbol': test_df.loc[idx, 'symbol'],
            'close': float(test_df.loc[idx, 'close']),
            'prediction': ['adverse', 'base', 'favorable'][fusion_pred[i]],
            'adverse': float(fusion_proba[i][0]),
            'base': float(fusion_proba[i][1]),
            'favorable': float(fusion_proba[i][2]),
        }
        pred_records.append(row)

    pred_df = pd.DataFrame(pred_records)

    results_table = 'backtest_results_5d_prob'
    daily_table = 'backtest_daily_5d_prob'
    bt_df = run_backtest(pred_df, '5d_prob', results_table, daily_table,
                         CRASH_CONFIG_STANDARD, 'probabilistic')

    all_results.append({
        'name': '5: 5日窗口+概率',
        'backtest': bt_df,
    })

    # ============================================================
    # 方案6: 深度学习算法
    # ============================================================
    print(f"\n{'=' * 80}")
    print("方案6: 深度学习算法")
    print(f"{'=' * 80}")

    # 准备数据
    df = base_df.copy()
    r = df['target_return_10d']
    df['scenario_label'] = make_scenario_label(r, 0.05, -0.05)
    df['scenario_int'] = df['scenario_label'].map({'adverse': 0, 'base': 1, 'favorable': 2})
    df = df.dropna(subset=base_feature_cols + ['scenario_int'])

    y = df['target_return_10d'].values
    X = df[base_feature_cols].values

    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 时间划分
    dates = pd.to_datetime(df['trade_date'])
    train_mask = dates < pd.Timestamp('2023-01-01')
    val_mask = (dates >= pd.Timestamp('2023-01-01')) & (dates < pd.Timestamp('2024-01-01'))
    test_mask = dates >= pd.Timestamp('2024-01-01')

    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_val, y_val = X_scaled[val_mask], y[val_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask]

    seq_len = 10
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_len)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    input_size = X_train.shape[1]

    # 训练 LSTM
    print("\n  训练 LSTM...")
    lstm_model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.2).to(device)
    lstm_model = train_dl_model(lstm_model, train_loader, val_loader, epochs=30, lr=0.001)

    # 训练 GRU
    print("  训练 GRU...")
    gru_model = GRUModel(input_size=input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.2).to(device)
    gru_model = train_dl_model(gru_model, train_loader, val_loader, epochs=30, lr=0.001)

    # 训练 Transformer
    print("  训练 Transformer...")
    transformer_model = TransformerModel(input_size=input_size, d_model=64, nhead=4, num_layers=1, output_size=1, dropout=0.2).to(device)
    transformer_model = train_dl_model(transformer_model, train_loader, val_loader, epochs=30, lr=0.001)

    # 生成预测
    def generate_dl_predictions(model, test_loader, test_df, seq_len):
        model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                predictions.extend(output.cpu().numpy().flatten())

        # 对齐预测和测试集
        pred_df = test_df.iloc[seq_len:].copy()
        pred_df['predicted_return'] = predictions

        # 转换为概率
        pred_records = []
        for idx in pred_df.index:
            pred_value = pred_df.loc[idx, 'predicted_return']
            if pred_value > 0.05:
                p_favorable = 0.7
                p_base = 0.2
                p_adverse = 0.1
            elif pred_value > 0.02:
                p_favorable = 0.5
                p_base = 0.3
                p_adverse = 0.2
            elif pred_value > 0:
                p_favorable = 0.3
                p_base = 0.5
                p_adverse = 0.2
            elif pred_value > -0.02:
                p_favorable = 0.2
                p_base = 0.5
                p_adverse = 0.3
            elif pred_value > -0.05:
                p_favorable = 0.1
                p_base = 0.3
                p_adverse = 0.6
            else:
                p_favorable = 0.1
                p_base = 0.2
                p_adverse = 0.7

            row = {
                'trade_date': str(pred_df.loc[idx, 'trade_date']),
                'symbol': pred_df.loc[idx, 'symbol'],
                'close': float(pred_df.loc[idx, 'close']),
                'prediction': 'base',
                'adverse': p_adverse,
                'base': p_base,
                'favorable': p_favorable,
            }
            pred_records.append(row)

        return pd.DataFrame(pred_records)

    test_df = df[test_mask].copy()

    # LSTM 预测
    print("\n  LSTM 回测...")
    pred_df = generate_dl_predictions(lstm_model, test_loader, test_df, seq_len)
    results_table = 'backtest_results_lstm'
    daily_table = 'backtest_daily_lstm'
    bt_df = run_backtest(pred_df, 'lstm', results_table, daily_table,
                         CRASH_CONFIG_STANDARD, 'probabilistic')
    all_results.append({
        'name': '6A: LSTM',
        'backtest': bt_df,
    })

    # GRU 预测
    print("  GRU 回测...")
    pred_df = generate_dl_predictions(gru_model, test_loader, test_df, seq_len)
    results_table = 'backtest_results_gru'
    daily_table = 'backtest_daily_gru'
    bt_df = run_backtest(pred_df, 'gru', results_table, daily_table,
                         CRASH_CONFIG_STANDARD, 'probabilistic')
    all_results.append({
        'name': '6B: GRU',
        'backtest': bt_df,
    })

    # Transformer 预测
    print("  Transformer 回测...")
    pred_df = generate_dl_predictions(transformer_model, test_loader, test_df, seq_len)
    results_table = 'backtest_results_transformer'
    daily_table = 'backtest_daily_transformer'
    bt_df = run_backtest(pred_df, 'transformer', results_table, daily_table,
                         CRASH_CONFIG_STANDARD, 'probabilistic')
    all_results.append({
        'name': '6C: Transformer',
        'backtest': bt_df,
    })

    # ============================================================
    # 汇总对比
    # ============================================================
    print(f"\n\n{'=' * 100}")
    print("汇总对比: 所有方案")
    print(f"{'=' * 100}")

    print(f"\n{'方案':<25s} {'总收益':>8s} {'年化':>8s} {'最大回撤':>8s} {'夏普':>6s} {'超额':>8s} {'平均仓位':>8s}")
    print("-" * 75)
    for r in all_results:
        bt = r['backtest']
        if len(bt) > 0:
            print(f"{r['name']:<25s} {bt['total_return'].mean()*100:>7.2f}% {bt['annual_return'].mean()*100:>7.2f}% "
                  f"{bt['max_drawdown'].mean()*100:>7.2f}% {bt['sharpe'].mean():>6.2f} "
                  f"{bt['excess_return'].mean()*100:>7.2f}% {bt['avg_position'].mean()*100:>7.1f}%")

    # 确定最优方案
    print(f"\n{'=' * 100}")
    print("结论")
    print(f"{'=' * 100}")

    bt_results_with_data = [r for r in all_results if len(r['backtest']) > 0]
    if bt_results_with_data:
        best_bt = max(bt_results_with_data, key=lambda x: x['backtest']['excess_return'].mean())
        print(f"  最高超额收益: {best_bt['name']} ({best_bt['backtest']['excess_return'].mean()*100:.2f}%)")

        best_sharpe = max(bt_results_with_data, key=lambda x: x['backtest']['sharpe'].mean())
        print(f"  最高夏普比率: {best_sharpe['name']} ({best_sharpe['backtest']['sharpe'].mean():.2f})")

        best_return = max(bt_results_with_data, key=lambda x: x['backtest']['total_return'].mean())
        print(f"  最高总收益: {best_return['name']} ({best_return['backtest']['total_return'].mean()*100:.2f}%)")

    # 保存结果
    results_summary = {
        'experiments': [],
    }
    for r in all_results:
        exp_data = {'name': r['name']}
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

    with open(os.path.join(BASE_MODEL_DIR, 'comprehensive_experiment_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {os.path.join(BASE_MODEL_DIR, 'comprehensive_experiment_results.json')}")


if __name__ == '__main__':
    main()
