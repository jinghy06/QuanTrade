"""
A股量化信号系统 - ML模型训练（多视野回归版）
LightGBM回归器 + 多步预测 + 时序交叉验证 + Optuna超参调优
对应 plan.md: "LightGBM预测未来走势（1/3/5/10日收益率）"
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from config.settings import (
    DB_PATH,
    LGB_PARAMS,
    MIN_TRAIN_DAYS,
    MODELS_DIR,
    N_SPLITS,
    SIGNAL_THRESHOLD,
)
from data.data_store import DataStore
from features.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)

# Optuna日志级别
optuna.logging.set_verbosity(optuna.logging.WARNING)


class PurgedGroupTimeSeriesSplit:
    """
    带清洗的时序交叉验证
    借鉴金融ML实践，避免未来信息泄露
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.purge_gap
            test_end = min(fold_size * (i + 2), n_samples)

            if test_start >= n_samples:
                break

            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class MLTrainer:
    """
    LightGBM多视野回归模型训练器
    同时训练4个时间尺度的回归模型 + 1个波动率预测模型
    """

    # 特征列（含新增趋势特征）
    FEATURE_COLS = [
        "return_5d", "return_10d", "return_20d",
        "rsi_14", "macd_dif", "macd_dea", "macd_hist",
        "std_5d", "std_20d", "atr_14",
        "volume_ma5", "volume_ma20", "obv", "turnover_ma5",
        "ma_alignment", "price_position", "trend_slope",
        "vol_percentile", "dist_to_support", "dist_to_resistance",
        "divergence_bear",
        "bond_yield_10y",
        "rd_factor_1", "rd_factor_2", "rd_factor_3",
    ]

    HORIZONS = [1, 3, 5, 10]

    def __init__(self):
        self.store = DataStore()
        self.models_dir = MODELS_DIR
        self.models: Dict[int, lgb.LGBMRegressor] = {}
        self.vol_model: Optional[lgb.LGBMRegressor] = None
        # 兼容旧分类模型
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = self.FEATURE_COLS.copy()
        self.version = datetime.now().strftime("v%Y%m%d")

    def _prepare_data(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据，对齐特征列，填充NaN"""
        available_cols = [
            c for c in self.feature_cols
            if c in df.columns and df[c].notna().sum() > 0
        ]
        if not available_cols:
            raise ValueError("没有可用的特征列")

        X = df[available_cols].copy()
        y = df[target_col].copy()

        # 删除target为NaN的行
        valid_mask = y.notna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        # 用列中位数填充特征NaN
        X = X.fillna(X.median())

        return X, y

    # ============================================================
    # 训练主流程（多视野回归）
    # ============================================================

    def train(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
        tune_hyperparams: bool = False,
    ) -> Dict:
        """
        训练多视野回归模型（1d/3d/5d/10d）+ 波动率预测器
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y%m%d")

        logger.info("开始多视野回归训练 | 股票池:%d只 | %s ~ %s", len(symbols), start_date, end_date)

        # 1. 加载训练数据
        df_train = self.store.get_training_data(symbols, start_date, end_date)
        if len(df_train) < MIN_TRAIN_DAYS:
            raise ValueError(f"训练数据不足: {len(df_train)}条，需要至少{MIN_TRAIN_DAYS}条")

        # 记录实际使用的特征列
        available_cols = [c for c in self.feature_cols if c in df_train.columns and df_train[c].notna().sum() > 0]
        self.feature_cols = available_cols
        logger.info("实际训练特征(%d): %s", len(self.feature_cols), self.feature_cols)

        # 2. 训练各horizon回归模型
        results = {"horizons": {}, "version": self.version}
        all_X = None

        for horizon in self.HORIZONS:
            target_col = f"target_return_{horizon}d"
            if target_col not in df_train.columns:
                logger.warning("目标列 %s 不存在，跳过", target_col)
                continue

            try:
                X, y = self._prepare_data(df_train, target_col)
                if len(X) < 100:
                    logger.warning("%s 有效样本不足: %d", target_col, len(X))
                    continue

                if all_X is None:
                    all_X = X  # 保存用于scaler训练

                # 超参调优（可选）
                params = self._get_regressor_params(tune_hyperparams, X, y)

                # 时序交叉验证
                cv_rmse = self._time_series_cv_regression(X, y, params)
                logger.info("H=%dd 时序CV RMSE: %.4f ± %.4f", horizon, cv_rmse.mean(), cv_rmse.std())

                # 全量训练
                model = lgb.LGBMRegressor(**params)
                model.fit(X, y)

                # 评估
                y_pred = model.predict(X)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)

                self.models[horizon] = model
                results["horizons"][horizon] = {
                    "rmse": round(float(rmse), 4),
                    "r2": round(float(r2), 4),
                    "cv_rmse_mean": round(float(cv_rmse.mean()), 4),
                    "n_samples": len(X),
                }
                logger.info("H=%dd 训练完成 | RMSE=%.4f | R²=%.4f", horizon, rmse, r2)

            except Exception as e:
                logger.error("H=%dd 训练失败: %s", horizon, e)

        # 3. 训练波动率预测器
        try:
            vol_target = "target_volatility_5d"
            if vol_target in df_train.columns:
                X_vol, y_vol = self._prepare_data(df_train, vol_target)
                vol_params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "num_leaves": 15,
                    "learning_rate": 0.05,
                    "n_estimators": 100,
                    "seed": 42,
                    "verbose": -1,
                }
                self.vol_model = lgb.LGBMRegressor(**vol_params)
                self.vol_model.fit(X_vol, y_vol)
                vol_pred = self.vol_model.predict(X_vol)
                vol_rmse = np.sqrt(mean_squared_error(y_vol, vol_pred))
                results["volatility_model"] = {"rmse": round(float(vol_rmse), 4), "n_samples": len(X_vol)}
                logger.info("波动率模型训练完成 | RMSE=%.4f", vol_rmse)
        except Exception as e:
            logger.error("波动率模型训练失败: %s", e)

        # 4. 统一scaler（用第一个horizon的数据fit）
        if all_X is not None and len(all_X) > 0:
            self.scaler.fit(all_X)

        # 5. 保存模型
        model_paths = self._save_models(results)
        results["model_paths"] = model_paths

        # 兼容旧接口：加载一个分类模型（用1d回归模型包装）
        if 1 in self.models:
            self._wrap_classification_model()

        logger.info("多视野训练全部完成 | 模型数: %d", len(self.models))
        return results

    def _get_regressor_params(self, tune: bool, X: pd.DataFrame, y: pd.Series) -> Dict:
        """获取回归模型参数"""
        base = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
            "n_estimators": 200,
        }
        if tune:
            best = self._tune_regressor_hyperparams(X, y)
            base.update(best)
        return base

    def _time_series_cv_regression(
        self, X: pd.DataFrame, y: pd.Series, params: Dict
    ) -> np.ndarray:
        """回归任务的时序交叉验证"""
        cv = PurgedGroupTimeSeriesSplit(n_splits=N_SPLITS, purge_gap=5)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(rmse)
            logger.debug("Fold %d RMSE: %.4f (%d train / %d test)", fold + 1, rmse, len(train_idx), len(test_idx))

        return np.array(scores)

    def _tune_regressor_hyperparams(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int = 30
    ) -> Dict:
        """Optuna回归超参调优"""

        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            cv = PurgedGroupTimeSeriesSplit(n_splits=3, purge_gap=5)
            scores = []
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model = lgb.LGBMRegressor(**{"objective": "regression", "metric": "rmse", "seed": 42, "verbose": -1, **params})
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        logger.info("Optuna最佳参数 (RMSE=%.4f): %s", study.best_value, study.best_params)
        return study.best_params

    # ============================================================
    # 模型保存与加载（多模型版）
    # ============================================================

    def _save_models(self, metrics: Dict) -> Dict:
        """保存多视野回归模型 + 波动率模型 + 统一配置"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = self.version
        paths = {}

        # 保存各horizon模型
        for horizon, model in self.models.items():
            model_file = self.models_dir / f"lgb_regressor_{horizon}d_{version}.txt"
            model.booster_.save_model(str(model_file))
            paths[f"regressor_{horizon}d"] = str(model_file)
            # 同时保存最新版本
            latest = self.models_dir / f"lgb_regressor_{horizon}d.txt"
            model.booster_.save_model(str(latest))

        # 保存波动率模型
        if self.vol_model is not None:
            vol_file = self.models_dir / f"lgb_volatility_{version}.txt"
            self.vol_model.booster_.save_model(str(vol_file))
            paths["volatility"] = str(vol_file)
            latest_vol = self.models_dir / "lgb_volatility.txt"
            self.vol_model.booster_.save_model(str(latest_vol))

        # 统一JSON配置
        config = {
            "version": version,
            "created_at": timestamp,
            "feature_names": list(self.feature_cols),
            "scaler_mean": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else [],
            "scaler_scale": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else [],
            "n_features_in_": getattr(self.scaler, "n_features_in_", len(self.feature_cols)),
            "horizons": self.HORIZONS,
            "metrics": metrics,
        }

        config_file = self.models_dir / f"model_config_{version}.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        latest_config = self.models_dir / "model_config.json"
        with open(latest_config, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        paths["config"] = str(config_file)
        logger.info("多模型已保存: %s", paths)
        return paths

    def load_models(self, config_path: str = None):
        """加载多视野回归模型 + 波动率模型"""
        if config_path is None:
            config_path = self.models_dir / "model_config.json"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.feature_cols = config["feature_names"]
        self.HORIZONS = config.get("horizons", [1, 3, 5, 10])

        # 重建scaler
        if config.get("scaler_mean") and config.get("scaler_scale"):
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(config["scaler_mean"])
            self.scaler.scale_ = np.array(config["scaler_scale"])
            self.scaler.n_features_in_ = config["n_features_in_"]

        # 加载各horizon模型
        for horizon in self.HORIZONS:
            model_path = self.models_dir / f"lgb_regressor_{horizon}d.txt"
            if model_path.exists():
                self.models[horizon] = lgb.Booster(model_file=str(model_path))
                logger.info("回归模型加载成功: H=%dd", horizon)

        # 加载波动率模型
        vol_path = self.models_dir / "lgb_volatility.txt"
        if vol_path.exists():
            self.vol_model = lgb.Booster(model_file=str(vol_path))
            logger.info("波动率模型加载成功")

        # 兼容旧接口
        if 1 in self.models:
            self._wrap_classification_model()

        return config

    def _wrap_classification_model(self):
        """将1d回归模型包装为旧分类接口（兼容）"""
        class RegressorAsClassifier:
            def __init__(self, regressor, scaler):
                self.regressor = regressor
                self.scaler = scaler

            def predict(self, X):
                X_s = self.scaler.transform(X)
                ret = self.regressor.predict(X_s)
                return (ret > 0).astype(int)

            def predict_proba(self, X):
                X_s = self.scaler.transform(X)
                ret = self.regressor.predict(X_s)
                # 将收益率映射到概率：sigmoid-like
                proba = 1 / (1 + np.exp(-ret * 10))  # 放大系数10让分界清晰
                return np.column_stack([1 - proba, proba])

        self.model = RegressorAsClassifier(self.models[1], self.scaler)

    # ============================================================
    # 新预测接口：多视野趋势预测
    # ============================================================

    def predict_trend(self, df_features: pd.DataFrame, current_price: float) -> Dict:
        """
        预测未来走势锚点（1/3/5/10日）

        Returns:
            {
                "current_price": float,
                "anchors": {1: {"return": x, "price": y, "confidence": z}, ...},
                "predicted_volatility": float,
            }
        """
        if not self.models:
            raise RuntimeError("模型未加载，请先调用train()或load_models()")

        available_cols = [c for c in self.feature_cols if c in df_features.columns]
        X_raw = df_features[available_cols].iloc[[-1]].copy()
        X_raw = X_raw.fillna(0)
        X = self.scaler.transform(X_raw)

        result = {
            "current_price": round(float(current_price), 2),
            "anchors": {},
            "predicted_volatility": 0.0,
        }

        for horizon, model in self.models.items():
            try:
                pred_return = float(model.predict(X)[0])
                pred_price = current_price * (1 + pred_return)
                # 置信度：用|预测收益率|/历史该horizon标准差估计（简化版）
                confidence = min(abs(pred_return) * 20, 1.0)  # 放大20倍，封顶1.0
                result["anchors"][horizon] = {
                    "return": round(pred_return, 4),
                    "price": round(pred_price, 2),
                    "confidence": round(confidence, 2),
                }
            except Exception as e:
                logger.error("H=%dd 预测失败: %s", horizon, e)

        # 波动率预测
        if self.vol_model is not None:
            try:
                pred_vol = float(self.vol_model.predict(X)[0])
                result["predicted_volatility"] = round(pred_vol, 4)
            except Exception as e:
                logger.error("波动率预测失败: %s", e)

        return result

    # ============================================================
    # 兼容旧接口
    # ============================================================

    def predict(self, df_features: pd.DataFrame) -> Dict:
        """兼容旧接口：返回涨跌概率"""
        if self.model is None:
            raise RuntimeError("模型未加载")

        available_cols = [c for c in self.feature_cols if c in df_features.columns]
        X = df_features[available_cols].iloc[[-1]].copy().fillna(0)

        proba = self.model.predict_proba(X)[0, 1]
        pred_class = 1 if proba > SIGNAL_THRESHOLD else 0

        return {
            "up_prob": round(float(proba), 4),
            "direction": int(pred_class),
            "threshold": SIGNAL_THRESHOLD,
            "confidence": round(float(abs(proba - 0.5) * 2), 4),
        }

    def batch_predict(self, symbols: List[str], top_n: int = 20) -> List[Dict]:
        """兼容旧接口"""
        results = []
        for symbol in symbols:
            try:
                df_feat = self.store.get_features(symbol)
                if df_feat.empty:
                    continue
                pred = self.predict(df_feat)
                if pred["up_prob"] > SIGNAL_THRESHOLD:
                    results.append({"symbol": symbol, **pred})
            except Exception as e:
                logger.error("预测失败 %s: %s", symbol, e)
        results.sort(key=lambda x: x["up_prob"], reverse=True)
        return results[:top_n]

    # ============================================================
    # 模型衰减监控（回归版）
    # ============================================================

    def evaluate_recent(
        self, symbols: List[str], n_days: int = 30, warn_rmse_threshold: float = 0.05
    ) -> Dict:
        """评估模型近N天表现（回归RMSE）"""
        if not self.models:
            raise RuntimeError("模型未加载")

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=n_days)).strftime("%Y%m%d")

        horizon_results = {}
        for horizon in self.HORIZONS:
            if horizon not in self.models:
                continue

            target_col = f"target_return_{horizon}d"
            all_y_true, all_y_pred = [], []

            for symbol in symbols:
                try:
                    df_feat = self.store.get_features(
                        symbol, start_date=start_date, end_date=end_date, with_target=True
                    )
                    if df_feat.empty or target_col not in df_feat.columns:
                        continue

                    available_cols = [c for c in self.feature_cols if c in df_feat.columns]
                    X = df_feat[available_cols].fillna(0)
                    y_true = df_feat[target_col]

                    valid = y_true.notna()
                    X = X.loc[valid]
                    y_true = y_true.loc[valid]

                    if len(X) == 0:
                        continue

                    X_scaled = self.scaler.transform(X)
                    y_pred = self.models[horizon].predict(X_scaled)

                    all_y_true.extend(y_true.values.tolist())
                    all_y_pred.extend(y_pred.tolist())

                except Exception as e:
                    logger.error("评估 %s H=%dd 失败: %s", symbol, horizon, e)

            if all_y_true:
                rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
                r2 = r2_score(all_y_true, all_y_pred) if len(set(all_y_true)) > 1 else 0
                needs_retrain = rmse > warn_rmse_threshold
                horizon_results[horizon] = {
                    "rmse": round(float(rmse), 4),
                    "r2": round(float(r2), 4),
                    "n_samples": len(all_y_true),
                    "needs_retrain": needs_retrain,
                }

        overall_needs_retrain = any(r.get("needs_retrain", False) for r in horizon_results.values())
        result = {
            "period_days": n_days,
            "horizon_results": horizon_results,
            "overall_needs_retrain": overall_needs_retrain,
        }
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = MLTrainer()

    store = DataStore()
    symbols = store.get_all_symbols()[:5]
    if len(symbols) >= 2:
        result = trainer.train(symbols, tune_hyperparams=False)
        print(f"\n训练结果:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
