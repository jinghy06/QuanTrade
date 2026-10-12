"""Frozen LightGBM regressors, date-based purged validation, consistent inputs."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_VERSION, FEATURES, compute_features
from .storage import write_json

MODEL_FORMAT_VERSION = 2


def purged_partition(features, train_end, validation_end, test_end, horizon):
    known = features.dropna(subset=[f"target_{horizon}", f"label_end_{horizon}"])
    train = known[(known.date <= train_end) & (known[f"label_end_{horizon}"] <= train_end)]
    validation = known[
        (known.date > train_end)
        & (known.date <= validation_end)
        & (known[f"label_end_{horizon}"] <= validation_end)
    ]
    test = known[
        (known.date > validation_end)
        & (known.date <= test_end)
        & (known[f"label_end_{horizon}"] <= test_end)
    ]
    return train, validation, test


def model_input(frame, medians):
    # Column order is frozen; missing fields cause an error, not silent reduction.
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"missing model features: {sorted(missing)}")
    return frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(pd.Series(medians)).fillna(0)


def fit_bundle(prices, config, cutoff, directory, snapshot_id="unspecified"):
    import lightgbm as lgb

    cfg = config["model"]
    horizons = cfg["horizons"]
    eligible = prices[prices.date <= cutoff]
    features = compute_features(eligible, horizons)
    dates = sorted(features.date.unique())
    needed = cfg["min_train_dates"] + cfg["validation_dates"] + cfg["test_dates"]
    if len(dates) < needed:
        raise ValueError(f"need {needed} dates, found {len(dates)}")
    train_end = dates[-cfg["validation_dates"] - cfg["test_dates"] - 1]
    validation_end = dates[-cfg["test_dates"] - 1]
    payload = {
        "config": cfg,
        "cutoff": cutoff,
        "snapshot": snapshot_id,
        "feature_version": FEATURE_VERSION,
        "features": FEATURES,
        "model_format": MODEL_FORMAT_VERSION,
    }
    # Include data content even when called outside the snapshot workflow.
    payload["prices_sha256"] = sha256(eligible.to_csv(index=False).encode()).hexdigest()
    version = "model-" + sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(directory) / version
    if (folder / "manifest.json").exists():
        return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": version,
        "feature_version": FEATURE_VERSION,
        "features": FEATURES,
        "model_format": MODEL_FORMAT_VERSION,
        "seed": cfg["seed"],
        "trained_through": cutoff,
        "train_end": train_end,
        "validation_end": validation_end,
        "test_end": dates[-1],
        "snapshot_id": snapshot_id,
        "config": cfg,
        "horizons": {},
        "status": "candidate",
        "validated": False,
    }
    for horizon in horizons:
        train, validation, test = purged_partition(
            features, train_end, validation_end, dates[-1], horizon
        )
        if min(len(train), len(validation), len(test)) < 20:
            raise ValueError(f"insufficient purged samples for horizon {horizon}")
        medians = train[FEATURES].median().fillna(0).to_dict()
        model = lgb.LGBMRegressor(
            n_estimators=cfg["n_estimators"],
            num_leaves=15,
            learning_rate=0.05,
            random_state=cfg["seed"],
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        model.fit(model_input(train, medians), train[f"target_{horizon}"])
        metrics = {}
        interval_width = None
        for name, sample in (("validation", validation), ("test", test)):
            truth = sample[f"target_{horizon}"].to_numpy()
            pred = model.predict(model_input(sample, medians))
            if name == "validation":
                interval_width = float(np.quantile(np.abs(pred - truth), 0.9))
            metrics[name] = {
                "n": len(sample),
                "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                "zero_rmse": float(np.sqrt(np.mean(truth**2))),
                "direction_accuracy": float(np.mean((pred > 0) == (truth > 0))),
                "interval_coverage_90": float(np.mean(np.abs(pred - truth) <= interval_width)),
            }
        # Fit only the same training interval. Model release is after the entire
        # validation/test observation period, never at the earlier train_end.
        model.booster_.save_model(str(folder / f"return_{horizon}.txt"))
        metadata["horizons"][str(horizon)] = dict(
            medians=medians, interval_width_90=interval_width, **metrics
        )
    write_json(folder / "manifest.json", metadata)
    return metadata


class ModelBundle:
    def __init__(self, folder):
        import lightgbm as lgb

        self.folder = Path(folder)
        self.metadata = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        if self.metadata.get("model_format") != MODEL_FORMAT_VERSION:
            raise ValueError("unsupported model artifact format; train a current model")
        if (
            self.metadata["features"] != FEATURES
            or self.metadata["feature_version"] != FEATURE_VERSION
        ):
            raise ValueError("unsupported feature contract")
        self.models = {
            int(h): lgb.Booster(model_file=str(self.folder / f"return_{h}.txt"))
            for h in self.metadata["horizons"]
        }
        self.available_at = None

    def can_predict(self, as_of):
        stamp = pd.Timestamp(as_of + "T15:30:00+08:00")
        trained = pd.Timestamp(self.metadata["trained_through"])
        if as_of <= self.metadata["trained_through"]:
            return False
        if (pd.Timestamp(as_of) - trained).days > self.metadata["config"].get("max_age_days", 120):
            return False
        return self.available_at is None or pd.Timestamp(self.available_at) <= stamp

    def predict(self, features, as_of):
        if not self.can_predict(as_of):
            raise ValueError("model unavailable, stale or includes future information")
        result = {}
        for horizon, model in self.models.items():
            medians = self.metadata["horizons"][str(horizon)]["medians"]
            values = model.predict(model_input(features, medians))
            result[horizon] = dict(zip(features.symbol, (float(v) for v in values)))
        return result


def validate_model(folder):
    bundle = ModelBundle(folder)
    checks = []
    for meta in bundle.metadata["horizons"].values():
        checks.append(
            all(np.isfinite(meta[s]["rmse"]) and meta[s]["n"] >= 20 for s in ("validation", "test"))
        )
        if not np.isfinite(list(meta["medians"].values())).all():
            raise ValueError("invalid preprocessing values")
    if not checks or not all(checks):
        raise ValueError("invalid model validation")
    bundle.metadata["validated"] = True
    bundle.metadata["validation_note"] = (
        "Technical validation only; trading superiority requires walk-forward and ablation."
    )
    write_json(Path(folder) / "manifest.json", bundle.metadata)
    return bundle.metadata


def publish_model(directory, version):
    folder = Path(directory) / version
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("validated"):
        raise ValueError("run model validate first")
    # Frozen artifacts aren't changed on publication; only the release pointer.
    write_json(
        Path(directory) / "published.json",
        {
            "version": version,
            "published_at": datetime.now(UTC).isoformat(),
            "trained_through": manifest["trained_through"],
        },
    )
    return {"version": version, "status": "published", "trading_admission": "not_established"}


def published_bundle(directory):
    pointer = Path(directory) / "published.json"
    if not pointer.exists():
        return None
    release = json.loads(pointer.read_text(encoding="utf-8"))
    bundle = ModelBundle(Path(directory) / release["version"])
    bundle.available_at = release["published_at"]
    return bundle
