"""Configuration paths are relative to the configuration file, never the shell cwd."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path or ROOT / "configs/default.toml").resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for key, value in config["paths"].items():
        config["paths"][key] = str((path.parent / value).resolve())
    config["config_file"] = str(path)
    strategy = config["strategy"]
    if strategy["frequency"] not in ("monthly", "quarterly"):
        raise ValueError("frequency must be monthly or quarterly")
    if not 0 <= strategy["cash_reserve"] < 1:
        raise ValueError("cash_reserve must be in [0, 1)")
    for key in ("max_weight", "max_theme_weight", "risk_reduction"):
        if not 0 < strategy[key] <= 1:
            raise ValueError(f"invalid {key}")
    if strategy["max_positions"] < 1 or strategy["lookback"] < 2:
        raise ValueError("invalid strategy size/window")
    for key in ("commission", "min_fee", "slippage"):
        if config["execution"][key] < 0:
            raise ValueError(f"negative {key}")
    return config
