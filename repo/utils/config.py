# utils/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import os

import torch

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e


_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}

_DEVICE_CHOICES = {"auto", "cpu", "cuda", "mps"}


def _resolve_device(device_str: str) -> torch.device:
    device_str = str(device_str).lower().strip()
    if device_str not in _DEVICE_CHOICES:
        raise ValueError(f"device must be one of {_DEVICE_CHOICES}, got '{device_str}'")

    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")
        return torch.device("cuda")
    if device_str == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device='mps' requested but MPS is not available.")
        return torch.device("mps")

    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = str(dtype_str).lower().strip()
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"dtype must be one of {list(_DTYPE_MAP.keys())}, got '{dtype_str}'")
    return _DTYPE_MAP[dtype_str]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML must be a mapping/dict, got {type(cfg)}")
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_nested(cfg: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested cfg value using dot path: e.g. "training.lr".
    Returns default if not found.
    """
    cur: Any = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def require_nested(cfg: Dict[str, Any], key_path: str) -> Any:
    val = get_nested(cfg, key_path, default=None)
    if val is None:
        raise KeyError(f"Missing required config key: '{key_path}'")
    return val


@dataclass(frozen=True)
class RuntimeEnv:
    device: torch.device
    dtype: torch.dtype
    seed: int


def resolve_runtime(cfg: Dict[str, Any]) -> RuntimeEnv:
    seed = int(cfg.get("seed", 0))
    device = _resolve_device(cfg.get("device", "auto"))
    dtype = _resolve_dtype(cfg.get("dtype", "float32"))
    return RuntimeEnv(device=device, dtype=dtype, seed=seed)


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Minimal runtime validation for stage1.yaml.

    Goal: catch missing keys and obviously invalid values early,
    without over-engineering a schema system.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict at top level.")

    # ---- Core runtime keys ----
    device = str(cfg.get("device", "auto")).lower().strip()
    if device not in _DEVICE_CHOICES:
        raise ValueError(f"device must be one of {_DEVICE_CHOICES}, got '{device}'")

    dtype = str(cfg.get("dtype", "float32")).lower().strip()
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"dtype must be one of {list(_DTYPE_MAP.keys())}, got '{dtype}'")

    seed = cfg.get("seed", 0)
    try:
        int(seed)
    except Exception:
        raise ValueError(f"seed must be an int-like value, got {seed!r}")

    # ---- Experiment ----
    exp = cfg.get("experiment", {})
    if not isinstance(exp, dict):
        raise ValueError("experiment must be a mapping/dict")
    if "out_dir" in exp and not isinstance(exp["out_dir"], str):
        raise ValueError("experiment.out_dir must be a string if provided")
    if "name" in exp and not isinstance(exp["name"], str):
        raise ValueError("experiment.name must be a string if provided")

    # ---- Problem basics ----
    prob = cfg.get("problem", {})
    if not isinstance(prob, dict):
        raise ValueError("problem must be a mapping/dict")
    T = get_nested(cfg, "problem.T", None)
    if T is None:
        raise KeyError("Missing required config key: 'problem.T'")
    if float(T) <= 0:
        raise ValueError("problem.T must be > 0")

    # ---- Models ----
    models = cfg.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models must be a mapping/dict")
    run_list = models.get("run", None)
    if run_list is None:
        raise KeyError("models.run is required (list of model IDs)")
    if not isinstance(run_list, list) or not all(isinstance(x, str) for x in run_list):
        raise ValueError("models.run must be a list of strings")
    if len(run_list) == 0:
        raise ValueError("models.run must not be empty")

    # ---- Training ----
    tr = cfg.get("training", {})
    if not isinstance(tr, dict):
        raise ValueError("training must be a mapping/dict")
    steps = get_nested(cfg, "training.steps", None)
    lr = get_nested(cfg, "training.lr", None)
    if steps is None or lr is None:
        raise KeyError("training.steps and training.lr are required")
    if int(steps) <= 0:
        raise ValueError("training.steps must be > 0")
    if float(lr) <= 0:
        raise ValueError("training.lr must be > 0")

    # ---- Loss ----
    loss = cfg.get("loss", {})
    if not isinstance(loss, dict):
        raise ValueError("loss must be a mapping/dict")
    lam_phys = get_nested(cfg, "loss.lambda_phys", None)
    lam_ic = get_nested(cfg, "loss.lambda_ic", None)
    if lam_phys is None or lam_ic is None:
        raise KeyError("loss.lambda_phys and loss.lambda_ic are required")
    if float(lam_phys) < 0 or float(lam_ic) < 0:
        raise ValueError("loss.lambda_phys and loss.lambda_ic must be >= 0")

    # ---- Collocation ----
    col = cfg.get("collocation", {})
    if not isinstance(col, dict):
        raise ValueError("collocation must be a mapping/dict")
    n_col = get_nested(cfg, "collocation.interior_points", None)
    n_ic = get_nested(cfg, "collocation.ic_points", None)
    if n_col is None or n_ic is None:
        raise KeyError("collocation.interior_points and collocation.ic_points are required")
    if int(n_col) <= 0:
        raise ValueError("collocation.interior_points must be > 0")
    if int(n_ic) <= 0:
        raise ValueError("collocation.ic_points must be > 0")


def validate_spectral(cfg: Dict[str, Any]) -> None:
    spectral = cfg.get("spectral_trunk", {})
    if not spectral.get("enabled", False):
        return

    fixed_basis = str(spectral.get("fixed_basis", "fourier")).lower().strip()

    if fixed_basis == "fourier":
        fourier = spectral.get("fourier", {})
        bands = fourier.get("bands", {})
        for band_name in ["low", "mid", "high"]:
            if band_name not in bands:
                raise KeyError(f"Missing spectral_trunk.fourier.bands.{band_name}")
            if "n" not in bands[band_name] or "w_min" not in bands[band_name] or "w_max" not in bands[band_name]:
                raise KeyError(f"Band '{band_name}' must have keys: n, w_min, w_max")
        K = int(bands["low"]["n"]) + int(bands["mid"]["n"]) + int(bands["high"]["n"])
        if K <= 0:
            raise ValueError("Fourier config invalid: sum(bands.*.n) must be > 0")

    elif fixed_basis == "chebyshev":
        cheb = spectral.get("chebyshev", {})
        degree = int(cheb.get("degree", 0))
        if degree <= 0:
            raise ValueError("Chebyshev config invalid: spectral_trunk.chebyshev.degree must be > 0")

    else:
        raise ValueError("spectral_trunk.fixed_basis must be 'fourier' or 'chebyshev'")


def set_global_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
