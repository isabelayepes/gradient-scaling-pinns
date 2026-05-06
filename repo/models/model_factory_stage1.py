# models/model_factory_stage1.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from utils.config import get_nested

from models.fourier_fixed_trunk import FourierFixedTrunkConfig, build_omegas_bands
from models.chebyshev_fixed_trunk import ChebyshevFixedTrunkConfig

from models.fixed_spectral_pinn import FixedSpectralPINNConfig, FixedSpectralPINN
from models.adaptive_spectral_pinn import AdaptiveSpectralPINNConfig, AdaptiveSpectralPINN

from models.baseline_pinn import BaselinePINNConfig, BaselinePINN

# cascaded optional
try:
    from models.cascaded_pinn import CascadedPINNConfig, CascadedPINN, MLPConfig  # type: ignore
except Exception:
    CascadedPINNConfig = None  # type: ignore
    CascadedPINN = None        # type: ignore
    MLPConfig = None           # type: ignore


def _get_spacing(cfg: Dict[str, Any]) -> str:
    # Support both:
    #  - spectral_trunk.spacing
    #  - spectral_trunk.fourier.spacing
    s = get_nested(cfg, "spectral_trunk.fourier.spacing", None)
    if s is None:
        s = get_nested(cfg, "spectral_trunk.spacing", "log")
    return str(s).lower().strip()


def _build_omegas_from_yaml(cfg: Dict[str, Any]) -> np.ndarray:
    bands = get_nested(cfg, "spectral_trunk.fourier.bands", None)
    if bands is None:
        raise KeyError("Missing spectral_trunk.fourier.bands in YAML")

    spacing = _get_spacing(cfg)

    low = bands["low"]
    mid = bands["mid"]
    high = bands["high"]

    return build_omegas_bands(
        low=(float(low["w_min"]), float(low["w_max"]), int(low["n"])),
        mid=(float(mid["w_min"]), float(mid["w_max"]), int(mid["n"])),
        high=(float(high["w_min"]), float(high["w_max"]), int(high["n"])),
        spacing=spacing,
    )


def make_model_stage1(model_id: str, cfg: Dict[str, Any]) -> torch.nn.Module:
    """
    Family-mode model factory (IDs from cfg.models.run).
    """
    T = float(get_nested(cfg, "problem.T", 10.0))
    r_min = float(get_nested(cfg, "problem.r_min", 1e-4))

    trunk_include_bias = bool(get_nested(cfg, "spectral_trunk.include_bias", False))
    fixed_basis = str(get_nested(cfg, "spectral_trunk.fixed_basis", "fourier")).lower().strip()

    omegas_np = _build_omegas_from_yaml(cfg)
    omegas_t = torch.as_tensor(omegas_np, dtype=torch.float32)

    if model_id == "baseline_pinn":
        b = cfg.get("baseline_pinn", {})
        model_cfg = BaselinePINNConfig(
            hidden_layers=tuple(int(x) for x in b.get("hidden_layers", [128, 128, 128])),
            activation=str(b.get("activation", "tanh")),
            out_dim=2,
            r_min=r_min,
            ic_embed=True,
        )
        return BaselinePINN(model_cfg)

    if model_id == "cascaded_pinn":
        if CascadedPINN is None:
            raise RuntimeError("models.cascaded_pinn not importable but 'cascaded_pinn' requested.")
        c = cfg.get("cascaded_pinn", {})
        blocks = int(c.get("blocks", 3))
        hidden_layers = c.get("hidden_layers", [128, 128, 128])
        activation = str(c.get("activation", "tanh"))
        bias = bool(c.get("bias", True))
        out_dim = 2

        model_cfg = CascadedPINNConfig(
            blocks=blocks,
            mlp=MLPConfig(
                in_dim=1,
                hidden_layers=tuple(int(x) for x in hidden_layers),
                activation=activation,
                out_dim=out_dim,
                bias=bias,
            ),
            r_min=r_min,
            out_dim=out_dim,
        )
        return CascadedPINN(model_cfg)

    if model_id == "fixed_spectral_pinn":
        head_bias = bool(get_nested(cfg, "fixed_spectral_pinn.head_bias", False))
        f = get_nested(cfg, "fixed_spectral_pinn", {}) or {}
        head_init = str(f.get("head_init", "small_normal"))
        fourier_std = float(f.get("fourier_init_std", 1e-2))
        cheby_std = float(f.get("cheby_init_std", 1e-3))

        if fixed_basis == "fourier":
            trunk_cfg = FourierFixedTrunkConfig(omegas=omegas_np, include_bias=trunk_include_bias)
            model_cfg = FixedSpectralPINNConfig(
                basis="fourier",
                fourier=trunk_cfg,
                chebyshev=None,
                out_dim=2,
                head_bias=head_bias,
                r_min=r_min,
                head_init=head_init,
                fourier_init_std=fourier_std,
                cheby_init_std=cheby_std,
            )
            return FixedSpectralPINN(model_cfg)

        if fixed_basis == "chebyshev":
            cheb = get_nested(cfg, "spectral_trunk.chebyshev", {}) or {}
            cheb_cfg = ChebyshevFixedTrunkConfig(
                degree=int(cheb.get("degree", 16)),
                T=T,
                include_bias=trunk_include_bias,
                normalize_features=bool(cheb.get("normalize_features", True)),
            )
            model_cfg = FixedSpectralPINNConfig(
                basis="chebyshev",
                fourier=None,
                chebyshev=cheb_cfg,
                out_dim=2,
                head_bias=head_bias,
                r_min=r_min,
                head_init=head_init,
                fourier_init_std=fourier_std,
                cheby_init_std=cheby_std,
            )
            return FixedSpectralPINN(model_cfg)

        raise ValueError(f"Unknown spectral_trunk.fixed_basis='{fixed_basis}'")

    if model_id == "adaptive_spectral_pinn":
        a = cfg.get("adaptive_spectral_pinn", {}) or {}
        head_bias = bool(a.get("head_bias", False))
        model_cfg = AdaptiveSpectralPINNConfig(
            omegas_init=omegas_t,
            include_bias=trunk_include_bias,
            learnable_omegas=True,
            positive_omegas=bool(a.get("positive_omegas", True)),
            softplus_beta=float(a.get("softplus_beta", 1.0)),
            out_dim=2,
            head_bias=head_bias,
            r_min=r_min,
        )
        return AdaptiveSpectralPINN(model_cfg)

    if model_id == "adaptive_multihead_pinn":
        m = cfg.get("adaptive_multihead_pinn", {}) or {}
        head_bias = bool(m.get("head_bias", False))
        model_cfg = AdaptiveMultiHeadPINNConfig(
            omegas_init=omegas_t,
            include_bias=trunk_include_bias,
            learnable_omegas=True,
            positive_omegas=bool(m.get("positive_omegas", True)),
            softplus_beta=float(m.get("softplus_beta", 1.0)),
            out_dim=2,
            head_bias=head_bias,
            r_min=r_min,
            task_ids=None,
        )
        return AdaptiveMultiHeadPINN(model_cfg)

    raise ValueError(f"Unknown model_id='{model_id}'")


def make_model_single(
    model: str,
    cfg: Dict[str, Any],
    basis: Optional[str] = None,
) -> torch.nn.Module:
    """
    Single-mode builder (argparse-friendly names), centralized here.

    model choices: baseline, fixed, adaptive, cascaded
    """
    model = str(model).lower().strip()
    if model == "baseline":
        return make_model_stage1("baseline_pinn", cfg)

    if model == "adaptive":
        return make_model_stage1("adaptive_spectral_pinn", cfg)

    if model == "cascaded":
        return make_model_stage1("cascaded_pinn", cfg)

    if model == "fixed":
        if basis is None:
            # use YAML fixed_basis if not provided
            basis = str(get_nested(cfg, "spectral_trunk.fixed_basis", "fourier")).lower().strip()
        # temporarily override fixed_basis in cfg by passing through a shallow copy
        cfg2 = dict(cfg)
        st = dict(cfg.get("spectral_trunk", {}) or {})
        st["fixed_basis"] = str(basis).lower().strip()
        cfg2["spectral_trunk"] = st
        return make_model_stage1("fixed_spectral_pinn", cfg2)

    raise ValueError(f"Unknown single model='{model}'. Use baseline|fixed|adaptive|cascaded.")
