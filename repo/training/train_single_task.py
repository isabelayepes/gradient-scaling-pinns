# training/train_single_task.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import time
import math
import torch
import torch.nn as nn

from training.losses import LossWeights, build_pinn_loss, PhysicsResidualEMA


@dataclass(frozen=True)
class SingleTaskTrainConfig:
    """
    Single-task training configuration.

    sampling:
      - "uniform": uniform on (0, T]
      - "chebyshev": Chebyshev-like on (0, T]
    """
    T: float = 10.0
    n_col: int = 2000
    n_ic: int = 1
    steps: int = 20_000
    lr: float = 1e-3
    weight_decay: float = 0.0
    log_every: int = 500
    seed: int = 0
    sampling: str = "uniform"


def _set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_time_points(
    T: float,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    include_t0: bool = False,
) -> torch.Tensor:
    """
    Uniform time sampling on (0, T] for collocation, and at t=0 for IC.
    Returns (n, 1).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if include_t0:
        return torch.zeros((n, 1), device=device, dtype=dtype)

    t = torch.rand((n, 1), device=device, dtype=dtype) * T
    return torch.clamp(t, min=1e-6)


def sample_time_points_chebyshev(
    T: float,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Chebyshev-like sampling on (0, T], clustered near endpoints.
    Returns (n, 1).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    u = torch.rand((n, 1), device=device, dtype=dtype)  # (0,1)
    x = torch.cos(math.pi * u)                          # [-1,1], clustered at ±1
    t = 0.5 * T * (x + 1.0)                             # map to [0,T]
    return torch.clamp(t, min=1e-6)


@torch.no_grad()
def residual_norm_on_times(
    model: nn.Module,
    t: torch.Tensor,
    k: torch.Tensor,
    params,
    residual_fn,
) -> torch.Tensor:
    """
    Compute per-sample L2 norm of physics residual at times t (N,1), WITHOUT
    building an optimization graph.
    """
    if t.ndim == 1:
        t_req = t[:, None].clone().detach()
    elif t.ndim == 2 and t.shape[1] == 1:
        t_req = t.clone().detach()
    else:
        raise ValueError(f"Expected t shape (N,) or (N,1); got {tuple(t.shape)}")

    with torch.enable_grad():
        t_req = t_req.requires_grad_(True)
        u, du_dt, d2u_dt2 = model.forward_with_derivatives(t_req, create_graph=False)
        res = residual_fn(
            t=t_req,
            u=u,
            du_dt=du_dt,
            d2u_dt2=d2u_dt2,
            k=k,
            params=params,
        )
        rn = torch.sqrt(torch.sum(res * res, dim=1) + 1e-12)
        return rn.detach()


def _make_steps_list(total_steps: int, blocks: int) -> List[int]:
    """
    Split total_steps across `blocks` using floor division, remainder on last.
    """
    if blocks < 1:
        raise ValueError("blocks must be >= 1")
    base = int(total_steps) // int(blocks)
    rem = int(total_steps) - base * int(blocks)
    steps_list = [base] * int(blocks)
    steps_list[-1] += rem
    return [int(s) for s in steps_list]


def _train_steps(
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    ema: PhysicsResidualEMA,
    cfg: SingleTaskTrainConfig,
    task_ic: Dict[str, float],
    k: torch.Tensor,
    params,
    residual_fn,
    ic_residual_fn,
    weights: LossWeights,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    global_step_start: int,
    stage_label: str,
    sampling_mode: str,
    residual_reweight_cfg: Optional[dict],
    t_start_global: float,  # <-- NEW: global monotonic timing anchor
) -> Tuple[List[Dict[str, float]], int]:
    """
    Run `steps` optimization iterations; return (history_chunk, global_step_end).

    sampling_mode:
      - "uniform": sample times using cfg.sampling
      - "reweight": sample times using cfg.sampling, and reweight physics per point by residual norm
    """
    history: List[Dict[str, float]] = []
    global_step = int(global_step_start)

    base_sampling = str(cfg.sampling).lower().strip()
    sampling_mode = str(sampling_mode).lower().strip()
    if sampling_mode not in {"uniform", "reweight"}:
        raise ValueError(f"Unknown sampling_mode='{sampling_mode}'. Use 'uniform' or 'reweight'.")

    rr = residual_reweight_cfg or {}
    rr_power = float(rr.get("power", 1.0))
    rr_eps = float(rr.get("eps", 1e-12))

    for _ in range(int(steps)):
        global_step += 1
        opt.zero_grad(set_to_none=True)

        if base_sampling == "chebyshev":
            t_col = sample_time_points_chebyshev(cfg.T, cfg.n_col, device=device, dtype=dtype)
        else:
            t_col = sample_time_points(cfg.T, cfg.n_col, device=device, dtype=dtype, include_t0=False)
        t0 = sample_time_points(cfg.T, cfg.n_ic, device=device, dtype=dtype, include_t0=True)

        point_weights = None
        if sampling_mode == "reweight":
            rn = residual_norm_on_times(
                model=model, t=t_col, k=k, params=params, residual_fn=residual_fn
            )
            w = torch.pow(torch.clamp(rn + rr_eps, min=rr_eps), rr_power)
            w = w / (torch.mean(w) + 1e-12)
            point_weights = w.to(device=device, dtype=dtype).detach()

        loss, logs = build_pinn_loss(
            model=model,
            t_col=t_col,
            t0=t0,
            k=k,
            ic=task_ic,
            params=params,
            residual_fn=residual_fn,
            ic_residual_fn=ic_residual_fn,
            weights=weights,
            ema=ema,
            point_weights=point_weights,
        )

        loss.backward()
        opt.step()

        last_step = global_step_start + steps
        if (global_step % cfg.log_every == 0) or (global_step == 1) or (global_step == last_step):
            elapsed_total = time.perf_counter() - t_start_global  # <-- FIXED
            logs_out = {
                **logs,
                "step": global_step,
                "elapsed_total_sec": float(elapsed_total),
                "lr": float(cfg.lr),
                "stage": stage_label,
                "sampling_mode": sampling_mode,
            }
            history.append(logs_out)

            ema_str = ""
            if "ema_rms_r" in logs_out:
                ema_str = (
                    f" ema=(r={logs_out['ema_rms_r']:.3e}, th={logs_out['ema_rms_theta']:.3e})"
                    f" batch=(r={logs_out['batch_rms_r']:.3e}, th={logs_out['batch_rms_theta']:.3e})"
                )

            print(
                f"[{stage_label} | step {global_step:6d}] "
                f"loss={logs_out['loss_total']:.3e}, "
                f"(phys_bal={logs_out['loss_phys_balanced']:.3e}, "
                f"phys_raw={logs_out['loss_phys_raw']:.3e}, "
                f"ic={logs_out['loss_ic_vel']:.3e}), "
                f"rms=(r={logs_out['rms_res_r']:.3e}, th={logs_out['rms_res_theta']:.3e}),"
                f"{ema_str} "
                f"elapsed_total={elapsed_total:.1f}s"
            )

    return history, global_step


def _ic_gate_smoking_gun(model: nn.Module, T: float, device: torch.device, dtype: torch.dtype) -> None:
    """
    Prints a decisive check of what gate is being used by the IC embedding.

    For T=10:
      linear -> [0, 5, 10]
      exp    -> [0, ~0.993, ~0.99995]
    """
    gate = getattr(model, "ic_gate", None)
    print(f"[IC_GATE] model.ic_gate = {gate!r}")

    # If you implemented ICEmbeddingMixin._ic_gate(t) as suggested:
    if hasattr(model, "_ic_gate"):
        t_test = torch.tensor([[0.0], [0.5 * T], [T]], device=device, dtype=dtype)
        with torch.no_grad():
            g_test = model._ic_gate(t_test)  # type: ignore[attr-defined]
        print(f"[IC_GATE_CHECK] t={t_test.squeeze().tolist()} g={g_test.squeeze().tolist()}")
    else:
        print("[IC_GATE_CHECK] model has no _ic_gate(t); cannot print g(t) values.")


def train_single_task(
    model: nn.Module,
    cfg: SingleTaskTrainConfig,
    task_ic: Dict[str, float],
    k_value: float,
    params,
    residual_fn,
    ic_residual_fn,
    weights: LossWeights = LossWeights(),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    cascaded: Optional[dict] = None,
) -> Dict[str, object]:
    """
    Train a PINN model on a single task.

    - Standard models: run cfg.steps of ordinary training.
    - Cascaded models: if `cascaded` is provided AND model supports block control,
      train blocks sequentially.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device=device, dtype=dtype)

    _set_seed(cfg.seed)

    if hasattr(model, "set_task_ic"):
        model.set_task_ic(task_ic)
    
    _ic_gate_smoking_gun(model, cfg.T, device, dtype)

    k = torch.tensor(float(k_value), device=device, dtype=dtype)
    ema = PhysicsResidualEMA(beta=0.99, eps=1e-12)

    supports_cascade = (
        hasattr(model, "set_active_stage")
        and hasattr(model, "freeze_only_block")
        and hasattr(model, "freeze_all")
        and hasattr(model, "unfreeze_all")
        and hasattr(getattr(model, "cfg", None), "blocks")
    )

    def _make_opt() -> torch.optim.Optimizer:
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

    history: List[Dict[str, float]] = []
    global_step = 0
    t_start_global = time.perf_counter()  # <-- FIXED: global anchor for entire training run

    # -------------------------
    # Standard training
    # -------------------------
    if (cascaded is None) or (not supports_cascade):
        opt = _make_opt()
        hist, global_step = _train_steps(
            model=model,
            opt=opt,
            ema=ema,
            cfg=cfg,
            task_ic=task_ic,
            k=k,
            params=params,
            residual_fn=residual_fn,
            ic_residual_fn=ic_residual_fn,
            weights=weights,
            device=device,
            dtype=dtype,
            steps=int(cfg.steps),
            global_step_start=global_step,
            stage_label="train",
            sampling_mode="uniform",
            residual_reweight_cfg=None,
            t_start_global=t_start_global,
        )
        history.extend(hist)
        final = history[-1] if history else {}
        return {"model": model, "history": history, "final_logs": final}

    # -------------------------
    # Cascaded training
    # -------------------------
    B = int(model.cfg.blocks)  # type: ignore[attr-defined]
    B = max(B, 1)

    freeze_prev = bool((cascaded or {}).get("freeze_previous", True))
    if not freeze_prev:
        raise ValueError("cascaded.freeze_previous must be true (gating is disabled).")

    sampling_mode = str((cascaded or {}).get("sampling_mode", "uniform")).lower().strip()
    residual_reweight_cfg = (cascaded or {}).get("residual_reweight", None)

    steps_list = _make_steps_list(int(cfg.steps), B)

    for b in range(B):
        model.set_active_stage(b)  # type: ignore[attr-defined]
        model.freeze_all()         # type: ignore[attr-defined]
        model.freeze_only_block(b) # type: ignore[attr-defined]

        opt = _make_opt()
        stage_label = f"block{b+1}/{B}"

        hist, global_step = _train_steps(
            model=model,
            opt=opt,
            ema=ema,
            cfg=cfg,
            task_ic=task_ic,
            k=k,
            params=params,
            residual_fn=residual_fn,
            ic_residual_fn=ic_residual_fn,
            weights=weights,
            device=device,
            dtype=dtype,
            steps=int(steps_list[b]),
            global_step_start=global_step,
            stage_label=stage_label,
            sampling_mode=sampling_mode,
            residual_reweight_cfg=residual_reweight_cfg,
            t_start_global=t_start_global,
        )
        history.extend(hist)

    model.set_active_stage(None)  # type: ignore[attr-defined]
    model.unfreeze_all()          # type: ignore[attr-defined]

    final = history[-1] if history else {}
    return {"model": model, "history": history, "final_logs": final}
