# utils/exp_logging.py
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_history_csv(run_dir: Path, history: List[Dict[str, Any]], filename: str = "history.csv") -> None:
    if not history:
        return
    keys = sorted({k for row in history for k in row.keys()})
    out = run_dir / filename
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow(row)


def _series(history: List[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for h in history:
        v = h.get(key, float("nan"))
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return out


def save_training_curves(
    run_dir: Path,
    history: List[Dict[str, Any]],
    tag: str,
    filename_loss: str = "training_losses.png",
    filename_rms: str = "training_residual_rms.png",
) -> None:
    if not history:
        return

    steps = [h.get("step", i) for i, h in enumerate(history)]

    plt.figure()
    plt.plot(steps, _series(history, "loss_total"), label="loss_total")
    plt.plot(steps, _series(history, "loss_phys_raw"), label="loss_phys_raw")
    plt.plot(steps, _series(history, "loss_ic_vel"), label="loss_ic_vel")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss (log)")
    plt.title(f"[{tag}] training losses")
    plt.legend()
    plt.savefig(run_dir / filename_loss, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(steps, _series(history, "rms_res_r"), label="rms_res_r")
    plt.plot(steps, _series(history, "rms_res_theta"), label="rms_res_theta")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("RMS residual (log)")
    plt.title(f"[{tag}] training residual RMS")
    plt.legend()
    plt.savefig(run_dir / filename_rms, dpi=150)
    plt.close()


def model_param_summary(model) -> Dict[str, Any]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"num_params_total": int(total), "num_params_trainable": int(trainable)}


def safe_asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def save_eval_plots(
    run_dir: Path,
    t: np.ndarray,
    r_ref: np.ndarray,
    r_pred: np.ndarray,
    th_ref: np.ndarray,
    th_pred: np.ndarray,
    res_norm: np.ndarray,
    wrap_err: np.ndarray,
    err_u_norm: np.ndarray,
    tag: str,
) -> None:
    # helper to save and report
    def _save(fig, path: Path):
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[save_eval_plots] Saved {path}  size={path.stat().st_size} bytes")
        except Exception as e:
            print(f"[save_eval_plots] ERROR saving {path}: {e}")

    # r(t)
    fig = plt.figure()
    plt.plot(t, r_ref, label="r_ref")
    plt.plot(t, r_pred, label="r_pred")
    plt.xlabel("t")
    plt.ylabel("r(t)")
    plt.title(f"[{tag}] r(t) prediction vs reference")
    plt.legend()
    _save(fig, run_dir / "pred_vs_ref_r.png")
    plt.close(fig)

    # theta(t)
    fig = plt.figure()
    plt.plot(t, th_ref, label="theta_ref")
    plt.plot(t, th_pred, label="theta_pred")
    plt.xlabel("t")
    plt.ylabel("theta(t)")
    plt.title(f"[{tag}] theta(t) prediction vs reference")
    plt.legend()
    _save(fig, run_dir / "pred_vs_ref_theta.png")
    plt.close(fig)

    # residual norm
    fig = plt.figure()
    plt.plot(t, res_norm, label="||res(t)||_2")
    plt.xlabel("t")
    plt.ylabel("residual norm")
    plt.title(f"[{tag}] physics residual norm vs time")
    plt.legend()
    _save(fig, run_dir / "residual_norm_vs_t.png")
    plt.close(fig)

    # wrap error
    fig = plt.figure()
    plt.plot(t, wrap_err, label="wrap(theta_pred - theta_ref)")
    plt.xlabel("t")
    plt.ylabel("wrapped angle error (rad)")
    plt.title(f"[{tag}] wrapped theta error vs time")
    plt.legend()
    _save(fig, run_dir / "theta_wrapped_error_vs_t.png")
    plt.close(fig)

    # error norm (vector) vs time
    fig = plt.figure()
    plt.plot(t, err_u_norm, label="|| [r-r_ref, wrap(theta-theta_ref)] ||_2")
    plt.xlabel("t")
    plt.ylabel("error norm")
    plt.title(f"[{tag}] solution error norm vs time")
    plt.legend()
    _save(fig, run_dir / "error_norm_vs_t.png")
    plt.close(fig)

def save_trunk_omegas(
    run_dir: Path,
    model,
    filename_json: str = "omegas.json",
    filename_fig: str = "omegas_init_vs_learned.png",
) -> None:
    """
    Save initialized vs learned omegas for models that expose a trunk with
    `omegas_init` and learnable `omegas` (or similar).

    We try a few common attribute names to be robust.
    """
    trunk = getattr(model, "trunk", None)
    if trunk is None:
        return

    # Try to find init and learned tensors
    # prefer exact names expected by trunk implementation
    omegas_init = getattr(trunk, "omegas_init", None)
    if omegas_init is None:
        # some older trunks might have stored initial omegas under a different name
        omegas_init = getattr(trunk, "_omegas_init", None)
        if omegas_init is None:
            # final fallback: check for a plain "omegas" buffer if present but not desirable
            omegas_init = getattr(trunk, "omegas", None)

    # Learned/current omegas: prefer property `omegas` or try common internals
    omegas_learned = getattr(trunk, "omegas", None)
    if omegas_learned is None:
        omegas_learned = getattr(trunk, "_get_omegas", None)
        if callable(omegas_learned):
            try:
                # call _get_omegas() without arguments, ensuring it's a tensor
                omegas_learned = omegas_learned()
            except Exception:
                omegas_learned = None
    if omegas_learned is None:
        # legacy raw names
        omegas_learned = getattr(trunk, "_omegas", None)
    if omegas_learned is None:
        omegas_learned = getattr(trunk, "omegas_raw", None)

    # If we still don't have both, give up silently
    if omegas_init is None or omegas_learned is None:
        return

    # Convert to numpy arrays for saving/plotting
    try:
        oi = omegas_init.detach().cpu().numpy().astype(float).ravel()
    except Exception:
        # if omegas_init is an array-like already
        oi = np.asarray(omegas_init).astype(float).ravel()

    try:
        ol = omegas_learned.detach().cpu().numpy().astype(float).ravel()
    except Exception:
        ol = np.asarray(omegas_learned).astype(float).ravel()

    # Save numeric arrays
    save_json(
        run_dir / filename_json,
        {"omegas_init": oi.tolist(), "omegas_learned": ol.tolist()},
    )

    # Simple sorted overlay plot (no explicit colors per style rules)
    oi_s = np.sort(oi)
    ol_s = np.sort(ol)

    plt.figure()
    plt.plot(np.arange(len(oi_s)), oi_s, label="init")
    plt.plot(np.arange(len(ol_s)), ol_s, label="learned")
    plt.yscale("log")
    plt.xlabel("feature index (sorted)")
    plt.ylabel("omega (log)")
    plt.title("Adaptive Fourier frequencies: init vs learned")
    plt.legend()
    plt.savefig(run_dir / filename_fig, dpi=150)
    plt.close()
