# experiments/stage1_spring_pendulum.py
"""
Unified Stage 1 runner with subcommands:

Single-mode:
  python3 -m experiments.stage1_spring_pendulum single --config configs/stage1.yaml --model fixed --basis fourier
  python3 -m experiments.stage1_spring_pendulum single --config configs/stage1.yaml --model adaptive
  python3 -m experiments.stage1_spring_pendulum single --config configs/stage1.yaml --model cascaded

Bundle-mode (single-task sweeps for multiple models + gates in one command):
  python3 -m experiments.stage1_spring_pendulum bundle --config configs/stage1_gradient_scale.yaml \
      --models baseline fixed_fourier adaptive \
      --gates linear exp \
      --lambda_ic 50.0
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import platform
import subprocess
import torch

from utils.config import (
    load_yaml,
    resolve_runtime,
    validate_config,
    validate_spectral,
    set_global_seed,
    get_nested,
)

from utils.exp_logging import (
    ensure_dir,
    save_json,
    save_history_csv,
    save_training_curves,
    save_eval_plots,
    model_param_summary,
    save_trunk_omegas,
)

from models.model_factory_stage1 import make_model_stage1, make_model_single

from problem.spring_pendulum import (
    SpringPendulumParams,
    residual_second_order,
    ic_residual_vel_only,
)
from ground_truth.solve_reference import solve_reference_task

from training.train_single_task import (
    SingleTaskTrainConfig,
    train_single_task,
    sample_time_points,
    sample_time_points_chebyshev,
)
from training.losses import LossWeights, build_pinn_loss, PhysicsResidualEMA


# ----------------------------
# Utils
# ----------------------------

def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _make_run_dir(base_out_dir: Path, exp_name: str) -> Path:
    return ensure_dir(base_out_dir / exp_name)


def wrap_angle_diff(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(pred - ref), np.cos(pred - ref))


def apply_ic_gate(
    net: torch.nn.Module,
    cfg: Dict[str, Any],
    *,
    gate_override: Optional[str] = None,
    alpha_override: Optional[float] = None,
) -> None:
    """
    Attach IC-gating attributes to the model. These are consumed by ICEmbeddingMixin._ic_gate.

    We keep this "out-of-model" so experiments can sweep gates without rebuilding cfg.
    """
    gate = str(gate_override if gate_override is not None else get_nested(cfg, "ic_embedding.gate", "exp")).lower().strip()
    alpha = float(alpha_override if alpha_override is not None else get_nested(cfg, "ic_embedding.alpha", 1.0))
    T = float(get_nested(cfg, "problem.T", 10.0))
    setattr(net, "ic_gate", gate)
    setattr(net, "ic_gate_alpha", alpha)
    setattr(net, "T", T)


def _make_run_label(
    *,
    model_tag: str,
    gate: str,
    lambda_ic: float,
    alpha: float,
) -> str:
    # Keep folder names stable and grep-friendly
    # Example: adaptive_gate-linear_lamIC50_a1
    lam = f"{lambda_ic:g}"
    a = f"{alpha:g}"
    return f"{model_tag}_gate-{gate}_lamIC{lam}_a{a}"


def _save_common_run_metadata(run_dir: Path, cfg: Dict[str, Any], rt) -> None:
    save_json(run_dir / "config_snapshot.json", cfg)
    save_json(
        run_dir / "resolved_runtime.json",
        {
            "timestamp": datetime.now().isoformat(),
            "git_commit": _git_commit_hash(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": int(rt.seed),
            "device": str(rt.device),
            "dtype": str(rt.dtype),
            "ic_gate": str(get_nested(cfg, "ic_embedding.gate", "exp")),
            "ic_gate_alpha": float(get_nested(cfg, "ic_embedding.alpha", 1.0)),
        },
    )

# ----------------------------
# Evaluation
# ----------------------------

def eval_task_metrics(
    model: torch.nn.Module,
    task: Dict[str, Any],
    params: SpringPendulumParams,
    T: float,
    n_eval: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    t_eval = np.linspace(0.0, T, n_eval)

    ref = solve_reference_task(
        ic=task["ic"],
        k=float(task["k"]),
        params=params,
        t_span=(0.0, T),
        t_eval=t_eval,
    )
    r_ref = ref["r"]
    th_ref = ref["theta"]

    model.eval()
    with torch.no_grad():
        t_t = torch.tensor(t_eval, device=device, dtype=dtype)[:, None]
        u = model(t_t)
        r_pred = u[:, 0].detach().cpu().numpy()
        th_pred = u[:, 1].detach().cpu().numpy()

    # Residual on eval grid
    t_req = torch.tensor(t_eval, device=device, dtype=dtype)[:, None].requires_grad_(True)
    u, du, d2u = model.forward_with_derivatives(t_req, create_graph=False)

    k_t = torch.tensor(float(task["k"]), device=device, dtype=dtype)
    res = residual_second_order(
        t=t_req,
        u=u,
        du_dt=du,
        d2u_dt2=d2u,
        k=k_t,
        params=params,
    )
    res_norm = torch.sqrt(torch.sum(res**2, dim=1)).detach().cpu().numpy()

    # Wrapped theta error (critical for angles)
    dtheta = wrap_angle_diff(th_pred, th_ref)

    # SV-SNN-style vector metrics
    dr = (r_pred - r_ref)
    e_u = np.stack([dr, dtheta], axis=1)
    u_ref = np.stack([r_ref, th_ref], axis=1)

    rel_l2_u = float(np.linalg.norm(e_u) / (np.linalg.norm(u_ref) + 1e-12))
    e_u_norm = np.linalg.norm(e_u, axis=1)
    max_ae_u = float(np.max(e_u_norm))

    # Component diagnostics
    rel_err_r = float(np.linalg.norm(dr) / (np.linalg.norm(r_ref) + 1e-12))
    rel_err_th_wrapped = float(np.linalg.norm(dtheta) / (np.linalg.norm(th_ref) + 1e-12))

    rmse_r = float(np.sqrt(np.mean(dr ** 2)))
    rmse_th = float(np.sqrt(np.mean(dtheta ** 2)))

    max_ae_r = float(np.max(np.abs(dr)))
    max_ae_theta_wrapped = float(np.max(np.abs(dtheta)))

    return {
        "rel_l2_u": rel_l2_u,
        "max_ae_u": max_ae_u,

        "rel_err_r": rel_err_r,
        "rel_err_theta_wrapped": rel_err_th_wrapped,
        "rmse_r": rmse_r,
        "rmse_theta_wrapped": rmse_th,
        "max_ae_r": max_ae_r,
        "max_ae_theta_wrapped": max_ae_theta_wrapped,

        "mean_res_norm": float(np.mean(res_norm)),
        "p95_res_norm": float(np.percentile(res_norm, 95)),
        "max_res_norm": float(np.max(res_norm)),

        "_arrays": {
            "t_eval": t_eval,
            "r_ref": r_ref,
            "theta_ref": th_ref,
            "r_pred": r_pred,
            "theta_pred": th_pred,
            "res_norm": res_norm,
            "wrap_err": dtheta,
            "err_u_norm": e_u_norm,
        },
    }

# ----------------------------
# Single mode (existing) — kept, but now includes gate/alpha/lambdas in metrics.json
# ----------------------------

def _run_single_sweep(
    *,
    run_root: Path,
    cfg: Dict[str, Any],
    rt,
    model: str,
    basis: Optional[str],
    gate: str,
    gate_alpha: float,
    lambda_phys: float,
    lambda_ic: float,
) -> Path:
    """
    Internal: run a single model sweep across k_values × seeds.
    Returns the model_dir written to.
    """
    single_root = ensure_dir(run_root / "single")

    params = SpringPendulumParams()
    T = float(get_nested(cfg, "problem.T", 10.0))
    n_eval = int(get_nested(cfg, "reference.eval_points", 2000))

    ic = dict(get_nested(cfg, "single.ic", {"r0": 1.0, "theta0": 0.2, "rdot0": 0.0, "thetadot0": 0.0}))
    k_values = list(get_nested(cfg, "single.k_values", [10.0, 20.0]))
    seeds = list(get_nested(cfg, "single.seeds", [0, 1]))

    n_col = int(get_nested(cfg, "collocation.interior_points", 2000))
    n_ic = int(get_nested(cfg, "collocation.ic_points", 1))
    sampling = str(get_nested(cfg, "collocation.sampling", "uniform")).lower().strip()

    steps = int(get_nested(cfg, "training.steps", 8000))
    lr = float(get_nested(cfg, "training.lr", 1e-3))
    wd = float(get_nested(cfg, "training.weight_decay", 0.0))
    log_every = int(get_nested(cfg, "training.log_every", 200))

    weights = LossWeights(phys=float(lambda_phys), ic_vel=float(lambda_ic))

    cascaded_training = None
    model_tag_base = None
    if model == "cascaded":
        base = get_nested(cfg, "cascaded_pinn.training", {}) or {}
        rr = get_nested(cfg, "cascaded_pinn.training.residual_reweight", None)
        if rr is None:
            rr = get_nested(cfg, "cascaded_pinn.training.reweight", {}) or {}
        cascaded_training = dict(base)
        cascaded_training["residual_reweight"] = dict(rr) if isinstance(rr, dict) else {}

        sm = cascaded_training.get("sampling_mode", "")
        if sm:
            sampling = str(sm).lower().strip()
        model_tag_base = f"cascaded_{sampling}"
    else:
        if model == "fixed":
            if basis is None:
                raise ValueError("fixed model requires basis")
            model_tag_base = f"fixed_{basis}"
        else:
            model_tag_base = model

    # fixed-chebyshev: collocation should be chebyshev
    if model == "fixed" and (basis == "chebyshev"):
        sampling = "chebyshev"

    # Include gate + lambda in tag so directory is self-describing
    model_tag = _make_run_label(
        model_tag=model_tag_base,
        gate=gate,
        lambda_ic=lambda_ic,
        alpha=gate_alpha,
    )
    model_dir = ensure_dir(single_root / model_tag)

    for k_value in k_values:
        for seed in seeds:
            run_dir = model_dir / f"k{int(k_value)}_seed{int(seed)}"
            if run_dir.exists():
                print(f"(skip) exists: {run_dir}")
                continue
            run_dir = ensure_dir(run_dir)

            save_json(
                run_dir / "run_config.json",
                {
                    "mode": "single",
                    "model": model,
                    "basis": basis,
                    "k": float(k_value),
                    "seed": int(seed),
                    "ic": dict(ic),
                    "T": float(T),
                    "train": {
                        "steps": int(steps),
                        "lr": float(lr),
                        "weight_decay": float(wd),
                        "log_every": int(log_every),
                        "n_col": int(n_col),
                        "n_ic": int(n_ic),
                        "sampling": str(sampling),
                    },
                    "loss_weights": {"lambda_phys": float(lambda_phys), "lambda_ic": float(lambda_ic)},
                    "ic_gate": {"mode": str(gate), "alpha": float(gate_alpha)},
                },
            )

            print(f"\n=== SINGLE: model={model_tag_base} gate={gate} lamIC={lambda_ic:g} k={k_value} seed={seed} ===")

            net = make_model_single(model=model, cfg=cfg, basis=basis)
            apply_ic_gate(net, cfg, gate_override=gate, alpha_override=gate_alpha)

            param_summary = model_param_summary(net)
            save_json(run_dir / "model_param_summary.json", param_summary)

            train_cfg = SingleTaskTrainConfig(
                T=T,
                n_col=n_col,
                n_ic=n_ic,
                steps=steps,
                lr=lr,
                weight_decay=wd,
                log_every=log_every,
                seed=int(seed),
                sampling=sampling,
            )

            t0 = time.perf_counter()
            out = train_single_task(
                model=net,
                cfg=train_cfg,
                task_ic=ic,
                k_value=float(k_value),
                params=params,
                residual_fn=residual_second_order,
                ic_residual_fn=ic_residual_vel_only,
                weights=weights,
                device=rt.device,
                dtype=rt.dtype,
                cascaded=cascaded_training,
            )
            train_time = time.perf_counter() - t0

            hist = out.get("history", [])
            save_history_csv(run_dir, hist)
            save_training_curves(run_dir, hist, tag=f"single:{model_tag}:k{int(k_value)}:seed{int(seed)}")

            trained = out["model"]

            ckpt = {
                "model_state_dict": trained.state_dict(),
                "model": model,              # "baseline" / "adaptive" / "fixed"
                "basis": basis,              # for fixed_fourier
                "gate": gate,
                "gate_alpha": gate_alpha,
                "lambda_phys": lambda_phys,
                "lambda_ic": lambda_ic,
                "k": float(k_value),
                "seed": int(seed),
                "T": float(T),
                "ic": dict(ic),
                "cfg": cfg,                  # optional: can be large, but convenient
            }
            torch.save(ckpt, run_dir / "model_ckpt.pt")
            save_trunk_omegas(run_dir, trained)

            task = {"task_id": "single", "ic": ic, "k": float(k_value)}
            eval_out = eval_task_metrics(
                model=trained,
                task=task,
                params=params,
                T=T,
                n_eval=n_eval,
                device=rt.device,
                dtype=rt.dtype,
            )

            arr = eval_out.pop("_arrays")
            save_eval_plots(
                run_dir,
                t=arr["t_eval"],
                r_ref=arr["r_ref"],
                r_pred=arr["r_pred"],
                th_ref=arr["theta_ref"],
                th_pred=arr["theta_pred"],
                res_norm=arr["res_norm"],
                wrap_err=arr["wrap_err"],
                err_u_norm=arr["err_u_norm"],
                tag=f"single:{model_tag}:k{int(k_value)}:seed{int(seed)}",
            )

            metrics = {
                "mode": "single",
                "model": model_tag_base,
                "model_tag": model_tag,
                "basis": basis,
                "k": float(k_value),
                "seed": int(seed),
                "ic": dict(ic),

                # include these so plotting can label automatically
                "lambda_phys": float(lambda_phys),
                "lambda_ic": float(lambda_ic),
                "ic_gate": str(gate),
                "ic_gate_alpha": float(gate_alpha),

                "train_time_sec": float(train_time),
                "steps": int(train_cfg.steps),
                "num_params_total": int(param_summary.get("num_params_total", 0)),
                "num_params_trainable": int(param_summary.get("num_params_trainable", 0)),
                "final_logs": out.get("final_logs", {}),
                **eval_out,
            }
            save_json(run_dir / "metrics.json", metrics)

    # Plot per-model + all-models for rel_l2_u and max_ae_u if we swept multiple k values
    if len(k_values) > 1:
        try:
            import subprocess as sp
            for metric in ["rel_l2_u", "max_ae_u"]:
                sp.check_call([
                    "python3",
                    "scripts/plot_score_vs_k.py",
                    "--exp_root",
                    str(run_root),
                    "--model_tag",
                    str(model_tag),
                    "--metric",
                    metric,
                ])
        except Exception as e:
            print(f"(warn) could not plot per-model curves: {e}")

    print(f"\nSingle artifacts saved under: {model_dir}")
    return model_dir


def run_single_mode(args, cfg: Dict[str, Any], rt, run_root: Path) -> None:
    model = str(args.model).lower().strip()
    basis = str(args.basis).lower().strip() if args.basis is not None else None

    if model == "fixed" and basis is None:
        raise ValueError("you must specify --basis fourier or chebyshev when --model is fixed")
    if model != "fixed" and basis is not None:
        raise ValueError("--basis is only valid when --model is fixed")

    gate = str(get_nested(cfg, "ic_embedding.gate", "exp")).lower().strip()
    gate_alpha = float(get_nested(cfg, "ic_embedding.alpha", 1.0))
    lambda_phys = float(get_nested(cfg, "loss.lambda_phys", 1.0))
    lambda_ic = float(get_nested(cfg, "loss.lambda_ic", 50.0))

    _run_single_sweep(
        run_root=run_root,
        cfg=cfg,
        rt=rt,
        model=model,
        basis=basis,
        gate=gate,
        gate_alpha=gate_alpha,
        lambda_phys=lambda_phys,
        lambda_ic=lambda_ic,
    )


# ----------------------------
# Bundle mode (NEW)
# ----------------------------

def run_bundle_mode(args, cfg: Dict[str, Any], rt, run_root: Path) -> None:
    """
    Runs multiple single-task sweeps under one experiment root.
    Intended usage: baseline/fixed_fourier/adaptive × gates (linear/exp)
    while keeping everything else identical.
    """
    # Defaults: what you described
    models = [str(m).lower().strip() for m in (args.models or ["baseline", "fixed_fourier", "adaptive"])]
    gates = [str(g).lower().strip() for g in (args.gates or ["linear", "exp"])]

    gate_alpha = float(args.alpha if args.alpha is not None else get_nested(cfg, "ic_embedding.alpha", 1.0))
    lambda_phys = float(args.lambda_phys if args.lambda_phys is not None else get_nested(cfg, "loss.lambda_phys", 1.0))
    lambda_ic = float(args.lambda_ic if args.lambda_ic is not None else get_nested(cfg, "loss.lambda_ic", 50.0))

    # Map bundle identifiers to make_model_single arguments
    # - baseline -> model="baseline"
    # - adaptive -> model="adaptive"
    # - fixed_fourier -> model="fixed", basis="fourier"
    def resolve_model(m: str) -> Tuple[str, Optional[str], str]:
        if m in {"baseline", "baseline_pinn"}:
            return ("baseline", None, "baseline")
        if m in {"adaptive", "adaptive_fourier", "adaptive_spectral"}:
            return ("adaptive", None, "adaptive")
        if m in {"fixed_fourier", "fixed", "fixed_spectral"}:
            return ("fixed", "fourier", "fixed_fourier")
        if m in {"fixed_chebyshev"}:
            return ("fixed", "chebyshev", "fixed_chebyshev")
        raise ValueError(f"Unknown bundle model '{m}'. Expected baseline, fixed_fourier, adaptive (or fixed_chebyshev).")

    written_model_tags: List[str] = []
    for gate in gates:
        for m in models:
            model_arg, basis_arg, pretty = resolve_model(m)
            model_dir = _run_single_sweep(
                run_root=run_root,
                cfg=cfg,
                rt=rt,
                model=model_arg,
                basis=basis_arg,
                gate=gate,
                gate_alpha=gate_alpha,
                lambda_phys=lambda_phys,
                lambda_ic=lambda_ic,
            )
            written_model_tags.append(model_dir.name)

    # After running all sweeps, produce a single all-model plot for key metrics.
    try:
        import subprocess as sp
        for metric in ["rel_l2_u", "max_ae_u"]:
            sp.check_call([
                "python3",
                "scripts/plot_score_vs_k.py",
                "--exp_root",
                str(run_root),
                "--all_models",
                "--metric",
                metric,
            ])
    except Exception as e:
        print(f"(warn) could not plot all-model curves: {e}")

    print("\nBundle complete.")
    print("Wrote model tags under single/:")
    for t in written_model_tags:
        print("  -", t)

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage1.yaml")

    sub = parser.add_subparsers(dest="mode", required=True)

    # single-mode
    p_single = sub.add_parser("single", help="Single task sweeps")
    p_single.add_argument("--model", type=str, required=True, choices=["baseline", "fixed", "adaptive", "cascaded"])
    p_single.add_argument("--basis", type=str, default=None, choices=["fourier", "chebyshev"])

    # bundle-mode
    p_bundle = sub.add_parser("bundle", help="Run a bundle of single-task sweeps (models × gates)")
    p_bundle.add_argument("--models", type=str, nargs="*", default=None,
                          help="Bundle models, e.g. baseline fixed_fourier adaptive (default: baseline fixed_fourier adaptive)")
    p_bundle.add_argument("--gates", type=str, nargs="*", default=None,
                          help="IC gates to run, e.g. linear exp (default: linear exp)")
    p_bundle.add_argument("--lambda_ic", type=float, default=None,
                          help="Override loss.lambda_ic for this bundle (otherwise from YAML).")
    p_bundle.add_argument("--lambda_phys", type=float, default=None,
                          help="Override loss.lambda_phys for this bundle (otherwise from YAML).")
    p_bundle.add_argument("--alpha", type=float, default=None,
                          help="Override ic_embedding.alpha for this bundle (otherwise from YAML).")

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    validate_config(cfg)
    validate_spectral(cfg)

    rt = resolve_runtime(cfg)
    set_global_seed(rt.seed)

    exp_name = str(get_nested(cfg, "experiment.name", "stage1_spring_pendulum"))
    out_dir = Path(get_nested(cfg, "experiment.out_dir", "outputs/stage1"))
    run_root = _make_run_dir(out_dir, exp_name)

    try:
        shutil.copyfile(args.config, run_root / "config_snapshot.yaml")
    except Exception:
        pass

    _save_common_run_metadata(run_root, cfg, rt)

    print(f"\n=== Stage 1: Spring–Pendulum ===")
    print(f"run_root: {run_root}")
    print(f"device={rt.device} dtype={rt.dtype} seed={rt.seed}")

    if args.mode == "single":
        run_single_mode(args, cfg, rt, run_root)
        return

    if args.mode == "bundle":
        run_bundle_mode(args, cfg, rt, run_root)
        return

    raise RuntimeError(f"Unknown mode '{args.mode}'")


if __name__ == "__main__":
    main()
