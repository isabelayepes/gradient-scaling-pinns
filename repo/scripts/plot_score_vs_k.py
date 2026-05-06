# scripts/plot_score_vs_k.py
#!/usr/bin/env python3
"""
Plot <metric> vs k for Stage1 single-mode outputs.

Expected layout:
  <exp_root>/single/<model_tag>/kXX_seedY/metrics.json

Writes per model:
  - <model_dir>/<metric>_vs_k.png
  - <model_dir>/<metric>_vs_k.json

Optionally writes all-models:
  - <exp_root>/single/<metric>_vs_k_all_models.png
  - (if filters/ylim used) also writes a suffixed version, e.g.
      <metric>_vs_k_all_models__zoom.png

New features:
  - --exclude_substr / --include_substr: filter model dirs for all-models plot
  - --ylim: y-axis limits
  - --out_suffix: custom suffix appended to all-models filename
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def ci95_halfwidth(x: np.ndarray) -> float:
    """
    95% confidence interval half-width for the mean using Student's t.
    halfwidth = t_{0.975, n-1} * (std / sqrt(n))
    """
    n = int(x.size)
    if n <= 1:
        return float("nan")
    s = float(np.std(x, ddof=1))
    se = s / math.sqrt(n)
    tcrit = float(student_t.ppf(0.975, df=n - 1))
    return tcrit * se


def safe_metric_filename(metric_name: str) -> str:
    out = []
    for ch in metric_name:
        if ch.isalnum() or ch in ["_", "-", "."]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _collect_setting(points: List[Dict[str, Any]], key: str) -> Optional[Any]:
    vals = []
    for p in points:
        if key in p and p[key] is not None:
            vals.append(p[key])
    if not vals:
        return None
    first = vals[0]
    if all(v == first for v in vals):
        return first
    return None


def gather_model_points(model_dir: Path, metric_name: str) -> List[Dict[str, Any]]:
    pts: List[Dict[str, Any]] = []
    for metrics_path in sorted(model_dir.glob("k*_seed*/metrics.json")):
        try:
            m = load_json(metrics_path)
        except Exception:
            continue

        if "k" not in m or metric_name not in m:
            continue

        try:
            k_val = float(m["k"])
            metric_val = float(m[metric_name])
        except Exception:
            continue

        pts.append(
            {
                "k": k_val,
                "seed": int(m.get("seed", -1)),
                metric_name: metric_val,
                # Optional experiment labels (may be missing depending on your writer)
                "lambda_ic": m.get("lambda_ic", None),
                "lambda_phys": m.get("lambda_phys", None),
                "ic_gate": m.get("ic_gate", None),
                "ic_gate_alpha": m.get("ic_gate_alpha", None),
                "path": str(metrics_path),
            }
        )
    return pts


def aggregate_by_k(points: List[Dict[str, Any]], metric_name: str) -> Dict[float, Dict[str, float]]:
    out: Dict[float, Dict[str, float]] = {}
    if not points:
        return out

    ks = sorted(set(p["k"] for p in points))
    for k in ks:
        vals = np.array([p[metric_name] for p in points if p["k"] == k], dtype=float)
        out[k] = {
            "n": float(vals.size),
            "mean": float(np.mean(vals)) if vals.size else float("nan"),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
            "ci95": ci95_halfwidth(vals),
        }
    return out


def _title_suffix(points: List[Dict[str, Any]]) -> str:
    lam_ic = _collect_setting(points, "lambda_ic")
    gate = _collect_setting(points, "ic_gate")
    alpha = _collect_setting(points, "ic_gate_alpha")

    parts = []
    if lam_ic is not None:
        try:
            parts.append(f"lambda_ic={float(lam_ic):g}")
        except Exception:
            parts.append(f"lambda_ic={lam_ic}")
    if gate is not None:
        parts.append(f"gate={gate}")
    if alpha is not None:
        try:
            parts.append(f"alpha={float(alpha):g}")
        except Exception:
            parts.append(f"alpha={alpha}")

    if not parts:
        return ""
    return " | " + ", ".join(parts)


# ----------------------
# Sanity print helper
# ----------------------
def print_ci_counts(model_name: str, metric_name: str, agg: Dict[float, Dict[str, float]]) -> None:
    """
    Print lines like:
      [CI] <model_tag> | metric=<metric> | k=<k> | n=<num_seeds>
    so you can confirm how many seeds were used to compute the CI for each (model,k).
    """
    for k in sorted(agg.keys()):
        n = int(agg[k].get("n", -1))
        print(f"[CI] {model_name} | metric={metric_name} | k={int(k)} | n={n}")


def plot_single_model(model_dir: Path, points: List[Dict[str, Any]], metric_name: str) -> None:
    agg = aggregate_by_k(points, metric_name)
    if not agg:
        return

    # ---- SANITY CHECK PRINT ----
    print_ci_counts(model_dir.name, metric_name, agg)

    ks = np.array(sorted(agg.keys()), dtype=float)
    means = np.array([agg[k]["mean"] for k in ks], dtype=float)
    errs = np.array([agg[k]["ci95"] for k in ks], dtype=float)

    metric_stem = safe_metric_filename(metric_name)

    plt.figure()
    plt.errorbar(ks, means, yerr=errs, fmt="-o", capsize=3)
    plt.xlabel("k")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs k ({model_dir.name}){_title_suffix(points)} [mean ± 95% CI]")
    plt.savefig(model_dir / f"{metric_stem}_vs_k.png", dpi=150)
    plt.close()

    payload = {
        "model_tag": model_dir.name,
        "metric": metric_name,
        "points": points,
        "agg_by_k": {str(k): v for k, v in agg.items()},
        "settings": {
            "lambda_ic": _collect_setting(points, "lambda_ic"),
            "lambda_phys": _collect_setting(points, "lambda_phys"),
            "ic_gate": _collect_setting(points, "ic_gate"),
            "ic_gate_alpha": _collect_setting(points, "ic_gate_alpha"),
        },
    }
    (model_dir / f"{metric_stem}_vs_k.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _passes_filters(name: str, include_substr: Sequence[str], exclude_substr: Sequence[str]) -> bool:
    if include_substr:
        # require at least one include substring to match
        if not any(s in name for s in include_substr):
            return False
    if exclude_substr:
        if any(s in name for s in exclude_substr):
            return False
    return True


def plot_all_models(
    single_root: Path,
    model_dirs: List[Path],
    metric_name: str,
    ylim: Optional[List[float]] = None,
    out_suffix: str = "",
) -> Optional[Path]:
    metric_stem = safe_metric_filename(metric_name)

    plt.figure()
    any_plotted = False

    # Infer shared settings for title by pooling points
    all_points_for_settings: List[Dict[str, Any]] = []
    for md in model_dirs:
        pts = gather_model_points(md, metric_name)
        if pts:
            all_points_for_settings.extend(pts)

    for md in model_dirs:
        pts = gather_model_points(md, metric_name)
        agg = aggregate_by_k(pts, metric_name)
        if not agg:
            continue

        # ---- SANITY CHECK PRINT for each model in all-models plot ----
        print_ci_counts(md.name, metric_name, agg)

        ks = np.array(sorted(agg.keys()), dtype=float)
        means = np.array([agg[k]["mean"] for k in ks], dtype=float)
        errs = np.array([agg[k]["ci95"] for k in ks], dtype=float)
        plt.errorbar(ks, means, yerr=errs, fmt="-o", label=md.name, capsize=3)
        any_plotted = True

    if not any_plotted:
        plt.close()
        return None

    plt.xlabel("k")
    plt.ylabel(metric_name)

    suffix = _title_suffix(all_points_for_settings)
    plt.title(f"{metric_name} vs k (all models){suffix} [mean ± 95% CI]")
    plt.legend()

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    suffix_part = f"__{out_suffix}" if out_suffix else ""
    out_path = single_root / f"{metric_stem}_vs_k_all_models{suffix_part}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp_root",
        type=str,
        required=True,
        help="Path to outputs/<...>/<exp_name> (must contain ./single/)",
    )
    ap.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="If set, plot only this single/<model_tag> dir.",
    )
    ap.add_argument(
        "--all_models",
        action="store_true",
        help="Also write single/<metric>_vs_k_all_models*.png across model tags.",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="rel_l2_u",
        help="Metric key in metrics.json to plot vs k (default: rel_l2_u).",
    )
    ap.add_argument(
        "--include_substr",
        type=str,
        nargs="*",
        default=[],
        help="For --all_models: include only model dirs whose name contains ANY of these substrings.",
    )
    ap.add_argument(
        "--exclude_substr",
        type=str,
        nargs="*",
        default=[],
        help="For --all_models: exclude model dirs whose name contains ANY of these substrings.",
    )
    ap.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits for --all_models, e.g. --ylim 0 0.4",
    )
    ap.add_argument(
        "--out_suffix",
        type=str,
        default="",
        help="Optional suffix appended to all-models filename (after '__').",
    )
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    single_root = exp_root / "single"
    if not single_root.exists():
        raise FileNotFoundError(f"Missing: {single_root}")

    metric_name = args.metric

    if args.model_tag:
        model_dir = single_root / args.model_tag
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing: {model_dir}")
        pts = gather_model_points(model_dir, metric_name)
        plot_single_model(model_dir, pts, metric_name)
        stem = safe_metric_filename(metric_name)
        print("Wrote:", model_dir / f"{stem}_vs_k.png")
        print("Wrote:", model_dir / f"{stem}_vs_k.json")
        return

    # Per-model plots (always for everything present)
    model_dirs_all = [p for p in sorted(single_root.iterdir()) if p.is_dir()]
    for md in model_dirs_all:
        pts = gather_model_points(md, metric_name)
        plot_single_model(md, pts, metric_name)

    # All-models plot (optionally filtered)
    if args.all_models:
        model_dirs = [
            md for md in model_dirs_all
            if _passes_filters(md.name, args.include_substr, args.exclude_substr)
        ]

        out_path = plot_all_models(
            single_root=single_root,
            model_dirs=model_dirs,
            metric_name=metric_name,
            ylim=list(args.ylim) if args.ylim is not None else None,
            out_suffix=args.out_suffix,
        )
        if out_path is not None:
            print("Wrote:", out_path)
        else:
            print("(warn) no models had plottable data for metric:", metric_name)


if __name__ == "__main__":
    main()
