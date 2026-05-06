# scripts/wilcoxon_gate_test.py
#!/usr/bin/env python3
"""
Paired Wilcoxon signed-rank tests for Stage1 outputs (+ diagnostics).

Adds:
  - symmetry/shape diagnostic plots for d = a - b
  - bootstrap CI for median(d) (and mean(d))
  - sign-flip permutation p-value for median(d)

Expected layout:
  <exp_root>/single/<model_tag>/kXX_seedY/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

try:
    from scipy.stats import wilcoxon, skew
except Exception as e:
    raise RuntimeError("scipy is required for this script. Install with: pip install scipy") from e

import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_seed_dirs(model_dir: Path, k: int) -> Dict[int, Path]:
    """Return {seed: metrics.json path} for a given model_dir and k."""
    out: Dict[int, Path] = {}
    for p in sorted(model_dir.glob(f"k{k}_seed*/metrics.json")):
        parent = p.parent.name  # e.g., "k20_seed7"
        if "seed" not in parent:
            continue
        try:
            seed_str = parent.split("seed", 1)[1]
            seed = int(seed_str)
        except Exception:
            continue
        out[seed] = p
    return out


def paired_values(
    exp_root: Path,
    model_a: str,
    model_b: str,
    k: int,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Load paired arrays (a_vals, b_vals) for the same seeds."""
    a_dir = exp_root / "single" / model_a
    b_dir = exp_root / "single" / model_b
    if not a_dir.exists():
        raise FileNotFoundError(f"Missing model dir: {a_dir}")
    if not b_dir.exists():
        raise FileNotFoundError(f"Missing model dir: {b_dir}")

    a_map = find_seed_dirs(a_dir, k)
    b_map = find_seed_dirs(b_dir, k)

    common_seeds = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not common_seeds:
        raise RuntimeError(f"No common seeds found for k={k} between '{model_a}' and '{model_b}'.")

    a_vals, b_vals, used = [], [], []
    for s in common_seeds:
        ma = load_json(a_map[s])
        mb = load_json(b_map[s])
        if metric not in ma or metric not in mb:
            continue
        try:
            a_vals.append(float(ma[metric]))
            b_vals.append(float(mb[metric]))
            used.append(s)
        except Exception:
            continue

    if len(used) < 3:
        raise RuntimeError(
            f"Too few paired points for k={k}, metric='{metric}'. Found {len(used)} paired seeds."
        )

    return np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float), used


def holm_adjust(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni correction. Returns adjusted p-values in original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        p = pvals[idx]
        a = min(1.0, (m - i) * p)
        a = max(a, prev)  # monotone non-decreasing
        adj[idx] = a
        prev = a
    return adj.tolist()


def summarize_diffs(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Summaries for a - b (negative => a better, assuming lower is better)."""
    d = a - b
    return {
        "n": int(d.size),
        "mean_diff": float(np.mean(d)),
        "median_diff": float(np.median(d)),
        "std_diff": float(np.std(d, ddof=1)) if d.size > 1 else float("nan"),
        "skew_diff": float(skew(d)) if d.size > 2 else float("nan"),
        "frac_a_better": float(np.mean(d < 0.0)),
        "frac_equal": float(np.mean(d == 0.0)),
        "frac_b_better": float(np.mean(d > 0.0)),
        "a_mean": float(np.mean(a)),
        "b_mean": float(np.mean(b)),
        "a_median": float(np.median(a)),
        "b_median": float(np.median(b)),
    }


def run_wilcoxon(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """
    Returns two-sided and both one-sided p-values.
    lower metric is better
      - "a_less" tests H1: median(a - b) < 0  (a better)
      - "b_less" tests H1: median(a - b) > 0  (b better)
    """
    res_two = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    res_a_less = wilcoxon(a, b, alternative="less", zero_method="wilcox")
    res_b_less = wilcoxon(a, b, alternative="greater", zero_method="wilcox")
    return {
        "statistic": float(res_two.statistic),
        "p_two_sided": float(res_two.pvalue),
        "p_a_less_b": float(res_a_less.pvalue),
        "p_b_less_a": float(res_b_less.pvalue),
    }


def bootstrap_ci(
    x: np.ndarray,
    stat_fn,
    n_boot: int = 20000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a statistic."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        stats[i] = float(stat_fn(samp))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def sign_flip_perm_pvalue_median(
    d: np.ndarray,
    n_perm: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Paired sign-flip permutation test for median(d).
    Null: median(d) == 0 under random sign flips.
    Two-sided p-value.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs = float(np.median(d))
    n = d.size
    if n == 0:
        return float("nan")
    stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        stats[i] = float(np.median(d * signs))
    p = (np.sum(np.abs(stats) >= abs(obs)) + 1.0) / (n_perm + 1.0)
    return float(p)


def save_diff_diagnostics_plot(
    out_dir: Path,
    *,
    k: int,
    metric: str,
    model_a: str,
    model_b: str,
    d: np.ndarray,
) -> Path:
    """
    Saves a compact plot to eyeball symmetry/shape:
      - histogram of d
      - scatter of sorted d vs -sorted d (mirror check)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_metric = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in metric)
    safe_a = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in model_a)
    safe_b = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in model_b)

    fig = plt.figure(figsize=(8, 3.2))

    # Left: histogram
    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(d, bins=min(10, max(3, d.size)), edgecolor="black")
    ax1.axvline(0.0, linestyle="--")
    ax1.set_title("diff histogram (A-B)")
    ax1.set_xlabel("d = A - B")
    ax1.set_ylabel("count")

    # Right: mirror plot
    ax2 = plt.subplot(1, 2, 2)
    sd = np.sort(d)
    ax2.scatter(sd, -sd)
    ax2.axhline(0.0, linestyle="--")
    ax2.axvline(0.0, linestyle="--")
    ax2.set_title("mirror check (sorted d vs -d)")
    ax2.set_xlabel("sorted d")
    ax2.set_ylabel("-sorted d")

    fig.suptitle(f"k={k} | {metric}\nA={model_a} vs B={model_b}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = out_dir / f"diffdiag__k{k}__{safe_metric}__A={safe_a}__B={safe_b}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", type=str, required=True)
    ap.add_argument("--model_a", type=str, required=True, help="Model tag A (in single/<tag>)")
    ap.add_argument("--model_b", type=str, required=True, help="Model tag B (in single/<tag>)")
    ap.add_argument("--ks", type=int, nargs="+", required=True, help="k values (e.g., 20 60)")
    ap.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["rel_l2_u", "max_ae_u"],
        help="Metric keys in metrics.json (default: rel_l2_u max_ae_u)",
    )
    ap.add_argument("--out_json", type=str, default=None, help="Optional path to write JSON results")
    ap.add_argument("--no_holm", action="store_true", help="Disable Holm correction across all tests")

    # New
    ap.add_argument("--diag_plots", action="store_true", help="Write per-test diff diagnostic plots")
    ap.add_argument("--boot", type=int, default=20000, help="Bootstrap resamples (default: 20000)")
    ap.add_argument("--perm", type=int, default=20000, help="Sign-flip permutations (default: 20000)")

    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    if not exp_root.exists():
        raise FileNotFoundError(f"Missing exp_root: {exp_root}")

    results: Dict[str, Any] = {
        "exp_root": str(exp_root),
        "model_a": args.model_a,
        "model_b": args.model_b,
        "lower_is_better": True,
        "tests": [],
        "holm_correction_applied": (not args.no_holm),
        "diagnostics": {
            "bootstrap_n": int(args.boot),
            "perm_n": int(args.perm),
        },
    }

    diag_dir: Optional[Path] = None
    if args.diag_plots:
        # Put diagnostics next to the JSON if possible, else under exp_root/single/
        if args.out_json:
            diag_dir = Path(args.out_json).parent / "wilcoxon_diagnostics"
        else:
            diag_dir = exp_root / "single" / "wilcoxon_diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

    # Collect p-values for optional Holm correction across all (k, metric) tests
    pvals_two_sided: List[float] = []
    idx_map: List[int] = []

    rng = np.random.default_rng(0)

    for k in args.ks:
        for metric in args.metrics:
            a, b, seeds = paired_values(exp_root, args.model_a, args.model_b, k, metric)

            d = a - b
            summ = summarize_diffs(a, b)
            w = run_wilcoxon(a, b)

            # Bootstrap CIs on diffs (median and mean)
            med_lo, med_hi = bootstrap_ci(d, np.median, n_boot=int(args.boot), rng=rng)
            mean_lo, mean_hi = bootstrap_ci(d, np.mean, n_boot=int(args.boot), rng=rng)

            # Sign-flip permutation p-value for median(d)
            p_perm_med = sign_flip_perm_pvalue_median(d, n_perm=int(args.perm), rng=rng)

            diag_plot_path = None
            if diag_dir is not None:
                diag_plot_path = str(
                    save_diff_diagnostics_plot(
                        diag_dir,
                        k=int(k),
                        metric=str(metric),
                        model_a=args.model_a,
                        model_b=args.model_b,
                        d=d,
                    )
                )

            test_row = {
                "k": int(k),
                "metric": metric,
                "seeds_used": seeds,
                "summary": summ,
                "wilcoxon": w,
                "robust": {
                    "bootstrap_ci_median_diff_95": [med_lo, med_hi],
                    "bootstrap_ci_mean_diff_95": [mean_lo, mean_hi],
                    "perm_signflip_p_two_sided_median_diff": float(p_perm_med),
                },
                "diagnostic_plot": diag_plot_path,
                "direction_hint": (
                    "A_better"
                    if summ["median_diff"] < 0
                    else ("B_better" if summ["median_diff"] > 0 else "tie")
                ),
            }

            idx_map.append(len(results["tests"]))
            pvals_two_sided.append(w["p_two_sided"])
            results["tests"].append(test_row)

    if not args.no_holm and len(pvals_two_sided) > 1:
        adj = holm_adjust(pvals_two_sided)
        for i, adj_p in enumerate(adj):
            test_idx = idx_map[i]
            results["tests"][test_idx]["wilcoxon"]["p_two_sided_holm"] = float(adj_p)

    # Pretty console output
    print("\n=== Paired Wilcoxon signed-rank tests (+ diagnostics) ===")
    print(f"exp_root: {exp_root}")
    print(f"A: {args.model_a}")
    print(f"B: {args.model_b}")
    print("Lower is better.\n")

    for t in results["tests"]:
        k = t["k"]
        metric = t["metric"]
        w = t["wilcoxon"]
        s = t["summary"]
        r = t["robust"]
        print(f"[k={k}] metric={metric}  n={s['n']}  direction={t['direction_hint']}")
        print(f"  A mean={s['a_mean']:.6g}, B mean={s['b_mean']:.6g}")
        print(f"  median(A-B)={s['median_diff']:.6g}  frac(A better)={s['frac_a_better']:.2f}")
        print(f"  skew(d)={s['skew_diff']:.4g}  std(d)={s['std_diff']:.4g}")
        print(f"  p(wilcox two-sided)={w['p_two_sided']:.6g}  p(A<B)={w['p_a_less_b']:.6g}  p(B<A)={w['p_b_less_a']:.6g}")
        if "p_two_sided_holm" in w:
            print(f"  p(wilcox two-sided, Holm)={w['p_two_sided_holm']:.6g}")
        print(f"  bootstrap median(d) 95% CI: [{r['bootstrap_ci_median_diff_95'][0]:.4g}, {r['bootstrap_ci_median_diff_95'][1]:.4g}]")
        print(f"  sign-flip perm p (median d, two-sided): {r['perm_signflip_p_two_sided_median_diff']:.6g}")
        if t.get("diagnostic_plot"):
            print(f"  diag plot: {t['diagnostic_plot']}")
        print()

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print("Wrote:", out_path)


if __name__ == "__main__":
    main()
