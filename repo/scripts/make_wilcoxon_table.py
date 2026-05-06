# scripts/make_wilcoxon_table.py
#!/usr/bin/env python3
"""
Create a LaTeX table summarizing Wilcoxon signed-rank tests.

Input: one or more JSON files produced by your wilcoxon script.
Output: a LaTeX table (.tex) you can \\input{}.

Example:
  python3 scripts/make_wilcoxon_table.py \
    --inputs outputs/tiny_paper/stage1_gradient_scale_noIC/single/wilcoxon_adaptive_exp_vs_linear_k20_k60.json \
            outputs/tiny_paper/stage1_gradient_scale_lIC50/single/wilcoxon_adaptive_exp_vs_linear_k20_k60.json \
    --labels noIC lIC50 \
    --out tables/wilcoxon_gate_crossover.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x: float, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def fmt_p(x: float) -> str:
    """Compact p formatting for tables (no math-mode wrappers besides the <1e-4 case)."""
    try:
        x = float(x)
    except Exception:
        return str(x)

    if x < 1e-4:
        return r"$<10^{-4}$"
    if x < 0.001:
        return f"{x:.1e}"
    return f"{x:.4f}"


def safe_tex(s: str) -> str:
    # minimal escaping
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def direction_arrow(direction_hint: str) -> str:
    # A_better means A < B (lower is better); B_better means B < A
    if direction_hint == "A_better":
        return r"$A \downarrow$"
    if direction_hint == "B_better":
        return r"$B \downarrow$"
    return r"--"


def build_block(payload: Dict[str, Any], label: str) -> List[str]:
    """
    Builds a block:
      - one multicolumn line describing A/B
      - followed by rows
    """
    rows: List[str] = []

    model_a = safe_tex(str(payload.get("model_a", "A")))
    model_b = safe_tex(str(payload.get("model_b", "B")))
    tests = payload.get("tests", [])

    # stable ordering: k then metric
    tests = sorted(tests, key=lambda t: (t.get("k", 0), str(t.get("metric", ""))))

    header_note = (
        r"\multicolumn{11}{l}{\small "
        + f"{safe_tex(label)}: $A$={model_a}, $B$={model_b}"
        + r"} \\"
    )
    rows.append(header_note)

    for t in tests:
        k = t.get("k", "")
        metric = safe_tex(str(t.get("metric", "")))
        summ = t.get("summary", {})
        wilc = t.get("wilcoxon", {})

        a_mean = summ.get("a_mean", float("nan"))
        b_mean = summ.get("b_mean", float("nan"))
        frac_a_better = float(summ.get("frac_a_better", float("nan")))
        frac_b_better = float(summ.get("frac_b_better", float("nan")))

        # Raw and Holm-adjusted two-sided p-values
        p_raw = wilc.get("p_two_sided", float("nan"))
        p_holm = wilc.get("p_two_sided_holm", p_raw)

        # Pick the one-sided p consistent with the direction hint we reported
        dh = str(t.get("direction_hint", ""))
        if dh == "A_better":
            p_one_sided = wilc.get("p_a_less_b", None)
        elif dh == "B_better":
            p_one_sided = wilc.get("p_b_less_a", None)
        else:
            p_one_sided = None

        win = direction_arrow(dh)
        n = summ.get("n", "")

        # fraction of seeds where the reported winner is better
        consistency = max(frac_a_better, frac_b_better)

        row = (
            f"{safe_tex(label)} & {k} & {metric} & {win} & {n} & "
            f"{fmt(a_mean)} & {fmt(b_mean)} & {fmt(consistency, nd=2)} & "
            f"{fmt_p(p_raw)} & {fmt_p(p_holm)} & "
        )
        row += (fmt_p(p_one_sided) if p_one_sided is not None else "--")
        row += r" \\"
        rows.append(row)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="Wilcoxon JSON file(s).")
    ap.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each input (e.g., noIC lIC50).",
    )
    ap.add_argument("--out", type=str, required=True, help="Output .tex path.")
    ap.add_argument("--caption", type=str, default=None, help="Optional caption override.")
    ap.add_argument("--label", type=str, default="tab:wilcoxon_gate", help="LaTeX label for the table.")
    args = ap.parse_args()

    in_paths = [Path(p) for p in args.inputs]
    labels = args.labels
    if labels is None:
        # default label from the exp root folder name
        labels = [p.parent.name for p in in_paths]
    if len(labels) != len(in_paths):
        raise ValueError("If provided, --labels must match number of --inputs.")

    all_rows: List[str] = []
    for p, lab in zip(in_paths, labels):
        payload = load_json(p)
        all_rows.extend(build_block(payload, lab))

    caption = args.caption or (
        "Paired Wilcoxon signed-rank tests across seeds comparing IC gates for the adaptive model. "
        "Lower is better for both metrics. Winner indicates whether $A$ or $B$ attains lower error. "
        "We report both raw two-sided p-values and Holm-adjusted two-sided p-values; Holm correction is "
        "applied across the tested $(k,\\text{metric})$ pairs within each setting."
    )

    # ICLR-ish: caption/title should appear before the table.
    tex = r"""
% Auto-generated by scripts/make_wilcoxon_table.py
\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{""" + args.label + r"""}

\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l c l c c c c c c c c}
\toprule
Setting & $k$ & Metric & Winner & $n$ & mean($A$) & mean($B$) & frac(win) & $p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & $p_{\mathrm{1s}}$ \\
\midrule
""" + "\n".join(all_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""".lstrip()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex, encoding="utf-8")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
