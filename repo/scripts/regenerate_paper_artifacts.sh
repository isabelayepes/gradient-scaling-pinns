# scripts/regenerate_paper_artifacts.sh
#!/usr/bin/env bash
set -euo pipefail

# Regenerate CI plots + Wilcoxon (+ diagnostics) + LaTeX table
# Place this script in repo/scripts/, run from repo/ (so ../.venv points to thesis_proj/thesis/.venv)
# e.g.
#   cd thesis_proj/thesis/repo
#   ./scripts/regenerate_paper_artifacts.sh

# Activate venv (adjust path if you run from somewhere else)
source ../.venv/bin/activate

# ----------------------------
# Experiment roots (after merge)
# ----------------------------
EXP_NOIC="outputs/stage1_gradient_scale_noIC"
EXP_LIC50="outputs/stage1_gradient_scale_lIC50"

# Model tags
A_NOIC="adaptive_gate-exp_lamIC0_a1"
B_NOIC="adaptive_gate-linear_lamIC0_a1"
A_LIC50="adaptive_gate-exp_lamIC50_a1"
B_LIC50="adaptive_gate-linear_lamIC50_a1"

# Where to save Wilcoxon JSON (and diagnostics folder will be created next to it)
W_NOIC="${EXP_NOIC}/single/wilcoxon_adaptive_exp_vs_linear_k20_k60.json"
W_LIC50="${EXP_LIC50}/single/wilcoxon_adaptive_exp_vs_linear_k20_k60.json"

# Where to save the LaTeX table
OUT_TEX="tables/wilcoxon_gate_crossover.tex"

# helper: expected seed list
EXPECTED_SEEDS=( $(seq 0 19) )

# ----------------------------
# Verify seeds: return PASS only if for each provided model tag,
# and for k in {20,60}, the seed directories k{k}_seed{0..19} all exist.
# Usage: check_seeds <exp_root> <model1> <model2> ...
# ----------------------------
check_seeds() {
  local exp_root="$1"; shift
  local ks=(20 60)
  local ok=true
  local missing_messages=()

  if [ $# -lt 1 ]; then
    echo "check_seeds: need at least one model tag"
    return 2
  fi

  for m in "$@"; do
    model_dir="${exp_root}/single/${m}"
    if [ ! -d "${model_dir}" ]; then
      missing_messages+=("MISSING model dir: ${model_dir}")
      ok=false
      continue
    fi

    for k in "${ks[@]}"; do
      # collect present seeds for this k
      present=()
      # iterate over matching dirs (glob may not match; handle that)
      shopt -s nullglob
      for d in "${model_dir}"/k"${k}"_seed*/ ; do
        base="$(basename "$d")"  # e.g., k20_seed7
        if [[ "$base" =~ seed([0-9]+) ]]; then
          present+=("${BASH_REMATCH[1]}")
        fi
      done
      shopt -u nullglob

      # sort and unique present entries
      if [ "${#present[@]}" -gt 0 ]; then
        IFS=$'\n' sorted_present=($(printf "%s\n" "${present[@]}" | sort -n -u))
        unset IFS
      else
        sorted_present=()
      fi

      # check each expected seed
      for s in "${EXPECTED_SEEDS[@]}"; do
        found=false
        for p in "${sorted_present[@]}"; do
          if [ "$p" = "$s" ]; then
            found=true
            break
          fi
        done
        if [ "$found" = false ]; then
          missing_messages+=("Missing: ${model_dir} k=${k} seed=${s}")
          ok=false
        fi
      done
    done
  done

  if [ "$ok" = true ]; then
    echo "Seed check: PASS (all specified models have seeds 0..19 for k=20 and k=60)"
    return 0
  else
    echo "Seed check: FAIL"
    for msg in "${missing_messages[@]}"; do
      echo "  $msg"
    done
    return 2
  fi
}

# Run the check only for the adaptive/linear pairs (both lambdas)
echo "==> Running seed check for noIC..."
check_seeds "${EXP_NOIC}" "${A_NOIC}" "${B_NOIC}"

echo "==> Running seed check for lIC50..."
check_seeds "${EXP_LIC50}" "${A_LIC50}" "${B_LIC50}"

# If both checks passed we continue. (The function exits non-zero on failure which stops the script.)
echo "==> Seed checks passed. Continuing to generate plots and tests..."

# ----------------------------
# Plots: 12 figures total
# ----------------------------
echo "=== Plotting (noIC) ==="
python3 scripts/plot_score_vs_k.py --exp_root "${EXP_NOIC}" --all_models --metric rel_l2_u
python3 scripts/plot_score_vs_k.py --exp_root "${EXP_NOIC}" --all_models --metric rel_l2_u \
  --include_substr adaptive_ fixed_fourier_ --ylim 0 0.4 --out_suffix zoom_spectralOnly
python3 scripts/plot_score_vs_k.py --exp_root "${EXP_NOIC}" --all_models --metric rel_l2_u \
  --exclude_substr baseline_ --include_substr adaptive_ fixed_fourier_ --ylim 0 0.4 --out_suffix zoom_noBaselineLinear

python3 scripts/plot_score_vs_k.py --exp_root "${EXP_NOIC}" --all_models --metric max_ae_u
python3 scripts/plot_score_vs_k.py --exp_root "${EXP_NOIC}" --all_models --metric max_ae_u \
  --include_substr adaptive_ fixed_fourier_ --ylim 0 0.5 --out_suffix zoom_spectralOnly
python3 scripts/plot_score_vs_k.py --exp_root "${EXP_NOIC}" --all_models --metric max_ae_u \
  --exclude_substr baseline_ --include_substr adaptive_ fixed_fourier_ --ylim 0 0.5 --out_suffix zoom_noBaselineLinear


echo "=== Plotting (lIC50) ==="
python3 scripts/plot_score_vs_k.py --exp_root "${EXP_LIC50}" --all_models --metric rel_l2_u
python3 scripts/plot_score_vs_k.py --exp_root "${EXP_LIC50}" --all_models --metric rel_l2_u \
  --include_substr adaptive_ fixed_fourier_ --ylim 0 0.4 --out_suffix zoom_spectralOnly
python3 scripts/plot_score_vs_k.py --exp_root "${EXP_LIC50}" --all_models --metric rel_l2_u \
  --exclude_substr baseline_ --include_substr adaptive_ fixed_fourier_ --ylim 0 0.4 --out_suffix zoom_noBaselineLinear

python3 scripts/plot_score_vs_k.py --exp_root "${EXP_LIC50}" --all_models --metric max_ae_u
python3 scripts/plot_score_vs_k.py --exp_root "${EXP_LIC50}" --all_models --metric max_ae_u \
  --include_substr adaptive_ fixed_fourier_ --ylim 0 0.5 --out_suffix zoom_spectralOnly
python3 scripts/plot_score_vs_k.py --exp_root "${EXP_LIC50}" --all_models --metric max_ae_u \
  --exclude_substr baseline_ --include_substr adaptive_ fixed_fourier_ --ylim 0 0.5 --out_suffix zoom_noBaselineLinear


# ----------------------------
# Wilcoxon + diagnostics (k=20,60; metrics=2)
# ----------------------------
echo "=== Wilcoxon (+ diagnostics) noIC ==="
python3 scripts/wilcoxon_gate_test.py \
  --exp_root "${EXP_NOIC}" \
  --model_a "${A_NOIC}" \
  --model_b "${B_NOIC}" \
  --ks 20 60 \
  --metrics rel_l2_u max_ae_u \
  --diag_plots \
  --boot 20000 \
  --perm 20000 \
  --out_json "${W_NOIC}"

echo "=== Wilcoxon (+ diagnostics) lIC50 ==="
python3 scripts/wilcoxon_gate_test.py \
  --exp_root "${EXP_LIC50}" \
  --model_a "${A_LIC50}" \
  --model_b "${B_LIC50}" \
  --ks 20 60 \
  --metrics rel_l2_u max_ae_u \
  --diag_plots \
  --boot 20000 \
  --perm 20000 \
  --out_json "${W_LIC50}"


# ----------------------------
# Make LaTeX table
# ----------------------------
echo "=== Make Wilcoxon LaTeX table ==="
python3 scripts/make_wilcoxon_table.py \
  --inputs \
    "${W_NOIC}" \
    "${W_LIC50}" \
  --labels noIC lIC50 \
  --out "${OUT_TEX}"

echo ""
echo "DONE."
echo "Plots written under:"
echo "  ${EXP_NOIC}/single/"
echo "  ${EXP_LIC50}/single/"
echo "Wilcoxon JSON (+ diag plots) written to:"
echo "  ${W_NOIC} (and ./wilcoxon_diagnostics/ next to it)"
echo "  ${W_LIC50} (and ./wilcoxon_diagnostics/ next to it)"
echo "LaTeX table:"
echo "  ${OUT_TEX}"
