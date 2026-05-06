# Gradient Scaling Effects in Adaptive Spectral PINNs for Stiff Nonlinear ODEs — Paper Reproduction

This repository contains code for spring–pendulum PINN experiments,
along with plotting and statistical analysis scripts used in the ICLR 2026 AI & PDE Workshop and arXiv
paper on gradient scaling effects. Precomputed experiment outputs are available in the GitHub release `v1.0-arxiv`.
Download `outputs.zip`, unzip it inside `repo/`

---

## Repository structure

- `experiments/stage1_spring_pendulum.py`  
  Main experiment runner (`single`, `bundle` modes)

- `configs/`  
  YAML configs controlling stiffness values, seeds, and training budgets

- `scripts/plot_score_vs_k.py`  
  Plot metrics vs stiffness \(k\)

- `scripts/wilcoxon_gate_test.py`  
  Paired Wilcoxon signed-rank tests across shared seeds

- `scripts/make_wilcoxon_table.py`  
  Convert Wilcoxon JSON outputs into LaTeX tables

---

## Paper reproduction

### 0) Create, activate, and install environment files

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
cd repo
```

---

### 1) Run the main gradient-scaling sweep

Run **all models × IC gates** in one command using bundle mode.

```bash
python3 -m experiments.stage1_spring_pendulum \
  --config configs/stage1_gradient_scale.yaml \
  bundle \
  --models baseline fixed_fourier adaptive \
  --gates linear exp \
  --lambda_ic 50.0 \
  --alpha 1.0
```

**What this does**
- Runs all `(k, seed)` combinations from the YAML
- Saves results under:
  ```
  outputs/<exp_name>/single/<model>_gate-<gate>_lamIC<...>_a<...>/
  ```
- Automatically produces:
  - per-run `metrics.json`
  - `model_param_summary.json`
  - per-model plots
  - combined all-model plots for `rel_l2_u` and `max_ae_u`

To stop a running sweep:
```bash
pkill -f stage1_spring_pendulum
```

---

### 2) Regenerate plots (optional)

Plots are auto-generated, but can be regenerated post-hoc.

```bash
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_lIC50 \
  --all_models \
  --metric rel_l2_u

python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_lIC50 \
  --all_models \
  --metric max_ae_u
```

For the no-IC setting:
```bash
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_noIC \
  --all_models \
  --metric rel_l2_u
```

---

### 3) Paired Wilcoxon signed-rank tests (adaptive exp vs linear)

All tests are **paired across identical random seeds**.

#### With IC velocity penalty (λ_IC = 50)

```bash
python3 scripts/wilcoxon_gate_test.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_lIC50 \
  --model_a adaptive_gate-exp_lamIC50_a1 \
  --model_b adaptive_gate-linear_lamIC50_a1 \
  --ks 20 60 \
  --metrics rel_l2_u max_ae_u \
  --out_json outputs/tiny_paper/stage1_gradient_scale_lIC50/single/wilcoxon_adaptive_exp_vs_linear_k20_k60.json
```

#### Without IC velocity penalty (λ_IC = 0)

```bash
python3 scripts/wilcoxon_gate_test.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_noIC \
  --model_a adaptive_gate-exp_lamIC0_a1 \
  --model_b adaptive_gate-linear_lamIC0_a1 \
  --ks 20 60 \
  --metrics rel_l2_u max_ae_u \
  --out_json outputs/tiny_paper/stage1_gradient_scale_noIC/single/wilcoxon_adaptive_exp_vs_linear_k20_k60.json
```

---

### 4) Generate LaTeX table from Wilcoxon results

```bash
python3 scripts/make_wilcoxon_table.py \
  --inputs \
    outputs/tiny_paper/stage1_gradient_scale_noIC/single/wilcoxon_adaptive_exp_vs_linear_k20_k60.json \
    outputs/tiny_paper/stage1_gradient_scale_lIC50/single/wilcoxon_adaptive_exp_vs_linear_k20_k60.json \
  --labels noIC lIC50 \
  --out tables/wilcoxon_gate_crossover.tex
```

Include in LaTeX:
```tex
\input{tables/wilcoxon_gate_crossover.tex}
```

---

### 5) Model sizes (parameter counts)

Each run directory contains:
```
model_param_summary.json
```

Example:
```bash
cat outputs/tiny_paper/stage1_gradient_scale_lIC50/single/baseline_gate-exp_lamIC50_a1/k10_seed7/model_param_summary.json
```

---

## Notes

- All Wilcoxon tests are **paired across seeds**
- Holm correction is applied **within each experimental setting**
- Lower metric values indicate better performance
- Bundle mode guarantees fair, apples-to-apples comparisons

---

## Modifying experiments

Edit YAML files in `configs/`:

- `single.k_values` — stiffness values
- `single.seeds` — random seeds
- `training.steps`, `training.lr` — training budget
- `loss.lambda_ic`, `loss.lambda_phys` — loss weights
- `ic_embedding.gate`, `ic_embedding.alpha` — defaults (overridden by bundle flags)

## Command to generate all 12 plots relevant to the paper:
```

########################################
# lamIC0 (lambda_ic = 0)  — 6 figures
########################################

# rel_l2_u — full
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_noIC \
  --all_models \
  --metric rel_l2_u

# rel_l2_u — zoom_spectralOnly (with y-lim)
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_noIC \
  --all_models \
  --metric rel_l2_u \
  --include_substr spectral fixed adaptive \
  --ylim 0 0.4 \
  --out_suffix zoom_spectralOnly

# rel_l2_u — zoom_noBaselineLinear (with y-lim)
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_noIC \
  --all_models \
  --metric rel_l2_u \
  --exclude_substr baseline_gate-linear \
  --ylim 0 0.4 \
  --out_suffix zoom_noBaselineLinear

# max_ae_u — full
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_noIC \
  --all_models \
  --metric max_ae_u

# max_ae_u — zoom_spectralOnly (with y-lim)
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_noIC \
  --all_models \
  --metric max_ae_u \
  --include_substr spectral fixed adaptive \
  --ylim 0 0.5 \
  --out_suffix zoom_spectralOnly

# max_ae_u — zoom_noBaselineLinear (with y-lim)
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_noIC \
  --all_models \
  --metric max_ae_u \
  --exclude_substr baseline_gate-linear \
  --ylim 0 0.5 \
  --out_suffix zoom_noBaselineLinear


########################################
# lamIC50 (lambda_ic = 50) — 6 figures
########################################

# rel_l2_u — full
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_lIC50 \
  --all_models \
  --metric rel_l2_u

# rel_l2_u — zoom_spectralOnly (with y-lim)
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_lIC50 \
  --all_models \
  --metric rel_l2_u \
  --include_substr spectral fixed adaptive \
  --ylim 0 0.4 \
  --out_suffix zoom_spectralOnly

# rel_l2_u — zoom_noBaselineLinear (with y-lim)
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_lIC50 \
  --all_models \
  --metric rel_l2_u \
  --exclude_substr baseline_gate-linear \
  --ylim 0 0.4 \
  --out_suffix zoom_noBaselineLinear

# max_ae_u — full
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_lIC50 \
  --all_models \
  --metric max_ae_u

# max_ae_u — zoom_spectralOnly (with y-lim)
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_lIC50 \
  --all_models \
  --metric max_ae_u \
  --include_substr spectral fixed adaptive \
  --ylim 0 0.5 \
  --out_suffix zoom_spectralOnly

# max_ae_u — zoom_noBaselineLinear (with y-lim)
python3 scripts/plot_score_vs_k.py \
  --exp_root outputs/tiny_paper/stage1_gradient_scale_lIC50 \
  --all_models \
  --metric max_ae_u \
  --exclude_substr baseline_gate-linear \
  --ylim 0 0.5 \
  --out_suffix zoom_noBaselineLinear

```
