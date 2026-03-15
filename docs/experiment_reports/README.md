# Experiment Reports

This folder contains a **lightweight public snapshot** of selected experiment outputs.

Why this folder exists:

- the full `results/experiments/` tree is large and is ignored by Git
- most runs contain many binary intermediates that are not necessary for GitHub readers
- this folder keeps only the most readable artifacts:
  - `master_report.md / .html`
  - `overall_summary.csv`
  - top-level `figures/`
  - `README_run.md`

## Included reports

### 1. Toy thesis study

Path:

- [toy_thesis_study_20260315](./toy_thesis_study_20260315)

What it shows:

- controlled toy datasets with increasing difficulty
- stable / drift / open-memory scenarios
- dynamic update comparison
- bit-budget ablation
- template-initialization ablation
- encoder comparison

Suggested first file:

- [toy_thesis_study_20260315/master_report.md](./toy_thesis_study_20260315/master_report.md)

### 2. Previous-work gap study

Path:

- [previous_work_gap_20260315](./previous_work_gap_20260315)

What it shows:

- why `previous work` looked much easier than the current thesis setting
- comparison between old encoded CSV, recreated toy data, and real data
- how protocol difficulty changes the observed accuracy

Suggested first file:

- [previous_work_gap_20260315/master_report.md](./previous_work_gap_20260315/master_report.md)
