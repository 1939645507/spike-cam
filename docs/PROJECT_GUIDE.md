# Spike CAM Detailed Guide

Chinese homepage: [../README.md](../README.md)  
English quick-start: [../README_en.md](../README_en.md)  
Chinese detailed guide: [PROJECT_GUIDE_zh.md](PROJECT_GUIDE_zh.md)

## 1. Project Overview

This project is a reproducible experiment platform for **online spike identification with binary templates stored in CAM (content-addressable memory)**.

The intended research question is:

> Under a fixed bit budget and fixed CAM capacity, how do different **template initialization**, **matching**, and **dynamic update** strategies trade off between accuracy, reject behavior, stability, and memory usage?

The important design choice is:

- the **encoder** is a frontend feature extractor
- the **CAM side** is the main research object
- the **default evaluation protocol is chronological / online**, not random split

The pipeline is:

1. Load raw extracellular recording data from an `.npz` file
2. Preprocess the continuous recording with configurable `bandpass / CMR / optional whitening`
3. Extract one multi-channel waveform for each spike, typically using `top-k` channels and flattening them
4. Encode each waveform into binary bits (`8 / 16 / 32` etc.)
5. Use an early warmup segment to build initial CAM templates
6. Feed later spikes into the CAM in chronological order
7. For each spike, run:
   - match
   - accept / reject
   - optional dynamic update
8. Save metrics, predictions, confusion matrices, and online curves

## 2. Directory Structure

```text
spike_cam/
├── README.md
├── requirements.txt
├── __init__.py
├── config.py
├── dataio.py
├── encoder.py
├── templates.py
├── match_strategies.py
├── update_strategies.py
├── cam_core.py
├── metrics.py
├── experiment_runner.py
├── run_experiment.py
├── configs/
│   ├── baseline_top10_ae16.json
│   ├── cam_compare_main.json
│   ├── bits_ablation.json
│   ├── encoder_compare.json
│   └── init_ablation.json
├── scripts/
│   ├── inspect_labels.py
│   ├── inspect_encoded_dataset.py
│   └── plot_results.py
├── dataset/
├── results/
├── external/
│   ├── Autoencoders-in-Spike-Sorting/
│   └── VAE/
└── create_dataset_spike_interface.ipynb
```

### Main files

- `config.py`: typed dataclasses, config loading, deep-merge of variants, path helpers
- `dataio.py`: raw `.npz` loading, continuous-signal preprocessing, multi-channel waveform extraction, chronological split helpers, `EncodedDataset`
- `encoder.py`: PCA baseline, lightweight numpy AE, external AE adapter, bit statistics
- `templates.py`: initial template construction
- `match_strategies.py`: matching algorithms
- `update_strategies.py`: dynamic update algorithms
- `cam_core.py`: CAM storage, row metadata, match/update result objects
- `metrics.py`: accuracy, reject metrics, online curves, result bundle
- `experiment_runner.py`: end-to-end runner, cache handling, result saving
- `run_experiment.py`: command-line entry point
- `scripts/run_previous_work_gap_study.py`: focused study script for reproducing and explaining the gap between `previous work` and the current thesis protocol
- `scripts/generate_toy_datasets.py`: generates multiple controlled toy datasets in the same raw `.npz` contract as the real data
- `scripts/run_toy_thesis_study.py`: full toy-data thesis study that packages datasets, experiments, figures, and a master report

## 3. Data Format

## Raw dataset format

The expected raw input is an `.npz` file that contains at least:

- `raw_data`: continuous extracellular voltage, shape like `(samples, channels)`
- `spike_times`: spike timestamps, shape `(N,)`
- `spike_clusters`: neuron / cluster id for each spike, shape `(N,)`

Optional keys such as `fs`, `duration_sec`, and `n_channels` are preserved in metadata when present.

### Frontend preprocessing and waveform extraction

The current default frontend is intentionally closer to a practical extracellular pipeline:

- `bandpass_enable = true`
- `common_reference_enable = true`
- `whitening_enable = false` by default
- `channel_selection = "topk_max_abs"`
- `topk_channels = 8`
- `flatten_order = "channel_major"`

This is important because a single-channel `max_abs` waveform often discards too much spatial footprint, especially on high-channel-count recordings.

## Encoded dataset format

After waveform extraction and encoding, the platform caches an encoded dataset as:

- `bits`: shape `(N, B)` with `B = code_size`
- `labels`: shape `(N,)`
- `spike_times`: shape `(N,)`
- `source_indices`: original spike index in the raw arrays
- `meta_json`: JSON metadata

This is wrapped by `dataio.EncodedDataset`.

The encoded dataset is intentionally cached so that:

- the encoder does **not** need to be retrained every time
- CAM experiments can be repeated many times with different update strategies
- bit statistics and separability checks can be run independently

## 4. Installation

## Recommended Python version

- Python `3.10+`

## Core dependencies

Install the core runtime:

```bash
pip install -r requirements.txt
```

The project is runnable with only `numpy` and `matplotlib`.

For the thesis workflow in this repository, the recommended environment name is:

```bash
conda activate spikecam_py310
```

## Optional external AE backend

If you want to use the bundled external repo under `external/Autoencoders-in-Spike-Sorting`, you also need its heavier dependencies such as:

- `scikit-learn`
- `tensorflow`

If `tensorflow` is missing in `spikecam_py310`, the external AE backend cannot run and you should either install it into that environment or temporarily use `backend="auto"` / `backend="numpy"` for debugging.

If these are not installed, the default config `backend="auto"` will fall back to the built-in lightweight numpy autoencoder.

This fallback is intentional so that the CAM platform remains usable even in minimal environments.

## Recommended external AE workflow

For thesis-style experiments, it is better to train the external AE once and then reuse it as a fixed frontend.

The project now supports reusable external AE artifacts that store:

- model weights
- saved bit thresholds
- saved scaling parameters
- artifact metadata

You can train one reusable artifact with:

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/train_external_ae.py \
  --config configs/cam_compare_main.json \
  --artifact-dir results/artifacts/external_ae_artifacts/fullset_ae16_normal \
  --subset-mode all \
  --epochs 6 \
  --scale minmax
```

Then point experiment configs to that artifact by setting:

- `encoder.backend = "external"`
- `encoder.artifact_path = "..."`
- `encoder.use_artifact = true`
- `encoder.save_artifact = true`
- `encoder.force_retrain_artifact = false`

This keeps the encoder fixed while you compare CAM-side strategies.

## Recommended toy-data workflow

If you want to study dynamic-update behavior in a controlled setting before going back to difficult real data, the recommended workflow is:

1. generate toy raw datasets
2. run the same Spike CAM pipeline on those toy datasets
3. summarize CAM-side patterns from toy results
4. carry those conclusions back to the real-data discussion

Generate toy datasets with:

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/generate_toy_datasets.py
```

This creates multiple raw `.npz` files under `dataset/toy/`, including:

- easy stable
- dense stable
- drift
- open memory

Run the full toy thesis study with:

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/run_toy_thesis_study.py
```

Outputs are automatically saved under:

- `results/experiments/<YYYY-MM-DD>/experiment_<timestamp>_toy_thesis_study/`

and include:

- `master_report.md / master_report.html`
- `overall_summary.csv`
- per-study subdirectories
- unified `figures/ / tables/ / logs/`

For normal `run_experiment.py` runs, if a config still uses the default `results_dir = "results"`, the runner now writes the actual experiment under:

- `results/experiments/<YYYY-MM-DD>/<experiment_name>/`

This keeps the top-level `results/` directory tidy.

## 5. Core Design Principles

### Config-driven

You should not edit main code to switch experiments.

All important variables are controlled through JSON config files:

- dataset path
- subset rule
- waveform extraction
- encoder method
- code size
- template initialization
- match strategy
- update strategy
- threshold
- capacity
- warmup ratio

### Decoupled modules

The system is split into independent layers:

- data loading / waveform extraction
- encoding
- template initialization
- matching
- updating
- CAM storage
- metrics
- experiment running

### Chronological evaluation first

Dynamic CAM updates are evaluated online:

1. sort spikes by `spike_times`
2. use the first segment as warmup
3. build initial templates from warmup data
4. feed the rest of the spikes one by one
5. match and optionally update at each step

`random_split` exists only as an optional baseline mode.

## 6. Supported Methods

## Subset rules

There are now two separate subset controls:

- `dataset.subset`: which labels enter waveform extraction, encoding, and the test stream
- `cam.memory_subset`: which labels are actually loaded into CAM templates

This is important for the memory-limited setting:

- the stream may contain more classes
- the CAM may remember only `topk` active classes
- out-of-memory classes should then be rejected

## Encoder methods

Configured in `encoder.method`:

- `ae`
- `pca`

For `ae`, the backend is controlled by `encoder.backend`:

- `auto`
- `numpy`
- `external`

For `pca`, the binarization rule is controlled by `encoder.binarize_mode`:

- `median`: current mainline default, binarize each PCA dimension by its median
- `zero`: previous-work-compatible `PCA > 0` binarization

## Template initialization

Configured in `template_init.method`:

- `majority_vote`
- `medoid`
- `stable_mask`
- `multi_template`

## Match strategies

Configured in `matcher.method`:

- `hamming_nearest`
- `weighted_hamming`
- `margin_reject`
- `top2_margin`

## Update strategies

Configured in `update.method`:

- `none`
- `counter`
- `margin_ema`
- `confidence_weighted`
- `dual_template`
- `probabilistic`
- `growing`
- `cooldown`
- `top2_margin`

## 7. How To Run

All commands below assume you are already inside the `spike_cam/` directory.

## Step 1: inspect raw label distribution

```bash
python scripts/inspect_labels.py --npz dataset/my_validation_subset_810000samples_27.00s.npz
```

This is the first thing you should do on a new dataset.

It helps answer:

- how many units are present
- how severe the long-tail problem is
- whether `topk` or `min_count` is a better starting point

## Step 2: prepare / inspect encoded dataset only

To run only the encoding stage and save bit statistics:

```bash
python run_experiment.py --config configs/baseline_top10_ae16.json --encode-only
```

Then inspect the cached encoded dataset:

```bash
python scripts/inspect_encoded_dataset.py --encoded results/cache/encoded_cache/<your_cached_file>.npz
```

This is useful when you want to sanity-check:

- bit balance
- bit entropy
- unique code count
- intra-class vs inter-class Hamming distance

## Step 3: run a sanity-check baseline

```bash
python run_experiment.py --config configs/baseline_top10_ae16.json
```

This runs:

- stream: top 50 most active units
- CAM memory: top 10 most active units
- 16-bit AE frontend
- majority-vote initialization
- static CAM
- chronological evaluation

This is the recommended first end-to-end check.

## Step 4: run the main CAM comparison

```bash
python run_experiment.py --config configs/cam_compare_main.json
```

This runs:

- stream: top 50 active units
- CAM memory: top 20 active units
- fixed 16-bit encoder
- majority-vote initialization
- multiple update strategies
- chronological online evaluation

This is the main experiment for the thesis.

## Step 5: run ablation studies

### Bit budget ablation

```bash
python run_experiment.py --config configs/bits_ablation.json
```

### Encoder comparison

```bash
python run_experiment.py --config configs/encoder_compare.json
```

### Initialization comparison

```bash
python run_experiment.py --config configs/init_ablation.json
```

## Step 5.1: reproduce and explain the previous-work gap

If you want to specifically study why `previous work` looked much stronger than the current thesis pipeline, run:

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 python scripts/run_previous_work_gap_study.py
```

This study automatically packages:

- replay of the saved `previous work` encoded CSV
- recreated spikeinterface toy data under a previous-work-style protocol
- real-data closed-set random-split comparison
- real-data chronological comparison
- real-data memory-limited open-set comparison

Outputs are saved under:

- `results/experiments/<YYYY-MM-DD>/experiment_<timestamp>_previous_work_gap/`

## Step 5.2: run the toy thesis study

If you want a cleaner, more interpretable mechanism study before relying on real data, run:

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 python scripts/run_toy_thesis_study.py
```

This study automatically packages:

- four toy datasets with increasing difficulty
- random-vs-chronological protocol comparison
- stable / drift / open-memory update comparisons
- bit-budget ablation
- template-initialization ablation
- PCA vs external-AE encoder comparison

Outputs are saved under:

- `results/experiments/<YYYY-MM-DD>/experiment_<timestamp>_toy_thesis_study/`

## Step 6: run only selected variants

If a config contains many variants, you can run only a subset:

```bash
python run_experiment.py --config configs/cam_compare_main.json --variant static --variant counter
```

## Step 7: plot saved online curves

```bash
python scripts/plot_results.py --result_dir results/experiments/<YYYY-MM-DD>/cam_compare_main
```

This will create:

- `results/experiments/<YYYY-MM-DD>/<experiment_name>/summary_plots.png`

## 8. Result Layout

Each experiment suite is saved under:

```text
results/experiments/<YYYY-MM-DD>/<experiment_name>/
├── config.json
├── summary.json
└── runs/
    ├── <variant_name>/
    │   ├── metrics.json
    │   ├── meta.json
    │   ├── predictions.npz
    │   ├── curves.npz
    │   ├── confusion.npy
    │   └── confusion_labels.npy
    └── ...
```

### Important saved files

- `config.json`: full suite config used for the run
- `summary.json`: key metrics for all variants
- `predictions.npz`: `y_true`, `y_pred`, distances, update flags, template counts
- `curves.npz`: cumulative and per-window online curves
- `confusion.npy`: confusion matrix including reject label

## 9. Metric Definitions

The platform saves more than final accuracy.

### Basic classification metrics

- `accuracy`: exact match over all online spikes, with reject counted as wrong
- `macro_f1`: class-balanced F1 across true labels
- `balanced_accuracy`: mean recall across true labels

### Reject metrics

- `reject_rate`: fraction of online spikes that were rejected
- `false_reject_rate`: true label was known to the initial CAM, but prediction was reject
- `false_accept_rate`: true label was not in the initial template set, but CAM still accepted some known label

### Online process metrics

- `cumulative_accuracy`
- `per_window_accuracy`
- `per_window_reject_rate`
- `cumulative_updates`
- `cumulative_wrong_updates`
- `template_count`

### Memory metrics

- `initial_template_count`
- `final_template_count`
- `max_template_count`
- `template_growth`
- `used_rows_over_capacity`

### Update quality

- `update_count`
- `wrong_update_count`
- `wrong_update_rate`

`wrong_update_rate` is defined here as:

- an update happened
- the updated row's neuron id does not match the true label of that spike

This is very useful for analyzing whether a dynamic strategy is adapting correctly or corrupting templates.

## 10. Recommended Experiment Plan

This project is organized around the following experimental stages.

### Stage A: baseline sanity check

Goal:

- make sure waveform extraction works
- make sure bits are not degenerate
- make sure static CAM works at all

Recommended config:

- `configs/baseline_top10_ae16.json`

### Stage B: compare dynamic update strategies

Goal:

- keep encoder fixed
- keep bit width fixed
- compare update algorithms fairly

Recommended config:

- `configs/cam_compare_main.json`

### Stage C: bit budget ablation

Goal:

- compare `8 / 16 / 32` bits
- see how update strategies behave under tighter or looser code budgets

Recommended config:

- `configs/bits_ablation.json`

### Stage D: encoder comparison

Goal:

- compare PCA and AE under the same CAM setting
- keep this as an ablation, not the main thesis contribution

Recommended config:

- `configs/encoder_compare.json`

### Stage E: initialization comparison

Goal:

- compare majority-vote, medoid, and stable-mask templates
- study how initialization affects later online adaptation

Recommended config:

- `configs/init_ablation.json`

### Stage F: error analysis

After you identify promising methods, analyze:

- which units are often rejected
- whether errors concentrate in certain classes
- whether updates help only early or throughout the stream
- whether wrong updates accumulate over time
- whether template count growth improves accuracy enough to justify memory usage

## 11. Why The Default Choices Look Like This

### Why `align_mode = min` by default?

Extracellular spikes often have a strong negative trough, so aligning to the minimum is a better default than aligning to the maximum.

### Why not use all units first?

Your dataset is long-tailed. In practice, many units have very few spikes. If you start from all units immediately:

- templates are weak for rare units
- warmup coverage is poor
- reject rate increases
- error analysis becomes noisy

That is why good starting points are usually:

- `top10`
- `top50`
- `min_count >= 500`

### Why chronological evaluation is mandatory?

Dynamic template updates are time-dependent. If you randomly shuffle spikes before evaluation:

- you destroy the temporal meaning of adaptation
- you let future distribution information leak into the past
- you make update algorithms look better than they really are

### Why cache encoded datasets?

Because the encoder is not the main research target here. Once you have a fixed frontend:

- CAM experiments become much faster
- comparisons become fairer
- you can inspect bit quality separately from CAM behavior

## 12. Config Notes

Each config file can define multiple `variants`.

The base config defines common settings.

Each variant only overrides the fields that change.

Example pattern:

```json
{
  "matcher": { "method": "hamming_nearest", "threshold": 4.0 },
  "update": { "method": "none" },
  "variants": [
    { "name": "static" },
    { "name": "counter", "update": { "method": "counter" } }
  ]
}
```

This is the preferred way to run many experiments without touching code.

## 13. Common Questions

### Why are results terrible on all units?

Usually because:

- the label distribution is too imbalanced
- many units have too few warmup samples
- your threshold is too strict or too loose
- the encoder bits are not sufficiently separable

Start with `top10` or `min_count >= 500`, then scale up.

### Why do I see unknown labels in the online stream?

Because a label may exist in the full subset but still not appear in the warmup segment.

This is normal in chronological evaluation and is one reason reject / false-accept metrics matter.

### Why use static CAM for some ablations?

Because some ablations are supposed to isolate one factor:

- encoder quality
- initialization quality

If you also change the update strategy, the interpretation becomes harder.

### How do I switch to the external AE repo?

Set:

```json
"encoder": {
  "method": "ae",
  "backend": "external"
}
```

But make sure optional dependencies are installed first.

### How do I inspect encoded separability before running CAM?

1. run `--encode-only`
2. locate the cached `.npz` under `results/cache/encoded_cache`
3. run `scripts/inspect_encoded_dataset.py`

## 14. Suggested Workflow For Thesis Writing

1. Run `inspect_labels.py`
2. Run `baseline_top10_ae16`
3. Verify bit statistics and online curves
4. Run `cam_compare_main`
5. Pick promising dynamic methods
6. Run `bits_ablation`
7. Run `encoder_compare`
8. Run `init_ablation`
9. Use `predictions.npz`, `curves.npz`, and `summary.json` to build plots and tables

This keeps the project aligned with the intended thesis story:

- main contribution: dynamic CAM update strategies
- supporting ablations: bit budget, encoder, initialization
