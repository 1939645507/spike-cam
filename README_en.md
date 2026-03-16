# Spike CAM

Chinese homepage: [README.md](README.md)  
Detailed guides: [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | [docs/PROJECT_GUIDE_zh.md](docs/PROJECT_GUIDE_zh.md)

Spike CAM is a reproducible experiment platform for **online spike identification with binary templates stored in CAM (content-addressable memory)**.

The main research question is:

> Under a fixed encoder and limited CAM memory, how do different template initialization, matching, and dynamic update strategies trade off between accuracy, reject behavior, stability, and memory usage?

## What this repo does

- loads raw spike datasets from `.npz`
- preprocesses continuous voltage with `bandpass / CMR / optional whitening`
- extracts multi-channel spike waveforms
- encodes waveforms into binary codes with `PCA` or `AE`
- builds CAM templates
- runs **chronological / online** evaluation
- compares dynamic update strategies
- saves metrics, predictions, curves, and reports under `results/`

## Repository layout

```text
spike_cam/
├── README.md
├── README_zh.md
├── README_en.md
├── docs/
├── configs/
├── dataset/
├── external/
├── results/
├── scripts/
├── config.py
├── dataio.py
├── encoder.py
├── templates.py
├── match_strategies.py
├── update_strategies.py
├── cam_core.py
├── metrics.py
├── experiment_runner.py
└── run_experiment.py
```

## Quick start

### 1. Environment

Recommended environment:

```bash
conda activate spikecam_py310
pip install -r requirements.txt
```

If you want to use the external `Autoencoders-in-Spike-Sorting` backend, make sure `tensorflow` and `scikit-learn` are available in `spikecam_py310`.

### 2. Run a baseline experiment

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python run_experiment.py --config configs/baseline_top10_ae16.json
```

### 3. Generate toy datasets

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/generate_toy_datasets.py
```

### 4. Run the toy thesis study

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/run_toy_thesis_study.py
```

### 5. Optional: train a reusable external AE artifact

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 \
  python scripts/train_external_ae.py \
  --config configs/cam_compare_main.json \
  --artifact-dir results/artifacts/external_ae_artifacts/fullset_ae16_normal \
  --subset-mode all \
  --epochs 6 \
  --scale minmax
```

## How to change experiments

The project is config-driven. In most cases you only edit `configs/*.json`.

Common fields to change:

- `dataset.npz_path`
- `dataset.subset`
- `cam.memory_subset`
- `encoder.method` / `encoder.backend` / `encoder.code_size`
- `template_init.method`
- `matcher.method`
- `update_strategy.method`
- `cam.match_threshold`
- `evaluation.warmup_ratio`

## Main commands

Inspect label distribution:

```bash
python scripts/inspect_labels.py --npz dataset/my_validation_subset_810000samples_27.00s.npz
```

Prepare encoded dataset only:

```bash
python run_experiment.py --config configs/baseline_top10_ae16.json --encode-only
```

Inspect encoded separability:

```bash
python scripts/inspect_encoded_dataset.py --encoded results/cache/encoded_cache/<file>.npz
```

Run only selected variants:

```bash
python run_experiment.py --config configs/cam_compare_main.json --variant static --variant counter
```

## Output layout

Regular runs are saved under:

```text
results/experiments/<YYYY-MM-DD>/<experiment_name>/
```

## Documentation

- Chinese homepage: [README.md](README.md)
- Chinese quick-start mirror: [README_zh.md](README_zh.md)
- Detailed English guide: [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)
- Detailed Chinese guide: [docs/PROJECT_GUIDE_zh.md](docs/PROJECT_GUIDE_zh.md)
- Public experiment reports: [docs/experiment_reports/README.md](docs/experiment_reports/README.md)
