# Dataset Folder

This directory is intentionally kept light in GitHub.

What goes here:

- raw real datasets in `.npz` format
- generated toy datasets under `dataset/toy/`

Typical expected keys in a raw `.npz` file:

- `raw_data`
- `spike_times`
- `spike_clusters`

Notes:

- large raw datasets are ignored by `.gitignore`
- toy datasets can be regenerated with:

```bash
MPLCONFIGDIR=.mplconfig conda run -n spikecam_py310 python scripts/generate_toy_datasets.py
```
