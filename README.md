# GCAT: Asynchronous SAR–Optical Crop Classification

This repository contains the core GCAT-RP-T implementation for the manuscript **“Robust Crop Classification Model Based on Asynchronous SAR and Optical Temporal Feature Fusion.”** It is a code-only release; datasets, prepared parcel caches, model checkpoints, and experimental outputs are not distributed here.

## Project Overview

GCAT classifies known crop parcels using unequal-length Sentinel-2, Sentinel-1 ascending, and Sentinel-1 descending acquisition sequences. The code exposes the model, random-prefix training, validation-only temperature scaling, and first-accept-and-lock operational evaluation.

## Main Features

- Real day-of-year encoding, packed GRU sequences, padding-aware bidirectional cross-attention, and complementary gated fusion.
- PASTIS-R patch-to-parcel mean/std extraction, official-fold normalization using Train folds only, and patch-disjoint Fold-2 manifests.
- Brandenburg 2018 Train/Validation/Calibration and frozen 2019 Test readers using 2018 Train-only normalization.
- Separate decision protocols: primary Fold-2 quality-aware SRC over the full 50⁴ policy family, secondary 50-threshold Global SRC, and an empirical Brandenburg 2018 development rule. These are **not interchangeable**. Formal binomial bounds depend on the adopted parcel-level calibration assumptions; the Brandenburg 2019 results are descriptive external-test results.

## Installation

Use Python 3.10 or newer. In a fresh virtual environment:

```bash
pip install -r requirements.txt
```

No pretrained weights are provided. Only load checkpoints that you trust.

## Dataset Preparation

Obtain PASTIS-R and the Brandenburg Fusion Competition data from their respective data providers, subject to their licenses. Set the dataset paths in `configs/pastis_fold2.yaml` and `configs/brandenburg_2018_2019.yaml`. No data files should be committed to this repository.

PASTIS-R: the included builder reads the original patch-format arrays and writes raw parcel mean/std features. The quality sidecar uses the dataset's original cloud-mask arrays and does **not** remove cloudy time steps. The split generator reproduces the paper's fixed patch-disjoint Fold-2 development search (seed 42, 20,000 trials). Run from the repository root:

```bash
python scripts/build_parcel_cache.py --config configs/pastis_fold2.yaml
python scripts/build_pastis_parcel_quality_cache.py --root datasets/PASTIS-R --base-cache cache/parcel_mean_std --output cache/parcel_quality
python scripts/make_pastis_splits.py --manifest cache/parcel_mean_std/manifest.csv --output splits --fold 2
```

The Brandenburg reader expects the **published project's prepared parcel NPZ schema** (`cache/parcel/train2018/parcel_<fid>.npz` and `cache/parcel/test2019/parcel_<fid>.npz`) plus the original spatial split manifests. The raw-imagery-to-parcel geospatial pipeline is **not included** in this compact release. Consequently, the Brandenburg branch is not end-to-end reproducible from raw imagery using this repository alone. Once these prepared parcels and immutable 2018 split manifests are available, create 2018 Train-only statistics:

```bash
python scripts/fit_brandenburg_train_stats.py --data-root datasets/Brandenburg --train-split datasets/Brandenburg/splits/train2018.csv --output datasets/Brandenburg/cache/normalization
```

Never estimate normalization statistics from Validation, Calibration, or 2019 Test. All data paths in the example configs are local placeholders and may be changed before execution.

## Training

Training uses random prefixes from the six operational checkpoints plus FullSeason. The best checkpoint maximizes mean Validation Macro-F1 over all seven views; Calibration and Test are not used for model selection.

```bash
python scripts/train_gcat.py --config configs/pastis_fold2.yaml
python scripts/train_gcat.py --config configs/brandenburg_2018_2019.yaml
```

Each run writes its own checkpoint beneath `runs/`. The Brandenburg command reads **2018 Train and Validation only**.

## Evaluation

Fit one temperature per checkpoint on Validation, and choose a decision policy using only the designated 2018 or Fold-2 development Calibration split:

```bash
python scripts/calibrate_gcat.py --config configs/pastis_fold2.yaml --checkpoint runs/pastis_fold2/<run>/best.pt --protocol primary-fold2 --output runs/pastis_fold2/<run>/policy_primary.json
python scripts/evaluate_gcat.py --config configs/pastis_fold2.yaml --checkpoint runs/pastis_fold2/<run>/best.pt --policy runs/pastis_fold2/<run>/policy_primary.json --output outputs/pastis_fold2_eval
```

Use `--protocol global-src` only when reproducing the distinct secondary global-threshold analysis. For Brandenburg, use `--protocol brandenburg-2018` with the 2018 configuration, then pass the resulting frozen policy to `evaluate_gcat.py`; evaluation reads 2019 Test without fitting a new temperature or threshold. The output contains monthly predictions, first accepted checkpoint/prediction, MCAC, accepted coverage, and empirical selective risk. `evaluation.operational.lead_time_gain` computes paired SAR lead-time gains if a separately frozen, parcel-aligned S2 comparator outcome is supplied; S2 comparator weights are not bundled.

## Citation

Citation details and the article DOI will be added after publication. Do not cite an unassigned DOI or a nonexistent repository URL.

**Release note:** The project authors must approve a code license before public distribution. Third-party dataset licenses and redistribution conditions are separate from this code package.
