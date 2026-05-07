# Nonlinear Joint Optimization for SASV

This repository contains the implementation and experimental pipelines for the IEEE paper "[Joint Optimization of Speaker and Spoof Detectors for Spoofing-Robust Automatic Speaker Verification](https://ieeexplore.ieee.org/document/11499447)" in SASV (speaker verification + spoofing countermeasure) settings.

The project includes three main experiment scripts (`fig3_a.py`, `fig3_b.py`, `fig3_c.py`) and shared metric/loss utilities (`adcf_utils.py`) to reduce code duplication.

## Repository Structure

- `fig3_a.py`: Main training/evaluation pipeline (variant A)
- `fig3_b.py`: Main training/evaluation pipeline (variant B)
- `fig3_c.py`: Main training/evaluation pipeline (variant C)
- `adcf_utils.py`: Shared aDCF loss and hard/soft aDCF metric helpers
- `dataset.py`: Dataset classes and data loading logic
- `metrics.py`: EER and related metric utilities
- `calculate_metrics.py`, `a_dcf.py`, `calculate_modules.py`: Additional metric utilities

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python fig3_a.py --output_dir ./results_a
```

Alternative experiment variants:

```bash
python fig3_b.py --output_dir ./results_b
python fig3_c.py --output_dir ./results_c
```

## How to Run

Examples:

```bash
python fig3_a.py --output_dir ./results_a
python fig3_b.py --output_dir ./results_b
python fig3_c.py --output_dir ./results_c
```

Each script supports additional arguments such as:

- `--batch_size`
- `--lr`
- `--num_epochs`
- `--embedding_dir`
- `--spk_meta_dir`
- `--sasv_dev_trial`
- `--sasv_eval_trial`

## Outputs

Depending on the script, outputs are written under `--output_dir` and may include:

- `training_log.txt`
- `best_model.pth`
- `dev_scores_kde.png`

## Notes

- GPU is used automatically if available; otherwise CPU is used.
- Ensure dataset protocol and embedding paths are configured correctly before running.
- Large model artifacts (`.pth`) should not be committed to Git unless intentionally versioned.
- Model embeddings should be extracted using the corresponding models, whose GitHub links are provided in the paper.

## Citation

If you use this repository, please cite:

O. Kurnaz, J. Mishra, T. H. Kinnunen and C. Hanilçi, "Joint Optimization of Speaker and Spoof Detectors for Spoofing-Robust Automatic Speaker Verification," in IEEE Transactions on Audio, Speech and Language Processing, doi: 10.1109/TASLPRO.2026.3688932
