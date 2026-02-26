# MicroForge: Physics-Informed Foundation Model Adaptation for Automated Concrete Microstructure Analysis

**Supplementary code and data repository for the manuscript submitted to *Automation in Construction*.**

---

## Overview

MicroForge is a fully automated, reproducible pipeline for concrete SEM microstructure analysis using the Segment Anything Model (SAM). This repository contains all source code, verification data, and generated figures required to reproduce every quantitative claim in the paper.

### Key Findings

| Metric | Value |
|--------|-------|
| Baseline SAM IoU (NIST concrete SEM) | 0.129 ± 0.123 |
| SAM calibration error (ECE, 21,469 masks) | 0.697 (236× overconfidence) |
| Annotation-style gap (manual ↔ auto GT) | IoU = 0.102 ± 0.065 |
| Microscopy-Aware Prompting improvement | +9.9% at N=36 |
| PIMP edge targeting improvement | +12.5% (sparse conditions) |
| LoRA mask quality improvement | 51× per-mask IoU |
| LoRA–AMG mask generation collapse | 156 → 1 mask/image (all 80 configs) |
| ASRC micro-crack expansion (14d → 28d) | +298% |
| ASHG expert inter-rater agreement (κ) | 0.71 (linearly weighted) |
| Deterministic reproducibility | max IoU diff = 0.000 |

---

## Repository Structure

```
ANONYMOUS_REPO/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── src/analysis/                      # Core analysis scripts
│   ├── 23_unified_verification.py     # Baseline SAM benchmark (Table 1)
│   ├── 24_lora_verification.py        # LoRA adaptation (Table 5)
│   ├── 26_clahe_baseline.py           # CLAHE preprocessing ablation
│   ├── 27_supervised_baselines.py     # U-Net / DeepLabv3 LOO CV
│   ├── 29_calibration_and_pareto.py   # Calibration + Pareto analysis (Table 6, Fig. 2)
│   ├── 30_prompt_efficiency.py        # Prompt budget analysis (Fig. 3)
│   ├── 31_csi_shuffle_ablation.py     # CSI shuffle ablation (10,000 perms)
│   ├── 32_supervised_stats.py         # Supervised baseline statistics
│   ├── 33_generate_final_figures_tables.py  # Generate all publication figures/tables
│   └── 34_json_to_csv.py             # Convert all JSON to CSV format
│
├── data/
│   ├── source_truth/                  # Source-of-truth data files
│   │   ├── *.json                     # Raw verification results (8 JSON files)
│   │   └── *.csv                      # Tabular versions for Excel/Origin (20 CSV files)
│   └── verification_tables/           # Publication tables (Tables 1–7, S1, S2)
│       ├── Table_1_Method_Comparison_v3.csv
│       ├── Table_2_ManualAutoGT_v3.csv
│       ├── ...
│       ├── Table_S1_PIMP_Full_Magnification.csv
│       └── Table_S2_ASHG_Expert_Evaluation.csv
│
├── figures/                           # Generated publication figures (300 DPI)
│   ├── Fig_1_Framework.png
│   ├── Fig_2_Calibration_Pareto.png
│   ├── Fig_3_Prompt_Efficiency.png
│   ├── Fig_4_Domain_Gap.png
│   └── Fig_5_ASRC_Morphology.png
│
└── supplementary/
    ├── SINGLE_SOURCE_OF_TRUTH.md      # Authoritative numerical reference
    ├── Table_S1_PIMP_Full_Magnification.csv
    └── Table_S2_ASHG_Expert_Evaluation.csv
```

---

## Reproduction Guide

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA 11.x+
- PyTorch 2.x
- SAM checkpoint (`sam_vit_h_4b8939.pth`)

### Setup

```bash
pip install -r requirements.txt
```

### External Data

- **NIST Concrete Damage SEM Dataset**: Download from [NIST ISG Platform](https://isg.nist.gov/deepzoomweb/concreteScoring/index.html)
  - Place images in `data/raw/nist_sem/`
  - Manual GT for 12 images + automated GT for 1,360 images
- **SAM Checkpoint**: Download from [Meta AI](https://github.com/facebookresearch/segment-anything)
  - Place in `data/checkpoints/sam_vit_h_4b8939.pth`

### Reproducing All Results

Execute scripts in order (each saves results to `data/source_truth/`):

```bash
# Step 1: Baseline SAM benchmark on NIST (Table 1, Fig. 4)
python src/analysis/23_unified_verification.py

# Step 2: CLAHE preprocessing ablation (Table 1)
python src/analysis/26_clahe_baseline.py

# Step 3: Supervised baselines U-Net/DeepLabv3 (Table 1-2)
python src/analysis/27_supervised_baselines.py
python src/analysis/32_supervised_stats.py

# Step 4: LoRA adaptation (Table 5)
python src/analysis/24_lora_verification.py

# Step 5: Calibration analysis + Pareto sweep (Table 6, Fig. 2)
python src/analysis/29_calibration_and_pareto.py

# Step 6: Prompt efficiency budget curve (Fig. 3)
python src/analysis/30_prompt_efficiency.py

# Step 7: CSI shuffle ablation - 10,000 permutations (Section 5.3)
python src/analysis/31_csi_shuffle_ablation.py

# Step 8: Generate all final figures and tables
python src/analysis/33_generate_final_figures_tables.py

# Step 9: Convert JSON results to CSV for Excel/Origin
python src/analysis/34_json_to_csv.py
```

### Deterministic Settings

All experiments use the following deterministic configuration:

```python
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

Verified: max IoU variation between repeated runs = **0.000** (bit-exact within the reported environment: NVIDIA RTX series, CUDA 11.x, PyTorch 2.x).

---

## Data File Reference

### Source-of-Truth JSON Files

| File | Paper Section | Content |
|------|--------------|---------|
| `calibration_corrected.json` | Table 6, Fig. 2, Section 5.1(F3) | ECE, reliability bins, pred/true IoU, Pareto summary |
| `unified_verification.json` | Table 1, Section 5.1-5.2 | Per-image SAM results (30 images × 3 methods) |
| `supervised_stats_corrected.json` | Table 1-2, Section 5.1(F2) | LOO CV + manual-auto GT statistics |
| `prompt_efficiency.json` | Fig. 3, Section 5.2 | Per-budget IoU (Laplacian texture variant) |
| `csi_shuffle_ablation.json` | Section 5.3 | 10,000-permutation test + lookup table |
| `lora_verification_real.json` | Table 5, Section 5.4 | LoRA results + training loss curve |
| `clahe_baseline_results.json` | Table 1, Section 5.1(F1) | 4 preprocessing variants × 30 images |
| `supervised_baselines_results.json` | Table 1, Section 5.1(F2) | U-Net/DeepLabv3 per-fold LOO results |

### CSV Versions (for Excel/Origin)

All JSON files have been converted to CSV format for convenience. See `data/source_truth/*.csv`. Key files:

| CSV File | Content |
|----------|---------|
| `unified_verification_per_image.csv` | 30 images × 3 methods (IoU, Dice, Precision, Recall) |
| `calibration_reliability_bins.csv` | 10-bin reliability diagram data |
| `clahe_comparison_per_image.csv` | 30 images × 4 CLAHE variants |
| `lora_training_loss.csv` | 30-epoch training loss curve |
| `csi_shuffle_lookup_table.csv` | 10 images × 6 ages lookup matrix |
| `manual_auto_gt_per_image.csv` | 12 images manual↔auto GT IoU |
| `prompt_efficiency_budget.csv` | 6 budget levels (N=4 to N=64) |

### Note on Fig. 3 vs Table 1

Fig. 3 (`prompt_efficiency.json`) uses Laplacian texture for the budget curve. Table 1's main result (0.074, +9.9%) uses local entropy. Both are correct; the methods differ in texture computation. For Table 1 absolute values, use `unified_verification.json`.

---

## Verification

All numerical claims in the paper are traceable to the JSON files in `data/source_truth/`. The complete mapping is documented in `supplementary/SINGLE_SOURCE_OF_TRUTH.md`.

To verify data integrity, compare the SHA-256 hashes listed in `SHA256SUMS.txt` against your local files.

---

## License

MIT License. See [LICENSE](LICENSE).
