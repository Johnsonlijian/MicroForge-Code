# MicroForge: Single Source of Truth (v3.2-final)
## All Numerical Claims — Verified & Corrected

**Last Updated**: 2026-02-24 (v3.2-final — AiC submission version. Full terminology/number consistency verified.)

---

### Domain Gap Benchmark
| Metric | Value | Source |
|--------|-------|--------|
| Baseline SAM AMG IoU | 0.129 ± 0.123 | unified_verification.json |
| CLAHE (clip=2.0) IoU | 0.115 ± 0.139 | clahe_baseline_results.json |
| CLAHE (clip=4.0) IoU | 0.105 ± 0.116 | clahe_baseline_results.json |
| CLAHE+Denoise IoU | 0.119 ± 0.118 | clahe_baseline_results.json |
| Uniform Points (N=36) IoU | 0.067 ± 0.076 | unified_verification.json |
| Microscopy-Aware (N=36) IoU | 0.074 ± 0.094 | unified_verification.json |
| Prompting improvement | +9.9% (0.074/0.067) | computed |

### Supervised Baselines (LOO CV, N=12)
| Model | LOO IoU (mean±std) | LOO 95% CI | Test IoU (N=30) | Test 95% CI |
|-------|-------------------|-----------|-----------------|-------------|
| U-Net | 0.999 ± 0.001 | [0.999, 1.000] | 0.153 ± 0.130 | [0.104, 0.202] |
| DeepLabv3 | 1.000 ± 0.000 | [1.000, 1.000] | 0.153 ± 0.130 | [0.104, 0.201] |

### Annotation-Style Gap
| Metric | Value | Source |
|--------|-------|--------|
| Manual↔Auto GT IoU | 0.102 ± 0.065 | supervised_stats_corrected.json |
| Manual↔Auto GT 95% CI | [0.061, 0.143] | supervised_stats_corrected.json |
| Manual↔Auto GT Dice | 0.180 ± 0.101 | supervised_stats_corrected.json |

### Calibration (CORRECTED — v3)
| Metric | Baseline | LoRA | Source |
|--------|----------|------|--------|
| N masks analyzed | 21,469 | 30 | calibration_corrected.json |
| pred_iou (raw mean) | 0.700 | 1.008 | calibration_corrected.json |
| pred_iou (clamped) | 0.700 | 1.000 | calibration_corrected.json |
| true_iou (mean) | 0.003 | 0.153 | calibration_corrected.json |
| ECE | **0.697** | **0.847** | calibration_corrected.json |
| Overconfidence ratio | 236× | 6.5× | calibration_corrected.json |
| Pearson r | 0.082 | NaN (constant pred) | calibration_corrected.json |
| Spearman rho | -0.019 | NaN (constant pred) | calibration_corrected.json |

### Deterministic Reproducibility
| Metric | Value | Source |
|--------|-------|--------|
| Max IoU diff (2 runs) | 0.000 | calibration_corrected.json |
| Is deterministic | YES | calibration_corrected.json |

### LoRA Adaptation
| Metric | AMG | Point | Source |
|--------|-----|-------|--------|
| Baseline IoU | 0.126 ± 0.126 | 0.087 ± 0.069 | lora_verification_real.json |
| LoRA IoU | 0.124 ± 0.146 | 0.090 ± 0.071 | lora_verification_real.json |
| Mask count (Baseline) | 4,685 | — | calibration_corrected.json |
| Mask count (LoRA) | 30 | — | calibration_corrected.json |

### Pareto Analysis
| Config | Avg Masks/Image | IoU | Source |
|--------|----------------|-----|--------|
| Baseline best (0.30/0.30) | 715.6 | 0.152 | pareto_baseline.csv |
| LoRA best (0.30/0.30) | **1.0** | 0.153 | pareto_lora.csv |
| LoRA at ALL 80 configs | **1.0** | — | pareto_lora.csv |

### PIMP / CSI
| Metric | Value | Source |
|--------|-------|--------|
| CSI at 14d | 0.0961 | XRD computation |
| CSI at 28d | 0.0043 | XRD computation |
| CSI ratio (14d/28d) | ~22× | computed |
| Edge improvement (28d 2000×) | +12.5% | PIMP_verification_real.csv |
| CSI shuffle p-value | p_emp ≥ 0.999, one-sided (10k perms) | csi_shuffle_ablation.json |
| CSI shuffle interpretation | Not significant; edge features dominate | csi_shuffle_ablation.json |

### Prompt Efficiency
| N points | Uniform IoU | Adaptive IoU | Gap | Source |
|----------|-------------|-------------|-----|--------|
| 4 | 0.069 | 0.016 | -76% | prompt_efficiency.json |
| 9 | 0.079 | 0.033 | -58% | prompt_efficiency.json |
| 16 | 0.079 | 0.049 | -38% | prompt_efficiency.json |
| 25 | 0.079 | 0.053 | -33% | prompt_efficiency.json |
| 36 | 0.067 | 0.057 | -15% | prompt_efficiency.json |
| 64 | 0.071 | 0.068 | -5% | prompt_efficiency.json |
| AUC | 4.38 | 3.23 | -26% | prompt_efficiency.json |

Note: Prompt efficiency uses Laplacian texture (fast); original +9.9% uses local entropy (slow but superior).

### ASRC Morphology
| Metric | 14d | 28d | Change | Source |
|--------|-----|-----|--------|--------|
| Total porosity (%) | 5.10 | 10.97 | +115% | Table_1_Key_Statistics_AiC.csv |
| True pore (%) | 4.32 | 7.85 | +82% | Table_1_Key_Statistics_AiC.csv |
| Micro-crack (%) | 0.78 | 3.11 | +298% | Table_1_Key_Statistics_AiC.csv |
| AFt decrease (3d→14d) | — | — | 98.6% | Table_Physics_Age_Analysis.csv |

### ASHG Expert Evaluation
| Metric | Value | Source |
|--------|-------|--------|
| Testability score | 5.0/5 (both raters) | Expert evaluation |
| Relevance score | 4.2 ± 0.4 | Expert evaluation |
| Novelty score | 3.8 ± 0.8 | Expert evaluation |
| Inter-rater agreement | Linearly weighted Cohen's κ = 0.71 (substantial) | Expert evaluation |

### Rounding Convention
| Context | ECE Baseline | ECE LoRA | Rule |
|---------|-------------|---------|------|
| Running text (Abstract, Intro, Discussion, Conclusions) | 0.70 | 0.85 | 2 decimal places |
| Tables (Table 6, Appendix A) | 0.697 | 0.847 | 3 decimal places |

### Terminology Convention
| Concept | Standard Term | Variants Avoided |
|---------|-------------|-----------------|
| Gap between manual/auto GT | annotation-style gap | annotation-protocol gap |
| The 0.102 IoU metric | annotation-consistency reference bound | ceiling, reference scale |
| General phenomenon | annotation-style mismatch | annotation-protocol mismatch |
| κ type | linearly weighted Cohen's κ | Cohen's κ (unqualified) |

### Threshold Convention
| Context | Masks/Image | Thresholds |
|---------|------------|------------|
| Standard AMG (Table 1, running text) | 156 | pred_iou=0.88, stability=0.95 |
| Permissive (calibration analysis, Table 6) | ~716 | pred_iou=0.30, stability=0.30 |
| LoRA (any threshold) | 1 | All 80 configs tested |
