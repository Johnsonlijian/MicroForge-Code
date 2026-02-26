#!/usr/bin/env python3
"""Save U-Net results from terminal output (script crashed before saving)."""
import json, numpy as np, pandas as pd
from pathlib import Path

out_dir = Path(__file__).parent.parent.parent / "outputs" / "tables"

unet_loo = [
    {'iou': 1.0000, 'dice': 1.0000, 'name': 'Sample__11_22_.ome.png'},
    {'iou': 1.0000, 'dice': 1.0000, 'name': 'Sample__12_37_.ome.png'},
    {'iou': 0.9984, 'dice': 0.9992, 'name': 'Sample__13_22_.ome.png'},
    {'iou': 1.0000, 'dice': 1.0000, 'name': 'Sample__3_33_.ome.png'},
    {'iou': 0.9998, 'dice': 0.9999, 'name': 'Sample__4_23_.ome.png'},
    {'iou': 0.9998, 'dice': 0.9999, 'name': 'Sample__4_31_.ome.png'},
    {'iou': 1.0000, 'dice': 1.0000, 'name': 'Sample__4_32_.ome.png'},
    {'iou': 0.9994, 'dice': 0.9997, 'name': 'Sample__4_33_.ome.png'},
    {'iou': 0.9987, 'dice': 0.9994, 'name': 'Sample__4_36_.ome.png'},
    {'iou': 0.9965, 'dice': 0.9982, 'name': 'Sample__6_38_.ome.png'},
    {'iou': 1.0000, 'dice': 1.0000, 'name': 'Sample__8_21_.ome.png'},
    {'iou': 1.0000, 'dice': 1.0000, 'name': 'Sample__9_35_.ome.png'},
]
loo_ious = [r['iou'] for r in unet_loo]

unet_data = {
    'loo_summary': {
        'model': 'U-Net',
        'cv_type': 'LOO',
        'n_folds': 12,
        'iou_mean': float(np.mean(loo_ious)),
        'iou_std': float(np.std(loo_ious)),
        'iou_ci95': float(1.96 * np.std(loo_ious) / np.sqrt(len(loo_ious))),
        'dice_mean': float(np.mean([r['dice'] for r in unet_loo])),
        'precision_mean': None,
        'recall_mean': None,
    },
    'loo_per_fold': unet_loo,
    'test_summary': {
        'model': 'U-Net',
        'test_set': 'auto_GT_30',
        'iou_mean': 0.1529,
        'iou_std': 0.1303,
        'dice_mean': None,
        'precision_mean': None,
        'recall_mean': None,
    },
    'test_per_image': [],
}

all_results = {'U-Net': unet_data}
with open(out_dir / "supervised_baselines_results.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"U-Net LOO IoU: {np.mean(loo_ious):.4f} +/- {np.std(loo_ious):.4f}")
print(f"U-Net Test IoU: 0.1529 +/- 0.1303")
print("[Saved] supervised_baselines_results.json")
