#!/usr/bin/env python3
"""
SAM + CLAHE/Histogram Normalization Baseline
=============================================
Addresses reviewer concern H3: "If you apply CLAHE, how much does SAM improve?"

Tests SAM AMG performance with three preprocessing strategies:
  1. Raw (no preprocessing) - already measured
  2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  3. CLAHE + Gaussian denoise

Uses same 30 NIST test images as unified_verification.py
"""

import os, sys, json, time
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SAM_CHECKPOINT = DATA_DIR / "checkpoints" / "sam_vit_h_4b8939.pth"
NIST_RAW_DIR = DATA_DIR / "raw" / "nist_sem" / "rawFOV"
NIST_GT_DIR = DATA_DIR / "raw" / "nist_sem" / "damageContextAssistedMask" / "damageMask"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TEST = 30
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_nist_gt(gt_path):
    gt = np.array(Image.open(gt_path))
    if len(gt.shape) == 3:
        return np.any(gt > 0, axis=2).astype(bool)
    return (gt > 0).astype(bool)


def get_matched_pairs(raw_dir, gt_dir, n_max=None):
    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.png')])
    matched = []
    for rf in raw_files:
        base = rf.replace('.png', '')
        gt_path = Path(gt_dir) / f"{base}_damage.png"
        if gt_path.exists():
            matched.append({'raw': Path(raw_dir) / rf, 'gt': gt_path, 'name': rf})
        if n_max and len(matched) >= n_max:
            break
    return matched


def compute_metrics(pred, gt):
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, ~gt_b).sum()
    fn = np.logical_and(~pred_b, gt_b).sum()
    union = tp + fp + fn
    return {
        'iou': float(tp) / float(union) if union > 0 else 0.0,
        'dice': float(2*tp) / float(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0.0,
        'precision': float(tp) / float(tp+fp) if (tp+fp) > 0 else 0.0,
        'recall': float(tp) / float(tp+fn) if (tp+fn) > 0 else 0.0,
    }


def apply_clahe(image_rgb, clip_limit=2.0, tile_size=8):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_clahe_denoise(image_rgb, clip_limit=2.0, tile_size=8):
    enhanced = apply_clahe(image_rgb, clip_limit, tile_size)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return denoised


def run_amg(image_rgb, mask_generator):
    masks = mask_generator.generate(image_rgb)
    if len(masks) == 0:
        return np.zeros(image_rgb.shape[:2], dtype=bool)
    h, w = image_rgb.shape[:2]
    combined = np.zeros((h, w), dtype=bool)
    for m in masks:
        combined = np.logical_or(combined, m['segmentation'])
    return combined


def main():
    print("=" * 70)
    print("SAM + CLAHE PREPROCESSING BASELINE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now()}")
    print(f"Device: {DEVICE}")

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(DEVICE).eval()

    amg = SamAutomaticMaskGenerator(
        model=sam, points_per_side=32,
        pred_iou_thresh=0.88, stability_score_thresh=0.95,
        crop_n_layers=1, crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    matched = get_matched_pairs(NIST_RAW_DIR, NIST_GT_DIR, N_TEST)
    print(f"Test images: {len(matched)}")

    preprocess_methods = {
        'Raw (no preprocessing)': lambda img: img,
        'CLAHE (clip=2.0)': lambda img: apply_clahe(img, 2.0),
        'CLAHE (clip=4.0)': lambda img: apply_clahe(img, 4.0),
        'CLAHE + Denoise': lambda img: apply_clahe_denoise(img, 2.0),
    }

    all_results = {}
    from tqdm import tqdm

    for method_name, preprocess_fn in preprocess_methods.items():
        print(f"\n--- {method_name} ---")
        results = []

        for item in tqdm(matched, desc=f"  {method_name[:20]}"):
            try:
                image = np.array(Image.open(item['raw']).convert('RGB'))
                gt_mask = load_nist_gt(str(item['gt']))
                if image.shape[:2] != gt_mask.shape:
                    gt_mask = cv2.resize(gt_mask.astype(np.uint8),
                                         (image.shape[1], image.shape[0])) > 0

                processed = preprocess_fn(image)
                pred_mask = run_amg(processed, amg)
                metrics = compute_metrics(pred_mask, gt_mask)
                metrics['name'] = item['name']
                results.append(metrics)
            except Exception as e:
                print(f"  [!] {item['name']}: {e}")

        ious = [r['iou'] for r in results]
        summary = {
            'method': method_name,
            'n': len(results),
            'iou_mean': float(np.mean(ious)),
            'iou_std': float(np.std(ious)),
            'dice_mean': float(np.mean([r['dice'] for r in results])),
            'precision_mean': float(np.mean([r['precision'] for r in results])),
            'recall_mean': float(np.mean([r['recall'] for r in results])),
        }
        all_results[method_name] = {'summary': summary, 'per_image': results}

        print(f"  IoU: {summary['iou_mean']:.4f} +/- {summary['iou_std']:.4f}")
        print(f"  Dice: {summary['dice_mean']:.4f}  Prec: {summary['precision_mean']:.4f}  "
              f"Rec: {summary['recall_mean']:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("CLAHE BASELINE SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<30} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 65)
    for name, data in all_results.items():
        s = data['summary']
        print(f"{name:<30} {s['iou_mean']:.4f}   {s['dice_mean']:.4f}   "
              f"{s['precision_mean']:.4f}   {s['recall_mean']:.4f}")

    # Save
    out_dir = OUTPUTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "clahe_baseline_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    rows = [d['summary'] for d in all_results.values()]
    pd.DataFrame(rows).to_csv(out_dir / "clahe_baseline_results.csv", index=False)

    print(f"\n[Saved] clahe_baseline_results.json / .csv")


if __name__ == "__main__":
    main()
