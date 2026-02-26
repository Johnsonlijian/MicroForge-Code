#!/usr/bin/env python3
"""
P0-3: Fix Supervised Baseline Reporting + Manual-Auto GT Consistency
====================================================================
- Report LOO mean ± std ± 95% CI (not "best fold")
- Compute manual↔auto GT IoU ceiling (label-style gap)
"""

import json, os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import stats
import cv2

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

NIST_RAW_DIR = DATA_DIR / "raw" / "nist_sem" / "rawFOV"
NIST_AUTO_GT_DIR = DATA_DIR / "raw" / "nist_sem" / "damageContextAssistedMask" / "damageMask"
NIST_MANUAL_GT_DIR = DATA_DIR / "raw" / "nist_sem" / "contextManualMaskGT" / "contextMaskGT"


def load_mask(path):
    img = np.array(Image.open(path))
    if len(img.shape) == 3:
        return np.any(img > 0, axis=2).astype(bool)
    return (img > 0).astype(bool)


def compute_iou_dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    iou = float(inter) / float(union) if union > 0 else 0.0
    total = a.sum() + b.sum()
    dice = float(2 * inter) / float(total) if total > 0 else 0.0
    return iou, dice


def ci95(arr):
    n = len(arr)
    if n < 2:
        return 0.0
    sem = np.std(arr, ddof=1) / np.sqrt(n)
    return stats.t.ppf(0.975, df=n - 1) * sem


def main():
    print("=" * 70)
    print("  FIX SUPERVISED STATS + MANUAL-AUTO GT CONSISTENCY")
    print("=" * 70)

    results_path = OUTPUTS_DIR / "tables" / "supervised_baselines_results.json"
    with open(results_path) as f:
        data = json.load(f)

    out = {"supervised_loo_stats": {}, "manual_auto_consistency": {}}

    # ── Part 1: Corrected LOO + test statistics ──
    print("\n[Part 1] Supervised Baseline LOO Cross-Validation Statistics")
    print("-" * 60)

    for model in ["U-Net", "DeepLabv3"]:
        md = data[model]
        loo_ious = [f["iou"] for f in md["loo_per_fold"]]
        n = len(loo_ious)
        m, s = np.mean(loo_ious), np.std(loo_ious, ddof=1)
        c = ci95(loo_ious)

        tm = md["test_summary"]["iou_mean"]
        ts = md["test_summary"]["iou_std"]
        nt = 30
        tc = stats.t.ppf(0.975, df=nt - 1) * ts / np.sqrt(nt)

        print(f"\n  {model}:")
        print(f"    LOO (n={n}): IoU = {m:.6f} ± {s:.6f},  95%CI [{m-c:.6f}, {m+c:.6f}]")
        print(f"    Test (n={nt}): IoU = {tm:.4f} ± {ts:.4f},  95%CI [{tm-tc:.4f}, {tm+tc:.4f}]")

        out["supervised_loo_stats"][model] = {
            "loo_n": n, "loo_iou_mean": round(m, 6), "loo_iou_std": round(s, 6),
            "loo_iou_ci95": round(c, 6),
            "test_n": nt, "test_iou_mean": round(tm, 4), "test_iou_std": round(ts, 4),
            "test_iou_ci95": round(tc, 4),
        }

    # ── Part 2: Manual ↔ Auto GT consistency (label-style gap ceiling) ──
    print(f"\n\n[Part 2] Manual vs Auto GT Consistency (Label-Style Gap Ceiling)")
    print("-" * 60)

    manual_files = sorted(
        f for f in os.listdir(NIST_MANUAL_GT_DIR) if f.endswith(".png")
    )

    records = []
    for mf in manual_files:
        base = mf.replace("annot_", "").replace(".png", "")
        auto_path = NIST_AUTO_GT_DIR / f"{base}.ome_damage.png"
        manual_path = NIST_MANUAL_GT_DIR / mf

        if not auto_path.exists():
            print(f"  [!] No auto GT for {mf}")
            continue

        manual_mask = load_mask(manual_path)
        auto_mask = load_mask(auto_path)

        if manual_mask.shape != auto_mask.shape:
            auto_mask = cv2.resize(
                auto_mask.astype(np.uint8),
                (manual_mask.shape[1], manual_mask.shape[0]),
            ) > 0

        iou, dice = compute_iou_dice(manual_mask, auto_mask)
        records.append({"sample": base, "iou": iou, "dice": dice,
                        "manual_px": int(manual_mask.sum()),
                        "auto_px": int(auto_mask.sum())})
        print(f"  {base}: IoU={iou:.4f}  Dice={dice:.4f}  "
              f"Manual={manual_mask.sum()}  Auto={auto_mask.sum()}")

    if records:
        ious = [r["iou"] for r in records]
        dices = [r["dice"] for r in records]
        n = len(ious)
        mi, si = np.mean(ious), np.std(ious, ddof=1)
        ci = ci95(ious)
        md, sd = np.mean(dices), np.std(dices, ddof=1)

        print(f"\n  SUMMARY (n={n}):")
        print(f"    Manual<->Auto IoU:  {mi:.4f} +/- {si:.4f},  95%CI [{mi-ci:.4f}, {mi+ci:.4f}]")
        print(f"    Manual<->Auto Dice: {md:.4f} +/- {sd:.4f}")
        print(f"\n  => This is the CEILING of agreement between annotation styles.")
        print(f"    Models trained on manual GT and tested on auto GT cannot exceed this.")

        out["manual_auto_consistency"] = {
            "n": n, "iou_mean": round(mi, 4), "iou_std": round(si, 4),
            "iou_ci95": round(ci, 4), "dice_mean": round(md, 4),
            "dice_std": round(sd, 4), "per_image": records,
        }

    out_path = OUTPUTS_DIR / "tables" / "supervised_stats_corrected.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
