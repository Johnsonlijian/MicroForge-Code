#!/usr/bin/env python3
"""
P1-5: Prompt Budget Efficiency Curve
======================================
Compare Uniform Grid vs Microscopy-Aware Prompting across varying N.
Shows that microscopy-aware prompting reaches target IoU with fewer points.
"""

import os, json
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
from scipy.ndimage import generic_filter
from skimage.filters import sobel
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SAM_CHECKPOINT = DATA_DIR / "checkpoints" / "sam_vit_h_4b8939.pth"
NIST_RAW_DIR = DATA_DIR / "raw" / "nist_sem" / "rawFOV"
NIST_GT_DIR = DATA_DIR / "raw" / "nist_sem" / "damageContextAssistedMask" / "damageMask"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TEST = 30
SEED = 42
POINT_BUDGETS = [4, 9, 16, 25, 36, 64]

np.random.seed(SEED)
torch.manual_seed(SEED)


def load_gt(path):
    gt = np.array(Image.open(path))
    if len(gt.shape) == 3:
        return np.any(gt > 0, axis=2).astype(bool)
    return (gt > 0).astype(bool)


def get_pairs(n_max=None):
    raw_files = sorted(f for f in os.listdir(NIST_RAW_DIR) if f.endswith(".png"))
    pairs = []
    for rf in raw_files:
        base = rf.replace(".png", "")
        gp = Path(NIST_GT_DIR) / f"{base}_damage.png"
        if gp.exists():
            pairs.append({"raw": Path(NIST_RAW_DIR) / rf, "gt": gp, "name": rf})
        if n_max and len(pairs) >= n_max:
            break
    return pairs


def compute_iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = inter + np.logical_and(p, ~g).sum() + np.logical_and(~p, g).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def compute_importance_map(image_gray):
    edge = sobel(image_gray.astype(float) / 255.0)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
    lap = cv2.Laplacian(image_gray, cv2.CV_64F)
    tex = cv2.GaussianBlur(np.abs(lap), (15, 15), 0)
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-8)
    h, w = image_gray.shape
    return 0.4 * edge + 0.4 * tex + 0.2 * np.random.rand(h, w)


def sample_uniform_grid(h, w, n_points):
    gs = max(2, int(np.sqrt(n_points)))
    pts = []
    for y in np.linspace(0, h - 1, gs).astype(int):
        for x in np.linspace(0, w - 1, gs).astype(int):
            pts.append([x, y])
    return np.array(pts[:n_points])


def sample_adaptive(importance, n_points, min_dist=15):
    h, w = importance.shape
    probs = importance.flatten()
    probs = probs / (probs.sum() + 1e-8)
    pts, selected = [], np.zeros((h, w), dtype=bool)
    attempts = 0
    while len(pts) < n_points and attempts < n_points * 10:
        idx = np.random.choice(len(probs), p=probs)
        y, x = idx // w, idx % w
        y0, y1 = max(0, y - min_dist), min(h, y + min_dist)
        x0, x1 = max(0, x - min_dist), min(w, x + min_dist)
        if not selected[y0:y1, x0:x1].any():
            pts.append([x, y])
            selected[y0:y1, x0:x1] = True
        attempts += 1
    if not pts:
        return sample_uniform_grid(h, w, n_points)
    return np.array(pts[:n_points])


def predict_with_points(predictor, points, h, w):
    all_masks = []
    for pt in points:
        try:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([pt]), point_labels=np.array([1]),
                multimask_output=True,
            )
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])
            ratio = best_mask.sum() / (h * w)
            if 0.0005 < ratio < 0.20 and best_score > 0.7:
                all_masks.append(best_mask)
        except Exception:
            continue
    if not all_masks:
        return np.zeros((h, w), dtype=bool)
    return np.any(np.stack(all_masks), axis=0)


def main():
    print("=" * 70)
    print("  PROMPT BUDGET EFFICIENCY CURVE")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Point budgets: {POINT_BUDGETS}")

    from segment_anything import sam_model_registry, SamPredictor
    from tqdm import tqdm

    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(DEVICE).eval()
    predictor = SamPredictor(sam)

    pairs = get_pairs(N_TEST)
    print(f"  Test images: {len(pairs)}")

    results = []

    for n_pts in POINT_BUDGETS:
        print(f"\n  === N = {n_pts} ===")
        uniform_ious, adaptive_ious = [], []

        for item in tqdm(pairs, desc=f"    N={n_pts}"):
            img = np.array(Image.open(item["raw"]).convert("RGB"))
            gt = load_gt(str(item["gt"]))
            if img.shape[:2] != gt.shape:
                gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0])) > 0
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            predictor.set_image(img)

            # Uniform
            np.random.seed(SEED)
            u_pts = sample_uniform_grid(h, w, n_pts)
            u_mask = predict_with_points(predictor, u_pts, h, w)
            uniform_ious.append(compute_iou(u_mask, gt))

            # Adaptive
            np.random.seed(SEED)
            imp = compute_importance_map(gray)
            a_pts = sample_adaptive(imp, n_pts)
            a_mask = predict_with_points(predictor, a_pts, h, w)
            adaptive_ious.append(compute_iou(a_mask, gt))

        u_mean, u_std = np.mean(uniform_ious), np.std(uniform_ious)
        a_mean, a_std = np.mean(adaptive_ious), np.std(adaptive_ious)
        improvement = (a_mean / u_mean - 1) * 100 if u_mean > 0 else 0

        print(f"    Uniform:    IoU = {u_mean:.4f} ± {u_std:.4f}")
        print(f"    Adaptive:   IoU = {a_mean:.4f} ± {a_std:.4f}")
        print(f"    Improvement: {improvement:+.1f}%")

        results.append({
            "n_points": n_pts,
            "uniform_iou_mean": round(u_mean, 4), "uniform_iou_std": round(u_std, 4),
            "adaptive_iou_mean": round(a_mean, 4), "adaptive_iou_std": round(a_std, 4),
            "improvement_pct": round(improvement, 1),
        })

    # AUC
    ns = [r["n_points"] for r in results]
    u_auc = np.trapz([r["uniform_iou_mean"] for r in results], ns)
    a_auc = np.trapz([r["adaptive_iou_mean"] for r in results], ns)
    auc_advantage = (a_auc / u_auc - 1) * 100 if u_auc > 0 else 0

    print(f"\n  AUC (Uniform):  {u_auc:.2f}")
    print(f"  AUC (Adaptive): {a_auc:.2f}")
    print(f"  AUC advantage:  {auc_advantage:+.1f}%")

    # ── Save ──
    out_dir = OUTPUTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "timestamp": datetime.now().isoformat(),
        "point_budgets": POINT_BUDGETS,
        "results": results,
        "auc_uniform": round(u_auc, 2),
        "auc_adaptive": round(a_auc, 2),
        "auc_advantage_pct": round(auc_advantage, 1),
    }
    with open(out_dir / "prompt_efficiency.json", "w") as f:
        json.dump(out, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "prompt_efficiency.csv", index=False)

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(8, 5))
    u_means = [r["uniform_iou_mean"] for r in results]
    u_stds  = [r["uniform_iou_std"] for r in results]
    a_means = [r["adaptive_iou_mean"] for r in results]
    a_stds  = [r["adaptive_iou_std"] for r in results]

    ax.errorbar(ns, u_means, yerr=u_stds, marker="s", capsize=4,
                color="steelblue", label=f"Uniform Grid (AUC={u_auc:.1f})")
    ax.errorbar(ns, a_means, yerr=a_stds, marker="o", capsize=4,
                color="coral", label=f"Microscopy-Aware (AUC={a_auc:.1f})")
    ax.fill_between(ns, [m - s for m, s in zip(u_means, u_stds)],
                    [m + s for m, s in zip(u_means, u_stds)], alpha=0.15, color="steelblue")
    ax.fill_between(ns, [m - s for m, s in zip(a_means, a_stds)],
                    [m + s for m, s in zip(a_means, a_stds)], alpha=0.15, color="coral")
    ax.set_xlabel("Number of Prompt Points", fontsize=12)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title("Prompt Budget Efficiency: Uniform vs Microscopy-Aware", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns)

    plt.tight_layout()
    fig_dir = OUTPUTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "Fig_Prompt_Efficiency.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(fig_dir / "Fig_Prompt_Efficiency.pdf", bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\n[Saved] prompt_efficiency.json / .csv")
    print(f"[Saved] Fig_Prompt_Efficiency.png/.pdf")
    print("Done!")


if __name__ == "__main__":
    main()
