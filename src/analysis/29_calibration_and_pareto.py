#!/usr/bin/env python3
"""
P0-1/P0-2/P1-4: Calibration Fix + AMG Pareto Sweep + Deterministic Verification
=================================================================================
Fixes:
  - Clamp pred_iou to [0,1] before ECE
  - ECE last bin inclusive  [0.9, 1.0]
  - Pearson & Spearman correlation
  - Reliability diagram + scatter plot
  - AMG Pareto sweep (lower thresholds → more masks → recover from tension)
  - Deterministic reproducibility check (CUDA deterministic mode)
"""

import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SAM_CHECKPOINT = DATA_DIR / "checkpoints" / "sam_vit_h_4b8939.pth"
NIST_RAW_DIR = DATA_DIR / "raw" / "nist_sem" / "rawFOV"
NIST_GT_DIR = DATA_DIR / "raw" / "nist_sem" / "damageContextAssistedMask" / "damageMask"
NIST_MANUAL_GT_DIR = DATA_DIR / "raw" / "nist_sem" / "contextManualMaskGT" / "contextMaskGT"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TEST = 30
SEED = 42
LORA_RANK = 16
LORA_ALPHA = 32
LORA_EPOCHS = 30
LORA_LR = 2e-4

# Pareto sweep grid
PARETO_IOU_THRESHS   = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.88, 0.90, 0.95]
PARETO_STAB_THRESHS  = [0.30, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
# Low thresholds for mask extraction (keep almost all masks)
EXTRACT_IOU_THRESH  = 0.30
EXTRACT_STAB_THRESH = 0.30


def set_deterministic(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_nist_gt(gt_path):
    gt = np.array(Image.open(gt_path))
    if len(gt.shape) == 3:
        return np.any(gt > 0, axis=2).astype(bool)
    return (gt > 0).astype(bool)


def get_matched_pairs(raw_dir, gt_dir, n_max=None):
    raw_files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".png"))
    matched = []
    for rf in raw_files:
        base = rf.replace(".png", "")
        gt_path = Path(gt_dir) / f"{base}_damage.png"
        if gt_path.exists():
            matched.append({"raw": Path(raw_dir) / rf, "gt": gt_path, "name": rf})
        if n_max and len(matched) >= n_max:
            break
    return matched


def get_manual_pairs():
    pairs = []
    for mf in sorted(os.listdir(NIST_MANUAL_GT_DIR)):
        if not mf.endswith(".png"):
            continue
        base = mf.replace("annot_", "").replace(".png", "")
        raw_path = NIST_RAW_DIR / f"{base}.ome.png"
        if raw_path.exists():
            pairs.append({"raw": raw_path, "gt": NIST_MANUAL_GT_DIR / mf, "name": raw_path.name})
    return pairs


def compute_iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = inter + np.logical_and(p, ~g).sum() + np.logical_and(~p, g).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# ── ECE (fixed) ──
def compute_ece(pred_ious, true_ious, n_bins=10):
    pred_c = np.clip(pred_ious, 0.0, 1.0)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(pred_c)
    bin_data = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (pred_c >= bins[i]) & (pred_c < bins[i + 1])
        else:
            mask = (pred_c >= bins[i]) & (pred_c <= bins[i + 1])
        cnt = mask.sum()
        if cnt == 0:
            bin_data.append({"bin_lo": bins[i], "bin_hi": bins[i+1], "count": 0,
                             "avg_pred": None, "avg_true": None})
            continue
        ap = float(pred_c[mask].mean())
        at = float(true_ious[mask].mean())
        ece += cnt / total * abs(ap - at)
        bin_data.append({"bin_lo": bins[i], "bin_hi": bins[i+1], "count": int(cnt),
                         "avg_pred": round(ap, 4), "avg_true": round(at, 4)})
    return ece, bin_data


# ── LoRA ──
class LoRALinear(nn.Module):
    def __init__(self, original, rank=16, alpha=32):
        super().__init__()
        self.original = original
        self.lora_A = nn.Parameter(torch.randn(rank, original.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def inject_lora(sam, rank=LORA_RANK, alpha=LORA_ALPHA):
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False
    count = 0
    for name, module in sam.mask_decoder.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.split(".")
            parent = sam.mask_decoder
            for part in parts[:-1]:
                parent = getattr(parent, part)
            lora = LoRALinear(module, rank, alpha).to(DEVICE)
            setattr(parent, parts[-1], lora)
            count += 1
    return count


def sample_points_from_gt(gt_mask, n_pos=8, n_neg=8):
    pos = np.argwhere(gt_mask)
    neg = np.argwhere(~gt_mask)
    pts, lbls = [], []
    if len(pos) > 0:
        idx = np.random.choice(len(pos), min(n_pos, len(pos)), replace=False)
        for i in idx:
            pts.append([pos[i][1], pos[i][0]])
            lbls.append(1)
    if len(neg) > 0:
        idx = np.random.choice(len(neg), min(n_neg, len(neg)), replace=False)
        for i in idx:
            pts.append([neg[i][1], neg[i][0]])
            lbls.append(0)
    return np.array(pts), np.array(lbls)


def train_lora(sam, manual_pairs, predictor):
    lora_params = [p for p in sam.mask_decoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, LORA_EPOCHS)
    sam.train()
    for ep in range(LORA_EPOCHS):
        ep_loss = 0
        for pair in manual_pairs:
            img = np.array(Image.open(pair["raw"]).convert("RGB"))
            gt = load_nist_gt(str(pair["gt"]))
            if img.shape[:2] != gt.shape:
                gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0])) > 0
            predictor.set_image(img)
            pts, lbls = sample_points_from_gt(gt, 8, 8)
            if len(pts) == 0:
                continue
            pts_t = torch.tensor(
                predictor.transform.apply_coords(pts, img.shape[:2]),
                dtype=torch.float32, device=DEVICE,
            ).unsqueeze(0)
            lbl_t = torch.tensor(lbls, dtype=torch.int, device=DEVICE).unsqueeze(0)
            sparse, dense = sam.prompt_encoder(points=(pts_t, lbl_t), boxes=None, masks=None)
            pred_masks, pred_ious = sam.mask_decoder(
                image_embeddings=predictor.features,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            gt_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            gt_r = nn.functional.interpolate(gt_t, size=pred_masks.shape[-2:], mode="nearest")
            bce = nn.functional.binary_cross_entropy_with_logits(pred_masks, gt_r)
            sig = torch.sigmoid(pred_masks)
            inter = (sig * gt_r).sum()
            dice = 1 - (2 * inter + 1) / (sig.sum() + gt_r.sum() + 1)
            loss = bce + dice
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    Epoch {ep+1}/{LORA_EPOCHS}, Loss: {ep_loss/len(manual_pairs):.4f}")


# ── AMG extraction (low thresh, one image at a time) ──
def extract_masks_for_image(sam, image_rgb, gt_mask, iou_thresh, stab_thresh):
    from segment_anything import SamAutomaticMaskGenerator
    amg = SamAutomaticMaskGenerator(
        model=sam, points_per_side=32,
        pred_iou_thresh=iou_thresh, stability_score_thresh=stab_thresh,
        crop_n_layers=1, crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    masks = amg.generate(image_rgb)
    infos = []
    for m in masks:
        seg = m["segmentation"]
        infos.append({
            "pred_iou": float(m["predicted_iou"]),
            "stability_score": float(m["stability_score"]),
            "true_iou": compute_iou(seg, gt_mask),
            "area": int(m["area"]),
        })
    return masks, infos


def pareto_for_image(masks, infos, gt_mask, iou_threshs, stab_threshs):
    """Compute union-IoU at each threshold combo from pre-extracted masks."""
    results = []
    for it in iou_threshs:
        for st in stab_threshs:
            kept_idx = [
                i for i, info in enumerate(infos)
                if info["pred_iou"] >= it and info["stability_score"] >= st
            ]
            if not kept_idx:
                results.append((it, st, 0, 0.0))
                continue
            union = np.zeros(gt_mask.shape, dtype=bool)
            for i in kept_idx:
                union |= masks[i]["segmentation"]
            results.append((it, st, len(kept_idx), compute_iou(union, gt_mask)))
    return results


# ── Plots ──
def make_figures(bl_pred, bl_true, lr_pred, lr_true,
                 bl_ece, lr_ece, bl_bins, lr_bins,
                 pareto_bl, pareto_lr, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Reliability diagram
    ax = axes[0, 0]
    for bins_data, lbl, clr in [(bl_bins, "Baseline", "steelblue"),
                                 (lr_bins, "LoRA", "coral")]:
        xs = [b["avg_pred"] for b in bins_data if b["avg_pred"] is not None]
        ys = [b["avg_true"] for b in bins_data if b["avg_true"] is not None]
        if xs:
            ax.plot(xs, ys, "o-", label=lbl, color=clr, markersize=6)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.set_xlabel("Predicted IoU (clamped to [0,1])")
    ax.set_ylabel("True IoU")
    ax.set_title(f"Reliability Diagram\nBaseline ECE={bl_ece:.3f}  |  LoRA ECE={lr_ece:.3f}")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, max(0.3, max(bl_true.max(), lr_true.max()) * 1.1))
    ax.grid(True, alpha=0.3)

    # Panel B: pred_iou vs true_iou scatter
    ax = axes[0, 1]
    ax.scatter(np.clip(bl_pred, 0, 1), bl_true, alpha=0.15, s=5, c="steelblue", label=f"Baseline (n={len(bl_pred)})")
    ax.scatter(np.clip(lr_pred, 0, 1), lr_true, alpha=0.6, s=20, c="coral", label=f"LoRA (n={len(lr_pred)})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Predicted IoU (clamped)")
    ax.set_ylabel("True IoU")
    ax.set_title("Predicted vs True IoU Scatter")
    ax.legend()
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.02, max(0.3, lr_true.max() * 1.3))
    ax.grid(True, alpha=0.3)

    # Panel C: Pareto – Baseline
    ax = axes[1, 0]
    if pareto_bl:
        df = pd.DataFrame(pareto_bl, columns=["iou_th", "stab_th", "n_masks", "union_iou"])
        grp = df.groupby(["iou_th", "stab_th"]).mean().reset_index()
        sc = ax.scatter(grp["n_masks"], grp["union_iou"], c=grp["iou_th"], cmap="viridis",
                        s=40, edgecolor="k", linewidth=0.3)
        plt.colorbar(sc, ax=ax, label="pred_iou_thresh")
    ax.set_xlabel("Avg Masks Per Image")
    ax.set_ylabel("Union IoU")
    ax.set_title("Baseline AMG: Pareto (Mask Count vs IoU)")
    ax.grid(True, alpha=0.3)

    # Panel D: Pareto – LoRA
    ax = axes[1, 1]
    if pareto_lr:
        df = pd.DataFrame(pareto_lr, columns=["iou_th", "stab_th", "n_masks", "union_iou"])
        grp = df.groupby(["iou_th", "stab_th"]).mean().reset_index()
        sc = ax.scatter(grp["n_masks"], grp["union_iou"], c=grp["iou_th"], cmap="magma",
                        s=40, edgecolor="k", linewidth=0.3)
        plt.colorbar(sc, ax=ax, label="pred_iou_thresh")
    ax.set_xlabel("Avg Masks Per Image")
    ax.set_ylabel("Union IoU")
    ax.set_title("LoRA-adapted AMG: Pareto (Threshold Re-tuning)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "Fig_Calibration_Pareto.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(fig_dir / "Fig_Calibration_Pareto.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [Saved] Fig_Calibration_Pareto.png/.pdf")


# ═══════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  CALIBRATION FIX + PARETO SWEEP + DETERMINISTIC CHECK")
    print("=" * 70)

    set_deterministic(SEED)
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    from tqdm import tqdm

    matched = get_matched_pairs(NIST_RAW_DIR, NIST_GT_DIR, N_TEST)
    manual_pairs = get_manual_pairs()
    print(f"  Test images: {len(matched)}, Manual train: {len(manual_pairs)}")
    print(f"  Device: {DEVICE}")

    # ═══ Phase 1: Baseline – low-threshold mask extraction ═══
    print(f"\n[Phase 1] Loading baseline SAM & extracting masks (thresh={EXTRACT_IOU_THRESH}/{EXTRACT_STAB_THRESH})...")
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(DEVICE).eval()

    bl_all_infos = []
    pareto_bl_raw = []
    for item in tqdm(matched, desc="  Baseline"):
        img = np.array(Image.open(item["raw"]).convert("RGB"))
        gt = load_nist_gt(str(item["gt"]))
        if img.shape[:2] != gt.shape:
            gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0])) > 0
        masks_obj, infos = extract_masks_for_image(
            sam, img, gt, EXTRACT_IOU_THRESH, EXTRACT_STAB_THRESH)
        for info in infos:
            info["image"] = item["name"]
        bl_all_infos.extend(infos)
        pareto_results = pareto_for_image(
            masks_obj, infos, gt, PARETO_IOU_THRESHS, PARETO_STAB_THRESHS)
        pareto_bl_raw.extend(pareto_results)
        del masks_obj
        gc.collect()

    bl_pred = np.array([m["pred_iou"] for m in bl_all_infos])
    bl_true = np.array([m["true_iou"] for m in bl_all_infos])
    print(f"  Baseline: {len(bl_all_infos)} total masks, "
          f"pred_iou range [{bl_pred.min():.3f}, {bl_pred.max():.3f}]")

    # ═══ Phase 2: Deterministic reproducibility (2nd run at standard thresholds) ═══
    print("\n[Phase 2] Deterministic reproducibility check...")
    set_deterministic(SEED)
    amg_std = SamAutomaticMaskGenerator(
        model=sam, points_per_side=32, pred_iou_thresh=0.88,
        stability_score_thresh=0.95, crop_n_layers=1,
        crop_n_points_downscale_factor=2, min_mask_region_area=100,
    )
    run1_ious = []
    for item in tqdm(matched[:10], desc="  Run1"):
        img = np.array(Image.open(item["raw"]).convert("RGB"))
        gt = load_nist_gt(str(item["gt"]))
        if img.shape[:2] != gt.shape:
            gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0])) > 0
        masks_out = amg_std.generate(img)
        combined = np.zeros(gt.shape, dtype=bool)
        for m in masks_out:
            combined |= m["segmentation"]
        run1_ious.append(compute_iou(combined, gt))

    set_deterministic(SEED)
    run2_ious = []
    for item in tqdm(matched[:10], desc="  Run2"):
        img = np.array(Image.open(item["raw"]).convert("RGB"))
        gt = load_nist_gt(str(item["gt"]))
        if img.shape[:2] != gt.shape:
            gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0])) > 0
        masks_out = amg_std.generate(img)
        combined = np.zeros(gt.shape, dtype=bool)
        for m in masks_out:
            combined |= m["segmentation"]
        run2_ious.append(compute_iou(combined, gt))

    max_diff = max(abs(a - b) for a, b in zip(run1_ious, run2_ious))
    print(f"  Max IoU difference between runs: {max_diff:.8f}")
    print(f"  {'DETERMINISTIC - PASS' if max_diff < 1e-6 else 'NON-DETERMINISTIC - FAIL'}")

    # Free baseline model for LoRA
    del amg_std
    del sam
    gc.collect()
    torch.cuda.empty_cache()

    # ═══ Phase 3: LoRA training ═══
    print(f"\n[Phase 3] Training LoRA (rank={LORA_RANK}, epochs={LORA_EPOCHS})...")
    set_deterministic(SEED)
    sam_lora = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
    sam_lora.to(DEVICE)
    n_lora = inject_lora(sam_lora, LORA_RANK, LORA_ALPHA)
    print(f"  Injected LoRA into {n_lora} layers")
    predictor = SamPredictor(sam_lora)
    train_lora(sam_lora, manual_pairs, predictor)

    # ═══ Phase 4: LoRA – low-threshold mask extraction ═══
    print(f"\n[Phase 4] LoRA mask extraction (thresh={EXTRACT_IOU_THRESH}/{EXTRACT_STAB_THRESH})...")
    sam_lora.eval()
    lr_all_infos = []
    pareto_lr_raw = []
    for item in tqdm(matched, desc="  LoRA"):
        img = np.array(Image.open(item["raw"]).convert("RGB"))
        gt = load_nist_gt(str(item["gt"]))
        if img.shape[:2] != gt.shape:
            gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0])) > 0
        masks_obj, infos = extract_masks_for_image(
            sam_lora, img, gt, EXTRACT_IOU_THRESH, EXTRACT_STAB_THRESH)
        for info in infos:
            info["image"] = item["name"]
        lr_all_infos.extend(infos)
        pareto_results = pareto_for_image(
            masks_obj, infos, gt, PARETO_IOU_THRESHS, PARETO_STAB_THRESHS)
        pareto_lr_raw.extend(pareto_results)
        del masks_obj
        gc.collect()

    lr_pred = np.array([m["pred_iou"] for m in lr_all_infos])
    lr_true = np.array([m["true_iou"] for m in lr_all_infos])
    print(f"  LoRA: {len(lr_all_infos)} total masks, "
          f"pred_iou range [{lr_pred.min():.3f}, {lr_pred.max():.3f}]")

    # ═══ Phase 5: Corrected calibration metrics ═══
    print("\n[Phase 5] Corrected calibration metrics (clamped pred_iou)...")
    bl_ece, bl_bins = compute_ece(bl_pred, bl_true)
    lr_ece, lr_bins = compute_ece(lr_pred, lr_true)

    bl_pred_c = np.clip(bl_pred, 0, 1)
    lr_pred_c = np.clip(lr_pred, 0, 1)
    bl_pearson = float(np.corrcoef(bl_pred_c, bl_true)[0, 1]) if len(bl_pred) > 1 else 0
    lr_pearson = float(np.corrcoef(lr_pred_c, lr_true)[0, 1]) if len(lr_pred) > 1 else 0
    bl_spearman = float(sp_stats.spearmanr(bl_pred_c, bl_true).statistic) if len(bl_pred) > 1 else 0
    lr_spearman = float(sp_stats.spearmanr(lr_pred_c, lr_true).statistic) if len(lr_pred) > 1 else 0

    print(f"  Baseline: ECE={bl_ece:.4f}, Pearson={bl_pearson:.4f}, Spearman={bl_spearman:.4f}, n={len(bl_pred)}")
    print(f"    pred_iou: {bl_pred.mean():.4f}±{bl_pred.std():.4f} → clamped: {bl_pred_c.mean():.4f}")
    print(f"    true_iou: {bl_true.mean():.6f}±{bl_true.std():.6f}")
    print(f"  LoRA:     ECE={lr_ece:.4f}, Pearson={lr_pearson:.4f}, Spearman={lr_spearman:.4f}, n={len(lr_pred)}")
    print(f"    pred_iou: {lr_pred.mean():.4f}±{lr_pred.std():.4f} → clamped: {lr_pred_c.mean():.4f}")
    print(f"    true_iou: {lr_true.mean():.6f}±{lr_true.std():.6f}")

    overconfidence_ratio = bl_pred_c.mean() / (bl_true.mean() + 1e-10)
    print(f"\n  Overconfidence ratio (baseline): {overconfidence_ratio:.0f}×")

    # ═══ Phase 6: Pareto aggregation ═══
    print("\n[Phase 6] Aggregating Pareto data...")
    def aggregate_pareto(raw_data, n_images):
        df = pd.DataFrame(raw_data, columns=["iou_th", "stab_th", "n_masks", "union_iou"])
        grp = df.groupby(["iou_th", "stab_th"]).agg(
            avg_masks=("n_masks", "mean"),
            avg_iou=("union_iou", "mean"),
            std_iou=("union_iou", "std"),
        ).reset_index()
        return grp

    pareto_bl_df = aggregate_pareto(pareto_bl_raw, N_TEST)
    pareto_lr_df = aggregate_pareto(pareto_lr_raw, N_TEST)

    best_bl = pareto_bl_df.loc[pareto_bl_df["avg_iou"].idxmax()]
    best_lr = pareto_lr_df.loc[pareto_lr_df["avg_iou"].idxmax()]
    print(f"  Baseline best: thresh={best_bl['iou_th']:.2f}/{best_bl['stab_th']:.2f}, "
          f"masks={best_bl['avg_masks']:.1f}, IoU={best_bl['avg_iou']:.4f}")
    print(f"  LoRA best:     thresh={best_lr['iou_th']:.2f}/{best_lr['stab_th']:.2f}, "
          f"masks={best_lr['avg_masks']:.1f}, IoU={best_lr['avg_iou']:.4f}")

    # ═══ Phase 7: Save everything ═══
    print("\n[Phase 7] Saving results and generating figures...")
    out_dir = OUTPUTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_corrected = {
        "timestamp": datetime.now().isoformat(),
        "ece_definition": "10 equal-width bins on [0,1]; pred_iou clamped to [0,1]; last bin inclusive",
        "baseline": {
            "n_masks": len(bl_pred), "ece": round(bl_ece, 4),
            "pearson_r": round(bl_pearson, 4), "spearman_rho": round(bl_spearman, 4),
            "pred_iou_mean": round(float(bl_pred.mean()), 4),
            "pred_iou_mean_clamped": round(float(bl_pred_c.mean()), 4),
            "pred_iou_std": round(float(bl_pred.std()), 4),
            "true_iou_mean": round(float(bl_true.mean()), 6),
            "true_iou_std": round(float(bl_true.std()), 6),
            "overconfidence_ratio": round(overconfidence_ratio, 1),
            "reliability_bins": bl_bins,
        },
        "lora": {
            "n_masks": len(lr_pred), "ece": round(lr_ece, 4),
            "pearson_r": round(lr_pearson, 4), "spearman_rho": round(lr_spearman, 4),
            "pred_iou_mean": round(float(lr_pred.mean()), 4),
            "pred_iou_mean_clamped": round(float(lr_pred_c.mean()), 4),
            "pred_iou_std": round(float(lr_pred.std()), 4),
            "true_iou_mean": round(float(lr_true.mean()), 6),
            "true_iou_std": round(float(lr_true.std()), 6),
        },
        "deterministic_check": {
            "max_iou_diff_between_runs": max_diff,
            "is_deterministic": max_diff < 1e-6,
        },
        "pareto_baseline_best": {
            "iou_thresh": float(best_bl["iou_th"]),
            "stab_thresh": float(best_bl["stab_th"]),
            "avg_masks": round(float(best_bl["avg_masks"]), 1),
            "avg_iou": round(float(best_bl["avg_iou"]), 4),
        },
        "pareto_lora_best": {
            "iou_thresh": float(best_lr["iou_th"]),
            "stab_thresh": float(best_lr["stab_th"]),
            "avg_masks": round(float(best_lr["avg_masks"]), 1),
            "avg_iou": round(float(best_lr["avg_iou"]), 4),
        },
    }

    with open(out_dir / "calibration_corrected.json", "w") as f:
        json.dump(calib_corrected, f, indent=2)

    pareto_bl_df.to_csv(out_dir / "pareto_baseline.csv", index=False)
    pareto_lr_df.to_csv(out_dir / "pareto_lora.csv", index=False)

    make_figures(bl_pred, bl_true, lr_pred, lr_true,
                 bl_ece, lr_ece, bl_bins, lr_bins,
                 pareto_bl_raw, pareto_lr_raw, OUTPUTS_DIR)

    print(f"\n[Saved] calibration_corrected.json")
    print(f"[Saved] pareto_baseline.csv, pareto_lora.csv")
    print("=" * 70)
    print("  ALL DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
