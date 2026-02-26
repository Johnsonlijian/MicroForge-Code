#!/usr/bin/env python3
"""
LoRA Calibration Analysis + AMG Threshold Re-optimization
==========================================================
Addresses reviewer concerns H5: "Quantify the protocol tension"

Analysis:
  1. Extract pred_iou from SAM before/after LoRA
  2. Compute calibration metrics (ECE, correlation with true IoU)
  3. Sweep AMG thresholds (pred_iou_thresh, stability_score_thresh)
     to find if tension can be mitigated
  4. Generate reliability diagram

Outputs:
  - calibration_analysis.json
  - amg_threshold_sweep.csv
  - Fig_Calibration_Analysis.png
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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


def compute_iou(pred, gt):
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred_b, gt_b).sum()
    union = tp + np.logical_and(pred_b, ~gt_b).sum() + np.logical_and(~pred_b, gt_b).sum()
    return float(tp) / float(union) if union > 0 else 0.0


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=16, alpha=32):
        super().__init__()
        self.original = original_linear
        in_f = original_linear.in_features
        out_f = original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def inject_lora(sam, rank=LORA_RANK, alpha=LORA_ALPHA):
    """Inject LoRA into mask decoder linear layers."""
    count = 0
    for name, module in sam.mask_decoder.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = sam.mask_decoder
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            lora = LoRALinear(module, rank, alpha)
            lora.lora_A = nn.Parameter(lora.lora_A.to(DEVICE))
            lora.lora_B = nn.Parameter(lora.lora_B.to(DEVICE))
            setattr(parent, child_name, lora)
            count += 1
    return count


def sample_points_from_gt(gt_mask, n_pos=8, n_neg=8):
    pos_coords = np.argwhere(gt_mask)
    neg_coords = np.argwhere(~gt_mask)
    points, labels = [], []
    if len(pos_coords) > 0:
        idx = np.random.choice(len(pos_coords), min(n_pos, len(pos_coords)), replace=False)
        for i in idx:
            points.append([pos_coords[i][1], pos_coords[i][0]])
            labels.append(1)
    if len(neg_coords) > 0:
        idx = np.random.choice(len(neg_coords), min(n_neg, len(neg_coords)), replace=False)
        for i in idx:
            points.append([neg_coords[i][1], neg_coords[i][0]])
            labels.append(0)
    return np.array(points), np.array(labels)


def get_manual_pairs():
    manual_files = sorted(os.listdir(NIST_MANUAL_GT_DIR))
    pairs = []
    for mf in manual_files:
        base = mf.replace('annot_', '').replace('.png', '')
        raw_path = NIST_RAW_DIR / f"{base}.ome.png"
        if raw_path.exists():
            pairs.append({'raw': raw_path, 'gt': NIST_MANUAL_GT_DIR / mf, 'name': raw_path.name})
    return pairs


def extract_amg_mask_info(sam, image_rgb, gt_mask, iou_thresh=0.88, stab_thresh=0.95):
    """Run AMG and extract per-mask pred_iou and true_iou for calibration analysis."""
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
        seg = m['segmentation']
        true_iou = compute_iou(seg, gt_mask)
        infos.append({
            'pred_iou': float(m['predicted_iou']),
            'stability_score': float(m['stability_score']),
            'true_iou': true_iou,
            'area': int(m['area']),
            'area_ratio': float(seg.sum() / (seg.shape[0] * seg.shape[1])),
        })
    return infos, masks


def compute_ece(pred_ious, true_ious, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(pred_ious)
    for i in range(n_bins):
        mask = (pred_ious >= bins[i]) & (pred_ious < bins[i+1])
        if mask.sum() == 0:
            continue
        avg_pred = pred_ious[mask].mean()
        avg_true = true_ious[mask].mean()
        ece += mask.sum() / total * abs(avg_pred - avg_true)
    return ece


def main():
    print("=" * 70)
    print("LORA CALIBRATION ANALYSIS + AMG THRESHOLD SWEEP")
    print("=" * 70)

    from segment_anything import sam_model_registry, SamPredictor
    from tqdm import tqdm

    # ===== Phase 1: Baseline SAM calibration =====
    print("\n[Phase 1] Loading baseline SAM...")
    sam_baseline = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
    sam_baseline.to(DEVICE).eval()

    matched = get_matched_pairs(NIST_RAW_DIR, NIST_GT_DIR, N_TEST)
    print(f"Test images: {len(matched)}")

    print("\nExtracting baseline AMG mask info...")
    baseline_mask_infos = []
    for item in tqdm(matched, desc="  Baseline"):
        try:
            image = np.array(Image.open(item['raw']).convert('RGB'))
            gt_mask = load_nist_gt(str(item['gt']))
            if image.shape[:2] != gt_mask.shape:
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (image.shape[1], image.shape[0])) > 0
            infos, _ = extract_amg_mask_info(sam_baseline, image, gt_mask)
            for info in infos:
                info['image'] = item['name']
            baseline_mask_infos.extend(infos)
        except Exception as e:
            print(f"  [!] {item['name']}: {e}")

    # ===== Phase 2: Train LoRA and get calibration =====
    print(f"\n[Phase 2] Training LoRA (rank={LORA_RANK}, LR=2e-4, 30 epochs)...")

    sam_lora = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
    sam_lora.to(DEVICE)

    for p in sam_lora.image_encoder.parameters():
        p.requires_grad = False
    for p in sam_lora.prompt_encoder.parameters():
        p.requires_grad = False

    n_lora = inject_lora(sam_lora, LORA_RANK, LORA_ALPHA)
    print(f"  Injected LoRA into {n_lora} layers")

    lora_params = [p for p in sam_lora.mask_decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)

    manual_pairs = get_manual_pairs()
    predictor = SamPredictor(sam_lora)

    sam_lora.train()
    for epoch in range(30):
        epoch_loss = 0
        for pair in manual_pairs:
            image = np.array(Image.open(pair['raw']).convert('RGB'))
            gt = load_nist_gt(str(pair['gt']))
            if image.shape[:2] != gt.shape:
                gt = cv2.resize(gt.astype(np.uint8), (image.shape[1], image.shape[0])) > 0

            predictor.set_image(image)
            points, labels = sample_points_from_gt(gt, 8, 8)
            if len(points) == 0:
                continue

            pts_t = torch.tensor(predictor.transform.apply_coords(points, image.shape[:2]),
                                dtype=torch.float32, device=DEVICE).unsqueeze(0)
            lbl_t = torch.tensor(labels, dtype=torch.int, device=DEVICE).unsqueeze(0)

            sparse, dense = sam_lora.prompt_encoder(points=(pts_t, lbl_t), boxes=None, masks=None)
            pred_masks, pred_ious = sam_lora.mask_decoder(
                image_embeddings=predictor.features,
                image_pe=sam_lora.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )

            gt_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            gt_resized = torch.nn.functional.interpolate(gt_t, size=pred_masks.shape[-2:], mode='nearest')

            bce = torch.nn.functional.binary_cross_entropy_with_logits(pred_masks, gt_resized)
            pred_sig = torch.sigmoid(pred_masks)
            smooth = 1.0
            inter = (pred_sig * gt_resized).sum()
            dice_loss = 1 - (2*inter + smooth) / (pred_sig.sum() + gt_resized.sum() + smooth)
            loss = bce + dice_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30, Loss: {epoch_loss/len(manual_pairs):.4f}")

    # ===== Phase 3: LoRA calibration extraction =====
    print("\n[Phase 3] Extracting LoRA AMG mask info...")
    sam_lora.eval()
    lora_mask_infos = []
    for item in tqdm(matched, desc="  LoRA"):
        try:
            image = np.array(Image.open(item['raw']).convert('RGB'))
            gt_mask = load_nist_gt(str(item['gt']))
            if image.shape[:2] != gt_mask.shape:
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (image.shape[1], image.shape[0])) > 0
            infos, _ = extract_amg_mask_info(sam_lora, image, gt_mask)
            for info in infos:
                info['image'] = item['name']
            lora_mask_infos.extend(infos)
        except Exception as e:
            print(f"  [!] {item['name']}: {e}")

    # ===== Phase 4: Calibration analysis =====
    print("\n[Phase 4] Computing calibration metrics...")

    bl_pred = np.array([m['pred_iou'] for m in baseline_mask_infos])
    bl_true = np.array([m['true_iou'] for m in baseline_mask_infos])
    lr_pred = np.array([m['pred_iou'] for m in lora_mask_infos])
    lr_true = np.array([m['true_iou'] for m in lora_mask_infos])

    bl_ece = compute_ece(bl_pred, bl_true)
    lr_ece = compute_ece(lr_pred, lr_true)
    bl_corr = float(np.corrcoef(bl_pred, bl_true)[0, 1]) if len(bl_pred) > 1 else 0
    lr_corr = float(np.corrcoef(lr_pred, lr_true)[0, 1]) if len(lr_pred) > 1 else 0

    print(f"  Baseline: ECE={bl_ece:.4f}, Correlation={bl_corr:.4f}, N_masks={len(bl_pred)}")
    print(f"  LoRA:     ECE={lr_ece:.4f}, Correlation={lr_corr:.4f}, N_masks={len(lr_pred)}")

    # ===== Phase 5: AMG threshold sweep =====
    print("\n[Phase 5] AMG threshold sweep for LoRA model...")

    iou_threshs = [0.80, 0.84, 0.86, 0.88, 0.90, 0.92]
    stab_threshs = [0.90, 0.92, 0.95, 0.97]

    sweep_results = []
    for iou_th in iou_threshs:
        for stab_th in stab_threshs:
            # Filter masks by threshold
            kept = [m for m in lora_mask_infos
                    if m['pred_iou'] >= iou_th and m['stability_score'] >= stab_th]
            if len(kept) == 0:
                continue

            # Compute per-image IoU
            per_image = {}
            for m in kept:
                img = m['image']
                if img not in per_image:
                    per_image[img] = {'true_ious': [], 'areas': []}
                per_image[img]['true_ious'].append(m['true_iou'])
                per_image[img]['areas'].append(m['area_ratio'])

            avg_iou = np.mean([np.mean(v['true_ious']) for v in per_image.values()])
            n_masks_avg = np.mean([len(v['true_ious']) for v in per_image.values()])

            sweep_results.append({
                'iou_thresh': iou_th,
                'stability_thresh': stab_th,
                'avg_mask_true_iou': float(avg_iou),
                'avg_masks_per_image': float(n_masks_avg),
                'n_total_masks': len(kept),
            })

    sweep_df = pd.DataFrame(sweep_results)
    if len(sweep_df) > 0:
        best = sweep_df.loc[sweep_df['avg_mask_true_iou'].idxmax()]
        print(f"  Best threshold: iou_thresh={best['iou_thresh']}, "
              f"stab_thresh={best['stability_thresh']}, "
              f"avg_true_iou={best['avg_mask_true_iou']:.4f}")

    # ===== Save & Plot =====
    out_dir = OUTPUTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_results = {
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'ece': bl_ece, 'correlation': bl_corr, 'n_masks': len(bl_pred),
            'pred_iou_mean': float(bl_pred.mean()), 'true_iou_mean': float(bl_true.mean()),
        },
        'lora': {
            'ece': lr_ece, 'correlation': lr_corr, 'n_masks': len(lr_pred),
            'pred_iou_mean': float(lr_pred.mean()), 'true_iou_mean': float(lr_true.mean()),
        },
        'threshold_sweep': sweep_results,
    }

    with open(out_dir / "calibration_analysis.json", 'w') as f:
        json.dump(calib_results, f, indent=2)
    sweep_df.to_csv(out_dir / "amg_threshold_sweep.csv", index=False)

    # Reliability diagram
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Reliability diagram
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    for pred_arr, true_arr, lbl, clr in [(bl_pred, bl_true, 'Baseline', 'steelblue'),
                                         (lr_pred, lr_true, 'LoRA', 'coral')]:
        bin_means_pred, bin_means_true = [], []
        for i in range(n_bins):
            mask = (pred_arr >= bins[i]) & (pred_arr < bins[i+1])
            if mask.sum() > 0:
                bin_means_pred.append(pred_arr[mask].mean())
                bin_means_true.append(true_arr[mask].mean())
        if bin_means_pred:
            axes[0].plot(bin_means_pred, bin_means_true, 'o-', label=lbl, color=clr, markersize=6)

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    axes[0].set_xlabel('Predicted IoU (SAM internal)')
    axes[0].set_ylabel('True IoU (vs GT)')
    axes[0].set_title(f'Reliability Diagram\nBaseline ECE={bl_ece:.3f}, LoRA ECE={lr_ece:.3f}')
    axes[0].legend()
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 0.6)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: pred_iou distributions
    axes[1].hist(bl_pred, bins=30, alpha=0.6, label=f'Baseline (n={len(bl_pred)})', color='steelblue')
    axes[1].hist(lr_pred, bins=30, alpha=0.6, label=f'LoRA (n={len(lr_pred)})', color='coral')
    axes[1].set_xlabel('Predicted IoU')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of SAM Predicted IoU Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Threshold sweep heatmap
    if len(sweep_df) > 0:
        pivot = sweep_df.pivot_table(index='iou_thresh', columns='stability_thresh',
                                      values='avg_mask_true_iou')
        im = axes[2].imshow(pivot.values, cmap='YlOrRd', aspect='auto',
                           extent=[pivot.columns.min()-0.01, pivot.columns.max()+0.01,
                                   pivot.index.max()+0.01, pivot.index.min()-0.01])
        axes[2].set_xlabel('Stability Score Threshold')
        axes[2].set_ylabel('Predicted IoU Threshold')
        axes[2].set_title('AMG Threshold Sweep\n(avg true IoU of kept masks)')
        plt.colorbar(im, ax=axes[2], label='Avg True IoU')

    plt.tight_layout()
    fig_path = OUTPUTS_DIR / "figures" / "Fig_Calibration_Analysis.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n[Saved] calibration_analysis.json")
    print(f"[Saved] amg_threshold_sweep.csv")
    print(f"[Saved] Fig_Calibration_Analysis.png/.pdf")
    print("\nDone!")


if __name__ == "__main__":
    main()
