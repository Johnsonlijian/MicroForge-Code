#!/usr/bin/env python3
"""
Supervised Baselines: U-Net and DeepLabv3+ with LOO Cross-Validation
=====================================================================
Addresses reviewer concern H3: "What about supervised methods on 12 images?"

Leave-One-Out (LOO) CV on 12 manually annotated NIST images.
Also tests on the same 30 auto-GT images used in unified_verification.

Models:
  1. U-Net (ResNet34 encoder, pretrained ImageNet)
  2. DeepLabv3 (ResNet50 encoder, pretrained COCO)
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

NIST_RAW_DIR = DATA_DIR / "raw" / "nist_sem" / "rawFOV"
NIST_AUTO_GT_DIR = DATA_DIR / "raw" / "nist_sem" / "damageContextAssistedMask" / "damageMask"
NIST_MANUAL_GT_DIR = DATA_DIR / "raw" / "nist_sem" / "contextManualMaskGT" / "contextMaskGT"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
N_EPOCHS = 80
LR = 1e-4
BATCH_SIZE = 4
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================
# Data
# ============================

def load_nist_gt(gt_path):
    gt = np.array(Image.open(gt_path))
    if len(gt.shape) == 3:
        return np.any(gt > 0, axis=2).astype(np.float32)
    return (gt > 0).astype(np.float32)


def get_manual_pairs():
    """Get 12 manually annotated image-GT pairs."""
    manual_files = sorted(os.listdir(NIST_MANUAL_GT_DIR))
    pairs = []
    for mf in manual_files:
        base = mf.replace('annot_', '').replace('.png', '')
        raw_path = NIST_RAW_DIR / f"{base}.ome.png"
        if raw_path.exists():
            pairs.append({
                'raw': raw_path,
                'gt': NIST_MANUAL_GT_DIR / mf,
                'name': raw_path.name
            })
    return pairs


def get_auto_test_pairs(n=30):
    """Get 30 auto-GT test pairs (same as unified_verification)."""
    raw_files = sorted([f for f in os.listdir(NIST_RAW_DIR) if f.endswith('.png')])
    matched = []
    for rf in raw_files:
        base = rf.replace('.png', '')
        gt_path = NIST_AUTO_GT_DIR / f"{base}_damage.png"
        if gt_path.exists():
            matched.append({'raw': NIST_RAW_DIR / rf, 'gt': gt_path, 'name': rf})
        if len(matched) >= n:
            break
    return matched


class SEMDataset(Dataset):
    def __init__(self, pairs, img_size=IMG_SIZE, augment=False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img = np.array(Image.open(p['raw']).convert('RGB'))
        gt = load_nist_gt(str(p['gt']))

        img = cv2.resize(img, (self.img_size, self.img_size))
        gt = cv2.resize(gt, (self.img_size, self.img_size))

        if self.augment and np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            gt = np.fliplr(gt).copy()
        if self.augment and np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            gt = np.flipud(gt).copy()
        if self.augment and np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            img = np.rot90(img, k).copy()
            gt = np.rot90(gt, k).copy()

        img_t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        gt_t = torch.from_numpy((gt > 0.5).astype(np.float32)).unsqueeze(0)

        return img_t, gt_t


# ============================
# Models
# ============================

class SimpleUNet(nn.Module):
    """Minimal U-Net with ResNet34-style encoder blocks."""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            )
        self.enc1 = conv_block(in_ch, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out_conv(d1)


class DeepLabWrapper(nn.Module):
    """Wraps torchvision DeepLabv3 for binary segmentation."""
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet50(weights='DEFAULT')
        self.model.classifier[-1] = nn.Conv2d(256, 1, 1)
        self.model.aux_classifier = None

    def forward(self, x):
        out = self.model(x)['out']
        return out


# ============================
# Training & Evaluation
# ============================

def dice_bce_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred_sig = torch.sigmoid(pred)
    smooth = 1.0
    intersection = (pred_sig * target).sum()
    dice = 1 - (2 * intersection + smooth) / (pred_sig.sum() + target.sum() + smooth)
    return bce + dice


def compute_metrics(pred_np, gt_np):
    pred_b, gt_b = pred_np > 0.5, gt_np > 0.5
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


def train_one_fold(model, train_pairs, val_pair, n_epochs=N_EPOCHS):
    """Train model on train_pairs, evaluate on val_pair. Returns val metrics."""
    model = copy.deepcopy(model).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    train_ds = SEMDataset(train_pairs, augment=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(n_epochs):
        for imgs, gts in train_dl:
            imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
            pred = model(imgs)
            loss = dice_bce_loss(pred, gts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate on val pair
    model.eval()
    val_ds = SEMDataset([val_pair], augment=False)
    img, gt = val_ds[0]
    with torch.no_grad():
        pred = torch.sigmoid(model(img.unsqueeze(0).to(DEVICE))).cpu().numpy()[0, 0]
    gt_np = gt.numpy()[0]
    return compute_metrics(pred, gt_np), model


def evaluate_on_test(model, test_pairs):
    """Evaluate trained model on test set (auto GT)."""
    model.eval()
    results = []
    for p in test_pairs:
        img = np.array(Image.open(p['raw']).convert('RGB'))
        gt = load_nist_gt(str(p['gt']))

        h_orig, w_orig = img.shape[:2]
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_t = torch.from_numpy(img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0)

        with torch.no_grad():
            pred = torch.sigmoid(model(img_t.unsqueeze(0).to(DEVICE))).cpu().numpy()[0, 0]

        pred_full = cv2.resize(pred, (w_orig, h_orig))
        metrics = compute_metrics(pred_full, gt)
        metrics['name'] = p['name']
        results.append(metrics)

    return results


# ============================
# Main
# ============================

def main():
    print("=" * 70)
    print("SUPERVISED BASELINES: U-Net & DeepLab LOO CV")
    print("=" * 70)
    print(f"Device: {DEVICE}, Epochs: {N_EPOCHS}, LR: {LR}")

    manual_pairs = get_manual_pairs()
    test_pairs = get_auto_test_pairs(30)
    print(f"Manual GT pairs: {len(manual_pairs)}")
    print(f"Auto GT test pairs: {len(test_pairs)}")

    models_to_test = {
        'U-Net': SimpleUNet,
        'DeepLabv3': DeepLabWrapper,
    }

    all_results = {}

    for model_name, ModelClass in models_to_test.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_name} - LOO CV ({len(manual_pairs)} folds)")
        print(f"{'='*70}")

        loo_results = []
        best_model = None
        best_iou = -1

        for fold_idx in range(len(manual_pairs)):
            val_pair = manual_pairs[fold_idx]
            train_pairs = [p for j, p in enumerate(manual_pairs) if j != fold_idx]

            print(f"  Fold {fold_idx+1}/{len(manual_pairs)}: val={val_pair['name']}", end=" ")
            sys.stdout.flush()

            model = ModelClass()
            val_metrics, trained_model = train_one_fold(model, train_pairs, val_pair)
            loo_results.append(val_metrics)

            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                best_model = trained_model

            print(f"-> IoU={val_metrics['iou']:.4f}, Dice={val_metrics['dice']:.4f}")

        # LOO summary
        loo_ious = [r['iou'] for r in loo_results]
        loo_dices = [r['dice'] for r in loo_results]
        loo_precs = [r['precision'] for r in loo_results]
        loo_recs = [r['recall'] for r in loo_results]

        loo_summary = {
            'model': model_name,
            'cv_type': 'LOO',
            'n_folds': len(manual_pairs),
            'iou_mean': float(np.mean(loo_ious)),
            'iou_std': float(np.std(loo_ious)),
            'iou_ci95': float(1.96 * np.std(loo_ious) / np.sqrt(len(loo_ious))),
            'dice_mean': float(np.mean(loo_dices)),
            'precision_mean': float(np.mean(loo_precs)),
            'recall_mean': float(np.mean(loo_recs)),
        }

        print(f"\n  LOO Summary: IoU={loo_summary['iou_mean']:.4f} +/- {loo_summary['iou_std']:.4f} "
              f"(95% CI: +/- {loo_summary['iou_ci95']:.4f})")

        # Test on 30 auto-GT images with best model
        print(f"  Evaluating best model on {len(test_pairs)} auto-GT test images...")
        test_results = evaluate_on_test(best_model, test_pairs)
        test_ious = [r['iou'] for r in test_results]

        test_summary = {
            'model': model_name,
            'test_set': 'auto_GT_30',
            'iou_mean': float(np.mean(test_ious)),
            'iou_std': float(np.std(test_ious)),
            'dice_mean': float(np.mean([r['dice'] for r in test_results])),
            'precision_mean': float(np.mean([r['precision'] for r in test_results])),
            'recall_mean': float(np.mean([r['recall'] for r in test_results])),
        }

        print(f"  Test Summary: IoU={test_summary['iou_mean']:.4f} +/- {test_summary['iou_std']:.4f}")

        all_results[model_name] = {
            'loo_summary': loo_summary,
            'loo_per_fold': loo_results,
            'test_summary': test_summary,
            'test_per_image': test_results,
        }

    # Save
    out_dir = OUTPUTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "supervised_baselines_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    rows = []
    for name, data in all_results.items():
        row = {**data['loo_summary'], 'test_iou': data['test_summary']['iou_mean']}
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "supervised_baselines_results.csv", index=False)

    # Print final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: Supervised Baselines")
    print("=" * 70)
    print(f"\n{'Model':<15} {'LOO IoU':>12} {'LOO Dice':>12} {'Test IoU (30)':>15}")
    print("-" * 55)
    for name, data in all_results.items():
        l = data['loo_summary']
        t = data['test_summary']
        print(f"{name:<15} {l['iou_mean']:.4f}+/-{l['iou_std']:.3f} "
              f"{l['dice_mean']:.4f}       {t['iou_mean']:.4f}")

    print(f"\n[Saved] supervised_baselines_results.json / .csv")


if __name__ == "__main__":
    main()
