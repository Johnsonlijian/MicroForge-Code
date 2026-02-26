#!/usr/bin/env python3
"""DeepLab LOO CV only (U-Net results already available). Fixes BatchNorm issue."""

import os, sys, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def load_nist_gt(gt_path):
    gt = np.array(Image.open(gt_path))
    if len(gt.shape) == 3:
        return np.any(gt > 0, axis=2).astype(np.float32)
    return (gt > 0).astype(np.float32)


def get_manual_pairs():
    pairs = []
    for mf in sorted(os.listdir(NIST_MANUAL_GT_DIR)):
        base = mf.replace('annot_', '').replace('.png', '')
        raw_path = NIST_RAW_DIR / f"{base}.ome.png"
        if raw_path.exists():
            pairs.append({'raw': raw_path, 'gt': NIST_MANUAL_GT_DIR / mf, 'name': raw_path.name})
    return pairs


def get_auto_test_pairs(n=30):
    matched = []
    for rf in sorted(f for f in os.listdir(NIST_RAW_DIR) if f.endswith('.png')):
        base = rf.replace('.png', '')
        gt_path = NIST_AUTO_GT_DIR / f"{base}_damage.png"
        if gt_path.exists():
            matched.append({'raw': NIST_RAW_DIR / rf, 'gt': gt_path, 'name': rf})
        if len(matched) >= n:
            break
    return matched


class SEMDataset(Dataset):
    def __init__(self, pairs, img_size=IMG_SIZE, augment=False, repeat=1):
        self.pairs = pairs * repeat
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
            img = np.fliplr(img).copy(); gt = np.fliplr(gt).copy()
        if self.augment and np.random.rand() > 0.5:
            img = np.flipud(img).copy(); gt = np.flipud(gt).copy()
        if self.augment and np.random.rand() > 0.5:
            k = np.random.randint(1, 4); img = np.rot90(img, k).copy(); gt = np.rot90(gt, k).copy()
        img_t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        gt_t = torch.from_numpy((gt > 0.5).astype(np.float32)).unsqueeze(0)
        return img_t, gt_t


class DeepLabWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet50(weights='DEFAULT')
        self.model.classifier[-1] = nn.Conv2d(256, 1, 1)
        self.model.aux_classifier = None

    def forward(self, x):
        return self.model(x)['out']


def dice_bce_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * target).sum()
    dice = 1 - (2 * inter + 1) / (pred_sig.sum() + target.sum() + 1)
    return bce + dice


def compute_metrics(pred_np, gt_np):
    pred_b, gt_b = pred_np > 0.5, gt_np > 0.5
    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, ~gt_b).sum()
    fn = np.logical_and(~pred_b, gt_b).sum()
    union = tp + fp + fn
    return {
        'iou': float(tp)/float(union) if union > 0 else 0.0,
        'dice': float(2*tp)/float(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0.0,
        'precision': float(tp)/float(tp+fp) if (tp+fp) > 0 else 0.0,
        'recall': float(tp)/float(tp+fn) if (tp+fn) > 0 else 0.0,
    }


def train_one_fold(train_pairs, val_pair):
    model = DeepLabWrapper().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

    # Repeat 4x to ensure batch_size=4 works with drop_last
    train_ds = SEMDataset(train_pairs, augment=True, repeat=4)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)

    for epoch in range(N_EPOCHS):
        model.train()
        for imgs, gts in train_dl:
            imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
            loss = dice_bce_loss(model(imgs), gts)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

    model.eval()
    val_ds = SEMDataset([val_pair], augment=False)
    img, gt = val_ds[0]
    with torch.no_grad():
        pred = torch.sigmoid(model(img.unsqueeze(0).to(DEVICE))).cpu().numpy()[0, 0]
    return compute_metrics(pred, gt.numpy()[0]), model


def evaluate_on_test(model, test_pairs):
    model.eval()
    results = []
    for p in test_pairs:
        img = np.array(Image.open(p['raw']).convert('RGB'))
        gt = load_nist_gt(str(p['gt']))
        h_orig, w_orig = img.shape[:2]
        img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_t = torch.from_numpy(img_r.transpose(2, 0, 1).astype(np.float32) / 255.0)
        with torch.no_grad():
            pred = torch.sigmoid(model(img_t.unsqueeze(0).to(DEVICE))).cpu().numpy()[0, 0]
        pred_full = cv2.resize(pred, (w_orig, h_orig))
        m = compute_metrics(pred_full, gt)
        m['name'] = p['name']
        results.append(m)
    return results


def main():
    print("=" * 60)
    print("DeepLabv3 LOO CV (BatchNorm fix: repeat 4x, batch=4)")
    print("=" * 60)

    manual_pairs = get_manual_pairs()
    test_pairs = get_auto_test_pairs(30)
    print(f"Manual: {len(manual_pairs)}, Test: {len(test_pairs)}")

    loo_results = []
    best_model, best_iou = None, -1

    for fold_idx in range(len(manual_pairs)):
        val_pair = manual_pairs[fold_idx]
        train_pairs = [p for j, p in enumerate(manual_pairs) if j != fold_idx]
        print(f"  Fold {fold_idx+1}/{len(manual_pairs)}: val={val_pair['name']}", end=" ", flush=True)
        val_metrics, trained_model = train_one_fold(train_pairs, val_pair)
        loo_results.append(val_metrics)
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            best_model = trained_model
        print(f"-> IoU={val_metrics['iou']:.4f}, Dice={val_metrics['dice']:.4f}")

    loo_ious = [r['iou'] for r in loo_results]
    print(f"\n  LOO: IoU={np.mean(loo_ious):.4f} +/- {np.std(loo_ious):.4f}")

    print(f"  Testing on {len(test_pairs)} auto-GT images...")
    test_results = evaluate_on_test(best_model, test_pairs)
    test_ious = [r['iou'] for r in test_results]
    print(f"  Test: IoU={np.mean(test_ious):.4f} +/- {np.std(test_ious):.4f}")

    # Save combined results (merge with U-Net if exists)
    out_dir = OUTPUTS_DIR / "tables"
    deeplab_data = {
        'loo_summary': {
            'model': 'DeepLabv3', 'cv_type': 'LOO', 'n_folds': len(manual_pairs),
            'iou_mean': float(np.mean(loo_ious)), 'iou_std': float(np.std(loo_ious)),
            'iou_ci95': float(1.96 * np.std(loo_ious) / np.sqrt(len(loo_ious))),
            'dice_mean': float(np.mean([r['dice'] for r in loo_results])),
            'precision_mean': float(np.mean([r['precision'] for r in loo_results])),
            'recall_mean': float(np.mean([r['recall'] for r in loo_results])),
        },
        'loo_per_fold': loo_results,
        'test_summary': {
            'model': 'DeepLabv3', 'test_set': 'auto_GT_30',
            'iou_mean': float(np.mean(test_ious)), 'iou_std': float(np.std(test_ious)),
            'dice_mean': float(np.mean([r['dice'] for r in test_results])),
            'precision_mean': float(np.mean([r['precision'] for r in test_results])),
            'recall_mean': float(np.mean([r['recall'] for r in test_results])),
        },
        'test_per_image': test_results,
    }

    # Try to merge with existing results
    json_path = out_dir / "supervised_baselines_results.json"
    if json_path.exists():
        with open(json_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results['DeepLabv3'] = deeplab_data

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    rows = []
    for name, data in all_results.items():
        row = {**data['loo_summary'], 'test_iou': data['test_summary']['iou_mean']}
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "supervised_baselines_results.csv", index=False)

    print(f"\n[Saved] supervised_baselines_results.json / .csv")


if __name__ == "__main__":
    main()
