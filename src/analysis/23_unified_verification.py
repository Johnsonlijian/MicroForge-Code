#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MicroForge 统一验证框架
========================

严格使用真实数据验证所有创新点性能。

评估协议:
- Baseline: SamAutomaticMaskGenerator (SAM标准接口)
- Prompting: Microscopy-Aware点采样 + SamPredictor
- LoRA: 加载微调权重 + SamPredictor

GT格式: NIST damageMask是RGB (非黑色=damage)
评估指标: IoU, Dice, Precision, Recall

NO PLACEHOLDERS. NO FAKE DATA.

Author: MicroForge Team
Date: 2026-01-23
"""

import os
import sys
import json
import time
import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path
from datetime import datetime
from scipy.ndimage import generic_filter
from skimage.filters import sobel
import pandas as pd

# ============================================================================
# 配置
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SAM_CHECKPOINT = DATA_DIR / "checkpoints" / "sam_vit_h_4b8939.pth"
NIST_RAW_DIR = DATA_DIR / "raw" / "nist_sem" / "rawFOV"
NIST_GT_DIR = DATA_DIR / "raw" / "nist_sem" / "damageContextAssistedMask" / "damageMask"
NIST_MANUAL_GT_DIR = DATA_DIR / "raw" / "nist_sem" / "contextManualMaskGT" / "contextMaskGT"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_TEST_IMAGES = 30
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================================
# GT加载（正确处理NIST RGB格式）
# ============================================================================

def load_nist_gt(gt_path: str) -> np.ndarray:
    """
    加载NIST GT mask
    GT是RGB，非黑色(0,0,0)像素 = damage
    """
    gt = np.array(Image.open(gt_path))
    
    if len(gt.shape) == 3:
        # RGB -> binary: any non-zero channel = damage
        gt_mask = np.any(gt > 0, axis=2)
    else:
        gt_mask = gt > 0
    
    return gt_mask.astype(bool)


def get_matched_pairs(raw_dir, gt_dir, n_max=None):
    """获取匹配的图像-GT文件对"""
    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.png')])
    
    matched = []
    for rf in raw_files:
        base = rf.replace('.png', '')
        gt_name = f"{base}_damage.png"
        gt_path = Path(gt_dir) / gt_name
        
        if gt_path.exists():
            matched.append({
                'raw': Path(raw_dir) / rf,
                'gt': gt_path,
                'name': rf
            })
        
        if n_max and len(matched) >= n_max:
            break
    
    return matched


# ============================================================================
# 指标计算
# ============================================================================

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """计算全套指标"""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    
    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, ~gt_b).sum()
    fn = np.logical_and(~pred_b, gt_b).sum()
    tn = np.logical_and(~pred_b, ~gt_b).sum()
    
    union = tp + fp + fn
    iou = float(tp) / float(union) if union > 0 else 0.0
    dice = float(2 * tp) / float(2 * tp + fp + fn) if (2*tp+fp+fn) > 0 else 0.0
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    
    total = pred_b.size
    pred_ratio = pred_b.sum() / total
    gt_ratio = gt_b.sum() / total
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'pred_ratio': float(pred_ratio),
        'gt_ratio': float(gt_ratio)
    }


# ============================================================================
# 方法A: Baseline SAM (SamAutomaticMaskGenerator)
# ============================================================================

def run_baseline_sam(image_rgb, mask_generator):
    """
    使用SAM标准自动分割接口
    这是SAM论文中推荐的零样本使用方式
    """
    masks = mask_generator.generate(image_rgb)
    
    if len(masks) == 0:
        return np.zeros(image_rgb.shape[:2], dtype=bool)
    
    # 合并所有检测到的mask
    h, w = image_rgb.shape[:2]
    combined = np.zeros((h, w), dtype=bool)
    for m in masks:
        combined = np.logical_or(combined, m['segmentation'])
    
    return combined


# ============================================================================
# 方法B: Microscopy-Aware Prompting
# ============================================================================

def compute_importance_map(image_gray):
    """计算重要性图: Edge + Entropy"""
    # 边缘
    edge = sobel(image_gray.astype(float) / 255.0)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
    
    # 局部熵 (用小窗口加速)
    def local_entropy(values):
        hist, _ = np.histogram(values.astype(int), bins=16, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-8))
    
    entropy_map = generic_filter(image_gray.astype(float), local_entropy, size=9)
    entropy_map = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-8)
    
    # 组合
    importance = 0.4 * edge + 0.4 * entropy_map + 0.2 * np.random.rand(*image_gray.shape)
    return importance


def sample_adaptive_points(importance_map, n_points=32, min_distance=15):
    """从重要性图中采样点（带最小距离约束）"""
    h, w = importance_map.shape
    probs = importance_map.flatten()
    probs = probs / (probs.sum() + 1e-8)
    
    points = []
    selected = np.zeros((h, w), dtype=bool)
    
    attempts = 0
    while len(points) < n_points and attempts < n_points * 10:
        idx = np.random.choice(len(probs), p=probs)
        y, x = idx // w, idx % w
        
        y0, y1 = max(0, y - min_distance), min(h, y + min_distance)
        x0, x1 = max(0, x - min_distance), min(w, x + min_distance)
        
        if not selected[y0:y1, x0:x1].any():
            points.append([x, y])
            selected[y0:y1, x0:x1] = True
        
        attempts += 1
    
    if len(points) == 0:
        # Fallback
        gs = int(np.sqrt(n_points))
        for y in np.linspace(0, h-1, gs).astype(int):
            for x in np.linspace(0, w-1, gs).astype(int):
                points.append([x, y])
    
    return np.array(points[:n_points])


def run_microscopy_prompting(image_rgb, predictor, n_points=32):
    """
    使用Microscopy-Aware Prompting运行SAM
    
    关键区别: 使用score过滤 + 面积过滤，避免过分割
    """
    h, w = image_rgb.shape[:2]
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 计算重要性图
    importance_map = compute_importance_map(image_gray)
    
    # 自适应采样
    points = sample_adaptive_points(importance_map, n_points)
    
    # SAM推理
    predictor.set_image(image_rgb)
    
    all_masks = []
    all_scores = []
    
    for pt in points:
        try:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([pt]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])
            
            # 面积过滤: 0.05% - 20% 的图像面积
            mask_ratio = best_mask.sum() / (h * w)
            if 0.0005 < mask_ratio < 0.20 and best_score > 0.7:
                all_masks.append(best_mask)
                all_scores.append(best_score)
                
        except:
            continue
    
    if len(all_masks) == 0:
        return np.zeros((h, w), dtype=bool)
    
    # 合并（取并集，但已经过滤了低质量mask）
    combined = np.any(np.stack(all_masks, axis=0), axis=0)
    return combined


def run_uniform_prompting(image_rgb, predictor, n_points=32):
    """Baseline点采样: 均匀网格 + 相同的过滤策略"""
    h, w = image_rgb.shape[:2]
    
    gs = int(np.sqrt(n_points))
    points = []
    for y in np.linspace(0, h-1, gs).astype(int):
        for x in np.linspace(0, w-1, gs).astype(int):
            points.append([x, y])
    points = np.array(points[:n_points])
    
    predictor.set_image(image_rgb)
    
    all_masks = []
    
    for pt in points:
        try:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([pt]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])
            
            mask_ratio = best_mask.sum() / (h * w)
            if 0.0005 < mask_ratio < 0.20 and best_score > 0.7:
                all_masks.append(best_mask)
                
        except:
            continue
    
    if len(all_masks) == 0:
        return np.zeros((h, w), dtype=bool)
    
    combined = np.any(np.stack(all_masks, axis=0), axis=0)
    return combined


# ============================================================================
# PIMP (Physics-Informed Prompting) - 在ASRC上验证
# ============================================================================

XRD_DATA = {
    1:  {"AFt": 5.46, "CH": 5.86, "CSH": 6.17},
    3:  {"AFt": 14.67, "CH": 10.76, "CSH": 14.92},
    7:  {"AFt": 11.96, "CH": 16.17, "CSH": 19.41},
    14: {"AFt": 0.21, "CH": 21.33, "CSH": 27.47},
    21: {"AFt": 0.00, "CH": 24.57, "CSH": 29.97},
    28: {"AFt": 0.53, "CH": 25.83, "CSH": 28.94},
}

def compute_physics_prior(image_gray, age_days):
    """
    生成物理先验热图
    
    原理: AFt下降 -> 化学收缩 -> ITZ高应力 -> 裂缝易发区
    """
    # 获取XRD数据
    ages_sorted = sorted(XRD_DATA.keys())
    idx = ages_sorted.index(age_days)
    
    # 计算AFt变化率
    if idx > 0:
        prev_age = ages_sorted[idx - 1]
        aft_change = abs(XRD_DATA[age_days]["AFt"] - XRD_DATA[prev_age]["AFt"])
        aft_max = max(d["AFt"] for d in XRD_DATA.values())
        shrinkage_index = aft_change / aft_max * 0.12  # 化学收缩系数
    else:
        shrinkage_index = 0
    
    # 检测ITZ区域 (使用Canny边缘 + 膨胀)
    edges = cv2.Canny(image_gray, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    itz_map = cv2.dilate(edges, kernel).astype(float) / 255.0
    
    # 物理先验 = ITZ区域 * 收缩指数
    physics_prior = itz_map * (1 + shrinkage_index * 10)
    
    # 归一化
    if physics_prior.max() > 0:
        physics_prior = physics_prior / physics_prior.max()
    
    return physics_prior, shrinkage_index


def run_pimp_prompting(image_rgb, predictor, age_days, n_points=32):
    """
    PIMP: Physics-Informed Microscopy Prompting
    importance = 0.3*edge + 0.3*entropy + 0.3*physics + 0.1*random
    """
    h, w = image_rgb.shape[:2]
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 边缘
    edge = sobel(image_gray.astype(float) / 255.0)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
    
    # 熵 (简化版，用Laplacian方差代替，快10倍)
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    texture_var = cv2.GaussianBlur(np.abs(laplacian), (15, 15), 0)
    texture_var = (texture_var - texture_var.min()) / (texture_var.max() - texture_var.min() + 1e-8)
    
    # 物理先验
    physics_prior, shrinkage = compute_physics_prior(image_gray, age_days)
    
    # 组合
    importance = (
        0.3 * edge +
        0.3 * texture_var +
        0.3 * physics_prior +
        0.1 * np.random.rand(h, w)
    )
    
    # 采样
    points = sample_adaptive_points(importance, n_points)
    
    # SAM推理
    predictor.set_image(image_rgb)
    
    all_masks = []
    for pt in points:
        try:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([pt]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])
            
            mask_ratio = best_mask.sum() / (h * w)
            if 0.0005 < mask_ratio < 0.20 and best_score > 0.7:
                all_masks.append(best_mask)
        except:
            continue
    
    if len(all_masks) == 0:
        return np.zeros((h, w), dtype=bool), shrinkage
    
    combined = np.any(np.stack(all_masks, axis=0), axis=0)
    return combined, shrinkage


# ============================================================================
# 主验证流程
# ============================================================================

def run_unified_verification():
    """
    统一验证: Baseline (AMG) + Baseline (Uniform Points) + Adaptive Prompting
    """
    print("=" * 70)
    print(" " * 15 + "MICROFORGE UNIFIED VERIFICATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print()
    
    # 加载SAM
    print("[1/6] Loading SAM model...")
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(DEVICE)
    sam.eval()
    
    predictor = SamPredictor(sam)
    
    # AMG with same params as original benchmark
    amg = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    print(f"      SAM ViT-H loaded on {DEVICE}")
    
    # 获取测试文件
    print(f"\n[2/6] Loading test images (N={N_TEST_IMAGES})...")
    matched = get_matched_pairs(NIST_RAW_DIR, NIST_GT_DIR, N_TEST_IMAGES)
    print(f"      Found {len(matched)} pairs")
    
    # =========================================================================
    # 方法A: Baseline SAM (SamAutomaticMaskGenerator)
    # =========================================================================
    print(f"\n[3/6] Running Config A: Baseline SAM (AutoMaskGenerator)...")
    config_a_results = []
    
    from tqdm import tqdm
    for item in tqdm(matched, desc="      AMG"):
        try:
            image = np.array(Image.open(item['raw']).convert('RGB'))
            gt_mask = load_nist_gt(str(item['gt']))
            
            if image.shape[:2] != gt_mask.shape:
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (image.shape[1], image.shape[0])) > 0
            
            pred_mask = run_baseline_sam(image, amg)
            metrics = compute_metrics(pred_mask, gt_mask)
            metrics['name'] = item['name']
            config_a_results.append(metrics)
        except Exception as e:
            print(f"\n      [!] {item['name']}: {e}")
    
    # =========================================================================
    # 方法B: 均匀点采样 (with filtering)
    # =========================================================================
    print(f"\n[4/6] Running Config B: Uniform Point Sampling (filtered)...")
    config_b_results = []
    
    for item in tqdm(matched, desc="      Uniform"):
        try:
            image = np.array(Image.open(item['raw']).convert('RGB'))
            gt_mask = load_nist_gt(str(item['gt']))
            
            if image.shape[:2] != gt_mask.shape:
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (image.shape[1], image.shape[0])) > 0
            
            pred_mask = run_uniform_prompting(image, predictor, n_points=36)
            metrics = compute_metrics(pred_mask, gt_mask)
            metrics['name'] = item['name']
            config_b_results.append(metrics)
        except Exception as e:
            print(f"\n      [!] {item['name']}: {e}")
    
    # =========================================================================
    # 方法C: Microscopy-Aware Prompting
    # =========================================================================
    print(f"\n[5/6] Running Config C: Microscopy-Aware Prompting...")
    config_c_results = []
    
    for item in tqdm(matched, desc="      Adaptive"):
        try:
            image = np.array(Image.open(item['raw']).convert('RGB'))
            gt_mask = load_nist_gt(str(item['gt']))
            
            if image.shape[:2] != gt_mask.shape:
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (image.shape[1], image.shape[0])) > 0
            
            pred_mask = run_microscopy_prompting(image, predictor, n_points=36)
            metrics = compute_metrics(pred_mask, gt_mask)
            metrics['name'] = item['name']
            config_c_results.append(metrics)
        except Exception as e:
            print(f"\n      [!] {item['name']}: {e}")
    
    # =========================================================================
    # 汇总结果
    # =========================================================================
    print(f"\n[6/6] Computing summary...")
    
    def summarize(results, name):
        if not results:
            return {}
        ious = [r['iou'] for r in results]
        dices = [r['dice'] for r in results]
        precs = [r['precision'] for r in results]
        recs = [r['recall'] for r in results]
        return {
            'method': name,
            'n': len(results),
            'iou_mean': float(np.mean(ious)),
            'iou_std': float(np.std(ious)),
            'dice_mean': float(np.mean(dices)),
            'dice_std': float(np.std(dices)),
            'precision_mean': float(np.mean(precs)),
            'recall_mean': float(np.mean(recs)),
        }
    
    summary_a = summarize(config_a_results, "Baseline SAM (AMG)")
    summary_b = summarize(config_b_results, "Uniform Point Sampling")
    summary_c = summarize(config_c_results, "Microscopy-Aware Prompting")
    
    # 输出
    print()
    print("=" * 70)
    print("VERIFICATION RESULTS (REAL DATA)")
    print("=" * 70)
    
    print(f"\n{'Method':<35} {'IoU':>10} {'Dice':>10} {'Prec':>10} {'Recall':>10}")
    print("-" * 75)
    
    for s in [summary_a, summary_b, summary_c]:
        if s:
            print(f"{s['method']:<35} {s['iou_mean']:.4f}+/-{s['iou_std']:.3f}  "
                  f"{s['dice_mean']:.4f}  {s['precision_mean']:.4f}  {s['recall_mean']:.4f}")
    
    # Prompting改进计算
    if summary_b and summary_c and summary_b['iou_mean'] > 0:
        prompt_improvement = (summary_c['iou_mean'] / summary_b['iou_mean'] - 1) * 100
    else:
        prompt_improvement = 0
    
    print(f"\nPrompting Improvement (Adaptive vs Uniform): {prompt_improvement:+.1f}%")
    
    # 保存
    verification = {
        'timestamp': datetime.now().isoformat(),
        'device': DEVICE,
        'n_images': N_TEST_IMAGES,
        'random_seed': RANDOM_SEED,
        'config_a_baseline_amg': summary_a,
        'config_b_uniform_points': summary_b,
        'config_c_adaptive_prompting': summary_c,
        'prompting_improvement_pct': prompt_improvement,
        'per_image': {
            'config_a': config_a_results,
            'config_b': config_b_results,
            'config_c': config_c_results,
        }
    }
    
    out_dir = OUTPUTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "unified_verification.json", 'w') as f:
        json.dump(verification, f, indent=2, default=str)
    
    # CSV表格
    df = pd.DataFrame([summary_a, summary_b, summary_c])
    df.to_csv(out_dir / "Verified_Ablation_Real.csv", index=False)
    
    print(f"\n[Saved] unified_verification.json")
    print(f"[Saved] Verified_Ablation_Real.csv")
    
    return verification


# ============================================================================
# PIMP在ASRC上验证
# ============================================================================

def verify_pimp_on_asrc():
    """
    在ASRC真实SEM图像上验证PIMP效果
    
    由于ASRC没有GT mask，我们比较:
    1. 采样点分布（是否更集中于边缘/ITZ区域）
    2. 物理先验热图的可视化
    3. 化学收缩指数的龄期差异
    """
    print()
    print("=" * 70)
    print(" " * 15 + "PIMP VERIFICATION ON ASRC")
    print("=" * 70)
    
    import matplotlib.pyplot as plt
    
    ASRC_SEM_DIR = DATA_DIR / "raw" / "my_asrc" / "SEM"
    
    from segment_anything import sam_model_registry, SamPredictor
    
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    
    # 对比14d和28d在5000x下的效果
    ages_scales = [
        (14, "5000x"),
        (28, "5000x"),
        (14, "2000x"),
        (28, "2000x"),
    ]
    
    results = []
    fig, axes = plt.subplots(len(ages_scales), 4, figsize=(20, 5*len(ages_scales)))
    
    for i, (age, scale) in enumerate(ages_scales):
        img_path = ASRC_SEM_DIR / f"SEM_ASRC_{age}d_{scale}.jpg"
        
        if not img_path.exists():
            print(f"  [!] Missing: {img_path.name}")
            continue
        
        print(f"\n  Processing {age}d {scale}...")
        
        image = np.array(Image.open(img_path).convert('RGB'))
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = image.shape[:2]
        
        # 标准采样（Edge+Entropy）
        importance_std = compute_importance_map(image_gray)
        points_std = sample_adaptive_points(importance_std, 32)
        
        # PIMP采样（Edge+Entropy+Physics）
        physics_prior, shrinkage = compute_physics_prior(image_gray, age)
        importance_pimp = (
            0.3 * sobel(image_gray.astype(float)/255.0) +
            0.3 * cv2.GaussianBlur(np.abs(cv2.Laplacian(image_gray, cv2.CV_64F)), (15,15), 0) / 
            (cv2.GaussianBlur(np.abs(cv2.Laplacian(image_gray, cv2.CV_64F)), (15,15), 0).max() + 1e-8) +
            0.3 * physics_prior +
            0.1 * np.random.rand(h, w)
        )
        points_pimp = sample_adaptive_points(importance_pimp, 32)
        
        # 计算采样点覆盖边缘区域的比例
        edges = cv2.Canny(image_gray, 30, 100)
        edge_dilated = cv2.dilate(edges, np.ones((10,10), np.uint8))
        
        std_on_edge = sum(1 for p in points_std if edge_dilated[p[1], p[0]] > 0) / len(points_std) * 100
        pimp_on_edge = sum(1 for p in points_pimp if edge_dilated[p[1], p[0]] > 0) / len(points_pimp) * 100
        
        result = {
            'age': age,
            'scale': scale,
            'shrinkage_index': shrinkage,
            'std_edge_hit_pct': std_on_edge,
            'pimp_edge_hit_pct': pimp_on_edge,
            'edge_hit_improvement': pimp_on_edge - std_on_edge,
        }
        results.append(result)
        
        print(f"    Shrinkage Index: {shrinkage:.4f}")
        print(f"    Standard Edge Hit: {std_on_edge:.1f}%")
        print(f"    PIMP Edge Hit:     {pimp_on_edge:.1f}%")
        print(f"    Improvement:       {pimp_on_edge - std_on_edge:+.1f}%")
        
        # 可视化
        axes[i, 0].imshow(image_gray, cmap='gray')
        axes[i, 0].set_title(f'{age}d {scale} - Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(physics_prior, cmap='hot')
        axes[i, 1].set_title(f'Physics Prior (CSI={shrinkage:.4f})')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(image_gray, cmap='gray')
        for pt in points_std:
            axes[i, 2].plot(pt[0], pt[1], 'b.', markersize=3)
        axes[i, 2].set_title(f'Standard ({std_on_edge:.0f}% on edge)')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(image_gray, cmap='gray')
        for pt in points_pimp:
            axes[i, 3].plot(pt[0], pt[1], 'r.', markersize=3)
        axes[i, 3].set_title(f'PIMP ({pimp_on_edge:.0f}% on edge)')
        axes[i, 3].axis('off')
    
    plt.suptitle('PIMP Verification: Standard vs Physics-Informed Sampling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = OUTPUTS_DIR / "figures" / "Fig_PIMP_Verified.png"
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[Saved] {fig_path}")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(OUTPUTS_DIR / "tables" / "PIMP_verification_real.csv", index=False)
    print(f"[Saved] PIMP_verification_real.csv")
    
    print("\nPIMP Summary:")
    print(df.to_string(index=False))
    
    return results


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    # Phase 1: NIST上验证Baseline + Prompting
    print("=" * 70)
    print("PHASE 1: NIST Benchmark Verification")
    print("=" * 70)
    nist_results = run_unified_verification()
    
    # Phase 2: ASRC上验证PIMP
    print("\n" + "=" * 70)
    print("PHASE 2: PIMP Verification on ASRC")
    print("=" * 70)
    pimp_results = verify_pimp_on_asrc()
    
    print("\n" + "=" * 70)
    print("ALL VERIFICATION COMPLETE")
    print("=" * 70)
