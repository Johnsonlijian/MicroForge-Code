#!/usr/bin/env python3
"""Generate calibration analysis figure from saved JSON data."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

out = Path(__file__).parent.parent.parent / "outputs"
with open(out / "tables/calibration_analysis.json") as f:
    data = json.load(f)

bl = data['baseline']
lr = data['lora']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Key calibration metrics
labels = ['Pred IoU\n(SAM internal)', 'True IoU\n(vs GT)', 'ECE', 'Correlation']
bl_vals = [bl['pred_iou_mean'], bl['true_iou_mean'], bl['ece'], bl['correlation']]
lr_vals = [lr['pred_iou_mean'], lr['true_iou_mean'], lr['ece'], lr['correlation']]
x = np.arange(len(labels))
w = 0.35
bl_n = bl['n_masks']
lr_n = lr['n_masks']
axes[0].bar(x - w/2, bl_vals, w, label=f'Baseline (n={bl_n})', color='steelblue', alpha=0.8)
axes[0].bar(x + w/2, lr_vals, w, label=f'LoRA (n={lr_n})', color='coral', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=9)
axes[0].set_ylabel('Value')
axes[0].set_title('Calibration Metrics Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=0, color='k', linewidth=0.5)

# Panel 2: Mask count comparison
categories = ['N masks\n(total)', 'Masks/image', 'True IoU\n(per mask)']
bl_v2 = [bl_n, bl_n/30, bl['true_iou_mean']]
lr_v2 = [lr_n, lr_n/30, lr['true_iou_mean']]
x2 = np.arange(len(categories))
bars1 = axes[1].bar(x2 - w/2, bl_v2, w, label='Baseline', color='steelblue', alpha=0.8)
bars2 = axes[1].bar(x2 + w/2, lr_v2, w, label='LoRA', color='coral', alpha=0.8)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(categories, fontsize=9)
axes[1].set_title('Protocol Tension: Mask Generation')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# Panel 3: Overconfidence illustration
ax3 = axes[2]
ax3.barh(['Predicted IoU\n(SAM says)', 'True IoU\n(actual)'],
         [bl['pred_iou_mean'], bl['true_iou_mean']],
         color=['#e74c3c', '#2ecc71'], alpha=0.8, height=0.4)
ax3.set_xlim(0, 1.1)
gap = bl['pred_iou_mean'] / bl['true_iou_mean']
ax3.set_title(f'Baseline SAM Overconfidence\n(pred={bl["pred_iou_mean"]:.3f} vs true={bl["true_iou_mean"]:.4f})')
ax3.axvline(x=0.88, color='gray', linestyle='--', label='AMG pred_iou_thresh')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='x')
ax3.annotate(f'{gap:.0f}x gap!',
             xy=(bl['true_iou_mean'], 1), xytext=(0.5, 0.8),
             fontsize=14, fontweight='bold', color='red',
             arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
fig.savefig(out / 'figures/Fig_Calibration_Analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out / 'figures/Fig_Calibration_Analysis.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print('Saved Fig_Calibration_Analysis.png/.pdf')
print(f'Baseline: {bl_n} masks, pred_iou={bl["pred_iou_mean"]:.3f}, true_iou={bl["true_iou_mean"]:.4f}')
print(f'LoRA: {lr_n} masks, pred_iou={lr["pred_iou_mean"]:.3f}, true_iou={lr["true_iou_mean"]:.4f}')
print(f'Overconfidence gap: {gap:.0f}x')
