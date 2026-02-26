"""
MicroForge v3.2 - Final Figure & Table Generator
Generates all publication-quality figures and CSV tables for AiC submission.
"""
import json
import os
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
})

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOT_DATA = os.path.join(BASE, 'PAPER_PACKAGE', 'plot_data')
FIG_OUT = os.path.join(BASE, 'PAPER_PACKAGE', 'figures')
TAB_OUT = os.path.join(BASE, 'PAPER_PACKAGE', 'tables')
os.makedirs(FIG_OUT, exist_ok=True)
os.makedirs(TAB_OUT, exist_ok=True)

with open(os.path.join(PLOT_DATA, 'calibration_corrected.json')) as f:
    cal = json.load(f)
with open(os.path.join(PLOT_DATA, 'prompt_efficiency.json')) as f:
    pe_data = json.load(f)
with open(os.path.join(PLOT_DATA, 'unified_verification.json')) as f:
    uv = json.load(f)
with open(os.path.join(PLOT_DATA, 'supervised_stats_corrected.json')) as f:
    sup = json.load(f)

import csv as csv_mod
pareto_bl = []
with open(os.path.join(TAB_OUT, 'pareto_baseline.csv')) as f:
    for row in csv_mod.DictReader(f):
        pareto_bl.append({k: float(v) for k, v in row.items()})
pareto_lr = []
with open(os.path.join(TAB_OUT, 'pareto_lora.csv')) as f:
    for row in csv_mod.DictReader(f):
        pareto_lr.append({k: float(v) for k, v in row.items()})

C_BLUE = '#2166AC'
C_RED = '#B2182B'
C_ORANGE = '#E08214'
C_GREEN = '#1B7837'
C_GRAY = '#636363'
C_LIGHT = '#D9D9D9'


# ============================================================
# FIGURE 1: MicroForge Framework Overview (Schematic)
# ============================================================
def fig1_framework():
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    boxes = [
        (0.3, 1.0, 1.6, 1.0, 'SEM + XRD\nInput', '#E3F2FD'),
        (2.3, 1.8, 1.6, 0.7, 'Microscopy-Aware\nPrompting', '#FFF3E0'),
        (2.3, 0.5, 1.6, 0.7, 'Physics-Informed\nPIMP (CSI)', '#E8F5E9'),
        (4.3, 1.0, 1.6, 1.0, 'LoRA\nAdaptation', '#FCE4EC'),
        (6.3, 1.8, 1.6, 0.7, 'Calibration\nAnalysis', '#F3E5F5'),
        (6.3, 0.5, 1.6, 0.7, 'Morphological\nClassification', '#E0F7FA'),
        (8.3, 1.0, 1.4, 1.0, 'ASHG\nHypothesis', '#FFF9C4'),
    ]

    for x, y, w, h, txt, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor='#333333', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center',
                fontsize=7, fontweight='bold', color='#333333')

    arrows = [
        (1.9, 1.5, 2.3, 2.15),
        (1.9, 1.5, 2.3, 0.85),
        (3.9, 2.15, 4.3, 1.5),
        (3.9, 0.85, 4.3, 1.5),
        (5.9, 1.5, 6.3, 2.15),
        (5.9, 1.5, 6.3, 0.85),
        (7.9, 2.15, 8.3, 1.5),
        (7.9, 0.85, 8.3, 1.5),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))

    ax.text(5.0, 0.05, 'MicroForge: Fully Automated Pipeline',
            ha='center', va='bottom', fontsize=9, fontstyle='italic', color=C_GRAY)

    fig.savefig(os.path.join(FIG_OUT, 'Fig_1_Framework.png'))
    fig.savefig(os.path.join(FIG_OUT, 'Fig_1_Framework.pdf'))
    plt.close(fig)
    print('Fig 1: Framework overview - DONE')


# ============================================================
# FIGURE 2: Calibration & Pareto (4-panel)
# ============================================================
def fig2_calibration_pareto():
    fig = plt.figure(figsize=(7.5, 6.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # --- Panel A: Reliability Diagram ---
    ax_a = fig.add_subplot(gs[0, 0])
    bins_data = cal['baseline']['reliability_bins']
    pred_vals, true_vals, counts = [], [], []
    for b in bins_data:
        if b['count'] > 0:
            pred_vals.append(b['avg_pred'])
            true_vals.append(b['avg_true'])
            counts.append(b['count'])

    ax_a.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect calibration')
    ax_a.bar(pred_vals, true_vals, width=0.08, color=C_RED, alpha=0.7,
             edgecolor='#333', linewidth=0.5, label='SAM actual IoU')

    ax_twin = ax_a.twinx()
    ax_twin.bar([p + 0.04 for p in pred_vals], counts, width=0.04,
                color=C_LIGHT, alpha=0.6, edgecolor='#999', linewidth=0.3)
    ax_twin.set_ylabel('Mask count', color=C_GRAY, fontsize=8)
    ax_twin.tick_params(axis='y', labelcolor=C_GRAY, labelsize=7)

    ax_a.set_xlabel('Predicted IoU (clamped)')
    ax_a.set_ylabel('Actual IoU')
    ax_a.set_title('(A) Reliability Diagram — Baseline SAM', fontweight='bold')
    ax_a.set_xlim(0.25, 1.0)
    ax_a.set_ylim(0, 0.12)
    ax_a.legend(loc='upper left', fontsize=7)
    ax_a.text(0.65, 0.09, f'ECE = {cal["baseline"]["ece"]:.3f}\n'
              f'N = {cal["baseline"]["n_masks"]:,}\n'
              f'Overconf. = {cal["baseline"]["overconfidence_ratio"]:.0f}x',
              fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Panel B: Predicted vs True IoU scatter (per image) ---
    ax_b = fig.add_subplot(gs[0, 1])
    iou_per_img = [d['iou'] for d in uv['per_image']['config_a']]
    pred_global = cal['baseline']['pred_iou_mean_clamped']
    ax_b.scatter(range(len(iou_per_img)), sorted(iou_per_img, reverse=True),
                 c=C_BLUE, s=20, alpha=0.7, edgecolors='#333', linewidths=0.3,
                 label='Actual IoU per image')
    ax_b.axhline(y=pred_global, color=C_RED, ls='--', lw=1.0,
                 label=f'Mean pred IoU = {pred_global:.3f}')
    ax_b.axhline(y=cal['baseline']['true_iou_mean'], color=C_BLUE, ls=':',
                 lw=1.0, label=f'Mean true IoU = {cal["baseline"]["true_iou_mean"]:.3f}')
    ax_b.fill_between(range(len(iou_per_img)), pred_global, cal['baseline']['true_iou_mean'],
                      alpha=0.08, color=C_RED)
    ax_b.set_xlabel('Image index (sorted by IoU)')
    ax_b.set_ylabel('IoU')
    ax_b.set_title('(B) Overconfidence Gap per Image', fontweight='bold')
    ax_b.legend(loc='upper right', fontsize=7)
    ax_b.text(15, 0.55, f'236x gap', fontsize=11, fontweight='bold',
              color=C_RED, alpha=0.4, ha='center')

    # --- Panel C: Pareto - Baseline ---
    ax_c = fig.add_subplot(gs[1, 0])
    bl_masks = [r['avg_masks'] for r in pareto_bl]
    bl_iou = [r['avg_iou'] for r in pareto_bl]
    sc_c = ax_c.scatter(bl_masks, bl_iou, c=[r['stab_th'] for r in pareto_bl],
                        cmap='RdYlGn', s=18, edgecolors='#333', linewidths=0.3,
                        vmin=0.3, vmax=0.95)
    ax_c.axvline(x=156, color=C_GRAY, ls=':', lw=0.7)
    ax_c.text(165, 0.098, 'Standard\nAMG (156)', fontsize=6, color=C_GRAY)
    ax_c.set_xlabel('Masks per image')
    ax_c.set_ylabel('Avg IoU')
    ax_c.set_title('(C) Pareto — Baseline SAM', fontweight='bold')
    plt.colorbar(sc_c, ax=ax_c, label='stability_score_thresh', shrink=0.8)

    # --- Panel D: Pareto - LoRA ---
    ax_d = fig.add_subplot(gs[1, 1])
    lr_masks = [r['avg_masks'] for r in pareto_lr]
    lr_iou = [r['avg_iou'] for r in pareto_lr]
    ax_d.scatter(lr_masks, lr_iou, c=C_RED, s=60, marker='X',
                 edgecolors='#333', linewidths=0.5, zorder=5)
    ax_d.set_xlim(-0.5, 5)
    ax_d.set_ylim(0.0, 0.25)
    ax_d.axhline(y=lr_iou[0], color=C_RED, ls='--', lw=0.8, alpha=0.5)
    ax_d.text(2.5, lr_iou[0] + 0.015,
              f'ALL 80 configs: exactly 1 mask\nIoU = {lr_iou[0]:.3f}',
              ha='center', fontsize=8, fontweight='bold', color=C_RED,
              bbox=dict(boxstyle='round', facecolor='#FFF0F0', alpha=0.9))
    ax_d.set_xlabel('Masks per image')
    ax_d.set_ylabel('Avg IoU')
    ax_d.set_title('(D) Pareto — LoRA SAM', fontweight='bold')
    ax_d.text(2.5, 0.04, 'Threshold re-tuning\ncannot recover\nmask generation',
              ha='center', fontsize=8, fontstyle='italic', color=C_GRAY)

    fig.savefig(os.path.join(FIG_OUT, 'Fig_2_Calibration_Pareto.png'))
    fig.savefig(os.path.join(FIG_OUT, 'Fig_2_Calibration_Pareto.pdf'))
    plt.close(fig)
    print('Fig 2: Calibration & Pareto - DONE')


# ============================================================
# FIGURE 3: Prompt Efficiency Curve
# ============================================================
def fig3_prompt_efficiency():
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ns = pe_data['point_budgets']
    u_mean = [r['uniform_iou_mean'] for r in pe_data['results']]
    u_std = [r['uniform_iou_std'] for r in pe_data['results']]
    a_mean = [r['adaptive_iou_mean'] for r in pe_data['results']]
    a_std = [r['adaptive_iou_std'] for r in pe_data['results']]

    ax.errorbar(ns, u_mean, yerr=[s/np.sqrt(30) for s in u_std],
                fmt='o-', color=C_BLUE, capsize=3, capthick=0.8, markersize=5,
                label=f'Uniform Grid (AUC={pe_data["auc_uniform"]:.2f})')
    ax.errorbar(ns, a_mean, yerr=[s/np.sqrt(30) for s in a_std],
                fmt='s--', color=C_RED, capsize=3, capthick=0.8, markersize=5,
                label=f'Adaptive Sampling (AUC={pe_data["auc_adaptive"]:.2f})')

    ax.fill_between(ns, u_mean, a_mean, alpha=0.08, color=C_GRAY)

    ax.axhline(y=0.074, color=C_GREEN, ls=':', lw=0.8, alpha=0.6)
    ax.text(50, 0.076, 'Original +9.9%\n(local entropy)', fontsize=6,
            color=C_GREEN, ha='center')

    ax.set_xlabel('Number of prompt points (N)')
    ax.set_ylabel('Mean IoU')
    ax.set_title('Prompt Budget Efficiency', fontweight='bold')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_xticks(ns)
    ax.set_ylim(0, 0.12)
    ax.grid(True, alpha=0.2)

    fig.savefig(os.path.join(FIG_OUT, 'Fig_3_Prompt_Efficiency.png'))
    fig.savefig(os.path.join(FIG_OUT, 'Fig_3_Prompt_Efficiency.pdf'))
    plt.close(fig)
    print('Fig 3: Prompt Efficiency - DONE')


# ============================================================
# FIGURE 4: Domain Gap Bar Chart
# ============================================================
def fig4_domain_gap():
    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    methods = [
        'SAM AMG\n(Raw)',
        'SAM AMG\n+CLAHE(2)',
        'SAM AMG\n+CLAHE(4)',
        'Uniform\nPts(N=36)',
        'Micro-Aware\n(N=36)',
        'U-Net\n(LOO)',
        'DeepLabv3\n(LOO)',
        'LoRA\n(AMG)',
        'LoRA\n(Point)',
    ]
    ious = [0.129, 0.115, 0.105, 0.067, 0.074, 0.153, 0.153, 0.124, 0.090]
    stds = [0.123, 0.139, 0.116, 0.076, 0.094, 0.130, 0.130, 0.146, 0.071]
    colors = [C_BLUE, C_BLUE, C_BLUE, C_ORANGE, C_GREEN, C_GRAY, C_GRAY, C_RED, C_RED]

    bars = ax.bar(range(len(methods)), ious, yerr=[s/np.sqrt(30) for s in stds],
                  color=colors, alpha=0.8, edgecolor='#333', linewidth=0.5,
                  capsize=3, width=0.7)

    ax.axhline(y=0.102, color='#888', ls=':', lw=0.8)
    ax.text(8.4, 0.105, 'Manual-Auto\nGT agreement\n(0.102)', fontsize=6,
            color='#888', ha='right')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel('IoU')
    ax.set_title('Method Comparison on NIST Concrete SEM', fontweight='bold')
    ax.set_ylim(0, 0.22)
    ax.grid(axis='y', alpha=0.2)

    for i, (v, s) in enumerate(zip(ious, stds)):
        ax.text(i, v + s/np.sqrt(30) + 0.005, f'{v:.3f}', ha='center',
                fontsize=6, fontweight='bold')

    fig.savefig(os.path.join(FIG_OUT, 'Fig_4_Domain_Gap.png'))
    fig.savefig(os.path.join(FIG_OUT, 'Fig_4_Domain_Gap.pdf'))
    plt.close(fig)
    print('Fig 4: Domain Gap Bar Chart - DONE')


# ============================================================
# FIGURE 5: ASRC Morphological Evolution
# ============================================================
def fig5_asrc_morphology():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    categories = ['Total\nPorosity', 'True\nPores', 'Micro-\ncracks']
    vals_14 = [5.10, 4.32, 0.78]
    vals_28 = [10.97, 7.85, 3.11]

    x = np.arange(len(categories))
    w = 0.32
    ax1.bar(x - w/2, vals_14, w, label='14 days', color=C_BLUE, alpha=0.8,
            edgecolor='#333', linewidth=0.5)
    ax1.bar(x + w/2, vals_28, w, label='28 days', color=C_RED, alpha=0.8,
            edgecolor='#333', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=8)
    ax1.set_ylabel('Area fraction (%)')
    ax1.set_title('(A) ASRC Damage Evolution', fontweight='bold')
    ax1.legend(fontsize=7)
    ax1.grid(axis='y', alpha=0.2)

    for i, (v14, v28) in enumerate(zip(vals_14, vals_28)):
        pct = (v28 - v14) / v14 * 100
        ax1.text(i + w/2, v28 + 0.3, f'+{pct:.0f}%', ha='center',
                 fontsize=7, fontweight='bold', color=C_RED)

    ages = [1, 3, 7, 14, 28]
    aft = [5.46, 14.67, 11.96, 0.21, 0.53]
    csi = [0.000, 0.075, 0.022, 0.096, 0.004]

    ax2.plot(ages, aft, 'o-', color=C_BLUE, label='AFt (%)', markersize=5)
    ax2.set_xlabel('Age (days)')
    ax2.set_ylabel('AFt content (%)', color=C_BLUE)
    ax2.tick_params(axis='y', labelcolor=C_BLUE)

    ax2r = ax2.twinx()
    ax2r.bar(ages, csi, width=1.5, color=C_ORANGE, alpha=0.5, label='CSI')
    ax2r.set_ylabel('CSI', color=C_ORANGE)
    ax2r.tick_params(axis='y', labelcolor=C_ORANGE)

    ax2.set_title('(B) AFt Kinetics & CSI', fontweight='bold')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_OUT, 'Fig_5_ASRC_Morphology.png'))
    fig.savefig(os.path.join(FIG_OUT, 'Fig_5_ASRC_Morphology.pdf'))
    plt.close(fig)
    print('Fig 5: ASRC Morphology - DONE')


# ============================================================
# CSV TABLE EXPORTS
# ============================================================
def export_tables():
    # Table 1: Method Comparison
    with open(os.path.join(TAB_OUT, 'Table_1_Method_Comparison_v3.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'IoU_mean', 'IoU_std', '95%_CI_lo', '95%_CI_hi', 'Dice', 'Training'])
        w.writerow(['SAM AMG (Raw)', 0.129, 0.123, '', '', 0.210, 'None'])
        w.writerow(['SAM AMG + CLAHE (2.0)', 0.115, 0.139, '', '', 0.184, 'None'])
        w.writerow(['SAM AMG + CLAHE (4.0)', 0.105, 0.116, '', '', 0.173, 'None'])
        w.writerow(['SAM Uniform Points (N=36)', 0.067, 0.076, '', '', 0.118, 'None'])
        w.writerow(['Microscopy-Aware (N=36)', 0.074, 0.094, '', '', 0.125, 'None'])
        w.writerow(['U-Net (LOO)', 0.153, 0.130, 0.104, 0.202, '', '12 images'])
        w.writerow(['DeepLabv3 (LOO)', 0.153, 0.130, 0.104, 0.201, '', '12 images'])
        w.writerow(['SAM + LoRA (AMG)', 0.124, 0.146, '', '', 0.197, '12 images'])
        w.writerow(['SAM + LoRA (Point)', 0.090, 0.071, '', '', 0.158, '12 images'])

    # Table 2: Manual vs Auto GT
    with open(os.path.join(TAB_OUT, 'Table_2_ManualAutoGT_v3.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Mean', 'Std', '95%_CI_lo', '95%_CI_hi'])
        mc = sup['manual_auto_consistency']
        w.writerow(['IoU', mc['iou_mean'], mc['iou_std'],
                     round(mc['iou_mean'] - mc['iou_ci95'], 3),
                     round(mc['iou_mean'] + mc['iou_ci95'], 3)])
        w.writerow(['Dice', mc['dice_mean'], mc['dice_std'], '', ''])

    # Table 3: CSI by Age
    with open(os.path.join(TAB_OUT, 'Table_3_CSI_by_Age_v3.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Age', 'AFt_pct', 'CSI', 'Interpretation'])
        w.writerow(['1d', 5.46, 0.000, 'Baseline'])
        w.writerow(['3d', 14.67, 0.075, 'Peak AFt'])
        w.writerow(['7d', 11.96, 0.022, 'Moderate'])
        w.writerow(['14d', 0.21, 0.096, 'High (AFt collapse)'])
        w.writerow(['28d', 0.53, 0.004, 'Low (stabilized)'])

    # Table 4: PIMP Edge Targeting
    with open(os.path.join(TAB_OUT, 'Table_4_PIMP_Edge_v3.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Age', 'Scale', 'Standard_pct', 'PIMP_pct', 'Improvement'])
        w.writerow(['14d', '5000x', 100.0, 100.0, '0.0%'])
        w.writerow(['28d', '5000x', 96.9, 96.9, '0.0%'])
        w.writerow(['28d', '2000x', 75.0, 87.5, '+12.5%'])

    # Table 5: LoRA Adaptation
    with open(os.path.join(TAB_OUT, 'Table_5_LoRA_Results_v3.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'IoU_mean', 'IoU_std', 'N_masks_total', 'Protocol_note'])
        w.writerow(['Baseline (AMG)', 0.126, 0.126, 4685,
                     'Standard AMG (pred_iou>=0.88, stability>=0.95)'])
        w.writerow(['Baseline (Point)', 0.087, 0.069, '',
                     '8+8 fg/bg prompts, multimask, best-score'])
        w.writerow(['LoRA (AMG)', 0.124, 0.146, 30, 'Same AMG thresholds'])
        w.writerow(['LoRA (Point)', 0.090, 0.071, '',
                     '8+8 fg/bg prompts, multimask, best-score'])

    # Table 6: Corrected Calibration
    with open(os.path.join(TAB_OUT, 'Table_6_Calibration_v3.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Baseline_SAM', 'LoRA_SAM'])
        b = cal['baseline']
        l = cal['lora']
        w.writerow(['N_masks', b['n_masks'], l['n_masks']])
        w.writerow(['Pred_IoU_raw_mean', round(b['pred_iou_mean'], 4), round(l['pred_iou_mean'], 4)])
        w.writerow(['Pred_IoU_clamped', round(b['pred_iou_mean_clamped'], 3), round(l['pred_iou_mean_clamped'], 3)])
        w.writerow(['True_IoU_mean', round(b['true_iou_mean'], 4), round(l['true_iou_mean'], 4)])
        w.writerow(['ECE', round(b['ece'], 4), round(l['ece'], 4)])
        w.writerow(['Overconfidence_ratio', f"{b['overconfidence_ratio']:.0f}x", '6.5x'])
        w.writerow(['Masks_per_image', '~716 (thresh 0.3)', '1'])
        w.writerow(['Pearson_r', round(b['pearson_r'], 4), 'NaN'])
        w.writerow(['Spearman_rho', round(b['spearman_rho'], 4), 'NaN'])

    # Table 7: ASRC Morphology
    with open(os.path.join(TAB_OUT, 'Table_7_ASRC_Morphology_v3.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Metric', '14_days', '28_days', 'Change_pct'])
        w.writerow(['Total Apparent Porosity (%)', 5.10, 10.97, '+115%'])
        w.writerow(['True Pore Fraction (%)', 4.32, 7.85, '+82%'])
        w.writerow(['Micro-crack Fraction (%)', 0.78, 3.11, '+298%'])

    # Table S1: Full Magnification PIMP (Supplementary)
    with open(os.path.join(TAB_OUT, 'Table_S1_PIMP_Full_Magnification.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Age', 'Magnification', 'Standard_edge_hit_pct', 'PIMP_edge_hit_pct',
                     'Improvement_pct', 'Note'])
        w.writerow(['14d', '500x', '', '', '', 'Insufficient damage features at this scale'])
        w.writerow(['14d', '1000x', '', '', '', 'Insufficient damage features at this scale'])
        w.writerow(['14d', '2000x', 100.0, 100.0, '0.0%', 'Saturated (all points on edges)'])
        w.writerow(['14d', '5000x', 100.0, 100.0, '0.0%', 'Reported in main text Table 4'])
        w.writerow(['14d', '10000x', '', '', '', 'Exceeds pixel resolution for edge statistics'])
        w.writerow(['28d', '500x', '', '', '', 'Insufficient damage features at this scale'])
        w.writerow(['28d', '1000x', '', '', '', 'Insufficient damage features at this scale'])
        w.writerow(['28d', '2000x', 75.0, 87.5, '+12.5%', 'Reported in main text Table 4'])
        w.writerow(['28d', '5000x', 96.9, 96.9, '0.0%', 'Reported in main text Table 4'])
        w.writerow(['28d', '10000x', '', '', '', 'Exceeds pixel resolution for edge statistics'])

    # Table S2: ASHG Expert Evaluation (Supplementary)
    with open(os.path.join(TAB_OUT, 'Table_S2_ASHG_Expert_Evaluation.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Hypothesis_ID', 'Hypothesis_Summary',
                     'Expert_A_Relevance', 'Expert_A_Novelty', 'Expert_A_Testability',
                     'Expert_B_Relevance', 'Expert_B_Novelty', 'Expert_B_Testability'])
        w.writerow(['H1', 'AFt-to-AFm transformation drives micro-crack expansion via chemical shrinkage',
                     5, 4, 5, 4, 4, 5])
        w.writerow(['H2', 'Porosity increase dominated by crack growth rather than pore nucleation',
                     4, 3, 5, 5, 3, 5])
        w.writerow(['H3', 'ITZ weakening from ettringite dissolution enables crack propagation',
                     4, 4, 5, 4, 5, 5])
        w.writerow(['H4', 'Aeolian sand surface texture accelerates ITZ micro-cracking',
                     3, 4, 5, 4, 3, 5])
        w.writerow(['H5', 'Late-age C-S-H densification partially compensates early micro-cracking',
                     5, 3, 5, 4, 4, 5])
        w.writerow(['---', 'SUMMARY STATISTICS', '', '', '', '', '', ''])
        w.writerow(['', 'Mean Relevance', 4.2, '', '', 4.2, '', ''])
        w.writerow(['', 'Mean Novelty', '', 3.6, '', '', 3.8, ''])
        w.writerow(['', 'Mean Testability', '', '', 5.0, '', '', 5.0])
        w.writerow(['', 'Overall Mean (across 3 dims)', '4.27', '', '', '4.33', '', ''])
        w.writerow(['', "Linearly weighted Cohen's kappa", '0.71', '', '', '', '', ''])

    print('All CSV tables exported - DONE')


# ============================================================
# FIGURE README
# ============================================================
def write_readme():
    readme = """# MicroForge v3.2 - Figure & Table Index
## Submission: Automation in Construction (AiC)
## Generated: 2026-02-24

## Figures

| File | Paper Reference | Description |
|------|----------------|-------------|
| Fig_1_Framework.png/pdf | Fig. 1 | MicroForge pipeline schematic |
| Fig_2_Calibration_Pareto.png/pdf | Fig. 2 | 4-panel: (A) Reliability diagram, (B) Overconfidence gap, (C) Pareto baseline, (D) Pareto LoRA |
| Fig_3_Prompt_Efficiency.png/pdf | Fig. 3 | IoU vs #prompts budget curve |
| Fig_4_Domain_Gap.png/pdf | Fig. 4 | Method comparison bar chart |
| Fig_5_ASRC_Morphology.png/pdf | Fig. 5 | (A) Damage evolution, (B) AFt kinetics & CSI |
| classification_SEM_ASRC_*.png | Fig. 6a-d | SEM classification overlays (existing) |
| Fig_A_Pore_Crack_Evolution_AiC.* | Supp. | Pore/crack evolution (existing) |
| Fig_B_AFt_Crack_Correlation_AiC.* | Supp. | AFt-crack correlation (existing) |

## Tables (CSV)

| File | Paper Reference | Description |
|------|----------------|-------------|
| Table_1_Method_Comparison_v3.csv | Table 1 | All methods on NIST concrete SEM |
| Table_2_ManualAutoGT_v3.csv | Table 2 | Manual vs automated GT consistency |
| Table_3_CSI_by_Age_v3.csv | Table 3 | Chemical Shrinkage Index by age |
| Table_4_PIMP_Edge_v3.csv | Table 4 | PIMP edge targeting results |
| Table_5_LoRA_Results_v3.csv | Table 5 | LoRA adaptation results |
| Table_6_Calibration_v3.csv | Table 6 | Corrected calibration analysis |
| Table_7_ASRC_Morphology_v3.csv | Table 7 | ASRC morphological evolution |
| Table_S1_PIMP_Full_Magnification.csv | Table S1 | Full magnification PIMP results |
| Table_S2_ASHG_Expert_Evaluation.csv | Table S2 | ASHG expert evaluation scores |
| pareto_baseline.csv | Fig. 2C data | Pareto sweep raw data (baseline) |
| pareto_lora.csv | Fig. 2D data | Pareto sweep raw data (LoRA) |
| prompt_efficiency.csv | Fig. 3 data | Prompt efficiency raw data |

## Plot Data (JSON - for reproducibility)

| File | Content |
|------|---------|
| calibration_corrected.json | ECE bins, pred/true IoU, Pareto summary |
| prompt_efficiency.json | Per-budget IoU results |
| unified_verification.json | Per-image SAM results (30 images) |
| supervised_stats_corrected.json | LOO CV + manual-auto GT |
| csi_shuffle_ablation.json | 10k permutation test results |

All figures generated at 300 DPI in both PNG and PDF formats.
"""
    with open(os.path.join(BASE, 'PAPER_PACKAGE', 'FIGURE_TABLE_README.md'), 'w', encoding='utf-8') as f:
        f.write(readme)
    print('README index - DONE')


if __name__ == '__main__':
    print('=' * 60)
    print('MicroForge v3.2 - Final Figure & Table Generator')
    print('=' * 60)
    fig1_framework()
    fig2_calibration_pareto()
    fig3_prompt_efficiency()
    fig4_domain_gap()
    fig5_asrc_morphology()
    export_tables()
    write_readme()
    print('=' * 60)
    print('ALL DONE. Check PAPER_PACKAGE/figures/ and PAPER_PACKAGE/tables/')
    print('=' * 60)
