#!/usr/bin/env python3
"""Convert all plot_data JSON files to user-friendly CSV tables."""

import json, csv, os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOT = os.path.join(BASE, "PAPER_PACKAGE", "plot_data")


def load(name):
    with open(os.path.join(PLOT, name), encoding="utf-8") as f:
        return json.load(f)


def write_csv(name, header, rows):
    path = os.path.join(PLOT, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> {name}  ({len(rows)} rows)")


def convert_calibration():
    """calibration_corrected.json -> 3 CSVs"""
    d = load("calibration_corrected.json")

    # 1) Summary
    b, l = d["baseline"], d["lora"]
    header = ["metric", "baseline", "lora"]
    rows = [
        ["n_masks", b["n_masks"], l["n_masks"]],
        ["pred_iou_mean_raw", b["pred_iou_mean"], l["pred_iou_mean"]],
        ["pred_iou_mean_clamped", b["pred_iou_mean_clamped"], l["pred_iou_mean_clamped"]],
        ["pred_iou_std", b["pred_iou_std"], l["pred_iou_std"]],
        ["true_iou_mean", b["true_iou_mean"], l["true_iou_mean"]],
        ["true_iou_std", b["true_iou_std"], l["true_iou_std"]],
        ["ECE", b["ece"], l["ece"]],
        ["overconfidence_ratio", b["overconfidence_ratio"], f'{l["true_iou_mean"]:.4f}' if l["true_iou_mean"] else ""],
        ["pearson_r", b["pearson_r"], "NaN"],
        ["spearman_rho", b["spearman_rho"], "NaN"],
        ["deterministic_max_diff", d["deterministic_check"]["max_iou_diff_between_runs"], ""],
    ]
    write_csv("calibration_summary.csv", header, rows)

    # 2) Reliability bins
    bins = b["reliability_bins"]
    header = ["bin_lo", "bin_hi", "count", "avg_predicted_iou", "avg_true_iou"]
    rows = [[x["bin_lo"], x["bin_hi"], x["count"],
             x["avg_pred"] if x["avg_pred"] is not None else "",
             x["avg_true"] if x["avg_true"] is not None else ""] for x in bins]
    write_csv("calibration_reliability_bins.csv", header, rows)

    # 3) Pareto summary
    header = ["model", "iou_thresh", "stab_thresh", "avg_masks_per_image", "avg_iou"]
    rows = [
        ["Baseline", d["pareto_baseline_best"]["iou_thresh"],
         d["pareto_baseline_best"]["stab_thresh"],
         d["pareto_baseline_best"]["avg_masks"],
         d["pareto_baseline_best"]["avg_iou"]],
        ["LoRA", d["pareto_lora_best"]["iou_thresh"],
         d["pareto_lora_best"]["stab_thresh"],
         d["pareto_lora_best"]["avg_masks"],
         d["pareto_lora_best"]["avg_iou"]],
    ]
    write_csv("calibration_pareto_best.csv", header, rows)


def convert_unified():
    """unified_verification.json -> per-image CSV with all 3 methods side by side"""
    d = load("unified_verification.json")
    a_imgs = d["per_image"]["config_a"]
    b_imgs = d["per_image"]["config_b"]
    c_imgs = d["per_image"]["config_c"]

    header = [
        "image",
        "baseline_amg_iou", "baseline_amg_dice", "baseline_amg_precision", "baseline_amg_recall",
        "uniform_N36_iou", "uniform_N36_dice", "uniform_N36_precision", "uniform_N36_recall",
        "microscopy_aware_N36_iou", "microscopy_aware_N36_dice", "microscopy_aware_N36_precision", "microscopy_aware_N36_recall",
    ]
    rows = []
    for a, b, c in zip(a_imgs, b_imgs, c_imgs):
        rows.append([
            a["name"],
            a["iou"], a["dice"], a["precision"], a["recall"],
            b["iou"], b["dice"], b["precision"], b["recall"],
            c["iou"], c["dice"], c["precision"], c["recall"],
        ])

    write_csv("unified_verification_per_image.csv", header, rows)

    # Summary row
    header2 = ["method", "n", "iou_mean", "iou_std", "dice_mean", "precision_mean", "recall_mean"]
    rows2 = []
    for key, label in [("config_a_baseline_amg", "Baseline SAM (AMG)"),
                        ("config_b_uniform_points", "Uniform Points N=36"),
                        ("config_c_adaptive_prompting", "Microscopy-Aware N=36")]:
        s = d[key]
        rows2.append([label, s["n"], s["iou_mean"], s["iou_std"],
                       s["dice_mean"], s["precision_mean"], s["recall_mean"]])
    write_csv("unified_verification_summary.csv", header2, rows2)


def convert_supervised_stats():
    """supervised_stats_corrected.json -> 2 CSVs"""
    d = load("supervised_stats_corrected.json")

    # Summary
    header = ["model", "loo_n", "loo_iou_mean", "loo_iou_std", "loo_iou_ci95",
              "test_n", "test_iou_mean", "test_iou_std", "test_iou_ci95"]
    rows = []
    for model in ["U-Net", "DeepLabv3"]:
        s = d["supervised_loo_stats"][model]
        rows.append([model, s["loo_n"], s["loo_iou_mean"], s["loo_iou_std"], s["loo_iou_ci95"],
                     s["test_n"], s["test_iou_mean"], s["test_iou_std"], s["test_iou_ci95"]])
    write_csv("supervised_loo_summary.csv", header, rows)

    # Manual-auto per image
    ma = d["manual_auto_consistency"]
    header = ["sample", "iou", "dice", "manual_px", "auto_px"]
    rows = [[x["sample"], x["iou"], x["dice"], x["manual_px"], x["auto_px"]]
            for x in ma["per_image"]]
    rows.append(["MEAN", ma["iou_mean"], ma["dice_mean"], "", ""])
    rows.append(["STD", ma["iou_std"], ma["dice_std"], "", ""])
    rows.append(["CI95_halfwidth", ma["iou_ci95"], "", "", ""])
    write_csv("manual_auto_gt_per_image.csv", header, rows)


def convert_prompt_efficiency():
    """prompt_efficiency.json -> CSV"""
    d = load("prompt_efficiency.json")
    header = ["n_points", "uniform_iou_mean", "uniform_iou_std",
              "adaptive_iou_mean", "adaptive_iou_std", "improvement_pct"]
    rows = [[r["n_points"], r["uniform_iou_mean"], r["uniform_iou_std"],
             r["adaptive_iou_mean"], r["adaptive_iou_std"], r["improvement_pct"]]
            for r in d["results"]]
    rows.append(["AUC", d["auc_uniform"], "", d["auc_adaptive"], "", d["auc_advantage_pct"]])
    write_csv("prompt_efficiency_budget.csv", header, rows)


def convert_csi_shuffle():
    """csi_shuffle_ablation.json -> 2 CSVs"""
    d = load("csi_shuffle_ablation.json")

    # Per-image
    header = ["age_days", "scale", "CSI", "expected_edge_hit_pct"]
    key = "expected_edge_hit_pct" if "expected_edge_hit_pct" in d["true_edge_hit_per_image"][0] else "edge_hit_pct"
    rows = [[x["age"], x["scale"], x["csi"], x[key]]
            for x in d["true_edge_hit_per_image"]]
    rows.append(["MEAN", "", "", d["true_edge_hit_mean"]])
    write_csv("csi_shuffle_per_image.csv", header, rows)

    # Lookup table (if present)
    if "lookup_table" in d:
        lt = d["lookup_table"]
        ages = lt["ages"]
        header = ["image"] + [f"age_{a}d_edge_hit_pct" for a in ages]
        rows = []
        for i, img in enumerate(lt["images"]):
            rows.append([img] + [round(v, 2) for v in lt["values"][i]])
        write_csv("csi_shuffle_lookup_table.csv", header, rows)

    # Summary
    header = ["metric", "value"]
    rows = [
        ["n_images", d["n_images"]],
        ["n_permutations", d["n_permutations"]],
        ["true_edge_hit_mean", d["true_edge_hit_mean"]],
        ["shuffled_edge_hit_mean", d["shuffled_edge_hit_mean"]],
        ["shuffled_edge_hit_std", d["shuffled_edge_hit_std"]],
        ["p_value", d["p_value"]],
        ["cohens_d", d["cohens_d"]],
    ]
    write_csv("csi_shuffle_summary.csv", header, rows)


def convert_lora():
    """lora_verification_real.json -> 2 CSVs"""
    d = load("lora_verification_real.json")

    # Summary
    header = ["method", "n", "iou_mean", "iou_std", "dice_mean", "dice_std",
              "precision_mean", "recall_mean"]
    rows = []
    for key, label in [("baseline_amg", "Baseline AMG"), ("baseline_point", "Baseline Point"),
                        ("lora_amg", "LoRA AMG"), ("lora_point", "LoRA Point")]:
        s = d[key]
        rows.append([label, s["n"], s["iou_mean"], s["iou_std"],
                     s["dice_mean"], s["dice_std"], s["precision_mean"], s["recall_mean"]])
    rows.append(["", "", "", "", "", "", "", ""])
    rows.append(["LoRA params", d["param_info"]["lora"], "", "", "", "", "", ""])
    rows.append(["Total params", d["param_info"]["total"], "", "", "", "", "", ""])
    rows.append(["Param ratio (%)", round(d["param_info"]["ratio"], 4), "", "", "", "", "", ""])
    rows.append(["AMG improvement (%)", round(d["amg_improvement_pct"], 2), "", "", "", "", "", ""])
    rows.append(["Point improvement (%)", round(d["point_improvement_pct"], 2), "", "", "", "", "", ""])
    write_csv("lora_summary.csv", header, rows)

    # Training loss curve
    header = ["epoch", "loss"]
    rows = [[i + 1, round(loss, 6)] for i, loss in enumerate(d["training_loss"])]
    write_csv("lora_training_loss.csv", header, rows)


def convert_clahe():
    """clahe_baseline_results.json -> per-image CSV"""
    d = load("clahe_baseline_results.json")

    methods = ["Raw (no preprocessing)", "CLAHE (clip=2.0)", "CLAHE (clip=4.0)", "CLAHE + Denoise"]
    n_images = len(d[methods[0]]["per_image"])

    header = ["image"]
    for m in methods:
        short = m.replace(" (no preprocessing)", "").replace(" ", "_")
        header += [f"{short}_iou", f"{short}_dice", f"{short}_precision", f"{short}_recall"]

    rows = []
    for i in range(n_images):
        row = [d[methods[0]]["per_image"][i]["name"]]
        for m in methods:
            img = d[m]["per_image"][i]
            row += [img["iou"], img["dice"], img["precision"], img["recall"]]
        rows.append(row)

    write_csv("clahe_comparison_per_image.csv", header, rows)

    # Summary
    header2 = ["method", "n", "iou_mean", "iou_std", "dice_mean", "precision_mean", "recall_mean"]
    rows2 = []
    for m in methods:
        s = d[m]["summary"]
        rows2.append([m, s["n"], s["iou_mean"], s["iou_std"],
                      s["dice_mean"], s["precision_mean"], s["recall_mean"]])
    write_csv("clahe_comparison_summary.csv", header2, rows2)


def convert_supervised_baselines():
    """supervised_baselines_results.json -> per-fold CSV"""
    d = load("supervised_baselines_results.json")

    for model in ["U-Net", "DeepLabv3"]:
        md = d[model]
        folds = md["loo_per_fold"]
        header = ["fold", "loo_iou", "loo_dice"]
        rows = []
        for i, f in enumerate(folds):
            label = f.get("name", f"fold_{i+1}")
            row = [label, f["iou"], f["dice"]]
            rows.append(row)
        s = md["loo_summary"]
        rows.append(["MEAN", s["iou_mean"], s["dice_mean"]])
        rows.append(["STD", s["iou_std"], ""])
        rows.append(["CI95_halfwidth", s["iou_ci95"], ""])

        # Test per-image (if available)
        test_imgs = md.get("test_per_image", [])
        if test_imgs:
            header2 = ["image", "test_iou", "test_dice", "test_precision", "test_recall"]
            rows2 = [[t.get("name", f"img_{j+1}"), t["iou"], t["dice"],
                       t.get("precision", ""), t.get("recall", "")]
                      for j, t in enumerate(test_imgs)]
            ts = md["test_summary"]
            rows2.append(["MEAN", ts["iou_mean"], ts.get("dice_mean", ""),
                          ts.get("precision_mean", ""), ts.get("recall_mean", "")])
            rows2.append(["STD", ts["iou_std"], "", "", ""])
            safe = model.replace("-", "")
            write_csv(f"supervised_{safe}_test_per_image.csv", header2, rows2)

        safe = model.replace("-", "")
        write_csv(f"supervised_{safe}_loo_per_fold.csv", header, rows)


if __name__ == "__main__":
    print("=" * 50)
    print("  Converting all JSON to CSV")
    print("=" * 50)

    print("\n[1/8] calibration_corrected.json")
    convert_calibration()

    print("\n[2/8] unified_verification.json")
    convert_unified()

    print("\n[3/8] supervised_stats_corrected.json")
    convert_supervised_stats()

    print("\n[4/8] prompt_efficiency.json")
    convert_prompt_efficiency()

    print("\n[5/8] csi_shuffle_ablation.json")
    convert_csi_shuffle()

    print("\n[6/8] lora_verification_real.json")
    convert_lora()

    print("\n[7/8] clahe_baseline_results.json")
    convert_clahe()

    print("\n[8/8] supervised_baselines_results.json")
    convert_supervised_baselines()

    print("\n" + "=" * 50)
    print("  All done! CSV files saved to plot_data/")
    print("=" * 50)
