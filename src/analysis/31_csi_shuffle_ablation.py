#!/usr/bin/env python3
"""
P1-6: CSI-Shuffle Ablation -- Prove Physics Prior Is Not Coincidental
=====================================================================
Ultra-lightweight: precompute E[edge-hit] for all (image, age) combos,
then permutation test is just a lookup table shuffle.
"""

import os, json, sys, time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from skimage.filters import sobel

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ASRC_SEM_DIR = DATA_DIR / "raw" / "my_asrc" / "SEM"

XRD_DATA = {
    1:  {"AFt": 5.46},
    3:  {"AFt": 14.67},
    7:  {"AFt": 11.96},
    14: {"AFt": 0.21},
    21: {"AFt": 0.00},
    28: {"AFt": 0.53},
}
ALL_AGES = sorted(XRD_DATA.keys())
N_PERMUTATIONS = 10000
MAX_DIM = 960


def compute_csi(age_days):
    idx = ALL_AGES.index(age_days)
    if idx == 0:
        return 0.0
    prev = ALL_AGES[idx - 1]
    aft_change = abs(XRD_DATA[age_days]["AFt"] - XRD_DATA[prev]["AFt"])
    aft_max = max(d["AFt"] for d in XRD_DATA.values())
    return aft_change / aft_max * 0.12


def main():
    t0 = time.time()
    print("=" * 60)
    print("  CSI-SHUFFLE ABLATION (Lookup-Table Method)")
    print("=" * 60)
    sys.stdout.flush()

    images_info = []
    for age in [14, 28]:
        for scale in ["1000x", "2000x", "5000x", "10000x", "30000x"]:
            p = ASRC_SEM_DIR / f"SEM_ASRC_{age}d_{scale}.jpg"
            if p.exists():
                images_info.append({"path": p, "true_age": age, "scale": scale})

    n_img = len(images_info)
    print(f"  Images: {n_img}, Ages: {ALL_AGES}")
    print(f"  CSI values: {[round(compute_csi(a), 4) for a in ALL_AGES]}")
    sys.stdout.flush()

    # Precompute E[edge-hit] for ALL (image, age) combinations
    print(f"\n[Phase 1] Precompute edge-hit lookup table ({n_img} x {len(ALL_AGES)})...")
    sys.stdout.flush()

    lookup = np.zeros((n_img, len(ALL_AGES)), dtype=np.float32)

    for i, info in enumerate(images_info):
        gray = np.array(Image.open(info["path"]).convert("L"))
        h, w = gray.shape
        if max(h, w) > MAX_DIM:
            sc = MAX_DIM / max(h, w)
            gray = cv2.resize(gray, (int(w * sc), int(h * sc)))

        # Image-dependent maps (compute once per image)
        edge = sobel(gray.astype(np.float32) / 255.0).astype(np.float32)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        tex = cv2.GaussianBlur(np.abs(lap), (15, 15), 0)
        tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-8)
        canny = cv2.Canny(gray, 30, 100)
        itz_map = cv2.dilate(canny, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0
        edge_binary = (cv2.dilate(canny, np.ones((10, 10), np.uint8)) > 0).astype(np.float32)

        for j, age in enumerate(ALL_AGES):
            csi = compute_csi(age)
            physics = itz_map * (1 + csi * 10)
            mx = physics.max()
            if mx > 0:
                physics = physics / mx
            importance = 0.3 * edge + 0.3 * tex + 0.3 * physics + 0.1
            lookup[i, j] = np.sum(importance * edge_binary) / (np.sum(importance) + 1e-10) * 100

        # Free image arrays
        del edge, tex, itz_map, edge_binary, lap, canny, gray

        true_age_idx = ALL_AGES.index(info["true_age"])
        print(f"    [{i+1}/{n_img}] {info['path'].name}: E[hit|true]={lookup[i, true_age_idx]:.2f}%")
        sys.stdout.flush()

    # True CSI results
    true_hits = []
    for i, info in enumerate(images_info):
        true_age_idx = ALL_AGES.index(info["true_age"])
        true_hits.append(float(lookup[i, true_age_idx]))
    true_mean = np.mean(true_hits)
    print(f"\n  TRUE mean E[edge-hit]: {true_mean:.2f}%")
    print(f"  Precompute time: {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # Permutation test (ultra-fast: just index lookups)
    print(f"\n[Phase 2] Permutation test ({N_PERMUTATIONS} shuffles, lookup-only)...")
    sys.stdout.flush()
    t1 = time.time()

    rng = np.random.RandomState(123)
    shuffled_means = np.zeros(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        age_indices = rng.randint(0, len(ALL_AGES), size=n_img)
        hits = [float(lookup[i, age_indices[i]]) for i in range(n_img)]
        shuffled_means[p] = np.mean(hits)

    p_value = float((shuffled_means >= true_mean).sum()) / N_PERMUTATIONS
    effect_size = (true_mean - shuffled_means.mean()) / (shuffled_means.std() + 1e-8)

    print(f"  Permutation time: {time.time()-t1:.1f}s")
    print(f"\n  Results:")
    print(f"    Shuffled mean: {shuffled_means.mean():.2f}% +/- {shuffled_means.std():.2f}%")
    print(f"    True mean:     {true_mean:.2f}%")
    print(f"    p-value:       {p_value:.4f}")
    print(f"    Cohen's d:     {effect_size:.2f}")
    if p_value < 0.05:
        print(f"    => SIGNIFICANT (p < 0.05)")
    elif p_value < 0.10:
        print(f"    => Marginally significant (p < 0.10)")
    else:
        print(f"    => Not significant at alpha=0.05")
    sys.stdout.flush()

    # Save
    out_dir = OUTPUTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "analytical_lookup_table",
        "n_images": n_img,
        "n_permutations": N_PERMUTATIONS,
        "true_edge_hit_mean": round(true_mean, 2),
        "true_edge_hit_per_image": [
            {"age": info["true_age"], "scale": info["scale"],
             "csi": round(compute_csi(info["true_age"]), 4),
             "expected_edge_hit_pct": round(h, 2)}
            for info, h in zip(images_info, true_hits)
        ],
        "lookup_table": {
            "ages": ALL_AGES,
            "images": [f"{info['true_age']}d_{info['scale']}" for info in images_info],
            "values": lookup.tolist(),
        },
        "shuffled_edge_hit_mean": round(float(shuffled_means.mean()), 2),
        "shuffled_edge_hit_std": round(float(shuffled_means.std()), 2),
        "p_value": round(p_value, 4),
        "cohens_d": round(effect_size, 2),
    }
    with open(out_dir / "csi_shuffle_ablation.json", "w") as f:
        json.dump(result, f, indent=2)

    rpt = OUTPUTS_DIR / "reports"
    rpt.mkdir(parents=True, exist_ok=True)
    with open(rpt / "CSI_Shuffle_Result.txt", "w") as f:
        f.write("CSI-Shuffle Ablation (Lookup-Table Method)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Images: {n_img}, Permutations: {N_PERMUTATIONS}\n")
        f.write(f"True E[hit]: {true_mean:.2f}%\n")
        f.write(f"Shuffled: {shuffled_means.mean():.2f}% +/- {shuffled_means.std():.2f}%\n")
        f.write(f"p-value: {p_value:.4f}, Cohen's d: {effect_size:.2f}\n")
        sig = "SIGNIFICANT" if p_value < 0.05 else ("Marginal" if p_value < 0.1 else "Not significant")
        f.write(f"Conclusion: {sig}\n")

    print(f"\n[Saved] csi_shuffle_ablation.json, CSI_Shuffle_Result.txt")
    print(f"Total time: {time.time()-t0:.1f}s")
    print("Done!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
