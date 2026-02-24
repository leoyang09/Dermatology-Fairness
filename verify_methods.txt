"""
Verify that each preprocessing method in generate_preprocessed.py
produces output that matches the saved baseline images in DDI/.

Run from your project root:
    python verify_methods.py
"""
import cv2
import numpy as np
import os
from generate_preprocessed import (
    apply_clahe,
    apply_adaptive_gamma,
    apply_white_balance,
    apply_msrcr,
    apply_percentile_norm,
    apply_local_contrast,
    apply_illumination_comp,
    apply_z_score_norm,
    apply_bilateral_filter,
    apply_non_local_means,
)

# Map each method to:
#   - the function and its default parameters
#   - the folder where baseline images were saved
methods = {
    "clahe":             (lambda img: apply_clahe(img),                          "images_clahe"),
    "adaptive_gamma":    (lambda img: apply_adaptive_gamma(img),                 "images_adaptive_gamma"),
    "white_balance":     (lambda img: apply_white_balance(img),                  "images_white_balance"),
    "msrcr":             (lambda img: apply_msrcr(img),                          "images_msrcr"),
    "percentile_norm":   (lambda img: apply_percentile_norm(img),                "images_percentile_norm"),
    "local_contrast":    (lambda img: apply_local_contrast(img),                 "images_local_contrast"),
    "illumination_comp": (lambda img: apply_illumination_comp(img),              "images_illumination_comp"),
    "z_score_norm":      (lambda img: apply_z_score_norm(img),                   "images_Z_score_norm"),
    "bilateral":         (lambda img: apply_bilateral_filter(img),               "images_bilateral"),
    "non_local_means":   (lambda img: apply_non_local_means(img),                "images_non_local_means"),
}

# Use a few sample images for the check
DDI_DIR = "DDI"
sample_files = ["000001.png", "000002.png", "000003.png"]

print("=" * 60)
print("Verifying preprocessing methods against saved baselines")
print("=" * 60)

for method_name, (func, folder) in methods.items():
    baseline_dir = os.path.join(DDI_DIR, folder)

    if not os.path.exists(baseline_dir):
        print(f"\n[SKIP] {method_name}: no baseline folder found at {baseline_dir}")
        continue

    mismatches = 0
    missing = 0
    mean_diffs = []

    for filename in sample_files:
        orig_path = os.path.join(DDI_DIR, "images", filename)
        base_path = os.path.join(baseline_dir, filename)

        if not os.path.exists(orig_path):
            continue
        if not os.path.exists(base_path):
            missing += 1
            continue

        orig = cv2.imread(orig_path)
        base = cv2.imread(base_path)

        if orig is None or base is None:
            print(f"  Could not read image: {filename}")
            continue

        recomputed = func(orig)

        if not np.array_equal(recomputed, base):
            mismatches += 1
            mean_diffs.append(abs(recomputed.mean() - base.mean()))

    print(f"\n[{'OK' if mismatches == 0 else 'MISMATCH'}] {method_name}")
    if mismatches > 0:
        print(f"  {mismatches}/{len(sample_files)} images differ")
        print(f"  Avg pixel mean difference: {np.mean(mean_diffs):.2f}")
        print(f"  --> Function has been modified since baseline was generated")
    if missing > 0:
        print(f"  {missing} sample files not found in baseline folder")

print("\n" + "=" * 60)
print("Done. Any [MISMATCH] methods need their baseline regenerated")
print("or their function reverted to the original version.")
print("=" * 60)