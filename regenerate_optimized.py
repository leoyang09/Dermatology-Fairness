import os
import glob
import cv2
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from generate_preprocessed import (
    apply_illumination_comp, apply_bilateral_filter,
    apply_non_local_means, apply_msrcr, apply_adaptive_gamma,
    apply_percentile_norm
)

IMAGE_DIR = os.path.join("DDI", "images")
image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
n_cores = cpu_count()

combos = [
    ("non_local_means+msrcr",            [lambda img: apply_non_local_means(img, h=14),                                                                         lambda img: apply_msrcr(img)]),
    ("bilateral+illumination_comp",      [lambda img: apply_bilateral_filter(img, sigma_color=38.17611636124784),                                                lambda img: apply_illumination_comp(img, sigma=10.468560122629444)]),
    ("bilateral+msrcr",                  [lambda img: apply_bilateral_filter(img, sigma_color=95.41493828699667),                                                lambda img: apply_msrcr(img)]),
    ("illumination_comp+adaptive_gamma", [lambda img: apply_illumination_comp(img, sigma=64.49739202603473),                                                     lambda img: apply_adaptive_gamma(img)]),
    ("illumination_comp",                [lambda img: apply_illumination_comp(img, sigma=36.144687286473655)]),
    ("percentile_norm",                  [lambda img: apply_percentile_norm(img, range_=80.01213439643897)]),
    ("bilateral",                        [lambda img: apply_bilateral_filter(img, sigma_color=57.79324817677015)]),
]

for combo_name, funcs in combos:
    out_dir = os.path.join("DDI", f"images_{combo_name}_optimized")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating {combo_name}...")

    def process(img_path):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            return f"Failed to read {img_path}"
        for func in funcs:
            img = func(img)
        cv2.imwrite(os.path.join(out_dir, filename), img)

    with ThreadPool(processes=n_cores) as pool:
        pool.map(process, image_paths)

    print(f"  Saved to {out_dir}")

print("Done.")
