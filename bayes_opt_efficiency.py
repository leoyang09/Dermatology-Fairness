"""
Bayesian Optimization for High-Impact Preprocessing Combinations
Only methods/combinations with >20% gap improvement in HAM10000 or DeepDerm
Requires: pip install optuna opencv-python
"""
import os
import glob
import shutil
import tempfile
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import optuna
import cv2
import numpy as np
import pandas as pd
import torch
from generate_preprocessed import method_map, process_image, process_image_pair, process_image_triple
from eval_ddi import load_model, DDI_Dataset, eval_model
from PIL import Image

# -------------------------------
# Preprocessing methods and parameter ranges
#
# Only methods whose baseline functions actually accept a tunable parameter
# are listed here with their correct parameter names matching the function
# signatures in generate_preprocessed.py.
#
# Fixed methods with no tunable parameter (white_balance, adaptive_gamma, msrcr)
# can still appear in combos but will use their default behaviour.
# -------------------------------

preprocessing_methods = {
    "clahe":             {"clip_limit":  (0.5, 4.0)},    # apply_clahe(clip_limit)
    "illumination_comp": {"sigma":       (10.0, 100.0)},  # apply_illumination_comp(sigma)
    "percentile_norm":   {"range_":      (80.0, 99.0)},   # apply_percentile_norm(range_)
    "local_contrast":    {"strength":    (0.1, 2.0)},     # apply_local_contrast(strength)
    "bilateral":         {"sigma_color": (10.0, 150.0)},  # apply_bilateral_filter(sigma_color)
    "non_local_means":   {"h":           (5, 15)},        # apply_non_local_means(h)
}

# Slow methods — reduce n_trials automatically for combos containing these
SLOW_METHODS = {}
N_TRIALS_DEFAULT = 20
N_TRIALS_SLOW    = 10

# -------------------------------
# Only methods/combinations with >20% gap reduction
# Use "+" as separator between methods (method names contain underscores)
# -------------------------------
high_impact_combos = [
    # Single methods
    # "illumination_comp",
    # "percentile_norm",
    # "local_contrast",
    # "bilateral",
    # "clahe",

    # Combinations — use + as separator
    # "illumination_comp+local_contrast",
    # "illumination_comp+adaptive_gamma",
    # "illumination_comp+clahe",
    # "clahe+adaptive_gamma",
    # "non_local_means+illumination_comp",
    # "Z_score_norm+percentile_norm",
    # "bilateral+Z_score_norm",
    # "bilateral+msrcr+Z_score_norm",
    "bilateral+illumination_comp",
    # "non_local_means+msrcr+adaptive_gamma",
    # "non_local_means+illumination_comp+clahe",
    # "non_local_means+illumination_comp+percentile_norm",
]

# Detect GPU once at startup
USE_GPU = torch.cuda.is_available()
print(f"GPU available: {USE_GPU}")
N_CORES = cpu_count()
print(f"CPU cores available: {N_CORES}")


# -------------------------------
# Picklable callable class
# -------------------------------
class PreprocessTransform:
    """
    Applies a sequence of preprocessing steps to an image then runs
    test_transform. Defined at module level so it can be pickled by
    multiprocessing workers.

    steps: list of (method_name, param_value) tuples.
           param_value is None for fixed methods.
    """
    def __init__(self, steps: list):
        self.steps = steps

    def _apply_method(self, cv_img, method_name, param_value):
        if method_name == "clahe":
            return method_map["clahe"](cv_img, clip_limit=param_value)
        elif method_name == "illumination_comp":
            return method_map["illumination_comp"](cv_img, sigma=param_value)
        elif method_name == "percentile_norm":
            return method_map["percentile_norm"](cv_img, range_=param_value)
        elif method_name == "local_contrast":
            return method_map["local_contrast"](cv_img, strength=param_value)
        elif method_name == "bilateral":
            return method_map["bilateral"](cv_img, sigma_color=param_value)
        elif method_name == "non_local_means":
            return method_map["non_local_means"](cv_img, h=param_value)
        else:
            # Fixed methods (white_balance, adaptive_gamma, msrcr, etc.)
            return method_map.get(method_name, lambda x: x)(cv_img)

    def __call__(self, img):
        from ddi_dataset import test_transform
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for method_name, param_value in self.steps:
            cv_img = self._apply_method(cv_img, method_name, param_value)
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        return test_transform(pil_img)


# -------------------------------
# Evaluate preprocessed dataset and extract skin-tone AUCs
# -------------------------------
def run_evaluation(dataset, model, num_workers=4):
    results = eval_model(model, dataset, use_gpu=USE_GPU, show_plot=False,
                         num_workers=num_workers)

    df = pd.DataFrame({
        "DDI_file": [os.path.basename(p) for p in results["images"]],
        "y_score":  results["predicted_labels"]
    })
    meta = pd.read_csv(os.path.join("DDI", "ddi_metadata.csv"))
    meta["y_true"] = meta["malignant"].astype(str).str.lower().map({"true": 1, "false": 0})

    df = df.merge(meta[["DDI_file", "skin_tone", "y_true"]], on="DDI_file", how="inner")
    df["skin_tone"] = pd.to_numeric(df["skin_tone"], errors="coerce")
    df["y_score"]   = pd.to_numeric(df["y_score"],   errors="coerce")
    df = df.dropna(subset=["skin_tone", "y_score", "y_true"])

    def auc_sub(df_sub):
        if df_sub["y_true"].nunique() < 2:
            return np.nan
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(df_sub["y_true"], df_sub["y_score"])

    auc_12      = auc_sub(df[df["skin_tone"] == 12])
    auc_56      = auc_sub(df[df["skin_tone"] == 56])
    overall_auc = auc_sub(df)

    return auc_12, auc_56, overall_auc


# -------------------------------
# Objective function generator
# -------------------------------
def make_objective(combo, model):
    split_methods = combo.split("+")

    def objective(trial):
        trial_params = {}
        for m in split_methods:
            if m not in preprocessing_methods:
                continue
            for param, (low, high) in preprocessing_methods[m].items():
                key = f"{m}_{param}"
                if isinstance(low, int) and isinstance(high, int):
                    trial_params[key] = trial.suggest_int(key, low, high)
                else:
                    trial_params[key] = trial.suggest_float(key, low, high)

        # Build steps list
        steps = []
        for m in split_methods:
            if m not in preprocessing_methods:
                steps.append((m, None))
                continue
            param_key   = list(preprocessing_methods[m].keys())[0]
            param_value = trial_params[f"{m}_{param_key}"]
            steps.append((m, param_value))

        tmp_dir     = tempfile.mkdtemp()
        tmp_img_dir = os.path.join(tmp_dir, "images")
        os.makedirs(tmp_img_dir)

        try:
            helper      = PreprocessTransform(steps)
            image_paths = glob.glob(os.path.join("DDI", "images", "*.png"))

            if len(image_paths) == 0:
                print("  ERROR: No images found in DDI/images/ — check path")
                return float("inf")

            # Parallel preprocessing using ThreadPool
            def preprocess_one(img_path):
                filename = os.path.basename(img_path)
                cv_img = cv2.imread(img_path)
                if cv_img is None:
                    print(f"  Failed to READ: {img_path}")
                    return
                for method_name, param_value in steps:
                    cv_img = helper._apply_method(cv_img, method_name, param_value)
                cv2.imwrite(os.path.join(tmp_img_dir, filename), cv_img)

            with ThreadPool(processes=N_CORES) as pool:
                pool.map(preprocess_one, image_paths)

            from ddi_dataset import test_transform
            dataset = DDI_Dataset(
                tmp_dir,
                img_dirname="images",
                csv_path=os.path.join("DDI", "ddi_metadata.csv"),
                transform=test_transform,
            )

            auc_12, auc_56, overall_auc = run_evaluation(dataset, model)

            gap     = auc_12 - auc_56
            penalty = max(0, 0.54 - overall_auc) * 2

            print(f"  params={trial_params}, auc_12={auc_12:.4f}, auc_56={auc_56:.4f}, "
                  f"overall={overall_auc:.4f}, gap={gap:.4f}")

            return gap + penalty

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return objective


# -------------------------------
# Run Bayesian Optimization
# -------------------------------
if __name__ == "__main__":
    model_name   = "HAM10000"  # or "DeepDerm"
    weights_path = f"DDI-models/{model_name.lower()}.pth"
    results      = []

    print("Loading model for evaluation...")
    model = load_model(model_name, weights_path=weights_path, save_dir="DDI-models", download=False)

    for combo in high_impact_combos:
        # Use fewer trials for combos containing slow methods
        split_methods = combo.split("+")
        n_trials = N_TRIALS_SLOW if any(m in SLOW_METHODS for m in split_methods) else N_TRIALS_DEFAULT

        print(f"\n=== Optimizing: {combo} (n_trials={n_trials}) ===")
        study = optuna.create_study(direction='minimize')
        study.optimize(make_objective(combo, model), n_trials=n_trials)
        print("Best params:", study.best_params)
        print(f"Best gap: {study.best_value:.4f}")
        results.append({"combo": combo, "best_params": study.best_params, "best_gap": study.best_value})

        # Save best params for this combo to CSV immediately after optimization
        csv_out_path = "optimization_results.csv"
        row = {"combo": combo, "best_gap": study.best_value}
        row.update(study.best_params)
        row_df = pd.DataFrame([row])
        if os.path.exists(csv_out_path):
            row_df.to_csv(csv_out_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(csv_out_path, mode="w", header=True, index=False)
        print(f"Best params appended to {csv_out_path}")
        
        # Save optimized images to disk
        print("Saving optimized images...")
        best_params   = study.best_params
        img_out_dir   = os.path.join("DDI", f"images_{combo}_optimized")
        os.makedirs(img_out_dir, exist_ok=True)

        image_paths   = glob.glob(os.path.join("DDI", "images", "*.png"))
        best_steps    = []
        for m in split_methods:
            if m not in preprocessing_methods:
                best_steps.append((m, None))
                continue
            param_key   = list(preprocessing_methods[m].keys())[0]
            param_value = best_params[f"{m}_{param_key}"]
            best_steps.append((m, param_value))

        helper = PreprocessTransform(best_steps)

        def save_image(img_path):
            filename = os.path.basename(img_path)
            try:
                cv_img = cv2.imread(img_path)
                if cv_img is None:
                    return f"Failed to read {img_path}"
                for method_name, param_value in best_steps:
                    cv_img = helper._apply_method(cv_img, method_name, param_value)
                cv2.imwrite(os.path.join(img_out_dir, filename), cv_img)
                return None
            except Exception as e:
                return f"{filename}: {str(e)}"

        with ThreadPool(processes=N_CORES) as pool:
            errors = list(pool.imap_unordered(save_image, image_paths))

        errors = [e for e in errors if e is not None]
        if errors:
            print(f"{len(errors)} images failed to save:")
            for e in errors[:10]:
                print(e)

    # Sort and print final results
    results = sorted(results, key=lambda x: x["best_gap"])
    print("\n=== Top Combinations ===")
    for r in results:
        print(f"Combo: {r['combo']}, Best Gap: {r['best_gap']:.4f}, Params: {r['best_params']}")
