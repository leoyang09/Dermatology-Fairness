import os
import subprocess
import pickle
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import f1_score, balanced_accuracy_score

# -------------------------------
# Configuration
# -------------------------------
preprocessed_dirs = [
    #"images_bilateral+illumination_comp_optimized",
    #"images_illumination_comp_optimized",
    #"images_illumination_comp+local_contrast_optimized",
    "images_illumination_comp+adaptive_gamma_optimized",
    #"images_z_score_norm+adaptive_gamma_optimized",
    #"images_z_score_norm+percentile_norm_optimized",
    #"images_bilateral+msrcr_optimized",
    "images_msrcr+local_contrast_optimized",
    #"images_clahe+adaptive_gamma_optimized",
    #"images_non_local_means+msrcr_optimized",
    "images_non_local_means+msrcr+adaptive_gamma_optimized",
    #images_non_local_means+illumination_comp+clahe_optimized",
    #"images"
]

data_dir = "DDI"
weights_paths = {
    "HAM10000": "DDI-models/ham10000.pth",
    #"DeepDerm": "DDI-models/deepderm.pth"
}

use_gpu = True
plot = True  # plots will be saved automatically, not shown interactively
summary_csv = "DDI_eval_summary.csv"
max_workers = 3  # Adjust to CPU/GPU resources

# -------------------------------
# Evaluation function
# -------------------------------
def run_eval(model_name, weights_path, img_dir):
    eval_dir = f"DDI-results_{img_dir}_{model_name}"
    os.makedirs(eval_dir, exist_ok=True)

    python_exe = r"C:\Users\LeoYa\GitHub\DDI-Code\.venv\Scripts\python.exe"
    cmd = [
        python_exe, "eval_ddi.py",
        "--model", model_name,
        "--weights_path", weights_path,
        "--data_dir", data_dir,
        "--img_dir", img_dir,
        "--eval_dir", eval_dir
    ]
    if use_gpu:
        cmd.append("--use_gpu")
    if plot:
        cmd.append("--plot")  # ensure eval_ddi.py saves plot as file

    print(f"Starting: Model={model_name}, Images={img_dir}")
    subprocess.run(cmd, check=True)  # raises if eval_ddi.py fails

    eval_file = os.path.join(eval_dir, f"{model_name}-evaluation.pkl")
    if os.path.exists(eval_file):
        with open(eval_file, "rb") as f:
            eval_results = pickle.load(f)

        return {
            "model": model_name,
            "preprocessing": img_dir,
            "ROC_AUC": eval_results.get("ROC_AUC"),
            "F1": f1_score(
                eval_results["true_labels"],
                (eval_results["predicted_labels"] > eval_results["threshold"]).astype(int)
            ),
            "Balanced_Acc": balanced_accuracy_score(
                eval_results["true_labels"],
                (eval_results["predicted_labels"] > eval_results["threshold"]).astype(int)
            )
        }
    else:
        print(f"Warning: Evaluation file missing for {model_name} on {img_dir}")
        return None

# -------------------------------
# Main execution (Windows-safe)
# -------------------------------
def main():
    results_summary = []

    # create all tasks
    tasks = [(m, w, d) for m, w in weights_paths.items() for d in preprocessed_dirs]

    # run evaluations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_eval, *task): task for task in tasks}

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result is not None:
                    results_summary.append(result)
            except Exception as e:
                print(f"Task {task} failed: {e}")

    # save summary CSV
    df = pd.DataFrame(results_summary)
    df.to_csv(summary_csv, index=False)
    print(f"\nSummary CSV saved to {summary_csv}")
    print(df)

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    main()