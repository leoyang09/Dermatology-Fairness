import os
import pickle
import subprocess

# -------------------------------
# Configuration
# -------------------------------
preprocessed_dirs = [
    "images_bilateral_optimized",
    "images_bilateral+msrcr_optimized",
    "images_non_local_means+msrcr_optimized",
    "images_bilateral+illumination_comp_optimized",
    "images_z_score_norm+adaptive_gamma_optimized",
    "images_z_score_norm+percentile_norm_optimized",
    "images_bilateral+z_score_norm_optimized",
]

models = ["HAM10000"]
# , "DeepDerm"
eval_base_dir = "DDI-results"
meta_file = "DDI/ddi_metadata.csv"

# Replace "python" with your venv Python path
venv_python = r"C:\Users\LeoYa\GitHub\DDI-Code\.venv\Scripts\python.exe"

for model in models:
    for preproc in preprocessed_dirs:
        eval_file = f"DDI-results_{preproc}_{model}/{model}-evaluation.pkl"
        if os.path.exists(eval_file):
            print(f"\n=== {model} | {preproc} ===")
            cmd = [
                venv_python, "auc_by_skin_tone.py",
                "--preds", eval_file,
                "--meta", meta_file
            ]
            subprocess.run(cmd)
        else:
            print(f"Missing eval file: {eval_file}")