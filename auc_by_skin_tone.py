import argparse
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score

def auc(df: pd.DataFrame) -> float:
    if df["y_true"].nunique() < 2:
        return float("nan")
    return roc_auc_score(df["y_true"], df["y_score"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV with DDI_file,y_true,y_score")
    ap.add_argument("--meta", required=True, help="ddi_metadata.csv")
    ap.add_argument("--save_dir", help="Directory to save skin tone specific dataframes as pkl files")
    args = ap.parse_args()

    if args.preds.endswith(".pkl"):
        data = pd.read_pickle(args.preds)
        # Extract y_score and DDI_file from pickle dict
        # Assuming predicted_labels is (N,) scores and images is list of paths
        y_score = data["predicted_labels"]
        # Convert full paths to filenames to match metadata
        ddi_files = [os.path.basename(p) for p in data["images"]]
        
        preds = pd.DataFrame({
            "DDI_file": ddi_files,
            "y_score": y_score
        })
    else:
        preds = pd.read_csv(args.preds)
    meta = pd.read_csv(args.meta)

    # Normalize target
    meta["y_true"] = meta["malignant"].astype(str).str.lower().map({"true": 1, "false": 0})
    if meta["y_true"].isna().any():
        raise ValueError("malignant column must be True/False (or true/false).")

    # Join on filename
    df = preds.merge(
        meta[["DDI_file", "skin_tone", "y_true"]],
        on="DDI_file",
        how="inner",
        suffixes=("", "_meta"),
    )

    # Use metadata truth to avoid accidental mismatch
    if "y_true_meta" in df.columns:
        df["y_true"] = df["y_true_meta"]
        df = df.drop(columns=["y_true_meta"])

    # Ensure types
    df["skin_tone"] = pd.to_numeric(df["skin_tone"], errors="coerce")
    df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce")
    df = df.dropna(subset=["skin_tone", "y_score", "y_true"])

    overall = auc(df)
    g12 = auc(df[df["skin_tone"] == 12])
    g56 = auc(df[df["skin_tone"] == 56])
    g34 = auc(df[df["skin_tone"] == 34])

    print(f"Overall AUC: {overall:.4f}")
    print(f"Skin tone 12 AUC: {g12:.4f} (n={len(df[df['skin_tone']==12])})")
    print(f"Skin tone 34 AUC: {g34:.4f} (n={len(df[df['skin_tone']==34])})")
    print(f"Skin tone 56 AUC: {g56:.4f} (n={len(df[df['skin_tone']==56])})")

    if not np.isnan(g12) and not np.isnan(g56):
        print(f"Gap (12 - 56): {(g12 - g56):.4f}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        df[df["skin_tone"] == 12].to_pickle(os.path.join(args.save_dir, "fst_12.pkl"))
        df[df["skin_tone"] == 34].to_pickle(os.path.join(args.save_dir, "fst_34.pkl"))
        df[df["skin_tone"] == 56].to_pickle(os.path.join(args.save_dir, "fst_56.pkl"))
        print(f"Saved skin tone splits to {args.save_dir}")

if __name__ == "__main__":
    main()
