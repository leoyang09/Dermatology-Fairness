"""
Inspect evaluation results from a .pkl file.
Usage: python inspect_results.py [path_to.pkl]
Default: DDI-results/DeepDerm-evaluation.pkl
"""
import pickle
import sys
import os

def main():
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else "DDI-results/DeepDerm-evaluation.pkl"
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        sys.exit(1)
    with open(pkl_path, "rb") as f:
        r = pickle.load(f)
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if hasattr(r, "columns") and hasattr(r, "shape"): # It's likely a DataFrame
        import pandas as pd
        from sklearn.metrics import roc_auc_score
        print(f"\nLoaded DataFrame with shape: {r.shape}")
        print("Columns:", list(r.columns))
        
        if "y_true" in r.columns and "y_score" in r.columns:
            try:
                auc = roc_auc_score(r["y_true"], r["y_score"])
                print(f"\n--- ROC AUC: {auc:.4f} ---")
            except Exception as e:
                print(f"\n--- ROC AUC: Error calculating ({e}) ---")
        
        print("\n--- Sample size ---")
        print("Number of samples:", len(r))
        print("\n--- Head ---")
        print(r.head())

    else: # It's a dictionary (original behavior)
        print("\nKeys in results:", list(r.keys()))
        print("\n--- Classification report ---")
        print(r.get("report", "(no report)"))
        print("\n--- ROC AUC ---")
        print(r.get("ROC_AUC", "(no AUC)"))
        print("\n--- Model / threshold ---")
        print("Model:", r.get("model", "?"), "| Threshold:", r.get("threshold", "?"))
        print("\n--- Sample size ---")
        true = r.get("true_labels")
        pred = r.get("predicted_labels")
        if true is not None:
            print("Number of samples:", len(true))
        if pred is not None:
            print("Predicted scores shape:", pred.shape if hasattr(pred, "shape") else len(pred))

if __name__ == "__main__":
    main()
