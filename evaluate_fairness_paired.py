import argparse
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.special import expit

# ---------------------------
# Utility Functions
# ---------------------------

def normalize_filename(x):
    return os.path.splitext(os.path.basename(str(x)))[0]

def bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    aucs = []
    attempts = 0
    max_attempts = n_bootstrap * 10

    while len(aucs) < n_bootstrap and attempts < max_attempts:
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            attempts += 1
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(aucs) == 0:
        raise ValueError("Unable to compute any bootstrap AUCs. Check subgroup size or labels.")

    mean_auc = np.mean(aucs)
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    return mean_auc, lower, upper, np.array(aucs)

def bootstrap_auc_gap(y_true, y_score, groups, group1, group2, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    groups = np.array(groups)
    
    mask1 = groups == group1
    mask2 = groups == group2
    y1, s1 = y_true[mask1], y_score[mask1]
    y2, s2 = y_true[mask2], y_score[mask2]

    gap_dist = []
    attempts = 0
    max_attempts = n_bootstrap * 10
    while len(gap_dist) < n_bootstrap and attempts < max_attempts:
        idx1 = rng.integers(0, len(y1), len(y1))
        idx2 = rng.integers(0, len(y2), len(y2))
        if len(np.unique(y1[idx1])) < 2 or len(np.unique(y2[idx2])) < 2:
            attempts += 1
            continue
        auc1 = roc_auc_score(y1[idx1], s1[idx1])
        auc2 = roc_auc_score(y2[idx2], s2[idx2])
        gap_dist.append(auc1 - auc2)

    if len(gap_dist) == 0:
        raise ValueError("Unable to compute any bootstrap AUC gaps. Check subgroup sizes.")
    
    gap_dist = np.array(gap_dist)
    return np.mean(gap_dist), np.percentile(gap_dist, 2.5), np.percentile(gap_dist, 97.5), gap_dist

def compute_tpr_gap(y_true, y_pred, groups):
    tprs = {}
    for g in np.unique(groups):
        mask = groups == g
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tprs[g] = tpr
    gap = max(tprs.values()) - min(tprs.values())
    return tprs, gap

def gap_significance_test(baseline_gap_dist, new_gap_dist):
    diff = np.array(baseline_gap_dist) - np.array(new_gap_dist)
    p_value = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    return p_value

def skin_group_label(g):
    mapping = {12: "1-2", 34: "3-4", 56: "5-6"}
    return mapping.get(g, str(g))

# ---------------------------
# New Paired Bootstrap Function
# ---------------------------
def paired_bootstrap_pvalue(y_true, new_probs, baseline_probs, groups, group1, group2, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    groups = np.array(groups)
    
    mask1 = groups == group1
    mask2 = groups == group2
    
    y1, y2 = y_true[mask1], y_true[mask2]
    new1, new2 = new_probs[mask1], new_probs[mask2]
    base1, base2 = baseline_probs[mask1], baseline_probs[mask2]
    
    diff_dist = []
    for _ in range(n_bootstrap):
        idx1 = rng.integers(0, len(y1), len(y1))
        idx2 = rng.integers(0, len(y2), len(y2))
        if len(np.unique(y1[idx1])) < 2 or len(np.unique(y2[idx2])) < 2:
            continue
        gap_new = roc_auc_score(y1[idx1], new1[idx1]) - roc_auc_score(y2[idx2], new2[idx2])
        gap_base = roc_auc_score(y1[idx1], base1[idx1]) - roc_auc_score(y2[idx2], base2[idx2])
        diff_dist.append(gap_base - gap_new)
    diff_dist = np.array(diff_dist)
    p_value = 2 * min(np.mean(diff_dist <= 0), np.mean(diff_dist >= 0))
    return p_value, diff_dist

# ---------------------------
# Main Evaluation
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare", type=str, default=None)
    parser.add_argument("--baseline_gap_file", type=str, default=None)
    parser.add_argument("--baseline_pkl", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---------------------------
    # Load new model pickle
    # ---------------------------
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame({
        "y_true": data["true_labels"],
        "y_score": data["predicted_labels"],
        "DDI_file": data["images"]
    })
    df["DDI_file"] = df["DDI_file"].apply(normalize_filename)
    df["y_prob"] = expit(df["y_score"])

    # ---------------------------
    # Load metadata and merge
    # ---------------------------
    metadata = pd.read_csv(args.metadata)
    metadata["DDI_file"] = metadata["DDI_file"].apply(normalize_filename)
    df = df.merge(metadata[["DDI_file","skin_tone"]], on="DDI_file", how="left")
    if df["skin_tone"].isnull().sum() > 0:
        raise ValueError("Missing skin tone labels after merge")

    y_true = df["y_true"].values
    y_prob = df["y_prob"].values
    groups = df["skin_tone"].values

    # ---------------------------
    # Optimal threshold (Youden's J)
    # ---------------------------
    fpr_all, tpr_all, thresholds = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr_all - fpr_all)
    optimal_thresh = thresholds[best_idx]
    df["y_pred"] = (df["y_prob"] >= optimal_thresh).astype(int)
    y_pred = df["y_pred"].values

    print(f"\nOptimal threshold (Youden J): {optimal_thresh:.4f}")

    # ---------------------------
    # Overall performance
    # ---------------------------
    print("\n--- Overall Performance ---")
    print(classification_report(y_true, y_pred))
    overall_auc = roc_auc_score(y_true, y_prob)
    print(f"Overall ROC AUC (raw): {overall_auc:.4f}")

    # ---------------------------
    # Per-group metrics + plots
    # ---------------------------
    results = []
    auc_distributions = {}
    roc_data = {}
    pr_data = {}

    for g in sorted(np.unique(groups)):
        mask = groups == g
        label = skin_group_label(g)
        raw_auc = roc_auc_score(y_true[mask], y_prob[mask])
        mean_auc, ci_lower, ci_upper, auc_dist = bootstrap_auc_ci(y_true[mask], y_prob[mask])
        auc_distributions[label] = auc_dist
        acc = accuracy_score(y_true[mask], y_pred[mask])
        prec = precision_score(y_true[mask], y_pred[mask], zero_division=0)
        rec = recall_score(y_true[mask], y_pred[mask], zero_division=0)
        f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)

        results.append({
            "skin_group": label,
            "raw_auc": raw_auc,
            "auc": mean_auc,
            "auc_ci_lower": ci_lower,
            "auc_ci_upper": ci_upper,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        # ROC per group
        fpr_g, tpr_g, _ = roc_curve(y_true[mask], y_prob[mask])
        roc_data[label] = (fpr_g, tpr_g, mean_auc)
        plt.figure()
        plt.plot(fpr_g, tpr_g, label=f"AUC={mean_auc:.3f}")
        plt.title(f"ROC Curve - Fitzpatrick {label}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(os.path.join(args.out, f"roc_skin_{label}.png"))
        plt.close()

        # PR per group
        precision_g, recall_g, _ = precision_recall_curve(y_true[mask], y_prob[mask])
        pr_data[label] = (recall_g, precision_g)
        plt.figure()
        plt.plot(recall_g, precision_g)
        plt.title(f"Precision-Recall - Fitzpatrick {label}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(args.out, f"pr_skin_{label}.png"))
        plt.close()

        # Confusion matrix per group
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0,1])
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=["Predicted Benign","Predicted Malignant"],
                    yticklabels=["Actual Benign","Actual Malignant"],
                    cmap="Blues")
        plt.title(f"Confusion Matrix - Fitzpatrick {label}")
        plt.savefig(os.path.join(args.out, f"confusion_matrix_skin_{label}.png"))
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.out,"fairness_metrics.csv"), index=False)

    # ---------------------------
    # Bootstrapped AUC Gap (1-2 minus 5-6)
    # ---------------------------
    if 12 in groups and 56 in groups:
        gap_mean, gap_lower, gap_upper, gap_dist = bootstrap_auc_gap(y_true, y_prob, groups, 12, 56)
        print(f"\nBootstrapped AUC Gap (1-2 minus 5-6): {gap_mean:.4f} (95% CI: {gap_lower:.3f}-{gap_upper:.3f})")
        np.save(os.path.join(args.out,"gap_distribution_1-2_vs_5-6.npy"), gap_dist)

        # ---------------------------
        # Paired bootstrap vs baseline model
        # ---------------------------
        if args.baseline_pkl:
            print("\nRunning paired bootstrap test vs baseline model...")
            with open(args.baseline_pkl, "rb") as f:
                baseline_data = pickle.load(f)
            df_baseline = pd.DataFrame({
                "DDI_file": baseline_data["images"],
                "y_prob": expit(baseline_data["predicted_labels"])
            })
            df_baseline["DDI_file"] = df_baseline["DDI_file"].apply(normalize_filename)

            merged = df.merge(df_baseline, on="DDI_file", how="inner", suffixes=("", "_baseline"))
            y_true_merge = merged["y_true"].values
            new_probs_merge = merged["y_prob"].values
            baseline_probs_merge = merged["y_prob_baseline"].values
            groups_merge = merged["skin_tone"].values

            p_val_paired, diff_dist = paired_bootstrap_pvalue(
                y_true_merge,
                new_probs_merge,
                baseline_probs_merge,
                groups_merge,
                12,
                56
            )
            print(f"Paired bootstrap gap reduction p-value: {p_val_paired:.4f}")
            np.save(os.path.join(args.out, "paired_gap_diff.npy"), diff_dist)

            plt.figure(figsize=(7,5))
            plt.hist(diff_dist, bins=30, color="lightblue", edgecolor="black")
            plt.axvline(np.mean(diff_dist), color="red", linestyle="--", label=f"Mean={np.mean(diff_dist):.4f}")
            plt.title("Paired Bootstrap Gap Difference (Baseline - New)")
            plt.xlabel("Gap Difference")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig(os.path.join(args.out,"paired_gap_diff_hist.png"))
            plt.close()

        # ---------------------------
        # Old gap significance test vs baseline file
        # ---------------------------
        if args.baseline_gap_file:
            baseline_gap_dist = np.load(args.baseline_gap_file)
            p_val = gap_significance_test(baseline_gap_dist, gap_dist)
            print(f"Gap reduction p-value vs baseline: {p_val:.4f}")

    # ---------------------------
    # Combined ROC & PR plots
    # ---------------------------
    plt.figure()
    plt.plot(fpr_all, tpr_all, label=f"Overall (AUC={overall_auc:.3f})", linewidth=2)
    for label, (fpr_g, tpr_g, mean_auc) in roc_data.items():
        plt.plot(fpr_g, tpr_g, label=f"{label} (AUC={mean_auc:.3f})")
    plt.title("ROC Curves by Fitzpatrick Group")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(args.out,"roc_combined.png"))
    plt.close()

    plt.figure()
    precision_all, recall_all, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall_all, precision_all, label="Overall", linewidth=2)
    for label, (recall_g, precision_g) in pr_data.items():
        plt.plot(recall_g, precision_g, label=label)
    plt.title("Precision-Recall by Fitzpatrick Group")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(os.path.join(args.out,"pr_combined.png"))
    plt.close()

    # ---------------------------
    # Bar plot of per-group ROC AUC
    # ---------------------------
    plt.figure(figsize=(7,5))
    means = results_df["auc"]
    lowers = results_df["auc_ci_lower"]
    uppers = results_df["auc_ci_upper"]
    errors = [means - lowers, uppers - means]
    plt.bar(results_df["skin_group"], means, yerr=errors, capsize=6, color="skyblue", edgecolor="black")
    plt.ylabel("ROC AUC")
    plt.title("Per-Group ROC AUC with 95% CI")
    plt.ylim(0,1.05)
    plt.savefig(os.path.join(args.out,"auc_bar_with_ci.png"))
    plt.close()

    # ---------------------------
    # Optional comparison with another model CSV
    # ---------------------------
    if args.compare:
        other = pd.read_csv(args.compare)
        merged_cmp = results_df.merge(other, on="skin_group", suffixes=("_Model1","_Model2"))
        print("\nComparison with second model:")
        print(merged_cmp[["skin_group","auc_Model1","auc_Model2","recall_Model1","recall_Model2"]])
        merged_cmp.to_csv(os.path.join(args.out,"model_comparison.csv"), index=False)

    print("\nEvaluation complete. All metrics and plots saved.")

if __name__ == "__main__":
    main()