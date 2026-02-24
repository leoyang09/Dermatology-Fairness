"""
Exploratory Data Analysis for the DDI dataset.

Analyzes class balance (benign vs malignant), skin tone distribution,
and diagnosis × skin tone cross-tabulation. Saves summary stats and plots.

Usage:
    python eda_ddi.py --data_dir=DDI --out_dir=EDA-output
    python eda_ddi.py  # uses defaults: data_dir=DDI, out_dir=EDA-output
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metadata(data_dir: str, csv_name: str = "ddi_metadata.csv") -> pd.DataFrame:
    """Load DDI metadata CSV and normalize malignant column."""
    csv_path = os.path.join(data_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metadata not found: {csv_path}\n"
            "Download DDI from https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965"
        )
    df = pd.read_csv(csv_path)
    # Match ddi_dataset.py: ensure 'malignant' column
    if "malignant" not in df.columns and "malignancy(malig=1)" in df.columns:
        df["malignant"] = (df["malignancy(malig=1)"] == 1).astype(int)
    return df


def run_eda(df: pd.DataFrame, out_dir: str) -> None:
    """Compute EDA stats and generate plots."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(df)

    # --- Class balance (benign vs malignant) ---
    class_counts = df["malignant"].value_counts().sort_index()
    benign_count = class_counts.get(0, 0)
    malignant_count = class_counts.get(1, 0)
    benign_pct = 100 * benign_count / n
    malignant_pct = 100 * malignant_count / n
    imbalance_ratio = max(benign_count, malignant_count) / max(1, min(benign_count, malignant_count))

    class_summary = [
        "=" * 60,
        "CLASS BALANCE (Diagnosis)",
        "=" * 60,
        f"  Total samples:     {n}",
        f"  Benign (0):        {benign_count:6d}  ({benign_pct:5.2f}%)",
        f"  Malignant (1):     {malignant_count:6d}  ({malignant_pct:5.2f}%)",
        f"  Imbalance ratio:   {imbalance_ratio:.2f} (max/min class size)",
        "",
    ]
    lines = "\n".join(class_summary)
    print(lines)

    # --- Skin tone distribution ---
    skin_counts = df["skin_tone"].value_counts().sort_index()
    skin_labels = {12: "Fitzpatrick 1-2", 34: "Fitzpatrick 3-4", 56: "Fitzpatrick 5-6"}
    skin_summary = [
        "",
        "=" * 60,
        "SKIN TONE DISTRIBUTION",
        "=" * 60,
        f"  Total samples:     {n}",
    ]
    for st in [12, 34, 56]:
        c = skin_counts.get(st, 0)
        pct = 100 * c / n
        skin_summary.append(f"  {skin_labels[st]:20s} {c:6d}  ({pct:5.2f}%)")
    skin_summary.append("")
    lines_skin = "\n".join(skin_summary)
    print(lines_skin)

    # --- Cross-tab: diagnosis × skin_tone ---
    ct = pd.crosstab(df["malignant"], df["skin_tone"], margins=True)
    ct.index = ["Benign", "Malignant", "Total"]
    ct.columns = [skin_labels.get(c, str(c)) for c in ct.columns]
    cross_tab = [
        "",
        "=" * 60,
        "CROSS-TAB: Diagnosis × Skin tone (counts)",
        "=" * 60,
        ct.to_string(),
        "",
    ]
    lines_cross = "\n".join(cross_tab)
    print(lines_cross)

    # Proportions within each skin tone (row per skin tone)
    ct_prop = pd.crosstab(df["skin_tone"], df["malignant"], normalize="index") * 100
    ct_prop.index = [skin_labels.get(i, i) for i in ct_prop.index]
    ct_prop.columns = ["Benign %", "Malignant %"]
    prop_tab = [
        "Proportion within each skin tone (%):",
        ct_prop.round(2).to_string(),
        "",
    ]
    print("\n".join(prop_tab))

    # --- Save summary to file ---
    summary_path = os.path.join(out_dir, "eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(class_summary))
        f.write("\n")
        f.write(lines_skin)
        f.write("\n")
        f.write(lines_cross)
        f.write("\n")
        f.write("\n".join(prop_tab))
    print(f"Summary saved to {summary_path}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Class balance bar chart
    ax = axes[0]
    classes = ["Benign", "Malignant"]
    counts = [benign_count, malignant_count]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(classes, counts, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Class balance (diagnosis)")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.02, str(c), ha="center", va="bottom", fontsize=11)

    # 2. Skin tone distribution
    ax = axes[1]
    st_labels = [skin_labels[k] for k in [12, 34, 56]]
    st_counts = [skin_counts.get(k, 0) for k in [12, 34, 56]]
    colors_st = ["#f1c40f", "#e67e22", "#8e44ad"]
    bars = ax.bar(st_labels, st_counts, color=colors_st, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Skin tone distribution")
    ax.tick_params(axis="x", rotation=15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    for b, c in zip(bars, st_counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(st_counts) * 0.02, str(c), ha="center", va="bottom", fontsize=10)

    # 3. Stacked bar: diagnosis by skin tone
    ax = axes[2]
    st_keys = [12, 34, 56]
    benign_by_st = [len(df[(df["skin_tone"] == st) & (df["malignant"] == 0)]) for st in st_keys]
    malignant_by_st = [len(df[(df["skin_tone"] == st) & (df["malignant"] == 1)]) for st in st_keys]
    x = np.arange(len(st_keys))
    w = 0.5
    ax.bar(x - w / 2, benign_by_st, w, label="Benign", color="#2ecc71", edgecolor="black", linewidth=0.6)
    ax.bar(x - w / 2, malignant_by_st, w, bottom=benign_by_st, label="Malignant", color="#e74c3c", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([skin_labels[k] for k in st_keys], rotation=15)
    ax.set_ylabel("Count")
    ax.set_title("Diagnosis by skin tone")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "eda_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="EDA for DDI dataset (class balance, skin tone).")
    parser.add_argument("--data_dir", type=str, default="DDI", help="Directory containing ddi_metadata.csv (and optionally images/).")
    parser.add_argument("--out_dir", type=str, default="EDA-output", help="Directory to write summary and plots.")
    parser.add_argument("--csv", type=str, default="ddi_metadata.csv", help="Metadata CSV filename.")
    args = parser.parse_args()

    df = load_metadata(args.data_dir, args.csv)
    run_eda(df, args.out_dir)


if __name__ == "__main__":
    main()
