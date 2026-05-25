"""
K-fold evaluation for fine-tuned models: per-fold predictions, pooled analysis,
bootstrap 95% CIs for AUC (overall + per skin tone), gap CIs between tone groups,
and fold consistency (mean ± std).
"""
import argparse
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eval_data import load_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default skin-tone groups used in this project (Fitzpatrick-style buckets).
DEFAULT_TONE_GROUPS = (12, 34, 56)


class FoldTestDataset(Dataset):
    """
    Loads ONLY rows listed in one fold's test.csv.
    Expected columns: DDI_file, malignant, skin_tone (same as project CSVs).
    """

    def __init__(self, csv_file: str, data_dir: str, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.df["malignant"] = self.df["malignant"].apply(self._convert_label)
        if "skin_tone" not in self.df.columns:
            raise ValueError(f"Missing skin_tone column in {csv_file}")

    @staticmethod
    def _convert_label(val):
        val_str = str(val).strip().lower()
        if val_str in ["true", "1"]:
            return 1
        if val_str in ["false", "0"]:
            return 0
        raise ValueError(f"Invalid label value: {val}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = row["DDI_file"]
        img_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.data_dir, rel_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["malignant"])
        tone = int(row["skin_tone"])
        return image, label, tone, rel_path


def _stratify_labels_for_cv(df: pd.DataFrame) -> np.ndarray:
    if "stratify" in df.columns:
        return pd.Categorical(df["stratify"]).codes
    return df["malignant"].astype(int).values


def regenerate_seed0_fold_test_csvs(k_folds: int, split_dir: str) -> None:
    full_df = pd.concat(
        [pd.read_csv("train.csv"), pd.read_csv("val.csv"), pd.read_csv("test.csv")],
        ignore_index=True,
    )
    full_df["malignant"] = full_df["malignant"].apply(FoldTestDataset._convert_label)

    y = _stratify_labels_for_cv(full_df)
    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)

    for fold_idx, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        fold_df = full_df.iloc[test_idx].reset_index(drop=True)
        fold_path = os.path.join(split_dir, "seed0", f"fold{fold_idx}")
        os.makedirs(fold_path, exist_ok=True)
        fold_df.to_csv(os.path.join(fold_path, "test.csv"), index=False)


def get_fold_test_csv(seed: int, fold: int, split_dir: str) -> str:
    return os.path.join(split_dir, f"seed{seed}", f"fold{fold}", "test.csv")


def load_finetuned_fold_model(model_name: str, seed: int, fold: int, weights_dir: str):
    model = load_model(model_name, download=False).to(DEVICE)
    weights_path = os.path.join(weights_dir, f"{model_name}_seed{seed}_fold{fold}.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing fold weights: {weights_path}")

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, weights_path


def evaluate_one_fold(
    model,
    fold_csv: str,
    data_dir: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Returns preds, labels, tones (parallel arrays) plus scalar metrics."""
    tf = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds = FoldTestDataset(fold_csv, data_dir=data_dir, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs: List[float] = []
    labels: List[int] = []
    tones: List[int] = []

    with torch.no_grad():
        for x, y, t, _ in loader:
            x = x.to(DEVICE)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            p = torch.softmax(out, dim=1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
            tones.extend(int(v) for v in t.numpy().tolist())

    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    tones_arr = np.asarray(tones, dtype=int)

    threshold = getattr(model, "_ddi_threshold", 0.5)
    pred_labels = (probs_arr > threshold).astype(int)

    metrics = {
        "n_test": int(len(labels_arr)),
        "accuracy": float(accuracy_score(labels_arr, pred_labels)),
        "f1": float(f1_score(labels_arr, pred_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_arr, probs_arr))
        if len(np.unique(labels_arr)) >= 2
        else float("nan"),
    }

    return {
        "preds": probs_arr,
        "labels": labels_arr,
        "tones": tones_arr,
        "metrics": metrics,
    }


def summarize(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


# --- Bootstrap AUC (95% CI = 2.5th / 97.5th percentiles) ---


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, Optional[Tuple[float, float]]]:
    """Resample rows with replacement, compute AUC each time; return point AUC and (lo, hi)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), None
    point = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, y_score[idx]))
    if not aucs:
        return point, None
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return point, (float(lo), float(hi))


def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def bootstrap_gap_ci_pair(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tones: np.ndarray,
    tone_a: int,
    tone_b: int,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Bootstrap both groups together: resample pooled rows with replacement,
    then AUC_a - AUC_b on each bootstrap (only rows with each tone).
    95% CI on the gap; includes zero => not significant at 5% (for this setup).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    tones = np.asarray(tones)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    ma = tones == tone_a
    mb = tones == tone_b
    gap_point = auc_safe(y_true[ma], y_score[ma]) - auc_safe(y_true[mb], y_score[mb])

    gaps: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        tt = tones[idx]
        m1 = tt == tone_a
        m2 = tt == tone_b
        if m1.sum() < 2 or m2.sum() < 2:
            continue
        if len(np.unique(yt[m1])) < 2 or len(np.unique(yt[m2])) < 2:
            continue
        g = roc_auc_score(yt[m1], ys[m1]) - roc_auc_score(yt[m2], ys[m2])
        gaps.append(float(g))

    if not gaps:
        return {
            "tone_a": tone_a,
            "tone_b": tone_b,
            "gap_point": gap_point,
            "ci_95": None,
            "ci_includes_zero": None,
            "n_boot_used": 0,
        }

    lo, hi = np.percentile(gaps, [2.5, 97.5])
    ci = (float(lo), float(hi))
    includes_zero = ci[0] <= 0.0 <= ci[1]
    return {
        "tone_a": tone_a,
        "tone_b": tone_b,
        "gap_point": gap_point,
        "ci_95": ci,
        "ci_includes_zero": includes_zero,
        "significant_at_05": not includes_zero,
        "n_boot_used": len(gaps),
    }


def pooled_analysis(
    fold_results: List[Dict[str, Any]],
    n_bootstrap: int,
    bootstrap_seed: int,
    tone_pairs: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    """Pool preds/labels/tones across folds; bootstrap AUCs and gap CIs."""
    preds = np.concatenate([fr["preds"] for fr in fold_results])
    labels = np.concatenate([fr["labels"] for fr in fold_results])
    tones = np.concatenate([fr["tones"] for fr in fold_results])

    overall_auc, overall_ci = bootstrap_auc_ci(labels, preds, n_boot=n_bootstrap, seed=bootstrap_seed)

    per_tone: Dict[int, Dict[str, Any]] = {}
    for g in DEFAULT_TONE_GROUPS:
        m = tones == g
        if m.sum() == 0:
            per_tone[g] = {"n": 0, "auc": float("nan"), "auc_ci_95": None}
            continue
        auc_pt, ci = bootstrap_auc_ci(
            labels[m], preds[m], n_boot=n_bootstrap, seed=bootstrap_seed + g
        )
        per_tone[g] = {"n": int(m.sum()), "auc": auc_pt, "auc_ci_95": ci}

    gap_results = []
    for a, b in tone_pairs:
        gap_results.append(
            bootstrap_gap_ci_pair(
                labels, preds, tones, a, b, n_boot=n_bootstrap, seed=bootstrap_seed + 100 + a + b
            )
        )

    return {
        "n_pooled": int(len(labels)),
        "overall_auc": overall_auc,
        "overall_auc_ci_95": overall_ci,
        "per_skin_tone": per_tone,
        "gap_cis": gap_results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate k-fold fine-tuned models by fold test CSVs.")
    parser.add_argument("--model", type=str, default="HAM10000")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--data_dir", type=str, default="DDI/images")
    parser.add_argument("--weights_dir", type=str, default=".")
    parser.add_argument("--split_dir", type=str, default="cv_splits")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="DDI-results/kfold_eval")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap replicates for AUC CIs.")
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    regenerate_seed0_fold_test_csvs(k_folds=args.k_folds, split_dir=args.split_dir)

    # All pairwise gaps among default tone groups (project uses 12, 34, 56).
    tone_pairs = [(12, 56)]

    all_rows = []
    for seed in args.seeds:
        print(f"\n=== Evaluating seed={seed} ===")
        fold_dicts: List[Dict[str, Any]] = []

        for fold in range(args.k_folds):
            fold_csv = get_fold_test_csv(seed=seed, fold=fold, split_dir=args.split_dir)
            if not os.path.exists(fold_csv):
                raise FileNotFoundError(
                    f"Missing fold CSV for seed={seed}, fold={fold}: {fold_csv}\n"
                    "Only seed0 splits are regenerated automatically. "
                    "Please provide existing split files for other seeds."
                )

            model, weights_path = load_finetuned_fold_model(
                model_name=args.model,
                seed=seed,
                fold=fold,
                weights_dir=args.weights_dir,
            )
            out = evaluate_one_fold(
                model=model,
                fold_csv=fold_csv,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
            )
            m = out["metrics"]
            fold_dicts.append(
                {
                    "fold": fold,
                    "preds": out["preds"],
                    "labels": out["labels"],
                    "tones": out["tones"],
                    "weights_path": weights_path,
                    **m,
                }
            )

            row = {
                "model": args.model,
                "seed": seed,
                "fold": fold,
                "fold_csv": fold_csv,
                "weights_path": weights_path,
                **m,
            }
            all_rows.append(row)
            print(
                f"seed={seed} fold={fold} "
                f"n_test={row['n_test']} acc={row['accuracy']:.4f} "
                f"f1={row['f1']:.4f} auc={row['roc_auc']:.4f}"
            )

        # --- Pooled + bootstrap (this seed) ---
        analysis = pooled_analysis(
            fold_dicts,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
            tone_pairs=tone_pairs,
        )

        print(f"\n--- Pooled predictions (seed={seed}) ---")
        print(f"n_pooled = {analysis['n_pooled']}")
        oc = analysis["overall_auc_ci_95"]
        print(f"Overall AUC = {analysis['overall_auc']:.4f}", end="")
        if oc:
            print(f"  95% CI [{oc[0]:.4f}, {oc[1]:.4f}] (bootstrap n={args.n_bootstrap})")
        else:
            print("  95% CI: N/A")

        print("Per skin tone (pooled):")
        for g in DEFAULT_TONE_GROUPS:
            pt = analysis["per_skin_tone"][g]
            ci = pt["auc_ci_95"]
            if ci:
                print(f"  FST {g}: n={pt['n']} AUC={pt['auc']:.4f} 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                print(f"  FST {g}: n={pt['n']} AUC={pt['auc']:.4f} CI N/A")

        print("AUC gap (tone_a - tone_b), bootstrap on pooled data:")
        for g in analysis["gap_cis"]:
            ci = g["ci_95"]
            sig = g.get("significant_at_05")
            if ci is None:
                print(f"  {g['tone_a']} vs {g['tone_b']}: gap={g['gap_point']:.4f} CI unavailable")
                continue
            inc = g["ci_includes_zero"]
            print(
                f"  {g['tone_a']} vs {g['tone_b']}: gap={g['gap_point']:.4f} "
                f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  "
                f"CI includes 0: {inc}  (5% sig: {sig})"
            )

        # --- Consistency across folds (mean ± std) ---
        fold_aucs = [fd["roc_auc"] for fd in fold_dicts]
        fold_acc = [fd["accuracy"] for fd in fold_dicts]
        fold_f1 = [fd["f1"] for fd in fold_dicts]
        m_auc, s_auc = summarize(fold_aucs)
        m_acc, s_acc = summarize(fold_acc)
        m_f1, s_f1 = summarize(fold_f1)
        print(f"\n--- Consistency across {args.k_folds} folds (seed={seed}) ---")
        print(f"ROC-AUC: {m_auc:.4f} ± {s_auc:.4f}")
        print(f"Accuracy: {m_acc:.4f} ± {s_acc:.4f}")
        print(f"F1: {m_f1:.4f} ± {s_f1:.4f}")

        payload = {
            "model": args.model,
            "seed": seed,
            "k_folds": args.k_folds,
            "fold_results": [
                {
                    "fold": fd["fold"],
                    "preds": fd["preds"],
                    "labels": fd["labels"],
                    "tones": fd["tones"],
                    "roc_auc": fd["roc_auc"],
                    "accuracy": fd["accuracy"],
                    "f1": fd["f1"],
                    "n_test": fd["n_test"],
                    "weights_path": fd["weights_path"],
                }
                for fd in fold_dicts
            ],
            "pooled": {
                "n": analysis["n_pooled"],
                "preds": np.concatenate([fd["preds"] for fd in fold_dicts]),
                "labels": np.concatenate([fd["labels"] for fd in fold_dicts]),
                "tones": np.concatenate([fd["tones"] for fd in fold_dicts]),
            },
            "bootstrap": {
                "n_replicates": args.n_bootstrap,
                "overall_auc": analysis["overall_auc"],
                "overall_auc_ci_95": analysis["overall_auc_ci_95"],
                "per_skin_tone": analysis["per_skin_tone"],
                "gap_cis": analysis["gap_cis"],
            },
            "fold_consistency": {
                "roc_auc_mean": m_auc,
                "roc_auc_std": s_auc,
                "accuracy_mean": m_acc,
                "accuracy_std": s_acc,
                "f1_mean": m_f1,
                "f1_std": s_f1,
            },
        }

        pkl_path = os.path.join(args.save_dir, f"{args.model}_seed{seed}_kfold_predictions.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"\nSaved predictions + analysis to: {pkl_path}")

    results_df = pd.DataFrame(all_rows)
    results_csv = os.path.join(args.save_dir, f"{args.model}_kfold_metrics.csv")
    results_df.to_csv(results_csv, index=False)

    print("\n=== Aggregate across all evaluated fold rows (table) ===")
    for metric in ["accuracy", "f1", "roc_auc"]:
        mean_v, std_v = summarize(results_df[metric].tolist())
        print(f"{metric}: mean={mean_v:.4f} std={std_v:.4f}")

    with open(os.path.join(args.save_dir, f"{args.model}_kfold_metrics.pkl"), "wb") as f:
        pickle.dump({"rows": all_rows}, f)

    print(f"\nSaved fold metrics table to: {results_csv}")


if __name__ == "__main__":
    main()
