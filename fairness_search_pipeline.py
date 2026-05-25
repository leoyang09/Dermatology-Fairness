"""
Automated fairness-driven optimization pipeline.

For each FST skin tone group (I-II, III-IV, V-VI), runs three sequential stages:

  Stage 1 — Preprocessing selection: benchmark every candidate preprocessing
             method on the group's data using default (baseline) hyperparameters.
             Methods that improve group AUC over the no-preprocessing baseline
             are declared "winners."

  Stage 2 — Pipeline combination: combine winners into 2-3 method chains
             following the ordering convention from the reference code
             (denoising -> color -> illumination -> contrast/normalization).
             Evaluate each chain with default params and keep the top-k.

  Stage 3 — Bayesian hyperparameter tuning (Optuna TPE): for each top-k
             pipeline, run an Optuna study that maximises AUC specifically on
             the current FST group. The globally best (pipeline, params) pair
             is selected.

  Final    — Finetune a dedicated InceptionV3 model on the full DDI dataset
             using the winning preprocessing pipeline. Saved as
             RESULTS_DIR/model_{group_name}.pth  (e.g. model_fst56.pth).
"""

import itertools
import json
import os
import random

import cv2
import numpy as np
import optuna
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eval_data import load_model
from generate_preprocessed import method_map

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME   = "HAM10000"                      # "HAM10000" or "DeepDerm"
DATA_DIR     = "DDI"
IMG_DIR      = os.path.join(DATA_DIR, "images")
META_CSV     = os.path.join(DATA_DIR, "ddi_metadata.csv")
RESULTS_DIR  = "fairness_pipeline_results"

BATCH_SIZE   = 16
LR           = 1e-5
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 200
PATIENCE     = 20
SEED         = 42

N_TRIALS_STAGE3 = 20    # Optuna trials per pipeline in Stage 3
TOP_K_COMBOS    = 3     # How many Stage-2 pipelines are passed to Stage 3
MIN_AUC_DELTA   = 0.005 # Minimum AUC improvement to call a Stage-1 method a "winner"

SKIN_TONE_GROUPS = {
    "fst12": 12,   # FST I-II   (light)
    "fst34": 34,   # FST III-IV (medium)
    "fst56": 56,   # FST V-VI   (dark)
}
GROUP_LABELS = {
    "fst12": "FST I-II   (light)",
    "fst34": "FST III-IV (medium)",
    "fst56": "FST V-VI   (dark)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing catalog
# ─────────────────────────────────────────────────────────────────────────────

# Tunable methods: {method_name: {param_name: (default_value, (lo, hi))}}
# Param names and ranges mirror bayes_opt_efficiency.py / bayes_opt_high_improvement.py
TUNABLE_PARAMS = {
    "clahe":             {"clip_limit":  (2.0,  (0.5,  4.0))},
    "illumination_comp": {"sigma":       (50.0, (10.0, 100.0))},
    "percentile_norm":   {"range_":     (98.0, (80.0, 99.0))},
    "local_contrast":    {"strength":   (1.25, (0.1,  2.0))},
    "bilateral":         {"sigma_color":(75.0, (10.0, 150.0))},
    "non_local_means":   {"h":          (10,   (5,    15))},
}

# Fixed methods (no tunable parameter — called with defaults from generate_preprocessed.py)
FIXED_METHODS = {"adaptive_gamma", "white_balance", "msrcr", "Z_score_norm"}

ALL_CANDIDATE_METHODS = list(TUNABLE_PARAMS.keys()) + sorted(FIXED_METHODS)

# ─────────────────────────────────────────────────────────────────────────────
# Semantic category ordering (mirrors generate_preprocessed.py triples convention)
# denoising -> color -> illumination -> contrast -> normalization
# ─────────────────────────────────────────────────────────────────────────────

METHOD_CATEGORY = {
    "non_local_means":   "denoising",
    "bilateral":         "denoising",
    "white_balance":     "color",
    "msrcr":             "illumination",
    "illumination_comp": "illumination",
    "clahe":             "contrast",
    "adaptive_gamma":    "contrast",
    "local_contrast":    "contrast",
    "percentile_norm":   "normalization",
    "Z_score_norm":      "normalization",
}

STAGE_ORDER = ["denoising", "color", "illumination", "contrast", "normalization"]

# ─────────────────────────────────────────────────────────────────────────────
# Transforms  (mirror finetune_ddi.py)
# ─────────────────────────────────────────────────────────────────────────────

class RandomRotateCrop:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return transforms.functional.rotate(img, angle, expand=False)


TRAIN_AUGMENT = transforms.Compose([
    RandomRotateCrop(degrees=30),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_method(cv_img, method_name, param_value=None):
    """Apply one preprocessing function (BGR in, BGR out).
    Tunable methods receive their param; fixed methods use internal defaults.
    """
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
        fn = method_map.get(method_name)
        if fn is None:
            raise ValueError(f"Unknown preprocessing method: {method_name!r}")
        return fn(cv_img)


def default_step(method_name):
    """Return (method_name, default_param_value) for baseline benchmarking."""
    if method_name in TUNABLE_PARAMS:
        pname = next(iter(TUNABLE_PARAMS[method_name]))
        return (method_name, TUNABLE_PARAMS[method_name][pname][0])
    return (method_name, None)


def combo_key(methods):
    """Join method list with '+' (matches reference code naming convention)."""
    return "+".join(methods) if methods else "none"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _to_label(val):
    s = str(val).strip().lower()
    if s in ("true", "1"):
        return 1
    if s in ("false", "0"):
        return 0
    raise ValueError(f"Cannot parse label: {val!r}")


class PreprocessDataset(Dataset):
    """
    Reads DDI images from a DataFrame, applies an OpenCV preprocessing pipeline
    (steps), then applies train augmentation or val transform.

    steps: list of (method_name, param_value) tuples.
           Pass an empty list for no preprocessing (baseline transform only).
    """

    def __init__(self, df, img_dir, steps, train=False):
        self.df      = df.copy().reset_index(drop=True)
        self.df["malignant"] = self.df["malignant"].apply(_to_label)
        self.img_dir = img_dir
        self.steps   = steps
        self.tf      = TRAIN_AUGMENT if train else VAL_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.img_dir, str(row["DDI_file"]))
        img  = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        for mname, pval in self.steps:
            img = apply_method(img, mname, pval)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return self.tf(pil), int(row["malignant"]), int(row["skin_tone"])

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dataset(model, dataset, batch_size=32):
    """Return (preds, labels, tones) numpy arrays over the full dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    model.to(DEVICE).eval()
    preds, labels, tones = [], [], []
    for x, y, t in loader:
        x   = x.to(DEVICE)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        prob = torch.softmax(out, dim=1)[:, 1]
        preds.extend(prob.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
        tones.extend(t.numpy().tolist())
    return np.array(preds), np.array(labels), np.array(tones)


def group_auc(preds, labels, tones, tone_value):
    """AUC restricted to one skin-tone value; returns nan if not computable."""
    mask = tones == tone_value
    y, p = labels[mask], preds[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def evaluate_steps(model, steps, df, img_dir, tone_value, batch_size=32):
    """Build a PreprocessDataset, run inference, return AUC for tone_value."""
    ds          = PreprocessDataset(df, img_dir, steps, train=False)
    p, l, t     = evaluate_dataset(model, ds, batch_size=batch_size)
    return group_auc(p, l, t, tone_value)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Single-method benchmark
# ─────────────────────────────────────────────────────────────────────────────

def stage1_benchmark(model, df, img_dir, tone_value, baseline_auc):
    """
    Test every candidate preprocessing method with default parameters.

    Returns:
        winners  : list of {"method", "auc", "delta"} dicts, filtered to those
                   that improved group AUC by at least MIN_AUC_DELTA, sorted
                   best-first.
        all_rows : full result list (used for JSON logging).
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1  Benchmark single methods  [tone={tone_value}]")
    print(f"  Baseline AUC: {baseline_auc:.4f}  (threshold delta >= {MIN_AUC_DELTA})")
    print(f"{'='*60}")

    all_rows = []
    for method in ALL_CANDIDATE_METHODS:
        steps = [default_step(method)]
        try:
            auc_val = evaluate_steps(model, steps, df, img_dir, tone_value)
        except Exception as exc:
            print(f"  {method:<24}  ERROR: {exc}")
            continue
        delta  = auc_val - baseline_auc
        marker = "WINNER" if delta >= MIN_AUC_DELTA else "      "
        print(f"  {method:<24}  AUC={auc_val:.4f}  delta={delta:+.4f}  {marker}")
        all_rows.append({"method": method, "auc": auc_val, "delta": delta})

    winners = [r for r in all_rows if r["delta"] >= MIN_AUC_DELTA]
    winners.sort(key=lambda r: r["delta"], reverse=True)
    print(f"\n  -> {len(winners)} winner(s): {[w['method'] for w in winners]}")
    return winners, all_rows

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Pipeline combination and ranking
# ─────────────────────────────────────────────────────────────────────────────

def _stage_idx(method):
    cat = METHOD_CATEGORY.get(method, "other")
    try:
        return STAGE_ORDER.index(cat)
    except ValueError:
        return len(STAGE_ORDER)


def _is_ordered(methods):
    """True iff methods follow the canonical stage ordering (non-decreasing index)."""
    idxs = [_stage_idx(m) for m in methods]
    return all(idxs[i] <= idxs[i + 1] for i in range(len(idxs) - 1))


def build_candidate_combos(winners):
    """
    Build 2- and 3-method pipelines from winners following the reference-code
    ordering convention:
      denoising -> color -> illumination -> contrast -> normalization

    Only permutations that respect this ordering are kept.
    Single-method entries are also included as fallback.
    """
    names  = [w["method"] for w in winners]
    combos = []

    for length in (2, 3):
        if len(names) < length:
            continue
        for perm in itertools.permutations(names, length):
            ordered = list(perm)
            if _is_ordered(ordered) and ordered not in combos:
                combos.append(ordered)

    # Single-method fallback (in case no ordered multi-method combo exists)
    for n in names:
        if [n] not in combos:
            combos.append([n])

    return combos


def stage2_rank_combos(model, combos, df, img_dir, tone_value, top_k):
    """
    Evaluate every candidate combo with default params.
    Returns top-k sorted by group AUC (best first).
    """
    print(f"\n{'='*60}")
    print(f"STAGE 2  Rank {len(combos)} candidate pipeline(s)  [tone={tone_value}]")
    print(f"{'='*60}")

    scored = []
    for methods in combos:
        steps = [default_step(m) for m in methods]
        try:
            auc_val = evaluate_steps(model, steps, df, img_dir, tone_value)
        except Exception as exc:
            print(f"  {combo_key(methods):<44}  ERROR: {exc}")
            continue
        print(f"  {combo_key(methods):<44}  AUC={auc_val:.4f}")
        scored.append({"methods": methods, "auc": auc_val})

    scored.sort(key=lambda r: r["auc"], reverse=True)
    top = scored[:top_k]
    print(f"\n  -> Top-{top_k}:")
    for r in top:
        print(f"     {combo_key(r['methods']):<44}  AUC={r['auc']:.4f}")
    return top

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Bayesian hyperparameter tuning (Optuna TPE)
# ─────────────────────────────────────────────────────────────────────────────

def _build_steps_from_trial(trial, methods):
    """Suggest hyperparameters for each method in an Optuna trial."""
    steps = []
    for m in methods:
        if m not in TUNABLE_PARAMS:
            steps.append((m, None))
            continue
        for pname, (default, (lo, hi)) in TUNABLE_PARAMS[m].items():
            key = f"{m}_{pname}"
            val = (trial.suggest_int(key, lo, hi)
                   if isinstance(lo, int) and isinstance(hi, int)
                   else trial.suggest_float(key, lo, hi))
            steps.append((m, val))
            break  # each method has exactly one tunable parameter


    return steps


def make_objective(model, methods, df, img_dir, tone_value):
    """Return an Optuna objective that maximises group AUC."""

    def objective(trial):
        steps = _build_steps_from_trial(trial, methods)
        try:
            auc_val = evaluate_steps(model, steps, df, img_dir, tone_value)
        except Exception as exc:
            print(f"  trial={trial.number} FAILED: {exc}")
            return 0.0  # worst outcome for a 'maximize' study

        readable = [
            (m, f"{v:.3f}" if isinstance(v, float) else v)
            for m, v in steps
        ]
        print(
            f"  trial={trial.number:3d}  "
            f"pipeline={combo_key(methods)}  "
            f"params={readable}  "
            f"AUC={auc_val:.4f}"
        )
        return auc_val

    return objective


def stage3_bayesian_tune(model, top_combos, df, img_dir, tone_value, n_trials):
    """
    Run one Optuna TPE study per top-k pipeline; return the globally best result:
      {"methods": [...], "params": {...}, "auc": float}
    """
    print(f"\n{'='*60}")
    print(
        f"STAGE 3  Bayesian tuning  [tone={tone_value}]  "
        f"({n_trials} trials per pipeline)"
    )
    print(f"{'='*60}")

    best = {"methods": None, "params": {}, "auc": -1.0}

    for combo_info in top_combos:
        methods = combo_info["methods"]
        ckey    = combo_key(methods)
        print(f"\n  Optimising: {ckey}")

        study = optuna.create_study(
            direction="maximize",
            study_name=f"tone{tone_value}_{ckey}",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        study.optimize(
            make_objective(model, methods, df, img_dir, tone_value),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        print(f"  Best AUC   : {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

        if study.best_value > best["auc"]:
            best = {
                "methods": methods,
                "params":  study.best_params,
                "auc":     study.best_value,
            }

    winner_key = combo_key(best["methods"] or [])
    print(f"\n  -> Globally best: {winner_key}  AUC={best['auc']:.4f}")
    return best

# ─────────────────────────────────────────────────────────────────────────────
# Final finetuning
# ─────────────────────────────────────────────────────────────────────────────

def _steps_from_best(best):
    """Reconstruct the concrete (method, param_value) step list from a best dict."""
    methods = best.get("methods") or []
    params  = best.get("params", {})
    steps   = []
    for m in methods:
        if m not in TUNABLE_PARAMS:
            steps.append((m, None))
            continue
        for pname, (default, _) in TUNABLE_PARAMS[m].items():
            key = f"{m}_{pname}"
            steps.append((m, params.get(key, default)))
            break
    return steps


def _mixup(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE, dtype=torch.long)
        x, ya, yb, lam = _mixup(x, y)
        out, aux = model(x)  # InceptionV3 returns (logits, aux_logits) in train mode
        loss = (
            criterion(out, ya) * lam + criterion(out, yb) * (1 - lam)
            + 0.4 * (criterion(aux, ya) * lam + criterion(aux, yb) * (1 - lam))
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def _val_pass(model, loader, criterion):
    """Return (val_loss, preds, labels, tones) for the full validation set."""
    model.eval()
    total = 0.0
    preds, labels, tones = [], [], []
    with torch.no_grad():
        for x, y, t in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE, dtype=torch.long)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            total += criterion(out, y).item()
            prob = torch.softmax(out, dim=1)[:, 1]
            preds.extend(prob.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
            tones.extend(t.numpy().tolist())
    return (
        total / len(loader),
        np.array(preds), np.array(labels), np.array(tones),
    )


def finetune_group_model(model_name, group_name, tone_value, best, df, img_dir):
    """
    Finetune a dedicated InceptionV3 model for one FST group.

    Trains on the full DDI dataset (all tones) using the winning preprocessing
    pipeline — which was selected to maximise accuracy for this group.
    Monitors val_loss for early stopping; also reports group-specific AUC.
    Saves best weights to RESULTS_DIR/model_{group_name}.pth.
    """
    steps    = _steps_from_best(best)
    pipeline = combo_key(best.get("methods") or [])
    print(f"\n{'='*60}")
    print(f"FINAL FINETUNE  [{group_name}]  pipeline={pipeline}")
    print(f"  Steps: {steps}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Stratified 80/20 train/val split (fall back to unstratified if needed)
    strat = df["malignant"].astype(str) + "_" + df["skin_tone"].astype(str)
    try:
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=SEED, stratify=strat
        )
    except ValueError:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    train_ds  = PreprocessDataset(train_df, img_dir, steps, train=True)
    val_ds    = PreprocessDataset(val_df,   img_dir, steps, train=False)
    train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = load_model(model_name)
    model.to(DEVICE)

    optimizer       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion       = torch.nn.CrossEntropyLoss()
    best_val_loss   = float("inf")
    patience_ctr    = 0
    save_path       = os.path.join(RESULTS_DIR, f"model_{group_name}.pth")

    for epoch in range(MAX_EPOCHS):
        tr_loss             = _train_epoch(model, train_ldr, optimizer, criterion)
        vl_loss, vp, vl, vt = _val_pass(model, val_ldr, criterion)
        vl_group_auc        = group_auc(vp, vl, vt, tone_value)

        auc_str = f"{vl_group_auc:.4f}" if not np.isnan(vl_group_auc) else "N/A"
        print(
            f"  epoch={epoch:3d}  train_loss={tr_loss:.4f}  "
            f"val_loss={vl_loss:.4f}  val_auc_tone{tone_value}={auc_str}"
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1

        if patience_ctr > PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    print(f"  -> Best model saved to {save_path}")
    return save_path


def evaluate_finetuned_model(model_path, group_name, tone_value, best, test_df, img_dir):
    """
    Load saved finetuned weights and evaluate on the held-out test set.

    Reports overall AUC, accuracy, F1, target-group AUC, and per-tone AUC.
    Returns a dict suitable for JSON serialisation (NaN -> None).
    """
    steps    = _steps_from_best(best)
    pipeline = combo_key(best.get("methods") or [])

    print(f"\n{'='*60}")
    print(f"TEST EVALUATION  [{group_name}]  pipeline={pipeline}")
    print(f"  n_test={len(test_df)}")
    print(f"{'='*60}")

    model = load_model(MODEL_NAME)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    test_ds = PreprocessDataset(test_df, img_dir, steps, train=False)
    preds, labels, tones = evaluate_dataset(model, test_ds)

    threshold   = 0.5
    pred_labels = (preds > threshold).astype(int)

    overall_auc = (
        float(roc_auc_score(labels, preds))
        if len(np.unique(labels)) >= 2 else float("nan")
    )
    acc = float(accuracy_score(labels, pred_labels))
    f1  = float(f1_score(labels, pred_labels, zero_division=0))

    per_tone_auc = {
        gname: group_auc(preds, labels, tones, gval)
        for gname, gval in SKIN_TONE_GROUPS.items()
    }
    target_auc = group_auc(preds, labels, tones, tone_value)

    # Fairness gap: light (fst12) vs dark (fst56)
    auc_light = per_tone_auc.get("fst12", float("nan"))
    auc_dark  = per_tone_auc.get("fst56", float("nan"))
    fairness_gap = (
        float(auc_light - auc_dark)
        if not (np.isnan(auc_light) or np.isnan(auc_dark))
        else float("nan")
    )

    print(f"  Overall AUC              : {overall_auc:.4f}")
    print(f"  Accuracy                 : {acc:.4f}")
    print(f"  F1 score                 : {f1:.4f}")
    print(f"  Target group AUC (tone={tone_value}): {target_auc:.4f}")
    print("  Per-tone AUC:")
    for gname, auc_val in per_tone_auc.items():
        auc_str = f"{auc_val:.4f}" if not np.isnan(auc_val) else "N/A"
        print(f"    {GROUP_LABELS[gname]}: {auc_str}")
    gap_str = f"{fairness_gap:+.4f}" if not np.isnan(fairness_gap) else "N/A"
    print(f"  Fairness gap (fst12-fst56): {gap_str}")

    def _safe(v):
        return None if isinstance(v, float) and np.isnan(v) else v

    return {
        "n_test":           int(len(labels)),
        "overall_auc":      _safe(overall_auc),
        "accuracy":         _safe(acc),
        "f1":               _safe(f1),
        "target_group_auc": _safe(target_auc),
        "fairness_gap_fst12_fst56": _safe(fairness_gap),
        "per_tone_auc":     {k: _safe(v) for k, v in per_tone_auc.items()},
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline loop
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load DDI metadata
    df = pd.read_csv(META_CSV)
    if "malignant" not in df.columns and "malignancy(malig=1)" in df.columns:
        df["malignant"] = df["malignancy(malig=1)"].apply(lambda x: 1 if x == 1 else 0)
    df["DDI_file"] = df["DDI_file"].astype(str)

    # ── Held-out test split (never used during preprocessing selection or training) ──
    strat_all = df["malignant"].astype(str) + "_" + df["skin_tone"].astype(str)
    try:
        fit_df, test_df = train_test_split(
            df, test_size=0.2, random_state=SEED, stratify=strat_all
        )
    except ValueError:
        fit_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    fit_df  = fit_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"\nData split: {len(fit_df)} fit samples, {len(test_df)} held-out test samples")

    summary = {}

    for group_name, tone_value in SKIN_TONE_GROUPS.items():
        print(f"\n\n{'#'*70}")
        print(f"#  GROUP: {GROUP_LABELS[group_name]}  (tone_value={tone_value})")
        print(f"{'#'*70}")

        # Load a fresh pretrained model for each group
        model = load_model(MODEL_NAME)
        model.to(DEVICE).eval()

        # ── Baseline (no preprocessing) ───────────────────────────────────────
        print("\nComputing baseline AUC (no preprocessing)...")
        baseline_ds          = PreprocessDataset(fit_df, IMG_DIR, steps=[], train=False)
        bp, bl, bt           = evaluate_dataset(model, baseline_ds)
        base_auc             = group_auc(bp, bl, bt, tone_value)
        print(f"  Baseline AUC tone={tone_value}: {base_auc:.4f}")

        # ── Stage 1 ───────────────────────────────────────────────────────────
        winners, stage1_all = stage1_benchmark(
            model, fit_df, IMG_DIR, tone_value, base_auc
        )

        if not winners:
            print(
                f"\n  No preprocessing method cleared the improvement threshold.\n"
                f"  Finetuning with no preprocessing (baseline transforms only)."
            )
            best = {"methods": [], "params": {}, "auc": base_auc}
        else:
            # ── Stage 2 ───────────────────────────────────────────────────────
            combos     = build_candidate_combos(winners)
            top_combos = stage2_rank_combos(
                model, combos, fit_df, IMG_DIR, tone_value, top_k=TOP_K_COMBOS
            )

            # ── Stage 3 ───────────────────────────────────────────────────────
            best = stage3_bayesian_tune(
                model, top_combos, fit_df, IMG_DIR, tone_value, n_trials=N_TRIALS_STAGE3
            )

        # Persist intermediate results to JSON
        group_log = {
            "group":           group_name,
            "tone_value":      tone_value,
            "baseline_auc":    float(base_auc),
            "stage1_results":  stage1_all,
            "winners":         [w["method"] for w in winners] if winners else [],
            "best_pipeline":   combo_key(best.get("methods") or []),
            "best_params":     best.get("params", {}),
            "best_auc_stage3": float(best["auc"]),
        }
        log_path = os.path.join(RESULTS_DIR, f"pipeline_{group_name}.json")
        with open(log_path, "w") as fh:
            json.dump(group_log, fh, indent=2)
        print(f"\n  Intermediate log -> {log_path}")

        # ── Final dedicated finetune ───────────────────────────────────────────
        model_path = finetune_group_model(
            MODEL_NAME, group_name, tone_value, best, fit_df, IMG_DIR
        )

        # ── Test evaluation ────────────────────────────────────────────────────
        test_results = evaluate_finetuned_model(
            model_path, group_name, tone_value, best, test_df, IMG_DIR
        )

        # Persist full per-group log (with test results)
        group_log["test_results"] = test_results
        with open(log_path, "w") as fh:
            json.dump(group_log, fh, indent=2)

        summary[group_name] = {**group_log, "model_path": model_path}

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("PIPELINE COMPLETE — Summary")
    print(f"{'='*70}")
    for gname, res in summary.items():
        delta    = res["best_auc_stage3"] - res["baseline_auc"]
        tr       = res.get("test_results", {})
        test_auc = tr.get("overall_auc")
        tgt_auc  = tr.get("target_group_auc")
        gap      = tr.get("fairness_gap_fst12_fst56")
        print(f"\n{GROUP_LABELS[gname]}:")
        print(f"  Pipeline     : {res['best_pipeline']}")
        print(f"  Params       : {res['best_params']}")
        print(
            f"  Stage-3 AUC  : {res['baseline_auc']:.4f} -> {res['best_auc_stage3']:.4f}"
            f"  (delta={delta:+.4f})"
        )
        print(f"  Test overall AUC      : {test_auc:.4f}" if test_auc is not None else "  Test overall AUC: N/A")
        print(f"  Test target-group AUC : {tgt_auc:.4f}" if tgt_auc is not None else "  Test target-group AUC: N/A")
        print(f"  Test fairness gap     : {gap:+.4f}" if gap is not None else "  Test fairness gap: N/A")
        print(f"  Model        : {res['model_path']}")

    # Write a combined summary JSON
    summary_path = os.path.join(RESULTS_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "stage1_results"}
             for k, v in summary.items()},
            fh, indent=2,
        )
    print(f"\nFull summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    run_pipeline()
