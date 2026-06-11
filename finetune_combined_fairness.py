"""
Finetune a single InceptionV3 model on all DDI images using per-group
preprocessing pipelines chosen by fairness_search_pipeline.py (Stages 1-3).

Each image is preprocessed by the pipeline that was optimised for its FST
skin-tone group (fst12 / fst34 / fst56). Training uses all 656 images with a
60 / 20 / 20 stratified train / val / test split.

Prerequisites:
  Place the three Stage-3 JSON logs from fairness_search_pipeline.py into
  RESULTS_DIR before running:
    fairness_pipeline_results/pipeline_fst12.json
    fairness_pipeline_results/pipeline_fst34.json
    fairness_pipeline_results/pipeline_fst56.json
"""

import json
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eval_data import load_model
from generate_preprocessed import method_map

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME   = "HAM10000"
DATA_DIR     = "DDI"
IMG_DIR      = os.path.join(DATA_DIR, "images")
META_CSV     = os.path.join(DATA_DIR, "ddi_metadata.csv")
RESULTS_DIR  = "fairness_pipeline_results"

SEED         = 42
BATCH_SIZE   = 16
LR           = 1e-5
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 200
PATIENCE     = 20

SKIN_TONE_GROUPS = {"fst12": 12, "fst34": 34, "fst56": 56}
GROUP_LABELS = {
    "fst12": "FST I-II   (light)",
    "fst34": "FST III-IV (medium)",
    "fst56": "FST V-VI   (dark)",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tunable parameter spec — must match fairness_search_pipeline.py
TUNABLE_PARAMS = {
    "clahe":             {"clip_limit":   (2.0,  (0.5,  4.0))},
    "illumination_comp": {"sigma":        (50.0, (10.0, 100.0))},
    "percentile_norm":   {"range_":      (98.0, (80.0, 99.0))},
    "local_contrast":    {"strength":    (1.25, (0.1,  2.0))},
    "bilateral":         {"sigma_color": (75.0, (10.0, 150.0))},
    "non_local_means":   {"h":           (10,   (5,    15))},
}

# ─────────────────────────────────────────────────────────────────────────────
# Transforms
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


def _to_label(val):
    s = str(val).strip().lower()
    if s in ("true", "1"):
        return 1
    if s in ("false", "0"):
        return 0
    raise ValueError(f"Cannot parse label: {val!r}")

# ─────────────────────────────────────────────────────────────────────────────
# Load Stage-3 results
# ─────────────────────────────────────────────────────────────────────────────

def load_group_best(results_dir, group_name):
    """Read pipeline_{group_name}.json and return methods, params, tone_value."""
    path = os.path.join(results_dir, f"pipeline_{group_name}.json")
    with open(path) as f:
        d = json.load(f)
    pipeline_str = d.get("best_pipeline", "none")
    methods = pipeline_str.split("+") if pipeline_str and pipeline_str != "none" else []
    return {
        "methods":    methods,
        "params":     d.get("best_params", {}),
        "tone_value": d["tone_value"],
    }


def steps_from_best(best):
    """Reconstruct [(method, param_value), ...] from a loaded best dict."""
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


def build_tone_to_steps(results_dir):
    """
    Returns tone_to_steps: {tone_int: [(method, param), ...]}
    and logs the loaded pipeline per group.
    """
    tone_to_steps = {}
    print("\nLoaded Stage-3 preprocessing pipelines:")
    for gname, tone_val in SKIN_TONE_GROUPS.items():
        best  = load_group_best(results_dir, gname)
        steps = steps_from_best(best)
        tone_to_steps[tone_val] = steps
        pipeline_str = "+".join(best["methods"]) if best["methods"] else "none"
        print(f"  {GROUP_LABELS[gname]} (tone={tone_val}):  {pipeline_str}")
        if steps:
            for m, v in steps:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"    {m}: {val_str}")
    return tone_to_steps

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MultiGroupPreprocessDataset(Dataset):
    """
    Loads raw DDI images and applies the preprocessing steps that correspond to
    each image's skin-tone group, then applies train augmentation or val transform.

    tone_to_steps: {tone_int: [(method_name, param_value), ...]}
    """

    def __init__(self, df, img_dir, tone_to_steps, train=False):
        self.df           = df.copy().reset_index(drop=True)
        self.df["malignant"] = self.df["malignant"].apply(_to_label)
        self.img_dir      = img_dir
        self.tone_to_steps = tone_to_steps
        self.tf           = TRAIN_AUGMENT if train else VAL_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.img_dir, str(row["DDI_file"]))
        img   = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        tone  = int(row["skin_tone"])
        steps = self.tone_to_steps.get(tone, [])
        for mname, pval in steps:
            img = apply_method(img, mname, pval)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return self.tf(pil), int(row["malignant"]), tone

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_auc_ci(y_true, y_score, n_boot=1000, alpha=0.95, seed=42):
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return None
    rng  = np.random.default_rng(seed)
    aucs = []
    n    = len(y_true)
    for _ in range(n_boot):
        idx  = rng.integers(0, n, n)
        y_b  = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, y_score[idx]))
    if not aucs:
        return None
    lo = (1 - alpha) / 2
    hi = 1 - lo
    return float(np.quantile(aucs, lo)), float(np.quantile(aucs, hi))


def group_auc(preds, labels, tones, tone_value):
    mask = tones == tone_value
    y, p = labels[mask], preds[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


@torch.no_grad()
def run_inference(model, loader):
    model.eval()
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

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _mixup(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x  = x.to(DEVICE)
        y  = y.to(DEVICE, dtype=torch.long)
        x, ya, yb, lam = _mixup(x, y)
        out, aux = model(x)
        loss = (
            criterion(out, ya) * lam + criterion(out, yb) * (1 - lam)
            + 0.4 * (criterion(aux, ya) * lam + criterion(aux, yb) * (1 - lam))
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def val_pass(model, loader, criterion):
    model.eval()
    total  = 0.0
    preds, labels, tones = [], [], []
    with torch.no_grad():
        for x, y, t in loader:
            x  = x.to(DEVICE)
            y  = y.to(DEVICE, dtype=torch.long)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            total += criterion(out, y).item()
            prob = torch.softmax(out, dim=1)[:, 1]
            preds.extend(prob.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
            tones.extend(t.numpy().tolist())
    vl  = total / len(loader)
    p   = np.array(preds)
    l   = np.array(labels)
    t   = np.array(tones)
    return vl, p, l, t

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ── Load Stage-3 preprocessing per group ─────────────────────────────────
    tone_to_steps = build_tone_to_steps(RESULTS_DIR)

    # ── Load metadata ─────────────────────────────────────────────────────────
    df = pd.read_csv(META_CSV)
    if "malignant" not in df.columns and "malignancy(malig=1)" in df.columns:
        df["malignant"] = df["malignancy(malig=1)"].apply(lambda x: 1 if x == 1 else 0)
    df["DDI_file"] = df["DDI_file"].astype(str)
    print(f"\nTotal images: {len(df)}")

    # ── 60 / 20 / 20 stratified split ────────────────────────────────────────
    strat = df["malignant"].astype(str) + "_" + df["skin_tone"].astype(str)
    try:
        fit_df, test_df = train_test_split(
            df, test_size=0.20, random_state=SEED, stratify=strat
        )
    except ValueError:
        fit_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED)

    strat_fit = fit_df["malignant"].astype(str) + "_" + fit_df["skin_tone"].astype(str)
    try:
        train_df, val_df = train_test_split(
            fit_df, test_size=0.25, random_state=SEED, stratify=strat_fit
        )
    except ValueError:
        train_df, val_df = train_test_split(fit_df, test_size=0.25, random_state=SEED)

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"Split  — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds  = MultiGroupPreprocessDataset(train_df, IMG_DIR, tone_to_steps, train=True)
    val_ds    = MultiGroupPreprocessDataset(val_df,   IMG_DIR, tone_to_steps, train=False)
    test_ds   = MultiGroupPreprocessDataset(test_df,  IMG_DIR, tone_to_steps, train=False)
    train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_ldr  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model     = load_model(MODEL_NAME)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    save_path      = os.path.join(RESULTS_DIR, "model_combined.pth")
    best_val_loss  = float("inf")
    patience_ctr   = 0

    print(f"\n{'='*60}")
    print(f"Training on {DEVICE} — MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}")
    print(f"{'='*60}")

    for epoch in range(MAX_EPOCHS):
        tr_loss              = train_epoch(model, train_ldr, optimizer, criterion)
        vl_loss, vp, vl, vt = val_pass(model, val_ldr, criterion)

        val_auc_overall = (
            float(roc_auc_score(vl, vp)) if len(np.unique(vl)) >= 2 else float("nan")
        )
        auc_parts = []
        for gname, tone_val in SKIN_TONE_GROUPS.items():
            ga = group_auc(vp, vl, vt, tone_val)
            auc_parts.append(f"val_auc_{gname}={ga:.4f}" if not np.isnan(ga) else f"val_auc_{gname}=N/A")

        overall_str = f"{val_auc_overall:.4f}" if not np.isnan(val_auc_overall) else "N/A"
        print(
            f"  epoch={epoch:3d}  train_loss={tr_loss:.4f}  "
            f"val_loss={vl_loss:.4f}  val_auc={overall_str}  "
            + "  ".join(auc_parts)
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

    print(f"\n  Best model saved to {save_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST EVALUATION")
    print(f"{'='*60}")

    state = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    preds, labels, tones = run_inference(model, test_ldr)

    overall_auc = (
        float(roc_auc_score(labels, preds)) if len(np.unique(labels)) >= 2 else float("nan")
    )
    overall_ci = bootstrap_auc_ci(labels, preds, seed=SEED)

    ci_str = (
        f" (95% CI: {overall_ci[0]:.4f}, {overall_ci[1]:.4f})"
        if overall_ci else ""
    )
    print(f"  Overall AUC: {overall_auc:.4f}{ci_str}")

    per_tone_auc = {}
    for gname, tone_val in SKIN_TONE_GROUPS.items():
        ga  = group_auc(preds, labels, tones, tone_val)
        gci = bootstrap_auc_ci(labels[tones == tone_val], preds[tones == tone_val], seed=SEED)
        per_tone_auc[gname] = ga
        auc_s = f"{ga:.4f}" if not np.isnan(ga) else "N/A"
        ci_s  = (
            f" (95% CI: {gci[0]:.4f}, {gci[1]:.4f})"
            if gci else ""
        )
        n = int((tones == tone_val).sum())
        print(f"  {GROUP_LABELS[gname]}: AUC={auc_s}{ci_s}  n={n}")

    auc_light = per_tone_auc.get("fst12", float("nan"))
    auc_dark  = per_tone_auc.get("fst56", float("nan"))
    fairness_gap = (
        float(auc_light - auc_dark)
        if not (np.isnan(auc_light) or np.isnan(auc_dark))
        else float("nan")
    )
    gap_str = f"{fairness_gap:+.4f}" if not np.isnan(fairness_gap) else "N/A"
    print(f"  Fairness gap (fst12 - fst56): {gap_str}")

    def _safe(v):
        return None if isinstance(v, float) and np.isnan(v) else v

    results = {
        "n_test":       int(len(labels)),
        "overall_auc":  _safe(overall_auc),
        "overall_ci":   list(overall_ci) if overall_ci else None,
        "per_tone_auc": {k: _safe(v) for k, v in per_tone_auc.items()},
        "fairness_gap_fst12_fst56": _safe(fairness_gap),
        "model_path":   save_path,
    }
    out_path = os.path.join(RESULTS_DIR, "combined_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
