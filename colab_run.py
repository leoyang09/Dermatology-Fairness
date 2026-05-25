#!/usr/bin/env python3
"""
colab_run.py  —  Standalone fairness optimization pipeline for Google Colab.

No external local modules are required; every helper is inlined.

═══════════════════════════════════════════════════════════════════
  HOW TO USE IN GOOGLE COLAB
═══════════════════════════════════════════════════════════════════
  1. Upload this file to your Colab session (Files → Upload).

  2. Place the DDI dataset so the path below is correct:
       DATA_DIR/images/   ← all .png images
       DATA_DIR/ddi_metadata.csv

     Recommended: upload your DDI folder to Google Drive, mount it,
     and set DATA_DIR = "/content/drive/MyDrive/DDI"

  3. Pre-trained weights (HAM10000.pth / DeepDerm.pth) are downloaded
     automatically from Zenodo on first run into MODEL_DIR.

  4. Run in a code cell:
       !python colab_run.py
     or:
       %run colab_run.py

  5. Outputs are written to RESULTS_DIR:
       pipeline_fst12.json / fst34 / fst56  ← per-group logs
       pipeline_summary.json                ← combined summary
       model_fst12.pth / fst34 / fst56      ← dedicated fine-tuned weights
═══════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Auto-install: only 'optuna' is missing from the standard Colab image
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna"],
                      stdout=subprocess.DEVNULL)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Optional Google Drive mount
# ─────────────────────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("Google Drive mounted at /content/drive")
except Exception:
    print("Not running in Colab or Drive already mounted — skipping mount.")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Standard imports
# ─────────────────────────────────────────────────────────────────────────────
import itertools
import json
import os
import random
import urllib.request

import cv2
import numpy as np
import optuna
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ═════════════════════════════════════════════════════════════════════════════
# ██  EDIT THIS SECTION TO MATCH YOUR FILE LAYOUT  ██
# ═════════════════════════════════════════════════════════════════════════════
#
#  If DDI lives in Google Drive:
#    DATA_DIR = "/content/drive/MyDrive/DDI"
#  If you uploaded directly to the Colab VM:
#    DATA_DIR = "/content/DDI"
#
DATA_DIR    = "/content/DDI"
MODEL_DIR   = "/content/DDI-models"
RESULTS_DIR = "/content/fairness_pipeline_results"
MODEL_NAME  = "HAM10000"    # "HAM10000" or "DeepDerm"
# ═════════════════════════════════════════════════════════════════════════════

IMG_DIR  = os.path.join(DATA_DIR, "images")
META_CSV = os.path.join(DATA_DIR, "ddi_metadata.csv")

# Training hyperparameters
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 16
LR           = 1e-5
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 200
PATIENCE     = 20
SEED         = 42

# Pipeline hyperparameters
N_TRIALS_STAGE3 = 20    # Optuna trials per pipeline in Stage 3
TOP_K_COMBOS    = 3     # Stage-2 pipelines forwarded to Stage 3
MIN_AUC_DELTA   = 0.005 # Min AUC gain for a Stage-1 method to be a "winner"

SKIN_TONE_GROUPS = {
    "fst12": 12,    # FST I-II   (light)
    "fst34": 34,    # FST III-IV (medium)
    "fst56": 56,    # FST V-VI   (dark)
}
GROUP_LABELS = {
    "fst12": "FST I-II   (light)",
    "fst34": "FST III-IV (medium)",
    "fst56": "FST V-VI   (dark)",
}

print(f"Device: {DEVICE}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Model loading  (inlined from eval_data.py)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_WEB_PATHS = {
    "HAM10000": "https://zenodo.org/record/6784279/files/HAM10000.pth",
    "DeepDerm": "https://zenodo.org/record/6784279/files/DeepDerm.pth",
    "GroupDRO": "https://drive.google.com/uc?id=193ippDUYpMaOaEyLjd1DNsOiW0aRXL75",
    "CORAL":    "https://drive.google.com/uc?id=18rMU0nRd4LiHN9WkXoDROJ2o2sG1_GD8",
    "CDANN":    "https://drive.google.com/uc?id=1PvvgQVqcrth840bFZ3ddLdVSL7NkxiRK",
}

_MODEL_THRESHOLDS = {
    "HAM10000": 0.733,
    "DeepDerm": 0.687,
    "GroupDRO": 0.980,
    "CORAL":    0.990,
    "CDANN":    0.980,
}


def load_model(model_name, save_dir=None, download=True):
    """Load pretrained InceptionV3 DDI model; download from Zenodo if missing."""
    if save_dir is None:
        save_dir = MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name.lower()}.pth")

    if not os.path.exists(model_path):
        if not download:
            raise FileNotFoundError(
                f"Weights not found at {model_path}. Set download=True or supply the file."
            )
        url = _MODEL_WEB_PATHS[model_name]
        print(f"Downloading {model_name} weights from {url} ...")
        if "drive.google.com" in url:
            import gdown
            gdown.download(url, model_path, quiet=False)
        else:
            urllib.request.urlretrieve(url, model_path)
        print("  Download complete.")

    # Build InceptionV3 (handle both old and new torchvision APIs)
    try:
        model = torchvision.models.inception_v3(weights=None, transform_input=True)
    except TypeError:
        model = torchvision.models.inception_v3(
            init_weights=False, pretrained=False, transform_input=True
        )
    model.fc            = torch.nn.Linear(2048, 2)
    model.AuxLogits.fc  = torch.nn.Linear(768,  2)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model._ddi_name      = model_name
    model._ddi_threshold = _MODEL_THRESHOLDS[model_name]
    model._ddi_web_path  = _MODEL_WEB_PATHS[model_name]
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Preprocessing functions  (inlined from generate_preprocessed.py)
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(img, clip_limit=2.0):
    """CLAHE on the L channel of LAB space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_adaptive_gamma(img):
    """Adaptive Gamma Correction via mean brightness of HSV Value channel."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = np.clip(np.mean(v) / 255.0, 1e-5, 1.0)
    gamma = np.log(0.5) / np.log(m)
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    v_corrected = cv2.LUT(v, table)
    return cv2.cvtColor(cv2.merge((h, s, v_corrected)), cv2.COLOR_HSV2BGR)


def apply_white_balance(img):
    """Gray World White Balance (LAB-based)."""
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def _apply_white_balance_simple(img):
    """Simple Gray World: scale channels so their means are equal."""
    b, g, r = cv2.split(img)
    avg = (np.mean(b) + np.mean(g) + np.mean(r)) / 3
    def _scale(ch):
        m = np.mean(ch)
        return cv2.convertScaleAbs(ch, alpha=avg / m) if m > 0 else ch
    return cv2.merge((_scale(b), _scale(g), _scale(r)))


def apply_msrcr(img, gamma=128, scales=(15, 80, 250), alpha=125.0, beta=46.0):
    """Multi-Scale Retinex with Color Restoration."""
    gain = gamma / 128.0
    img_f = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img_f)
    for s in scales:
        blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=float(s), sigmaY=float(s))
        retinex += np.log(img_f) - np.log(blur + 1.0)
    retinex /= float(len(scales))
    sum_ch = np.sum(img_f, axis=2, keepdims=True)
    color_restore = beta * (np.log(alpha * img_f) - np.log(sum_ch + 1.0))
    msrcr = gain * (retinex * color_restore)
    out = np.zeros_like(msrcr)
    for c in range(3):
        ch = msrcr[:, :, c] - np.min(msrcr[:, :, c])
        ch = (ch / (np.max(ch) + 1e-6)) * 255.0
        out[:, :, c] = ch
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_homomorphic(img, cutoff=0.35, order=2.0, boost=1.6):
    """Homomorphic filtering for illumination normalisation (V channel)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    v = np.clip(v, 1.0, 255.0)
    v_log = np.log(v)
    v_fft_shift = np.fft.fftshift(np.fft.fft2(v_log))
    rows, cols = v.shape
    cy, cx = rows // 2, cols // 2
    X, Y = np.meshgrid(np.arange(cols) - cx, np.arange(rows) - cy)
    D = np.sqrt(X * X + Y * Y)
    D0 = max(cutoff * np.max(D), 1e-6)
    H = 1.0 - 1.0 / (1.0 + (D / D0) ** (2.0 * order))
    v_filt = v_fft_shift * (1.0 + (boost - 1.0) * H)
    v_out = np.real(np.fft.ifft2(np.fft.ifftshift(v_filt)))
    v_out = np.exp(v_out)
    v_out = (v_out - np.min(v_out)) / (np.max(v_out) + 1e-6) * 255.0
    hsv_out = cv2.merge((h.astype(np.uint8), s.astype(np.uint8), v_out.astype(np.uint8)))
    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)


def apply_percentile_norm(img, range_=98.0):
    """Robust brightness normalisation using HSV Value channel percentiles."""
    low_p, high_p = (100.0 - range_) / 2.0, 100.0 - (100.0 - range_) / 2.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_f = v.astype(np.float32)
    lo, hi = np.percentile(v_f, low_p), np.percentile(v_f, high_p)
    if hi <= lo + 1e-6:
        return img
    v_f = np.clip((v_f - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return cv2.cvtColor(cv2.merge((h, s, v_f.astype(np.uint8))), cv2.COLOR_HSV2BGR)


def apply_local_contrast(img, sigma=2.0, strength=1.25):
    """Unsharp-masking on the L channel (LAB) to boost local contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    blur   = cv2.GaussianBlur(l, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    l_enh  = np.clip(l + strength * (l - blur), 0, 255)
    return cv2.cvtColor(cv2.merge((l_enh, a, b)).astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_illumination_comp(img, sigma=50.0, eps=1e-6):
    """Illumination compensation via large-scale Gaussian on the L channel (LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    illum  = cv2.GaussianBlur(l, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma)) + eps
    l_corr = l / illum
    l_corr = (l_corr - np.min(l_corr)) / (np.max(l_corr) + eps) * 255.0
    return cv2.cvtColor(cv2.merge((l_corr, a, b)).astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_z_score_norm(img, scale=1.0, eps=1e-6):
    """Per-channel Z-score normalisation, rescaled to [0, 255]."""
    img_f = img.astype(np.float32)
    out   = np.zeros_like(img_f)
    for c in range(3):
        ch   = img_f[:, :, c]
        z    = (ch - np.mean(ch)) / max(np.std(ch) * scale, eps)
        z   -= np.min(z)
        z    = z / (np.max(z) + eps) * 255.0
        out[:, :, c] = z
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_color_constancy(img, power=6, sigma=0):
    """Shades of Gray colour constancy (Minkowski norm generalisation)."""
    img_f = img.astype(np.float32) + 1e-6
    if sigma > 0:
        img_f = cv2.GaussianBlur(img_f, (0, 0), sigma)
    norm   = np.power(np.mean(np.power(img_f, power), axis=(0, 1)), 1 / power)
    result = img.astype(np.float32)
    mean_n = np.mean(norm)
    for i in range(3):
        if norm[i] > 0:
            result[:, :, i] = result[:, :, i] / norm[i] * mean_n
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_lab_color_normalization(img, target_a=128, target_b=128):
    """Shift a and b LAB channels to their target means."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] = lab[:, :, 1] - np.mean(lab[:, :, 1]) + target_a
    lab[:, :, 2] = lab[:, :, 2] - np.mean(lab[:, :, 2]) + target_b
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_clahe_mild(img):
    """CLAHE with clip_limit=1.0 (gentle contrast enhancement)."""
    return apply_clahe(img, clip_limit=1.0)


def apply_clahe_very_mild(img):
    """CLAHE with clip_limit=0.5 (minimal contrast adjustment)."""
    return apply_clahe(img, clip_limit=0.5)


def apply_clahe_blended(img, strength=0.5):
    """Blend CLAHE-enhanced image with original (strength controls CLAHE weight)."""
    return cv2.addWeighted(img, 1 - strength, apply_clahe(img), strength, 0)


def apply_white_balance_mild(img, strength=0.3):
    """Partial white balance: 30% corrected + 70% original."""
    return cv2.addWeighted(img, 1 - strength, _apply_white_balance_simple(img), strength, 0)


def apply_white_balance_very_mild(img, strength=0.15):
    """Very subtle white balance: 15% corrected + 85% original."""
    return cv2.addWeighted(img, 1 - strength, _apply_white_balance_simple(img), strength, 0)


def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """Edge-preserving bilateral filter."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def apply_non_local_means(img, h=10, template_window=7, search_window=21):
    """Non-local means denoising (colour)."""
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window, search_window)


def apply_skin_tone_shift(img, shift_amount=20):
    """Hue shift to simulate appearance on different skin tones."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift_amount) % 180
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.9, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# Lookup table: method name -> function  (matches generate_preprocessed.py)
method_map = {
    "clahe":                 apply_clahe,
    "adaptive_gamma":        apply_adaptive_gamma,
    "white_balance":         apply_white_balance,
    "msrcr":                 apply_msrcr,
    "homomorphic":           apply_homomorphic,
    "percentile_norm":       apply_percentile_norm,
    "local_contrast":        apply_local_contrast,
    "illumination_comp":     apply_illumination_comp,
    "Z_score_norm":          apply_z_score_norm,
    "color_constancy":       apply_color_constancy,
    "lab_color_norm":        apply_lab_color_normalization,
    "clahe_mild":            apply_clahe_mild,
    "clahe_very_mild":       apply_clahe_very_mild,
    "clahe_blended":         apply_clahe_blended,
    "white_balance_mild":    apply_white_balance_mild,
    "white_balance_very_mild": apply_white_balance_very_mild,
    "bilateral":             apply_bilateral_filter,
    "non_local_means":       apply_non_local_means,
    "skin_tone_shift":       apply_skin_tone_shift,
}

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Preprocessing catalog
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

# Fixed methods: called with their internal defaults, no tunable parameter
FIXED_METHODS = {"adaptive_gamma", "white_balance", "msrcr", "Z_score_norm"}

ALL_CANDIDATE_METHODS = list(TUNABLE_PARAMS.keys()) + sorted(FIXED_METHODS)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Semantic category ordering
#     (mirrors the triple ordering convention in generate_preprocessed.py:
#      denoising -> color -> illumination -> contrast -> normalization)
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
# 7.  Transforms  (mirror finetune_ddi.py)
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
# 8.  Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_method(cv_img, method_name, param_value=None):
    """Apply one preprocessing step (BGR in, BGR out).
    Tunable methods receive param_value; fixed methods use their internal defaults.
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
    """Join a method list with '+', matching the reference code naming convention."""
    return "+".join(methods) if methods else "none"

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Dataset
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
    (steps), then applies train augmentation or val-time transform.

    steps : list of (method_name, param_value) tuples.
            Pass [] for no preprocessing (baseline transform only).
    train : if True, uses TRAIN_AUGMENT; otherwise uses VAL_TRANSFORM.
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
# 10. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dataset(model, dataset, batch_size=32):
    """Return (preds, labels, tones) numpy arrays over the full dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)
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
    """AUC restricted to one skin-tone group; returns nan if not computable."""
    mask = tones == tone_value
    y, p = labels[mask], preds[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def evaluate_steps(model, steps, df, img_dir, tone_value, batch_size=32):
    """Build dataset with preprocessing steps, evaluate, return group AUC."""
    ds      = PreprocessDataset(df, img_dir, steps, train=False)
    p, l, t = evaluate_dataset(model, ds, batch_size=batch_size)
    return group_auc(p, l, t, tone_value)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Stage 1 — Single-method benchmark
# ─────────────────────────────────────────────────────────────────────────────

def stage1_benchmark(model, df, img_dir, tone_value, baseline_auc):
    """
    Test every candidate preprocessing method with default parameters.

    Returns:
        winners  : list of {"method", "auc", "delta"} dicts, filtered to those
                   that improved group AUC by >= MIN_AUC_DELTA, sorted best-first.
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
# 12. Stage 2 — Pipeline combination and ranking
# ─────────────────────────────────────────────────────────────────────────────

def _stage_idx(method):
    cat = METHOD_CATEGORY.get(method, "other")
    try:
        return STAGE_ORDER.index(cat)
    except ValueError:
        return len(STAGE_ORDER)


def _is_ordered(methods):
    """True iff methods respect the canonical category ordering (non-decreasing)."""
    idxs = [_stage_idx(m) for m in methods]
    return all(idxs[i] <= idxs[i + 1] for i in range(len(idxs) - 1))


def build_candidate_combos(winners):
    """
    Build 2- and 3-method pipelines from the winner list following the
    reference-code ordering:
      denoising -> color -> illumination -> contrast -> normalization

    Only permutations that respect this ordering are kept.
    Single-method entries are appended as fallback.
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
# 13. Stage 3 — Bayesian hyperparameter tuning (Optuna TPE)
# ─────────────────────────────────────────────────────────────────────────────

def _build_steps_from_trial(trial, methods):
    """Suggest hyperparameters for each method inside an Optuna trial."""
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

    print(f"\n  -> Globally best: {combo_key(best['methods'] or [])}  AUC={best['auc']:.4f}")
    return best

# ─────────────────────────────────────────────────────────────────────────────
# 14. Final finetuning
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
    """Return (val_loss, preds, labels, tones) over the validation set."""
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

    Trains on the full DDI dataset (all tones) with the winning preprocessing
    pipeline selected to maximise accuracy for this group.
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

    # Stratified 80/20 train/val split; fall back to unstratified if a stratum
    # is too small for sklearn's train_test_split.
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
    train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = load_model(model_name)
    model.to(DEVICE)

    optimizer     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion     = torch.nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    patience_ctr  = 0
    save_path     = os.path.join(RESULTS_DIR, f"model_{group_name}.pth")

    for epoch in range(MAX_EPOCHS):
        tr_loss              = _train_epoch(model, train_ldr, optimizer, criterion)
        vl_loss, vp, vl, vt = _val_pass(model, val_ldr, criterion)
        vl_group_auc         = group_auc(vp, vl, vt, tone_value)

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

# ─────────────────────────────────────────────────────────────────────────────
# 15. Main pipeline loop
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = pd.read_csv(META_CSV)
    if "malignant" not in df.columns and "malignancy(malig=1)" in df.columns:
        df["malignant"] = df["malignancy(malig=1)"].apply(lambda x: 1 if x == 1 else 0)
    df["DDI_file"] = df["DDI_file"].astype(str)

    print(f"\nLoaded {len(df)} rows from {META_CSV}")
    print(f"Skin tone distribution:\n{df['skin_tone'].value_counts().sort_index()}\n")

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
        baseline_ds = PreprocessDataset(df, IMG_DIR, steps=[], train=False)
        bp, bl, bt  = evaluate_dataset(model, baseline_ds)
        base_auc    = group_auc(bp, bl, bt, tone_value)
        print(f"  Baseline AUC tone={tone_value}: {base_auc:.4f}")

        # ── Stage 1 ───────────────────────────────────────────────────────────
        winners, stage1_all = stage1_benchmark(
            model, df, IMG_DIR, tone_value, base_auc
        )

        if not winners:
            print(
                "\n  No preprocessing method cleared the improvement threshold.\n"
                "  Finetuning with baseline transforms only (no preprocessing)."
            )
            best = {"methods": [], "params": {}, "auc": base_auc}
        else:
            # ── Stage 2 ───────────────────────────────────────────────────────
            combos     = build_candidate_combos(winners)
            top_combos = stage2_rank_combos(
                model, combos, df, IMG_DIR, tone_value, top_k=TOP_K_COMBOS
            )

            # ── Stage 3 ───────────────────────────────────────────────────────
            best = stage3_bayesian_tune(
                model, top_combos, df, IMG_DIR, tone_value, n_trials=N_TRIALS_STAGE3
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
            MODEL_NAME, group_name, tone_value, best, df, IMG_DIR
        )

        summary[group_name] = {**group_log, "model_path": model_path}

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("PIPELINE COMPLETE — Summary")
    print(f"{'='*70}")
    for gname, res in summary.items():
        delta = res["best_auc_stage3"] - res["baseline_auc"]
        print(f"\n{GROUP_LABELS[gname]}:")
        print(f"  Pipeline : {res['best_pipeline']}")
        print(f"  Params   : {res['best_params']}")
        print(
            f"  AUC      : {res['baseline_auc']:.4f} -> {res['best_auc_stage3']:.4f}"
            f"  (delta={delta:+.4f})"
        )
        print(f"  Model    : {res['model_path']}")

    # Write combined summary JSON (strip per-method stage1 detail to keep it small)
    summary_path = os.path.join(RESULTS_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "stage1_results"}
             for k, v in summary.items()},
            fh, indent=2,
        )
    print(f"\nFull summary -> {summary_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 16. Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
