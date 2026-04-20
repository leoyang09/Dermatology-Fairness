import os
import re
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, classification_report
from torchvision import transforms
from PIL import Image
from eval_data import load_model  # your existing load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_tone_value(tone):
    if isinstance(tone, torch.Tensor):
        return int(tone.item())
    return tone

def bootstrap_auc_ci(y_true, y_score, n_boot=1000, alpha=0.95, seed=42):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return None
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_b, y_score[idx])
        aucs.append(auc(fpr_b, tpr_b))
    if not aucs:
        return None
    lo = (1 - alpha) / 2
    hi = 1 - lo
    return float(np.quantile(aucs, lo)), float(np.quantile(aucs, hi))

# ---------------------------
# Dataset for test CSV
# ---------------------------
class DDITestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["DDI_file"])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["malignant"])
        return image, label, img_path, row["skin_tone"]

# ---------------------------
# Evaluation
# ---------------------------
def eval_model_on_csv(model, csv_file, data_dir):
    model.to(DEVICE).eval()
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = DDITestDataset(csv_file, data_dir, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    preds, labels, paths, tones = [], [], [], []

    with torch.no_grad():
        for x, y, p, t in loader:
            x = x.to(DEVICE)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]  # handle Inception aux output
            prob = torch.softmax(out, dim=1)[:, 1]
            preds.extend(prob.cpu().numpy())
            labels.extend(y.numpy())
            paths.extend(p)
            tones.extend([_normalize_tone_value(v) for v in t])

    preds = np.array(preds)
    labels = np.array(labels)
    fpr, tpr, _ = roc_curve(labels, preds)
    auc_score = auc(fpr, tpr)
    auc_ci = bootstrap_auc_ci(labels, preds)

    # Per-skin-tone ROC-AUC (only valid when both classes exist for that tone)
    tone_auc = {}
    tone_auc_ci = {}
    unique_tones = sorted(set(tones))
    for tone in unique_tones:
        tone_mask = np.array([t == tone for t in tones], dtype=bool)
        tone_labels = labels[tone_mask]
        tone_preds = preds[tone_mask]

        if len(np.unique(tone_labels)) < 2:
            tone_auc[str(tone)] = None
            tone_auc_ci[str(tone)] = None
            continue

        tone_fpr, tone_tpr, _ = roc_curve(tone_labels, tone_preds)
        tone_auc[str(tone)] = auc(tone_fpr, tone_tpr)
        tone_auc_ci[str(tone)] = bootstrap_auc_ci(tone_labels, tone_preds)

    report = classification_report(labels, (preds > model._ddi_threshold).astype(int),
                                   target_names=["benign", "malignant"])

    results = {
        "predicted_labels": preds,
        "true_labels": labels,
        "images": paths,
        "skin_tones": tones,
        "ROC_AUC": auc_score,
        "ROC_AUC_95CI": auc_ci,
        "ROC_AUC_by_skin_tone": tone_auc,
        "ROC_AUC_by_skin_tone_95CI": tone_auc_ci,
        "report": report,
        "threshold": model._ddi_threshold,
        "model": model._ddi_name,
    }

    return results

# ---------------------------
# Load fine-tuned weights per seed
# ---------------------------
import os
import torch

def load_seed_model(model_name, seed, epoch=None, model_dir="."):
    base_model = load_model(model_name, download=False)

    # Build filename
    if epoch is None:
        weights_path = os.path.join(model_dir, f"{model_name}_seed{seed}.pth")
        model_tag = f"{model_name}_seed{seed}"
    else:
        weights_path = os.path.join(
            model_dir, f"{model_name}_seed{seed}_epoch{epoch}.pth"
        )
        model_tag = f"{model_name}_seed{seed}_epoch{epoch}"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    base_model.load_state_dict(state_dict)

    base_model._ddi_name = model_tag
    return base_model

def find_available_epochs(model_name, seed, model_dir):
    pattern = re.compile(rf"{model_name}_seed{seed}_epoch(\d+)\.pth")
    epochs = []

    for fname in os.listdir(model_dir):
        match = pattern.match(fname)
        if match:
            epochs.append(int(match.group(1)))

    return sorted(epochs)

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    TEST_CSV = "test.csv"
    DATA_DIR = "DDI/images"
    WEIGHTS_DIR = "."
    EVAL_DIR = "DDI-results/baseline_finetuned_models"
    os.makedirs(EVAL_DIR, exist_ok=True)

    MODEL_NAMES = [
    #"DeepDerm", 
    "HAM10000"
    ]
    NUM_SEEDS = 1

    for model_name in MODEL_NAMES:
        for seed in range(NUM_SEEDS):

            epochs = find_available_epochs(model_name, seed, WEIGHTS_DIR)

            # fallback if no epoch files exist
            if not epochs:
                epochs = [None]

            for epoch in epochs:
                tag = f"seed {seed}" if epoch is None else f"seed {seed} epoch {epoch}"
                print(f"Evaluating {model_name} {tag}...")

                try:
                    model = load_seed_model(
                        model_name,
                        seed,
                        epoch=epoch,
                        model_dir=WEIGHTS_DIR
                    )
                except FileNotFoundError as e:
                    print(e)
                    continue

                results = eval_model_on_csv(model, TEST_CSV, DATA_DIR)

                # Save path
                if epoch is None:
                    save_name = f"{model_name}_seed{seed}-evaluation.pkl"
                else:
                    save_name = f"{model_name}_seed{seed}_epoch{epoch}-evaluation.pkl"

                save_path = os.path.join(EVAL_DIR, save_name)

                with open(save_path, "wb") as f:
                    pickle.dump(results, f)

                print(f"Done. AUC: {results['ROC_AUC']:.4f}. Saved to {save_path}")