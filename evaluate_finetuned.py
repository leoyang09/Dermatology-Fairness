import os
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

    # Per-skin-tone ROC-AUC (only valid when both classes exist for that tone)
    tone_auc = {}
    unique_tones = sorted(set(tones))
    for tone in unique_tones:
        tone_mask = np.array([t == tone for t in tones], dtype=bool)
        tone_labels = labels[tone_mask]
        tone_preds = preds[tone_mask]

        if len(np.unique(tone_labels)) < 2:
            tone_auc[str(tone)] = None
            continue

        tone_fpr, tone_tpr, _ = roc_curve(tone_labels, tone_preds)
        tone_auc[str(tone)] = auc(tone_fpr, tone_tpr)

    report = classification_report(labels, (preds > model._ddi_threshold).astype(int),
                                   target_names=["benign", "malignant"])

    results = {
        "predicted_labels": preds,
        "true_labels": labels,
        "images": paths,
        "skin_tones": tones,
        "ROC_AUC": auc_score,
        "ROC_AUC_by_skin_tone": tone_auc,
        "report": report,
        "threshold": model._ddi_threshold,
        "model": model._ddi_name,
    }

    return results

# ---------------------------
# Load fine-tuned weights per seed
# ---------------------------
def load_seed_model(model_name, seed, model_dir="."):
    base_model = load_model(model_name, download=False)
    weights_path = os.path.join(model_dir, f"{model_name}_seed{seed}.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")
    state_dict = torch.load(weights_path, map_location=DEVICE)
    base_model.load_state_dict(state_dict)
    base_model._ddi_name = f"{model_name}_seed{seed}"
    return base_model

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    TEST_CSV = "test.csv"
    DATA_DIR = "DDI/images"
    WEIGHTS_DIR = "baseline_finetuned_models"
    EVAL_DIR = "DDI-results/baseline_finetuned_models"
    os.makedirs(EVAL_DIR, exist_ok=True)

    MODEL_NAMES = ["DeepDerm", "HAM10000"]
    NUM_SEEDS = 5

    for model_name in MODEL_NAMES:
        for seed in range(NUM_SEEDS):
            print(f"Evaluating {model_name} seed {seed}...")
            try:
                model = load_seed_model(model_name, seed, model_dir=WEIGHTS_DIR)
            except FileNotFoundError as e:
                print(e)
                continue

            results = eval_model_on_csv(model, TEST_CSV, DATA_DIR)

            save_path = os.path.join(EVAL_DIR, f"{model_name}_seed{seed}-evaluation.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(results, f)

            print(f"Done. AUC: {results['ROC_AUC']:.4f}. Saved to {save_path}")