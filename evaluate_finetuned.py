import os
import torch
import torchvision
from torchvision import transforms, datasets
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import tqdm

# ---------------------------
# CONFIG
# ---------------------------

MODEL_DIR = "DDI-finetuned-models"           # folder containing DeepDerm_seed{0-4}.pth
DATA_DIR = "DDI"                             # folder containing images
METADATA_CSV = "DDI/ddi_dataset/metadata.csv" # metadata CSV with skin_tone
TEST_CSV = "DDI/ddi_dataset/test.csv"

NUM_SEEDS = 5
THRESHOLD = 0.687

# ---------------------------
# HELPERS
# ---------------------------

def load_skin_tone_map(csv_path):
    """Return dict mapping image filename -> skin tone"""
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        filename = os.path.basename(row["DDI_file"])
        mapping[filename] = row["skin_tone"]
    return mapping

def load_test_image_set(csv_path):
    """Return a set of filenames listed in the test CSV"""
    df = pd.read_csv(csv_path)
    test_filenames = set(os.path.basename(f) for f in df["DDI_file"])
    return test_filenames


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset returning (image, label, path)"""
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = os.path.abspath(self.imgs[index][0])
        return original_tuple + (path,)


def eval_model(model, image_dir, use_gpu=False, show_plot=False):
    """Evaluate a model on the image_dir, returns dict with predictions, true labels, paths, report, ROC AUC."""
    device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = ImageFolderWithPaths(
        image_dir,
        transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize
        ])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=use_gpu)

    model.to(device).eval()

    hat, star, all_paths = [], [], []
    for i, (images, target, paths) in enumerate(tqdm.tqdm(dataloader)):
        images = images.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(images)

        hat.append(output[:,1].detach().cpu().numpy())
        star.append(target.cpu().numpy())
        all_paths.append(paths)

    hat = np.concatenate(hat)
    star = np.concatenate(star)
    all_paths = np.concatenate(all_paths)

    threshold = getattr(model, "_ddi_threshold", THRESHOLD)
    m_name = getattr(model, "_ddi_name", "DeepDerm")

    fpr, tpr, _ = roc_curve(star, hat, pos_label=1)
    auc_est = auc(fpr, tpr)
    report = classification_report(star, (hat>threshold).astype(int), target_names=["benign","malignant"])

    if show_plot:
        plt.plot(fpr, tpr, color="blue", linestyle="-", linewidth=2, marker="o", markersize=2, label=f"AUC={auc_est:.3f}")
        plt.show()
        plt.close()

    eval_results = {
        "predicted_labels": hat,
        "true_labels": star,
        "images": all_paths,
        "report": report,
        "ROC_AUC": auc_est,
        "threshold": threshold,
        "model": m_name,
    }
    return eval_results


def compute_auc_per_skin_tone(results, skin_tone_map):
    """Return dict of skin_tone -> ROC AUC for this model"""
    hat = results["predicted_labels"]
    star = results["true_labels"]
    paths = results["images"]

    tone_to_preds = defaultdict(list)
    tone_to_labels = defaultdict(list)

    for pred, label, path in zip(hat, star, paths):
        filename = os.path.basename(path)
        if filename not in skin_tone_map:
            continue
        tone = skin_tone_map[filename]
        tone_to_preds[tone].append(pred)
        tone_to_labels[tone].append(label)

    tone_auc = {}
    for tone in tone_to_preds:
        try:
            tone_auc[tone] = roc_auc_score(tone_to_labels[tone], tone_to_preds[tone])
        except ValueError:
            tone_auc[tone] = np.nan
    return tone_auc

# ---------------------------
# MAIN
# ---------------------------

def main():
    skin_tone_map = load_skin_tone_map(METADATA_CSV)
    test_filenames = load_test_image_set(TEST_CSV)
    all_seed_results = []

    for seed in range(NUM_SEEDS):
        print(f"\n=== Evaluating seed {seed} ===")

        # Load model architecture
        model = torchvision.models.inception_v3(pretrained=False, transform_input=True)
        model.fc = torch.nn.Linear(2048,2)
        model.AuxLogits.fc = torch.nn.Linear(768,2)

        # Load weights
        weights_path = os.path.join(MODEL_DIR, f"DeepDerm_seed{seed}.pth")
        weights = torch.load(weights_path)
        model.load_state_dict(weights)
        model._ddi_threshold = THRESHOLD
        model._ddi_name = f"DeepDerm_seed{seed}"

        # Evaluate
        results = eval_model(model, DATA_DIR)
        print(f"Seed {seed} overall ROC AUC: {results['ROC_AUC']:.4f}")

        # Compute per-skin-tone AUC
        tone_auc = compute_auc_per_skin_tone(results, skin_tone_map)
        all_seed_results.append(tone_auc)
        print(f"Seed {seed} per-skin-tone AUC: {tone_auc}")

    # Average across seeds
    avg_auc = defaultdict(list)
    for seed_result in all_seed_results:
        for tone, auc_val in seed_result.items():
            if not np.isnan(auc_val):
                avg_auc[tone].append(auc_val)

    avg_auc = {tone: np.mean(vals) for tone, vals in avg_auc.items()}
    print("\n=== Average AUC per skin tone across seeds ===")
    for tone, val in avg_auc.items():
        print(f"{tone}: {val:.4f}")

if __name__ == "__main__":
    main()