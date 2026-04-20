import random
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from eval_data import load_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
LR = 1e-5
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 500
PATIENCE = 20

SEEDS = [0,1,2,3,4]
N_FOLDS = 5

DATA_DIR = "DDI"


# ---------------------------------------------------
# Data transforms
# ---------------------------------------------------
class RandomRotateCrop:
    """
    Rotate image and crop to largest possible rectangle (approximation).
    torchvision doesn't provide exact largest-inscribed-rectangle,
    so we use expand=False to keep dimensions and avoid black borders.
    """
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return transforms.functional.rotate(img, angle, expand=False)

train_tf = transforms.Compose([

    RandomRotateCrop(degrees=30),
    transforms.RandomVerticalFlip(p=0.5),

    transforms.Resize(299),
    transforms.CenterCrop(299),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),

    transforms.GaussianBlur(
        kernel_size=(5, 9),
        sigma=(0.1, 5)
    ),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_tf = transforms.Compose([

    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------

class DDIDataset(Dataset):
    def __init__(self, csv_file=None, transform=None, data_dir="DDI/images", dataframe=None):
        if dataframe is not None:
            self.df = dataframe.copy()
        elif csv_file is not None:
            self.df = pd.read_csv(csv_file)
        else:
            raise ValueError("Provide csv_file or dataframe")
        self.transform = transform
        self.data_dir = data_dir

        # Ensure labels are 0 or 1 integers
        self.df['malignant'] = self.df['malignant'].apply(self._convert_label)

    def _convert_label(self, val):
        """Convert CSV value to integer 0 or 1"""
        val_str = str(val).strip().lower()
        if val_str in ["true", "1"]:
            return 1
        elif val_str in ["false", "0"]:
            return 0
        else:
            raise ValueError(f"Invalid label value: {val}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["DDI_file"])
        image = Image.open(img_path).convert("RGB")

        label = row["malignant"]

        if self.transform:
            image = self.transform(image)

        return image, label, row["skin_tone"]


# ---------------------------------------------------
# Mixup
# ---------------------------------------------------

def mixup(x, y, alpha=1.0):

    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam*x + (1-lam)*x[index]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


# ---------------------------------------------------
# Train
# ---------------------------------------------------

def train_epoch(model,loader,optimizer,criterion):

    model.train()

    total_loss = 0

    for x,y,_ in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE, dtype=torch.long)

        x,y_a,y_b,lam = mixup(x,y)

        outputs,aux = model(x)

        loss1 = criterion(outputs,y_a)*lam + criterion(outputs,y_b)*(1-lam)

        loss2 = criterion(aux,y_a)*lam + criterion(aux,y_b)*(1-lam)

        loss = loss1 + 0.4*loss2

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(loader)


# ---------------------------------------------------
# Validation
# ---------------------------------------------------

def evaluate(model,loader):

    model.eval()

    preds = []
    labels = []
    tones = []

    with torch.no_grad():

        for x,y,t in loader:

            x = x.to(DEVICE)

            out = model(x)

            if isinstance(out,tuple):
                out = out[0]

            prob = torch.softmax(out,dim=1)[:,1]

            preds.extend(prob.cpu().numpy())

            labels.extend(y.numpy())

            tones.extend(t)

    return np.array(preds),np.array(labels),tones


# ---------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------
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
        aucs.append(roc_auc_score(y_b, y_score[idx]))
    if not aucs:
        return None
    lo = (1 - alpha) / 2
    hi = 1 - lo
    return float(np.quantile(aucs, lo)), float(np.quantile(aucs, hi))


def baseline_evaluation(model, test_loader, model_name, seed):
    preds, labels, tones = evaluate(model, test_loader)
    overall_auc = roc_auc_score(labels, preds)
    overall_ci = bootstrap_auc_ci(labels, preds, seed=seed)

    print("\n" + "=" * 60)
    print(f"BASELINE EVALUATION BEFORE FINE-TUNING: {model_name} seed={seed}")
    print(f"Overall AUC: {overall_auc:.4f}", end="")
    if overall_ci is not None:
        print(f" (95% CI: {overall_ci[0]:.4f}, {overall_ci[1]:.4f})")
    else:
        print(" (95% CI: N/A)")

    tones_arr = np.asarray(tones)
    for tone in [12, 34, 56]:
        mask = tones_arr == tone
        y_tone = labels[mask]
        p_tone = preds[mask]
        if len(y_tone) == 0 or len(np.unique(y_tone)) < 2:
            print(f"FST {tone} AUC: N/A (insufficient class diversity)")
            continue
        tone_auc = roc_auc_score(y_tone, p_tone)
        tone_ci = bootstrap_auc_ci(y_tone, p_tone, seed=seed)
        if tone_ci is not None:
            print(
                f"FST {tone} AUC: {tone_auc:.4f} "
                f"(95% CI: {tone_ci[0]:.4f}, {tone_ci[1]:.4f}; n={mask.sum()})"
            )
        else:
            print(f"FST {tone} AUC: {tone_auc:.4f} (95% CI: N/A; n={mask.sum()})")
    print("=" * 60 + "\n")

    if overall_auc < 0.55:
        print(
            "WARNING: Baseline AUC is very low; confirm you are loading the paper's "
            "Zenodo checkpoints."
        )


# ---------------------------------------------------
# Training: best checkpoint by validation loss (full MAX_EPOCHS)
# ---------------------------------------------------

def _stratify_labels_for_cv(df):
    """Match finetuning_setup joint strata when present; else binary malignant."""
    if "stratify" in df.columns:
        return pd.Categorical(df["stratify"]).codes
    return df["malignant"].values


def train_val_split_for_fold(seed, fold):
    """Combine train.csv + val.csv; return train/val DataFrames for one CV fold."""
    full_df = pd.concat(
        [pd.read_csv("train.csv"), pd.read_csv("val.csv")],
        ignore_index=True,
    )
    aligned = DDIDataset(dataframe=full_df, transform=train_tf).df
    y = _stratify_labels_for_cv(aligned)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    train_idx, val_idx = splits[fold]
    train_df = aligned.iloc[train_idx].reset_index(drop=True)
    val_df = aligned.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def train_model(seed, model_name, fold):
  
    train_losses = []
    val_losses = []
    val_aucs = []
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_model(model_name)
    model = model.to(DEVICE)

    train_df, val_df = train_val_split_for_fold(seed, fold)
    print(
        f"CV fold {fold}/{N_FOLDS - 1}: train_n={len(train_df)} val_n={len(val_df)} "
        f"(train+val from train.csv + val.csv; test.csv held out)"
    )

    train_ds = DDIDataset(dataframe=train_df, transform=train_tf)
    val_ds = DDIDataset(dataframe=val_df, transform=val_tf)
    test_ds = DDIDataset("test.csv", val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    baseline_evaluation(model, test_loader, model_name=model_name, seed=seed)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = torch.nn.CrossEntropyLoss()
    patience_counter = 0
    best_val_loss = float("inf")  # track minimum validation loss (checkpoint selection)

    for epoch in range(MAX_EPOCHS):

        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        # Validation: loss + AUC (single pass)
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE, dtype=torch.long)
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = criterion(out, y)
                val_loss += loss.item()
                prob = torch.softmax(out, dim=1)[:, 1]
                val_preds.extend(prob.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        val_loss /= len(val_loader)
        val_preds = np.asarray(val_preds)
        val_labels = np.asarray(val_labels)
        if len(np.unique(val_labels)) >= 2:
            val_auc = roc_auc_score(val_labels, val_preds)
        else:
            val_auc = float("nan")
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print(
            f"seed={seed} fold={fold} epoch={epoch} val_loss={val_loss:.6f} val_auc={val_auc:.4f} "
            f"(best_val_loss={best_val_loss:.6f})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model based on val loss
            torch.save(model.state_dict(), f"{model_name}_seed{seed}.pth")
        else:
            patience_counter += 1
        if patience_counter > PATIENCE:
            print(f"Early stopping at epoch {epoch} for seed {seed} fold {fold}")
            break

    plt.figure(figsize=(10,5))

    # Loss plot
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Seed {seed} Fold {fold} Loss")
    plt.legend()

    # AUC plot
    plt.subplot(1,2,2)
    plt.plot(val_aucs, label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"{model_name} Seed {seed} Fold {fold} AUC")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{model_name}_seed{seed}_fold{fold}_learning_curve.png")
    plt.close()

# ---------------------------------------------------
# Run experiments
# ---------------------------------------------------

for model_name in [#"DeepDerm",
"HAM10000"]:

    for seed in SEEDS:

        for fold in range(N_FOLDS):

            train_model(seed, model_name, fold)