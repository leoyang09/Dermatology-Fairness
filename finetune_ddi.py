import torch
import torchvision
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from sklearn.metrics import roc_auc_score
from eval_data import load_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
LR = 0.05
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 500
PATIENCE = 20

SEEDS = [0,1,2,3,4]

DATA_DIR = "DDI"


# ---------------------------------------------------
# Data transforms
# ---------------------------------------------------

train_tf = transforms.Compose([

    transforms.RandomRotation(180),

    transforms.RandomHorizontalFlip(0.5),

    transforms.RandomVerticalFlip(0.5),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1,2.0)
    ),

    transforms.Resize((299,299)),
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
    def __init__(self, csv_file, transform=None, data_dir="DDI/images"):
        self.df = pd.read_csv(csv_file)
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

        x,y = x.to(DEVICE), y.to(DEVICE)

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
# Early stopping using validation loss
# ---------------------------------------------------

def train_model(seed, model_name):

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_model(model_name)
    model = model.to(DEVICE)

    train_ds = DDIDataset("train.csv", train_tf)
    val_ds = DDIDataset("val.csv", val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")  # track minimum validation loss
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):

        # Train one epoch
        train_epoch(model, train_loader, optimizer, criterion)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(seed, epoch, val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model based on val loss
            torch.save(model.state_dict(), f"{model_name}_seed{seed}.pth")
        else:
            patience_counter += 1

        if patience_counter > PATIENCE:
            print(f"Early stopping at epoch {epoch} for seed {seed}")
            break


# ---------------------------------------------------
# Run experiments
# ---------------------------------------------------

for model_name in ["DeepDerm","HAM10000"]:

    for seed in SEEDS:

        train_model(seed,model_name)