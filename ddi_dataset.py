"""Code for loading DDI Dataset."""

from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image

means = [0.485, 0.456, 0.406]
stds  = [0.229, 0.224, 0.225]


def clahe_on_luminance(pil_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to the L channel in LAB space; keep color unchanged.
    Args:
        pil_img: PIL Image (RGB).
        clip_limit: CLAHE contrast limit (higher = more contrast).
        tile_grid_size: Grid size for adaptive equalization.
    Returns:
        PIL Image (RGB) with contrast-enhanced luminance.
    """
    img = np.array(pil_img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def build_transform(use_clahe=False, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
    """Build the standard DDI image transform, optionally with CLAHE preprocessing."""
    steps = [
        lambda x: x.convert('RGB'),
    ]
    if use_clahe:
        steps.append(lambda x: clahe_on_luminance(x, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size))
    steps.extend([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=means, std=stds),
    ])
    return T.Compose(steps)


# Default transform (no CLAHE) for backward compatibility
test_transform = build_transform(use_clahe=False)

# Transform with CLAHE preprocessing (e.g. for training or ablation)
test_transform_clahe = build_transform(use_clahe=True)


class DDI_Dataset(ImageFolder):
    _DDI_download_link = "https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965"
    """DDI Dataset.

    Note: assumes DDI data is organized as
        ./DDI
            /images
                /000001.png
                /000002.png
                ...
            /ddi_metadata.csv

    (After downloading from the Stanford AIMI repository, this requires moving all .png files into a new subdirectory titled "images".)

    Args:
        root     (str): Root directory of dataset.
        csv_path (str): Path to the metadata CSV file. Defaults to `{root}/ddi_metadata.csv`
        transform     : Function to transform and collate image input. (can use test_transform from this file) 
    """
    def __init__(self, root, img_dirname="images", csv_path=None, download=True, transform=None, *args, **kwargs):
        self.img_dirname = img_dirname
        if csv_path is None:
            csv_path = os.path.join(root, "ddi_metadata.csv")
        if not os.path.exists(csv_path) and download:
            raise Exception(f"Please visit <{DDI_Dataset._DDI_download_link}> to download the DDI dataset.")
        assert os.path.exists(csv_path), f"Path not found <{csv_path}>."
        super(DDI_Dataset, self).__init__(root, *args, transform=transform, **kwargs)
        self.annotations = pd.read_csv(csv_path)
        m_key = 'malignant'
        if m_key not in self.annotations:
            self.annotations[m_key] = self.annotations['malignancy(malig=1)'].apply(lambda x: x==1)

    def find_classes(self, directory):
        """Override to only find the specific image directory."""
        path = os.path.join(directory, self.img_dirname)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Image directory not found: {path}")
        return [self.img_dirname], {self.img_dirname: 0}

    def __getitem__(self, index):
        img, target = super(DDI_Dataset, self).__getitem__(index)
        path = self.imgs[index][0]
        # use first matching row if multiple rows share the same DDI_file (e.g. duplicate filenames)
        filename = os.path.basename(path)
        match = self.annotations[self.annotations.DDI_file == filename]
        if len(match) == 0:
            raise KeyError(f"No annotation for DDI_file: {filename}")
        row = match.iloc[0]
        target = int(row['malignant'])  # 1 if malignant, 0 if benign
        skin_tone = int(row['skin_tone'])  # Fitzpatrick 12, 34, or 56
        return path, img, target, skin_tone

    """Return a subset of the DDI dataset based on skin tones and malignancy of lesion.

    Args:
        skin_tone    (list of int): Which skin tones to include in the subset. Options are {12, 34, 56}.
        diagnosis    (list of str): Include malignant and/or benign images. Options are {"benign", "malignant"}
    """
    def subset(self, skin_tone=None, diagnosis=None):
        skin_tone = [12, 34, 56] if skin_tone is None else skin_tone
        diagnosis = ["benign", "malignant"] if diagnosis is None else diagnosis
        for si in skin_tone: 
            assert si in [12,34,56], f"{si} is not a valid skin tone"
        for di in diagnosis: 
            assert di in ["benign", "malignant"], f"{di} is not a valid diagnosis"
        indices = np.where(self.annotations['skin_tone'].isin(skin_tone) & \
                           self.annotations['malignant'].isin([di=="malignant" for di in diagnosis]))[0]
        return Subset(self, indices)