import os
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, Spacingd,
    ResizeWithPadOrCropd,
    RandFlipd, RandRotate90d, ToTensord
)

from monai.data import DataLoader, CacheDataset
from monai.networks.nets import UNETR, SwinUNETR


# ---------------- TRANSFORMS ----------------
def get_transforms(train=True):

    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        # 🔥 unify voxel spacing
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),

        ScaleIntensityd(keys=["image"]),

        # 🔥 FIX shape for UNETR
        ResizeWithPadOrCropd(
            keys=["image", "label"],
            spatial_size=(96,96,96)
        ),
    ]

    if train:
        transforms += [
            RandFlipd(keys=["image", "label"], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5),
        ]

    transforms += [ToTensord(keys=["image", "label"])]

    return Compose(transforms)


# ---------------- MODEL ----------------
def create_model(name, device):

    if name == "swinunetr":

        model = SwinUNETR(

            img_size=(96, 96, 96),

            in_channels=1,

            out_channels=1,

            feature_size=48

        )


    else:
        raise ValueError("model must be 'unetr' or 'swinunetr'")

    return model.to(device)


# ---------------- DATASET ----------------
def prepare_dataloaders(root, batch):

    img_dir = os.path.join(root, "imagesTr")
    lab_dir = os.path.join(root, "labelsTr")

    files = []

    for f in sorted(os.listdir(img_dir)):

        if not f.endswith(".nii"):
            continue

        img_path = os.path.join(img_dir, f)

        base = f.replace(".nii", "")
        label_path = os.path.join(lab_dir, base + ".nii")

        if os.path.exists(label_path):
            files.append({
                "image": img_path,
                "label": label_path
            })

    split = int(0.8 * len(files))
    train_files = files[:split]
    val_files = files[split:]

    train_ds = CacheDataset(train_files, get_transforms(True), cache_rate=0.2)
    val_ds = CacheDataset(val_files, get_transforms(False), cache_rate=0.2)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

    return train_loader, val_loader


# ---------------- TRAIN ----------------
def train_one_epoch(loader, model, optimizer, loss_fn, device):

    model.train()
    total_loss = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_fn(pred, y)   # ⚠️ no sigmoid here

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------- VALIDATION ----------------
def validate(loader, model, metric, device):

    model.eval()
    metric.reset()

    with torch.no_grad():
        for b in loader:

            x = b["image"].to(device)
            y = b["label"].to(device)

            y = (y > 0).float()   # 🔥 مهم

            out = torch.sigmoid(model(x))
            out = (out > 0.5).float()

            metric(y_pred=out, y=y)

    return metric.aggregate().item()