import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from nnAudio.Spectrogram import CQT1992v2
from torch.utils.data import DataLoader, Dataset
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

BS = 64
FOLD = 0
qtransform_params = {
    "sr": 2048,
    "fmin": 20,
    "fmax": 1024,
    "hop_length": 32,  # count of signal points used for 1 tranformed point
    "bins_per_octave": 32,  # y-axe of transformed image depends from this parameter, for example [91,] if bins=16 and 42 if bins=8
}
INPUT_PATH = Path("/home/trytolose/rinat/kaggle/g2net/input")
STEPS_PER_EPOCH = 2000  # 7000 total if BS=64


class TrainDataset(Dataset):
    def __init__(self, df, transform=None, steps_per_epoch=150, mode="train"):
        self.df = df
        self.file_names = df["path"].values
        self.labels = df["target"].values
        self.wave_transform = CQT1992v2(**qtransform_params)
        self.transform = transform
        self.steps_per_epoch = steps_per_epoch * BS
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return self.steps_per_epoch
        else:
            return len(self.df)

    def apply_qtransform(self, waves, transform):
        waves = np.hstack(waves)
        waves = waves / np.max(waves)
        waves = torch.from_numpy(waves).float()
        image = transform(waves)
        return image

    def __getitem__(self, idx):
        if self.mode == "train":
            rand_id = np.random.randint(len(self.df), size=1)[0]
            file_path = self.file_names[rand_id]
            label = float(self.labels[rand_id])
        else:
            file_path = self.file_names[idx]
            label = float(self.labels[idx])
        waves = np.load(file_path)
        waves = np.hstack(waves)
        waves = waves / np.max(waves)
        # image = self.apply_qtransform(waves, self.wave_transform)

        return waves, label


class CustomModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, in_chans=1
        )
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, 1)
        self.cqt = CQT1992v2(
            **qtransform_params
        )  # qtransform_params={"sr": 2048, "fmin": 20, "fmax": 1024, "hop_length": 32, "bins_per_octave": 8}

    def forward(self, x):
        x = self.cqt(x).unsqueeze(1)  # 3ch: [64, 1, 46, 385];  # 1ch: [64, 1, 46, 129]
        x = nn.functional.interpolate(x, (256, 386))
        output = self.model(x)
        return output


df = pd.read_csv(INPUT_PATH / "training_labels.csv")

files = list((INPUT_PATH / "train").rglob("*.npy"))
FILE_PATH_DICT = {x.stem: str(x) for x in files}
df["path"] = df["id"].apply(lambda x: FILE_PATH_DICT[x])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
df["fold"] = -1
for f, (train_ids, val_ids) in enumerate(skf.split(df.index, y=df["target"])):
    df.loc[val_ids, "fold"] = f


train_ds = TrainDataset(
    df[df["fold"] != FOLD].reset_index(drop=True),
    steps_per_epoch=STEPS_PER_EPOCH,
    mode="train",
)
val_ds = TrainDataset(df[df["fold"] == FOLD].reset_index(drop=True), mode="val")

train_loader = DataLoader(
    train_ds, shuffle=True, num_workers=12, batch_size=BS, pin_memory=False
)
val_loader = DataLoader(
    val_ds, shuffle=False, num_workers=12, batch_size=BS * 2, pin_memory=False
)

model = CustomModel()

model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.BCEWithLogitsLoss()
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", verbose=True, patience=5, factor=0.5, eps=1e-12
)


best_score = 0
for e in range(100):

    # Training:
    train_loss = []
    model.train()

    for x, y in tqdm(train_loader, ncols=70, leave=False):
        optimizer.zero_grad()
        x = x.cuda().float().unsqueeze(1)
        y = y.cuda().float().unsqueeze(1)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    val_loss = []

    val_true = []
    val_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(val_loader, ncols=50, leave=False):
            x = x.cuda().float().unsqueeze(1)
            y = y.cuda().float().unsqueeze(1)
            pred = model(x)
            loss = loss_fn(pred, y)
            val_loss.append(loss.item())

            pred = pred.sigmoid().cpu().data.numpy()
            val_pred.append(pred)
            val_true.append(y.cpu().numpy())

    val_loss = np.mean(val_loss)

    val_true = np.concatenate(val_true).reshape(-1,)
    val_pred = np.concatenate(val_pred).reshape(-1,)

    final_score = metrics.roc_auc_score(val_true, val_pred)
    # print(f'Epoch: {e:03d}; lr: {lr:.06f}; train_loss: {np.mean(train_loss):.05f}; val_loss: {val_loss:.05f}; ', end='')
    print(
        f"Epoch: {e:03d}; train_loss: {np.mean(train_loss):.05f} val_loss: {val_loss:.05f}; roc: {final_score:.5f}",
        end=" ",
    )

    if final_score > best_score:
        best_score = final_score
        torch.save(model.state_dict(), f"baseline_f0_part_epoch.pt")
        print("+")
    else:
        print()

    scheduler.step(final_score)
