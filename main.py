import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2, transforms
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from src.dataset import TrainDataset
from src.model import CustomModel_v0, CustomModel_v1
from omegaconf import DictConfig, OmegaConf
from pydoc import locate
import hydra
from functools import partial


qtransform_params = {
    "sr": 2048,
    "fmin": 30,
    "fmax": 400,
    "hop_length": 64,  # count of signal points used for 1 tranformed point
    #     "bins_per_octave": 32,  # y-axe of transformed image depends from this parameter, for example [91,] if bins=16 and 42 if bins=8
}
INPUT_PATH = Path("/home/trytolose/rinat/kaggle/grav_waves_detection/input")
STEPS_PER_EPOCH = 2000  # 7000 total if BS=64


def get_loaders(cfg):
    df = pd.read_csv(INPUT_PATH / "training_labels.csv")

    files = list((INPUT_PATH / "train").rglob("*.npy"))
    FILE_PATH_DICT = {x.stem: str(x) for x in files}
    df["path"] = df["id"].apply(lambda x: FILE_PATH_DICT[x])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
    df["fold"] = -1
    for f, (train_ids, val_ids) in enumerate(skf.split(df.index, y=df["target"])):
        df.loc[val_ids, "fold"] = f

    transform_f = partial(locate(cfg.TRANSFORM), params=cfg.bandpass_param)

    train_ds = TrainDataset(
        df[df["fold"] != cfg.FOLD].reset_index(drop=True),
        steps_per_epoch=cfg.STEPS_PER_EPOCH,
        mode="train",
        transform=transform_f,
    )
    val_ds = TrainDataset(
        df[df["fold"] == cfg.FOLD].reset_index(drop=True),
        mode="val",
        transform=transform_f,
    )

    train_loader = DataLoader(
        train_ds, shuffle=True, num_workers=12, batch_size=cfg.BS, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, num_workers=12, batch_size=cfg.BS * 2, pin_memory=False
    )
    return train_loader, val_loader


def train(cfg):
    train_loader, val_loader = get_loaders(cfg)
    model = CustomModel_v0(cfg)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", verbose=True, patience=5, factor=0.5, eps=1e-12
    )

    best_score = 0
    for e in range(cfg.EPOCH):

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

        val_true = np.concatenate(val_true).reshape(
            -1,
        )
        val_pred = np.concatenate(val_pred).reshape(
            -1,
        )

        final_score = metrics.roc_auc_score(val_true, val_pred)
        # print(f'Epoch: {e:03d}; lr: {lr:.06f}; train_loss: {np.mean(train_loss):.05f}; val_loss: {val_loss:.05f}; ', end='')
        print(
            f"Epoch: {e:03d}; train_loss: {np.mean(train_loss):.05f} val_loss: {val_loss:.05f}; roc: {final_score:.5f}",
            end=" ",
        )

        if final_score > best_score:
            best_score = final_score
            torch.save(model.state_dict(), f"{cfg.EXP_NAME}_fold_{cfg.FOLD}_w.pt")
            print("+")
        else:
            print()

        scheduler.step(final_score)


@hydra.main(config_path="./configs", config_name="default")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()
