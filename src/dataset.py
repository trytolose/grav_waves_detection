import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from functools import partial
from pydoc import locate
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import torch
from dataclasses import dataclass
from gwpy.timeseries import TimeSeries
from src.transforms import min_max_scale
import cv2


class TrainDataset(Dataset):
    def __init__(self, df, transform=None, steps_per_epoch=150, mode="train", bs=64):
        self.df = df
        self.file_names = df["path"].values
        self.labels = df["target"].values
        self.transform = transform
        self.steps_per_epoch = steps_per_epoch * bs
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]

        waves = np.load(file_path)
        if self.transform is not None:
            waves = self.transform(waves)
        if self.mode == "train" or self.mode == "val":
            label = float(self.labels[idx])
            return waves, label
        if self.mode == "test":
            return waves


class InMemoryDataset(Dataset):
    def __init__(self, path, folds, transform=None, mode="train"):
        self.mode = mode
        self.transform = transform
        self.data = np.concatenate([np.load(f"{path}/fold_{f}.npy") for f in folds])
        self.target = np.concatenate([np.load(f"{path}/fold_{f}_target.npy") for f in folds])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waves = self.data[idx]
        if self.transform is not None:
            waves = self.transform(waves)
        if self.mode == "train" or self.mode == "val":
            label = float(self.target[idx])
            return waves, label
        if self.mode == "test":
            return waves


def gwpy_qtransform(x, img_size=(256, 256)):
    x
    x = min_max_scale(x)
    images = []
    for ii in range(3):
        strain = TimeSeries(x[ii, :], t0=0, dt=1 / 2048)

        hq = strain.q_transform(qrange=(4, 32), frange=(20, 400), logf=False, whiten=True, fduration=1, tres=2 / 1000)
        images.append(hq)

    img = np.stack([np.array(x).T[:-200] for x in images], axis=2)
    img = cv2.resize(img, img_size)  # .astype(np.float16)
    return img


class TrainFromDiskDataset(Dataset):
    def __init__(self, df, path, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode
        self.path = Path(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.path / f"{self.df.loc[idx, 'id']}.npy"
        img = np.load(file_path).transpose(2, 0, 1)
        if self.mode == "train" or self.mode == "val":
            label = float(self.df.loc[idx, "target"])
            return (
                img,
                label,
            )
        if self.mode == "test":
            return img


INPUT_PATH = Path("/home/trytolose/rinat/kaggle/grav_waves_detection/input")


def get_loaders(cfg):
    df = pd.read_csv(INPUT_PATH / "training_labels.csv")
    # df = pd.read_csv(INPUT_PATH / "train_oof_overfit.csv")
    files = list((INPUT_PATH / "train").rglob("*.npy"))
    FILE_PATH_DICT = {x.stem: str(x) for x in files}
    df["path"] = df["id"].apply(lambda x: FILE_PATH_DICT[x])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
    df["fold"] = -1
    for f, (train_ids, val_ids) in enumerate(skf.split(df.index, y=df["target"])):
        df.loc[val_ids, "fold"] = f

    transform_f = partial(locate(cfg.TRANSFORM.NAME), **cfg.TRANSFORM.CFG)

    if cfg.TRAIN_FOLD == -1:
        df_train = df[df["fold"] != cfg.FOLD].reset_index(drop=True)
    else:
        df_train = df[df["fold"] == cfg.TRAIN_FOLD].reset_index(drop=True)

    df_val = df[df["fold"] == cfg.FOLD].reset_index(drop=True)

    train_ds = TrainDataset(
        df_train,
        steps_per_epoch=cfg.STEPS_PER_EPOCH,
        mode="train",
        transform=transform_f,
    )
    val_ds = TrainDataset(
        df_val,
        mode="val",
        transform=transform_f,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        batch_size=cfg.BS,
        pin_memory=False,  # sampler=sampler
    )
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=cfg.NUM_WORKERS, batch_size=cfg.BS, pin_memory=False)
    return train_loader, val_loader


def get_test_loader(cfg):

    df = pd.read_csv(INPUT_PATH / "sample_submission.csv")

    files = list((INPUT_PATH / "test").rglob("*.npy"))
    FILE_PATH_DICT = {x.stem: str(x) for x in files}
    df["path"] = df["id"].apply(lambda x: FILE_PATH_DICT[x])

    transform_f = partial(locate(cfg.TRANSFORM.NAME), **cfg.TRANSFORM.CFG)

    test_ds = TrainDataset(
        df,
        mode="test",
        transform=transform_f,
    )
    test_loader = DataLoader(test_ds, shuffle=False, num_workers=cfg.NUM_WORKERS, batch_size=cfg.BS, pin_memory=False)
    return test_loader
