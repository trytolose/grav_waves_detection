import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from functools import partial
from pydoc import locate
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from src.transforms import min_max_scale


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

def get_loaders(cfg):
    INPUT_PATH = Path(cfg.INPUT_PATH)
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
    print(df_train.head())
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
    INPUT_PATH = Path(cfg.INPUT_PATH)

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
