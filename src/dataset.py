import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import partial
from pydoc import locate

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


def get_in_memory_loaders(cfg):
 
    folds = [0, 1, 2, 3, 4]
    folds.remove(cfg.FOLD)
  
    transform_f = partial(locate(cfg.TRANSFORM.NAME), params=cfg.TRANSFORM.CFG)

    train_ds = InMemoryDataset(
        path="input/fp16/train",
        folds=folds,
        mode="train",
        transform=transform_f,
    )
    val_ds = InMemoryDataset(
        path="input/fp16/train",
        folds=[cfg.FOLD],
        mode="val",
        transform=transform_f,
    )

    train_loader = DataLoader(train_ds, shuffle=True, num_workers=cfg.NUM_WORKERS, batch_size=cfg.BS, pin_memory=False)
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=cfg.NUM_WORKERS, batch_size=cfg.BS, pin_memory=False)
    return train_loader, val_loader