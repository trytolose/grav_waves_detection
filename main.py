import os
import time
from functools import partial
from pathlib import Path
from pydoc import locate

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import TrainDataset, get_in_memory_loaders
from src.models import get_model
from src.utils.checkpoint import ModelCheckpoint
from src.utils.utils import get_lr

from src.losses import WeightedFocalLoss

INPUT_PATH = Path("/home/clover_rtx/ds_users/bulat/kaggle")

def get_loaders(cfg):
    
    df = pd.read_csv(INPUT_PATH / "training_labels.csv")
    files = list((INPUT_PATH / "train").rglob("*.npy"))
    FILE_PATH_DICT = {x.stem: str(x) for x in files}
    df["path"] = df["id"].apply(lambda x: FILE_PATH_DICT[x])

    # train_test splits
    n_splits = cfg.SPLITS.n_splits
    random_state = cfg.SPLITS.random_state
    print('n_splits', n_splits, 'random_state', random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    df["fold"] = -1
    for f, (train_ids, val_ids) in enumerate(skf.split(df.index, y=df["target"])):
        df.loc[val_ids, "fold"] = f

    # datasets
    transform_f = partial(locate(cfg.TRANSFORM.NAME), params=cfg.TRANSFORM.CFG)

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
    print(df[df["fold"] != cfg.FOLD].head())
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=cfg.NUM_WORKERS, batch_size=cfg.BS, pin_memory=False)
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=cfg.NUM_WORKERS, batch_size=cfg.BS, pin_memory=False)
    return train_loader, val_loader


def train(cfg):
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoints_path = f"weights/{cfg.MODEL.NAME}/{cfg.EXP_NAME}_{time_str}/fold_{cfg.FOLD}"
    tensorboard_logs = f"logs/tensorboard/{cfg.EXP_NAME}_{time_str}"
    if cfg.DEBUG is False:
        Path(tensorboard_logs).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_logs)
        checkpoints = ModelCheckpoint(dirname=checkpoints_path, n_saved=cfg.N_SAVED)
    train_loader, val_loader = get_loaders(cfg)
    # train_loader, val_loader = get_in_memory_loaders(cfg)

    model = get_model(cfg)
    model.cuda()

    loss_cfg_dict = dict(cfg.LOSS.CFG)
    #print('loss_cfg_dict', loss_cfg_dict)
    if "pos_weight" in loss_cfg_dict:
        loss_cfg_dict["pos_weight"] = torch.tensor(loss_cfg_dict["pos_weight"]).cuda()

    optimizer = locate(cfg.OPTIMIZER.NAME)(model.parameters(), **cfg.OPTIMIZER.CFG)
    loss_fn = locate(cfg.LOSS.NAME)(**loss_cfg_dict)
    scheduler = locate(cfg.SCHEDULER.NAME)(optimizer, **cfg.SCHEDULER.CFG)
    scaler = GradScaler()

    best_score = 0
    iters = len(train_loader)
    for e in range(cfg.EPOCH):

        # Training:
        train_loss = []
        model.train()

        for i, (x, y) in tqdm(enumerate(train_loader), total=iters, ncols=70, leave=False):

            optimizer.zero_grad()
            x = x.cuda().float()
            y = y.cuda().float().unsqueeze(1)
            if cfg.FP16 is True:
                with autocast():
                    pred = model(x)
                    loss = loss_fn(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
            train_loss.append(loss.item())
            scheduler.step(e + i / iters)

        val_loss = []

        val_true = []
        val_pred = []
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(val_loader, ncols=50, leave=False):
                x = x.cuda().float()
                y = y.cuda().float().unsqueeze(1)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss.append(loss.item())

                pred = pred.sigmoid().cpu().data.numpy()
                val_pred.append(pred)
                val_true.append(y.cpu().numpy())

        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)
        val_true = np.concatenate(val_true).reshape(
            -1,
        )
        val_pred = np.concatenate(val_pred).reshape(
            -1,
        )

        final_score = metrics.roc_auc_score(val_true, val_pred)
        print(
            f"Epoch: {e:03d}; train_loss: {train_loss:.05f} val_loss: {val_loss:.05f}; roc: {final_score:.5f}",
            end=" ",
        )
        if cfg.DEBUG is False:
            tensorboard_writer.add_scalar("Learning rate", get_lr(optimizer), e)
            tensorboard_writer.add_scalar("Loss/train", train_loss, e)
            tensorboard_writer.add_scalar("Loss/valid", val_loss, e)
            tensorboard_writer.add_scalar("ROC_AUC/valid", final_score, e)
            checkpoints(e, final_score, model)
        if final_score > best_score:
            best_score = final_score
            print("+")
        else:
            print()


@hydra.main(config_path="./configs", config_name="default")
def main(cfg: DictConfig):

    torch.cuda.set_device(cfg.DEVICE) 
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()
