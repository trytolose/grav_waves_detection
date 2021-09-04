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

from src.dataset import TrainDataset, get_in_memory_loaders, get_disk_loader, get_loaders
from src.models import get_model
from src.utils.checkpoint import ModelCheckpoint
from src.utils.utils import get_lr
from src.loops import get_dataset_statistics
import warnings

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def train(cfg):
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoints_path = (
        f"/home/trytolose/rinat/kaggle/grav_waves_detection/weights/{cfg.MODEL.NAME}/{cfg.EXP_NAME}/fold_{cfg.FOLD}"
    )
    tensorboard_logs = (
        f"/home/trytolose/rinat/kaggle/grav_waves_detection/logs/tensorboard/{cfg.EXP_NAME}_{time_str}_f{cfg.FOLD}"
    )
    if cfg.DEBUG is False:
        Path(tensorboard_logs).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_logs)
        checkpoints = ModelCheckpoint(dirname=checkpoints_path, n_saved=cfg.N_SAVED)
    if cfg.MODEL.NAME == "timm":
        train_loader, val_loader = get_disk_loader(cfg)
    else:
        train_loader, val_loader = get_loaders(cfg)

    # train_loader, val_loader = get_in_memory_loaders(cfg)

    model = get_model(cfg)
    model.cuda()

    loss_cfg_dict = dict(cfg.LOSS.CFG)
    if "pos_weight" in loss_cfg_dict:
        loss_cfg_dict["pos_weight"] = torch.tensor(loss_cfg_dict["pos_weight"]).cuda()

    optimizer = locate(cfg.OPTIMIZER.NAME)(model.parameters(), **cfg.OPTIMIZER.CFG)
    loss_fn = locate(cfg.LOSS.NAME)(**loss_cfg_dict)
    scheduler = locate(cfg.SCHEDULER.NAME)(optimizer, **cfg.SCHEDULER.CFG)
    scaler = GradScaler()

    best_score = 0
    iters = len(train_loader)

    if cfg.MODEL.USE_SCALER is True and cfg.MODEL.NAME=="CustomModel_v1":
        stats = get_dataset_statistics(train_loader, val_loader, model)
        scaler = locate(cfg.SCALER.NAME)(cfg.SCALER.MODE, cfg.SCALER.CHANNELS)
        scaler.set_stats(stats)
        model.set_scaler(scaler)
        for k, v in stats.items():
            print(f"{k}: {v}")

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
            if e < scheduler.T_max:
                scheduler.step(e + i / iters)

        val_loss = []

        val_true = []
        val_pred = []
        model.eval()

        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(val_loader), ncols=50, leave=False, total=len(val_loader)):
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

    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()
