import os
import time
import warnings
from pathlib import Path
from pydoc import locate

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import get_loaders
from src.loops import get_dataset_statistics
from src.models import get_model
from src.utils.checkpoint import ModelCheckpoint
from src.utils.utils import get_gradient_norm, get_lr

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def train(cfg):
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoints_path = f"{cfg.CH_PATH}/{cfg.MODEL.NAME}/{cfg.EXP_NAME}/fold_{cfg.FOLD}"
    tensorboard_logs = f"{cfg.TENSORBOARD_DIR}/{cfg.EXP_NAME}_{time_str}_f{cfg.FOLD}"
    if cfg.DEBUG is False:
        Path(tensorboard_logs).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_logs, flush_secs=30)
        checkpoints = ModelCheckpoint(dirname=checkpoints_path, n_saved=cfg.N_SAVED)

    train_loader, val_loader = get_loaders(cfg)

    # train_loader, val_loader = get_in_memory_loaders(cfg)

    model = get_model(cfg)
    model.cuda()

    if cfg.MODEL.CHECKPOINT != "":
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

    loss_cfg_dict = dict(cfg.LOSS.CFG)
    if "pos_weight" in loss_cfg_dict:
        loss_cfg_dict["pos_weight"] = torch.tensor(loss_cfg_dict["pos_weight"]).cuda()

    optimizer = locate(cfg.OPTIMIZER.NAME)(model.parameters(), **cfg.OPTIMIZER.CFG)
    loss_fn = locate(cfg.LOSS.NAME)(**loss_cfg_dict)
    scheduler = locate(cfg.SCHEDULER.NAME)(optimizer, **cfg.SCHEDULER.CFG)
    scaler = GradScaler()

    best_score = 0
    iters = len(train_loader)

    if cfg.MODEL.USE_SCALER is True and "CustomModel" in cfg.MODEL.NAME:
        print("calc_statistics")
        stats = get_dataset_statistics(train_loader, val_loader, model)
        spec_scaler = locate(cfg.SCALER.NAME)(cfg.SCALER.MODE, cfg.SCALER.CHANNELS)
        spec_scaler.set_stats(stats)
        model.set_scaler(spec_scaler)
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
                # if cfg.GRAD_CLIP !=0 :
                #     clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                if e >= cfg.GRAD_CLIP.START_EPOCH:
                    clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP.THR)
                optimizer.step()
            # if i % 30 == 0:
            #     print(get_gradient_norm(model))
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
        # grad_norm = get_gradient_norm(model)
        print(
            f"Epoch: {e:03d}; lr: {get_lr(optimizer):.10f}; train_loss: {train_loss:.05f} val_loss: {val_loss:.05f}; roc: {final_score:.5f}",
            end=" ",
        )
        if cfg.DEBUG is False:
            tensorboard_writer.add_scalar("Learning rate", get_lr(optimizer), e)
            # tensorboard_writer.add_scalar("Grad_norm", grad_norm, e)
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
    os.environ["CUDA_VISIBLE_FEVICES"] = str(cfg.GPU)
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()
