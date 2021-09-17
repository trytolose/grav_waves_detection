import os
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List
from pydoc import locate
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from sklearn import metrics

from src.dataset import get_loaders, get_test_loader
from src.models import get_model

warnings.filterwarnings("ignore")


def average_weights(state_dicts: List[dict]):
    everage_dict = OrderedDict()
    for k in state_dicts[0].keys():
        everage_dict[k] = sum([state_dict[k] for state_dict in state_dicts]) / len(state_dicts)
    return everage_dict


def avg(model, cfg, fold):
    checkpoints_path = (
        f"/home/trytolose/rinat/kaggle/grav_waves_detection/weights/{cfg.MODEL.NAME}/{cfg.EXP_NAME}/fold_{fold}"
    )
    checkpoints_weights_paths = list(Path(checkpoints_path).glob("*.pth"))
    checkpoints_weights_paths = [x for x in checkpoints_weights_paths if "avg" not in x.name and "last" not in x.name]
    checkpoints_weights_paths = sorted(
        checkpoints_weights_paths, key=lambda x: float(x.stem.split("score")[-1]), reverse=True
    )
    for ch in checkpoints_weights_paths:
        print(ch)
    cfg.FOLD = fold
    _, val_loader = get_loaders(cfg)

    all_weights = [torch.load(path, map_location="cuda") for path in checkpoints_weights_paths]
    best_score = 0
    best_weights = []
    for w in all_weights:
        current_weights = best_weights + [w]
        average_dict = average_weights(current_weights)
        model.load_state_dict(average_dict)
        score = get_score(model, val_loader)
        print(score)
        if score > best_score:
            best_score = score
            best_weights.append(w)

    torch.save(average_weights(best_weights), Path(checkpoints_path) / f"best_avg_score{best_score:.5f}.pth")
    print("avg weight saved")


def get_score(model, loader):
    val_true = []
    val_pred = []
    model.eval()

    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(loader), ncols=50, leave=False, total=len(loader)):
            x = x.cuda().float()
            y = y.cuda().float().unsqueeze(1)
            pred = model(x)
            pred = pred.sigmoid().cpu().data.numpy()
            val_pred.append(pred)
            val_true.append(y.cpu().numpy())

    val_true = np.concatenate(val_true).reshape(
        -1,
    )
    val_pred = np.concatenate(val_pred).reshape(
        -1,
    )
    final_score = metrics.roc_auc_score(val_true, val_pred)
    return final_score


def get_model_paths(exp_path, crietion="best"):
    folders = [x for x in os.listdir(exp_path) if "fold" in x]
    folds = sorted(folders, key=lambda x: int(x.split("_")[-1]))
    result = []
    for f in folds:
        weights = list((Path(exp_path) / f).glob("*.pth"))
        weights = [x for x in weights if "score" in x.name]
        if crietion == "last":
            weights = sorted(weights, key=lambda x: int(x.stem.split("epoch")[-1].split("_")[0]))
            result.append(weights[-1])
        if crietion == "best":
            weights = sorted(weights, key=lambda x: float(x.stem.split("score")[-1]))
            result.append(weights[-1])
    return result


def get_oof(cfg, model):
    checkpoints_path = f"/home/trytolose/rinat/kaggle/grav_waves_detection/weights/{cfg.MODEL.NAME}/{cfg.EXP_NAME}"
    paths = get_model_paths(checkpoints_path)

    dfs = []
    for fold in range(5):
        cfg.FOLD = fold
        _, val_loader = get_loaders(cfg)

        val_pred = []
        print(paths[fold])
        model.load_state_dict(torch.load(paths[fold]))
        model.eval()

        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(val_loader), ncols=50, leave=True, total=len(val_loader)):
                x = x.cuda().float()
                y = y.cuda().float().unsqueeze(1)
                pred = model(x)
                pred = pred.sigmoid().cpu().data.numpy()
                val_pred.append(pred)

        val_pred = np.concatenate(val_pred).reshape(
            -1,
        )
        df_pred = val_loader.dataset.df.copy()
        df_pred["pred"] = val_pred
        dfs.append(df_pred)
    dfs = pd.concat(dfs, ignore_index=True)

    dfs.to_csv(f"oof_{cfg.EXP_NAME}.csv", index=False)


def inference(cfg, model, paths=None):
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoints_path = f"/home/trytolose/rinat/kaggle/grav_waves_detection/weights/{cfg.MODEL.NAME}/{cfg.EXP_NAME}"
    if paths is None:
        paths = get_model_paths(checkpoints_path)
    for p in paths:
        print(p)

    test_loader = get_test_loader(cfg)
    df = test_loader.dataset.df

    predicts = []
    for path in paths:
        model.load_state_dict(torch.load(path))
        model.eval()
        val_pred = []
        with torch.no_grad():
            for i, x in tqdm(enumerate(test_loader), ncols=50, leave=True, total=len(test_loader)):
                x = x.cuda().float()
                pred = model(x)
                # pred = pred.sigmoid().cpu().data.numpy()
                pred = pred.softmax(dim=1).cpu().data.numpy()
                val_pred.append(pred)

        # val_pred = np.concatenate(val_pred).reshape(
        #     -1,
        # )
        val_pred = np.concatenate(val_pred)
        val_pred = val_pred[:, 1:].sum(axis=1)
        predicts.append(val_pred)

    predicts = np.stack(predicts).T  # .mean(axis=0)
    df[[f"fold_{i}" for i in range(len(paths))]] = predicts
    df = df.drop("path", axis=1)
    df.to_csv(Path(checkpoints_path) / "submission.csv", index=False)


@hydra.main(config_path="./configs", config_name="default")
def main(cfg: DictConfig):
    model = get_model(cfg)
    model.cuda()

    if cfg.MODEL.USE_SCALER is True and "CustomModel" in cfg.MODEL.NAME:
        stats = {}
        stats["min"] = cfg.SCALER.MIN
        stats["max"] = cfg.SCALER.MAX
        stats["mean"] = cfg.SCALER.MEAN
        stats["std"] = cfg.SCALER.STD

        spec_scaler = locate(cfg.SCALER.NAME)(cfg.SCALER.MODE)
        spec_scaler.set_stats(stats)
        model.set_scaler(spec_scaler)

    # print(OmegaConf.to_yaml(cfg))

    # avg(model, cfg, cfg.FOLD)
    # get_oof(cfg, model)
    inference(cfg, model)


# 0.8712827783072145

if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()
