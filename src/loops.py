import torch
from tqdm import tqdm
import numpy as np


def get_dataset_statistics(train_loader, val_loader, model):
    mins, maxes, means, stds = [], [], [], []
    result = {}
    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(train_loader, ncols=100):
            x = x.cuda().float()
            # spec = model.spec(x)
            spec = model.get_full_spec(x)
            # means.append(torch.mean(spec, dim=(2, 3)).cpu().data.numpy())
            # stds.append(torch.std(spec, dim=(2, 3)).cpu().data.numpy())
            maxes.append(torch.amax(spec, dim=(2, 3)).cpu().data.numpy())
            mins.append(torch.amin(spec, dim=(2, 3)).cpu().data.numpy())

        for x, _ in tqdm(val_loader, ncols=100):
            x = x.cuda().float()
            # spec = model.spec(x)
            spec = model.get_full_spec(x)
            # means.append(torch.mean(spec, dim=(2, 3)).cpu().data.numpy())
            # stds.append(torch.std(spec, dim=(2, 3)).cpu().data.numpy())
            maxes.append(torch.amax(spec, dim=(2, 3)).cpu().data.numpy())
            mins.append(torch.amin(spec, dim=(2, 3)).cpu().data.numpy())
    mins = np.concatenate(mins)
    maxes = np.concatenate(maxes)
    result["min"] = mins.min(axis=0)
    result["max"] = maxes.max(axis=0)

    min_t = torch.tensor(result["min"]).cuda().reshape(1, -1, 1, 1)
    max_t = torch.tensor(result["max"]).cuda().reshape(1, -1, 1, 1)

    with torch.no_grad():
        for x, _ in tqdm(train_loader, ncols=100):
            x = x.cuda().float()
            # spec = model.spec(x)
            spec = model.get_full_spec(x)
            spec = (spec - min_t) / (max_t - min_t)
            means.append(torch.mean(spec, dim=(2, 3)).cpu().data.numpy())
            stds.append(torch.std(spec, dim=(2, 3)).cpu().data.numpy())
            # maxes.append(torch.amax(spec, dim=(2, 3)).cpu().data.numpy())
            # mins.append(torch.amin(spec, dim=(2, 3)).cpu().data.numpy())

        for x, _ in tqdm(val_loader, ncols=100):
            x = x.cuda().float()
            # spec = model.spec(x)
            spec = model.get_full_spec(x)
            spec = (spec - min_t) / (max_t - min_t)
            means.append(torch.mean(spec, dim=(2, 3)).cpu().data.numpy())
            stds.append(torch.std(spec, dim=(2, 3)).cpu().data.numpy())
            # maxes.append(torch.amax(spec, dim=(2, 3)).cpu().data.numpy())
            # mins.append(torch.amin(spec, dim=(2, 3)).cpu().data.numpy())
    means = np.concatenate(means)
    stds = np.concatenate(stds)

    result["mean"] = means.mean(axis=0)
    result["std"] = stds.mean(axis=0)

    # result["max"] = maxes.max()
    # result["min"] = mins.min()
    # result["mean"] = means.mean()
    # result["std"] = stds.mean()

    return result
