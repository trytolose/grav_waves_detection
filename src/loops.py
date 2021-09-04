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
            spec = model.spec(x)
            means.append(torch.mean(spec, dim=(2, 3)).cpu().data.numpy())
            stds.append(torch.std(spec, dim=(2, 3)).cpu().data.numpy())
            maxes.append(torch.amax(spec, dim=(2, 3)).cpu().data.numpy())
            mins.append(torch.amin(spec, dim=(2, 3)).cpu().data.numpy())

        for x, _ in tqdm(val_loader, ncols=100):
            x = x.cuda().float()
            spec = model.spec(x)
            means.append(torch.mean(spec, dim=(2, 3)).cpu().data.numpy())
            stds.append(torch.std(spec, dim=(2, 3)).cpu().data.numpy())
            maxes.append(torch.amax(spec, dim=(2, 3)).cpu().data.numpy())
            mins.append(torch.amin(spec, dim=(2, 3)).cpu().data.numpy())
    mins = np.concatenate(mins)
    maxes = np.concatenate(maxes)
    means = np.concatenate(means)
    stds = np.concatenate(stds)

    result["max"] = maxes.max()
    result["min"] = mins.min()
    result["mean"] = means.mean()
    result["std"] = stds.std()
    
    result["max_3"] = maxes.max(axis=0)
    result["min_3"] = mins.min(axis=0)
    result["mean_3"] = means.mean(axis=0)
    result["std_3"] = stds.std(axis=0)
    return result
