import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from pickle import dump, load
from sklearn.model_selection import StratifiedKFold
import torch.utils.data as utils
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
from sklearn import metrics
from collections import deque

BS = 200
FOLD = 0

INPUT_PATH = Path("/home/trytolose/rinat/kaggle/g2net/input")
df = pd.read_csv(INPUT_PATH / "training_labels.csv")

files = list((INPUT_PATH / "train").rglob("*.npy"))
FILE_PATH_DICT = {x.stem: str(x) for x in files}
df["path"] = df["id"].apply(lambda x: FILE_PATH_DICT[x])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
df["fold"] = -1
for f, (train_ids, val_ids) in enumerate(skf.split(df.index, y=df["target"])):
    df.loc[val_ids, "fold"] = f


class Wave_Block_true(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(Wave_Block_true, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.dil_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.dil_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation_rate,
                    dilation=dilation_rate,
                )
            )
            self.filter_convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))
            self.gate_convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

        self.end_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.convs[0](x)
        #         res = x
        skip = 0
        for i in range(self.num_rates):

            res = x
            x = self.dil_convs[i](x)
            x = torch.mul(
                torch.tanh(self.filter_convs[i](x)),
                torch.sigmoid(self.gate_convs[i](x)),
            )
            x = self.convs[i + 1](x)
            skip = skip + x
            # x += res
            x = x + res

        x = self.end_block(skip)
        return x


class Andrewnet_v3_true(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.first_conv = nn.Sequential(nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.waveblock_1 = nn.Sequential(Wave_Block_true(64, 16, 12), nn.BatchNorm1d(16))
        self.waveblock_2 = nn.Sequential(Wave_Block_true(16, 32, 8), nn.BatchNorm1d(32))
        self.waveblock_3 = nn.Sequential(Wave_Block_true(32, 64, 4), nn.BatchNorm1d(64))
        self.waveblock_4 = nn.Sequential(Wave_Block_true(64, 128, 1), nn.BatchNorm1d(128))
        self.waveblock_5 = nn.Sequential(nn.Conv1d(128, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU())

        self.dropout = nn.Dropout(p=0.2)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.waveblock_1(x)
        x = self.waveblock_2(x)
        x = self.waveblock_3(x)
        x = self.waveblock_4(x)
        x = self.waveblock_5(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = self.fc(x.view(-1, 128))
        return x


class SignalDataset(Dataset):
    def __init__(self, df, scaler):
        self.df = df
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        target = float(self.df.loc[idx, "target"])
        x = np.load(path)
        x = x / np.max(x)
        # x = self.scaler.transform(np.load(path).T).T
        return x, target


scaler = load(open("scaler.pkl", "rb"))
train_ds = SignalDataset(df[df["fold"] != FOLD].reset_index(drop=True), scaler)
val_ds = SignalDataset(df[df["fold"] == FOLD].reset_index(drop=True), scaler)

train_loader = utils.DataLoader(train_ds, shuffle=True, num_workers=11, batch_size=BS, pin_memory=True)
val_loader = utils.DataLoader(val_ds, shuffle=False, num_workers=11, batch_size=BS, pin_memory=True)


model = Andrewnet_v3_true(3)

model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.BCEWithLogitsLoss()
scheduler = ReduceLROnPlateau(optimizer, mode="max", verbose=True, patience=5, factor=0.5, eps=1e-12)


best_score = 0
best_models = deque(maxlen=5)
for e in range(100):

    # Training:
    train_loss = []
    model.train()

    for x, y in tqdm(train_loader, ncols=70):
        optimizer.zero_grad()
        x = x.cuda().float()
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
        for x, y in tqdm(val_loader, ncols=50):
            x = x.cuda().float()
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
        torch.save(model.state_dict(), f"baseline_f0.pt")
    else:
        print()

    scheduler.step(final_score)
