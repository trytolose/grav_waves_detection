from scipy import signal
import numpy as np
import torch
from typing import List, Optional

TRAIN_MAX_VAL = np.array([4.61521162e-20, 4.14383536e-20, 1.11610637e-20])
TRAIN_MIN_VAL = np.array([-4.42943562e-20, -4.23039083e-20, -1.08631992e-20])

TEST_MAX_VAL = np.array([4.16750054e-20, 4.16596419e-20, 1.09645901e-20])
TEST_MIN_VAL = np.array([-4.12508703e-20, -4.17404094e-20, -1.07887724e-20])

TOTAL_MAX_VAL = np.array([4.61521162e-20, 4.16596419e-20, 1.11610637e-20])
TOTAL_MIN_VAL = np.array([-4.42943562e-20, -4.23039083e-20, -1.08631992e-20])

# 1. Смещать  временные данные по врвемени чуть вправо влево и далее строить ку траснофрмы
# 2. выпиливать из временных данных случайный узкий диапазон частот и далее строить ку транс
# 3. применять турки траснформ и не применять итп


def apply_bandpass(x, lf=30, hf=400, order=8, sr=2048):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    return signal.sosfiltfilt(sos, x) / normalization


def min_max_scale(waves, min_val=-1, max_val=1):
    X_std = (waves.T - TOTAL_MIN_VAL) / (TOTAL_MAX_VAL - TOTAL_MIN_VAL)
    X_scaled = X_std * (max_val - min_val) + min_val
    return X_scaled.T


def apply_win(x):
    xr = x * signal.tukey(4096, 0.1)
    return xr


def minmax_transform(waves):
    waves = min_max_scale(waves)
    return waves


def minmax_plus_minmax_bandpass(waves, params):
    waves = min_max_scale(waves)
    waves_add = apply_bandpass(waves, **params)
    return np.concatenate([waves, waves_add])


def stack_minmax_transform(waves, params):
    waves = np.hstack(waves)
    waves = min_max_scale(waves)
    return waves


def minmax_turkey_transform(waves, params):
    waves = min_max_scale(waves)
    waves = apply_win(waves)
    return waves


def minmax_bandpass_transform(waves, lf=30, hf=400):
    waves = min_max_scale(waves)
    waves = apply_bandpass(waves, lf=lf, hf=hf)
    return waves


def bandpass_transform(waves, lf=30, hf=400):
    # waves = min_max_scale(waves)
    waves = apply_bandpass(waves, lf=lf, hf=hf)
    return waves


def turkey_bandpass_transform(waves, lf=30, hf=400):
    waves = apply_win(waves)
    waves = apply_bandpass(waves, lf=lf, hf=hf)
    return waves


def minmax_turkey_bandpass_transform(waves, lf=30, hf=400):
    waves = min_max_scale(waves)
    waves = apply_win(waves)
    waves = apply_bandpass(waves, lf=lf, hf=hf)
    return waves


class Scaler:
    def __init__(
        self,
        mode: str = "none",
        channels: int = 1,
        min_val=-1,
        max_val=1,
    ) -> None:
        self.mode = mode
        self.channels = channels
        self.min_val = min_val
        self.max_val = max_val

    def set_stats(self, stats):
        if self.mode == "minmax":
            self.min_val = torch.tensor(self.min_val).cuda()
            self.max_val = torch.tensor(self.max_val).cuda()

            if self.channels == 1:
                self.ds_min = torch.tensor([stats["min"]]).cuda()
                self.ds_max = torch.tensor([stats["max"]]).cuda()
            elif self.channels == 3:
                self.ds_min = torch.tensor(stats["min_3"]).cuda().reshape(1, 3, 1, 1)
                self.ds_max = torch.tensor(stats["max_3"]).cuda().reshape(1, 3, 1, 1)
            self.scaler_fn = self.minmax_scaler
        elif self.mode == "standart":
            if self.channels == 1:
                self.ds_mean = torch.tensor([stats["mean"]]).cuda()
                self.ds_std = torch.tensor([stats["std"]]).cuda()
            elif self.channels == 3:
                self.ds_mean = torch.tensor(stats["mean_3"]).cuda().reshape(1, 3, 1, 1)
                self.ds_std = torch.tensor(stats["std_3"]).cuda().reshape(1, 3, 1, 1)

            self.scaler_fn = self.standart_scaler
        else:
            self.scaler_fn = self.get_wave

    def get_wave(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def minmax_scaler(self, x: torch.Tensor) -> torch.Tensor:
        X_std = (x - self.ds_min) / (self.ds_max - self.ds_min)
        X_scaled = X_std * (self.max_val - self.min_val) + self.min_val
        return X_scaled

    def standart_scaler(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.ds_mean) / self.ds_std

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.scaler_fn(X)
