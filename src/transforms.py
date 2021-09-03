from scipy import signal
import numpy as np

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


def minmax_transform(waves, params):
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


def single_lab_transform(waves, params):
    waves = waves[params.CH]
    waves = min_max_scale(waves)
    return waves


def minmax_turkey_transform(waves, params):
    waves = min_max_scale(waves)
    waves = apply_win(waves)
    return waves


def minmax_bandpass_transform(waves, params):
    waves = min_max_scale(waves)
    waves = apply_bandpass(waves, **params)
    return waves


def minmax_turkey_bandpass_transform(waves, params):
    waves = min_max_scale(waves)
    waves = apply_win(waves)
    waves = apply_bandpass(waves, **params)
    return waves
