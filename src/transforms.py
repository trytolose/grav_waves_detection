from scipy import signal
import numpy as np
import torch
from typing import List, Optional
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
from skimage.restoration import denoise_tv_chambolle


TRAIN_MAX_VAL = np.array([4.61521162e-20, 4.14383536e-20, 1.11610637e-20])
TRAIN_MIN_VAL = np.array([-4.42943562e-20, -4.23039083e-20, -1.08631992e-20])

TEST_MAX_VAL = np.array([4.16750054e-20, 4.16596419e-20, 1.09645901e-20])
TEST_MIN_VAL = np.array([-4.12508703e-20, -4.17404094e-20, -1.07887724e-20])

TOTAL_MAX_VAL = np.array([4.61521162e-20, 4.16596419e-20, 1.11610637e-20])
TOTAL_MIN_VAL = np.array([-4.42943562e-20, -4.23039083e-20, -1.08631992e-20])

# 1. Смещать  временные данные по врвемени чуть вправо влево и далее строить ку траснофрмы
# 2. выпиливать из временных данных случайный узкий диапазон частот и далее строить ку транс
# 3. применять турки траснформ и не применять итп

def iir_bandstops(fstops, fs, order=4):
    """ellip notch filter
    fstops is a list of entries of the form [frequency (Hz), df, df2]
    where df is the pass width and df2 is the stop width (narrower
    than the pass width). Use caution if passing more than one freq at a time,
    because the filter response might behave in ways you don't expect.
    """
    nyq = 0.5 * fs

    # Zeros zd, poles pd, and gain kd for the digital filter
    zd = np.array([])
    pd = np.array([])
    kd = 1

    # Notches
    for fstopData in fstops:
        fstop = fstopData[0]
        df = fstopData[1]
        df2 = fstopData[2]
        low = (fstop - df) / nyq
        high = (fstop + df) / nyq
        low2 = (fstop - df2) / nyq
        high2 = (fstop + df2) / nyq
        z, p, k = iirdesign([low, high], [low2, high2], gpass=1, gstop=6,
                            ftype='ellip', output='zpk')
        zd = np.append(zd, z)
        pd = np.append(pd, p)

    # Set gain to one at 100 Hz...better not notch there
    bPrelim, aPrelim = zpk2tf(zd, pd, 1)
    outFreq, outg0 = freqz(bPrelim, aPrelim, 100 / nyq)

    # Return the numerator and denominator of the digital filter
    b, a = zpk2tf(zd, pd, k)
    return b, a


def get_filter_coefs(fs=2048, ch=0):
    # assemble the filter b,a coefficients:
    coefs = []

    # bandpass filter parameters
    lowcut = 30
    highcut = 1023
    order = 4

    # bandpass filter coefficients
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    bb, ab = butter(order, [low, high], btype='band')
    coefs.append((bb, ab))

    # Frequencies of notches at known instrumental spectral line frequencies.
    # You can see these lines in the ASD above, so it is straightforward to make this list.
    # notchesAbsolute = np.array(
    #     [14.0,34.70, 35.30, 35.90, 36.70, 37.30, 40.95, 60.00,
    #      120.00, 179.99, 304.99, 331.49, 510.02, 1009.99])

    # [13.5, 14.0, 38, 60.00, 306.5, 495]_
    if ch == 2:
        notchesAbsolute = np.array([438])
    else:
        notchesAbsolute = np.array([13.5, 14.0, 38, 60.00, 306.5, 495])

    # notch filter coefficients:
    for notchf in notchesAbsolute:
        bn, an = iir_bandstops(np.array([[notchf, 1, 0.1]]), fs, order=4)
        coefs.append((bn, an))

    '''
    # Manually do a wider notch filter around 510 Hz etc.          
    bn, an = iir_bandstops(np.array([[510,200,20]]), fs, order=4)
    coefs.append((bn, an))

    # also notch out the forest of lines around 331.5 Hz
    bn, an = iir_bandstops(np.array([[331.5,10,1]]), fs, order=4)
    coefs.append((bn, an))
    '''
    return coefs

coefs_list = [get_filter_coefs(fs=2048, ch=ch) for ch in range(3)]
def filter_data(data_in):
    data = data_in.copy()
    for ch in range(3):
        coefs = coefs_list[ch]
        for coef in coefs:
            b, a = coef
            # filtfilt applies a linear filter twice, once forward and once backwards.
            # The combined filter has linear phase.
            data[ch] = filtfilt(b, a, data[ch])
    return data

def apply_bandpass(x, lf=30, hf=400, order=8, sr=2048):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    return signal.sosfiltfilt(sos, x) / normalization


def min_max_scale(waves, min_val=-1, max_val=1):
    X_std = (waves.T - TOTAL_MIN_VAL) / (TOTAL_MAX_VAL - TOTAL_MIN_VAL)
    X_scaled = X_std * (max_val - min_val) + min_val
    return X_scaled.T


def apply_win(x):
    xr = x * signal.tukey(4096, 0.2)
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
    waves = filter_data(waves)
    waves = apply_win(waves)
    waves = apply_bandpass(waves)
    waves = denoise_tv_chambolle(waves)
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

    def max_scaler(self, x):
        x /= self.ds_max

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.scaler_fn(X)
