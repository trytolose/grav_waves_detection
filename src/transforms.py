from scipy import signal
import numpy as np

def apply_bandpass(x, lf=30, hf=400, order=8, sr=2048):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    return signal.sosfiltfilt(sos, x) / normalization

def apply_win(x):
    xr = x*signal.tukey(len(x), 0.1)
    return xr


def stack_turkey(waves, params):
    waves = np.hstack(waves)
    waves = waves / np.max(waves)
    waves = apply_win(waves)
    return waves    


def stack_bandpass_transform(waves, params):
    waves = np.hstack(waves)
    waves = waves / np.max(waves)
    waves = apply_bandpass(waves, **params)
    return waves


def stack_bandpass_turkey_transform(waves, params):
    
    waves = np.hstack(waves)
    waves = waves / np.max(waves)
    waves = apply_win(waves)
    waves = apply_bandpass(waves, **params)
    return waves


