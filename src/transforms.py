from scipy import signal
import numpy as np

TRAIN_MAX_VAL = np.array([4.61521162e-20, 4.14383536e-20, 1.11610637e-20])
TRAIN_MIN_VAL = np.array([-4.42943562e-20, -4.23039083e-20, -1.08631992e-20])

TEST_MAX_VAL = np.array([4.16750054e-20, 4.16596419e-20, 1.09645901e-20])
TEST_MIN_VAL = np.array([-4.12508703e-20, -4.17404094e-20, -1.07887724e-20])

TOTAL_MAX_VAL = np.array([4.61521162e-20, 4.16596419e-20, 1.11610637e-20])
TOTAL_MIN_VAL = np.array([-4.42943562e-20, -4.23039083e-20, -1.08631992e-20])

#spec_noise = np.load('/home/clover_rtx/ds_users/bulat/kaggle/grav_waves_detection/eda_notebooks/spec_avg2.npy')
#print('spec_noise', spec_noise.shape)


def apply_bandpass(x, lf=30, hf=400, order=8, sr=2048):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    return signal.sosfiltfilt(sos, x) / normalization

def min_max_scale(waves, min_val=-1, max_val=1):
    X_std = (waves.T - TOTAL_MIN_VAL) / (TOTAL_MAX_VAL - TOTAL_MIN_VAL)
    X_scaled = X_std * (max_val - min_val) + min_val
    return X_scaled.T

def apply_win(x, edge=0.2):
    """
    apply turkey window for time signal
    """
    xr = x * signal.tukey(4096, edge)
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

def substract(x, spec_noise):
    """
    substract magnitude spectrum
    x - time signal
    spec_noise = estimated noise spectrum
    return x_sub - time signal
    """
    spec = np.fft.rfft(x)
    
    spec_amp = np.abs(spec)
    spec_sub = spec_amp - spec_noise

    spec_sub = np.clip(spec_sub, a_min=0, a_max=None)
    orig_phase = np.arctan2(spec.imag, spec.real)
    real_part = spec_sub * np.cos(orig_phase)
    imag_part = spec_sub * np.sin(orig_phase)
    
    spec_res = np.array([ complex(r, im) for r, im in list(zip(real_part, imag_part)) ])
    x_sub = np.fft.irfft(spec_res)
    return  x_sub

def substract2(x_, spec_noise):
    x = x_.copy()
    #x = apply_win(x)
    spec = np.fft.rfft(x)
    
    spec_amp = np.abs(spec)**2
    spec_sub = spec_amp - spec_noise**2
    #spec_sub = spec_amp
    spec_sub = np.clip(spec_sub, a_min=0, a_max=None)
    spec_sub = np.sqrt(spec_sub)
    
    orig_phase = np.arctan2(spec.imag, spec.real)
    real_part = spec_sub * np.cos(orig_phase)
    imag_part = spec_sub * np.sin(orig_phase)
    
    spec_res = np.array([ complex(r, im) for r, im in list(zip(real_part, imag_part)) ])
    x_sub = np.fft.irfft(spec_res)
    return  x_sub

def substract_waves(waves, specs_noise, sub_func=substract):
    waves = apply_win(waves, 0.2)
    sub_waves = np.zeros(waves.shape)
    for i in range(3):
        sub_waves[i, :] = sub_func(waves[i,:], specs_noise[i,:])
    return sub_waves

def minmax_sub_bandpass_transform(waves, params):
    waves = min_max_scale(waves)
    waves = substract_waves(waves, spec_noise)
    waves = apply_bandpass(waves, **params)
    return waves