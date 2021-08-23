import numpy as np
import timm
import torch
import torch.nn as nn
from nnAudio.Spectrogram import CQT1992v2


class CustomModel_v0(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=pretrained, in_chans=1)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, 1)
        self.cqt = CQT1992v2(**cfg.qtransform_params)

    def forward(self, x):
        x = x.unsqueeze(1)
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.cqt(x).unsqueeze(1)
        x = nn.functional.interpolate(x, (256, 386))
        output = self.model(x)
        return output


class CustomModel_v1(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=pretrained, in_chans=3)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, 1)
        self.cqt = CQT1992v2(**cfg.qtransform_params)

    def forward(self, x):
        bs, ch, sig_len = x.shape
        x = x.view(-1, sig_len)
        x = self.cqt(x).unsqueeze(1)
        x = nn.functional.interpolate(x, (256, 256))
        _, _, h, w = x.shape
        x = x.view(bs, 3, h, w)
        output = self.model(x)
        return output


def get_model(cfg):
    if cfg.MODEL.NAME == "CustomModel_CWT":
        return CustomModel_CWT(cfg)


class CustomModel_CWT(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=pretrained, in_chans=3, num_classes=1)
        cwt_params = dict(cfg.cwt_params)
        start, end, step = tuple(cwt_params["widths"])
        cwt_params["widths"] = np.arange(start, end, step)
        self.cwt = CWT(**cwt_params)
        self.h, self.w = cfg.MODEL.IMG_SIZE

    def forward(self, x):
        x = self.cwt(x)
        x = torch.absolute(x)
        x = nn.functional.interpolate(x, (self.h, self.w))
        output = self.model(x)
        return output


class CWT(nn.Module):
    def __init__(
        self,
        widths,
        wavelet="ricker",
        channels=1,
        filter_len=2000,
    ):
        """PyTorch implementation of a continuous wavelet transform.

        Args:
            widths (iterable): The wavelet scales to use, e.g. np.arange(1, 33)
            wavelet (str, optional): Name of wavelet. Either "ricker" or "morlet".
            Defaults to "ricker".
            channels (int, optional): Number of audio channels in the input. Defaults to 3.
            filter_len (int, optional): Size of the wavelet filter bank. Set to
            the number of samples but can be smaller to save memory. Defaults to 2000.
        """
        super().__init__()
        self.widths = widths
        self.wavelet = getattr(self, wavelet)
        self.filter_len = filter_len
        self.channels = channels
        self.wavelet_bank = self._build_wavelet_bank()

    def ricker(self, points, a):
        # https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/wavelets.py#L262-L306
        A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
        wsq = a ** 2
        vec = torch.arange(0, points) - (points - 1.0) / 2
        xsq = vec ** 2
        mod = 1 - xsq / wsq
        gauss = torch.exp(-xsq / (2 * wsq))
        total = A * mod * gauss
        return total

    def morlet(self, points, s):
        x = torch.arange(0, points) - (points - 1.0) / 2
        x = x / s
        # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#morlet-wavelet
        wavelet = torch.exp(-(x ** 2.0) / 2.0) * torch.cos(5.0 * x)
        output = np.sqrt(1 / s) * wavelet
        return output

    def cmorlet(self, points, s, wavelet_width=1, center_freq=1):
        # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#complex-morlet-wavelets
        x = torch.arange(0, points) - (points - 1.0) / 2
        x = x / s
        norm_constant = np.sqrt(np.pi * wavelet_width)
        exp_term = torch.exp(-(x ** 2) / wavelet_width)
        kernel_base = exp_term / norm_constant
        kernel = kernel_base * torch.exp(1j * 2 * np.pi * center_freq * x)
        return kernel

    def _build_wavelet_bank(self):
        """This function builds a 2D wavelet filter using wavelets at different scales

        Returns:
            tensor: Tensor of shape (num_widths, 1, channels, filter_len)
        """
        wavelet_bank = [torch.conj(torch.flip(self.wavelet(self.filter_len, w), [-1])) for w in self.widths]
        wavelet_bank = torch.stack(wavelet_bank)
        wavelet_bank = wavelet_bank.view(wavelet_bank.shape[0], 1, 1, wavelet_bank.shape[1])
        wavelet_bank = torch.cat([wavelet_bank] * self.channels, 2)
        return wavelet_bank

    def forward(self, x):
        """Compute CWT arrays from a batch of multi-channel inputs

        Args:
            x (torch.tensor): Tensor of shape (batch_size, channels, time)

        Returns:
            torch.tensor: Tensor of shape (batch_size, channels, widths, time)
        """
        x = x.unsqueeze(1)
        if self.wavelet_bank.is_complex():
            wavelet_real = self.wavelet_bank.real.cuda()  # .to(device=x.device, dtype=x.dtype)
            wavelet_imag = self.wavelet_bank.imag.cuda()  # .to(device=x.device, dtype=x.dtype)

            output_real = nn.functional.conv2d(x, wavelet_real, padding="same")
            output_imag = nn.functional.conv2d(x, wavelet_imag, padding="same")
            output_real = torch.transpose(output_real, 1, 2)
            output_imag = torch.transpose(output_imag, 1, 2)
            return torch.complex(output_real, output_imag)
            # return output_real, output_imag
        else:
            self.wavelet_bank = self.wavelet_bank.cuda()  # .to(device=x.device, dtype=x.dtype)
            output = nn.functional.conv2d(x, self.wavelet_bank, padding="same")
            return torch.transpose(output, 1, 2)
