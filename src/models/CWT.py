import torch
import torch.nn as nn
import numpy as np
import timm
import math


class CustomModel_CWT(nn.Module):
    def __init__(
        self,
        encoder="efficientnet_b0",
        pretrained=True,
        nv=8,  # number of voices
        sr=2048,  # sample rate (Hz)
        flow=8,  # lowest frequency of interest (Hz)
        fhigh=500,  # highest frequency of interest (Hz)
        img_h=256,
        img_w=256,
    ):
        super().__init__()
        self.model = timm.create_model(encoder, pretrained=pretrained, in_chans=3, num_classes=1)
        self.cwt = CWT_TF(
            nv=nv,
            sr=sr,
            flow=flow,
            fhigh=fhigh,
        )
        self.h, self.w = img_h, img_w

        self.scaler = None

    def set_scaler(self, scaler):
        self.scaler = scaler

    def forward(self, x):
        x = self.get_full_spec(x)
        if self.scaler is not None:
            x = self.scaler(x)
        output = self.model(x)
        return output

    def get_full_spec(self, x):
        bs, ch, sig_len = x.shape
        x = x.view(-1, sig_len)
        x = self.cwt(x).unsqueeze(1)
        x = nn.functional.interpolate(x, (self.h, self.w)).float()
        _, _, h, w = x.shape
        x = x.view(bs, 3, h, w)
        return x


class CWT_TF(nn.Module):
    def __init__(
        self,
        nv=8,  # number of voices
        sr=2048,  # sample rate (Hz)
        flow=8,  # lowest frequency of interest (Hz)
        fhigh=500,  # highest frequency of interest (Hz)
    ):
        super().__init__()
        self.nv = nv
        self.sr = sr
        self.flow = flow
        self.fhigh = fhigh

        self.get_const()

    def kron(self, a, b):
        """
        Kronecker product of matrices a and b with leading batch dimensions.
        Batch dimensions are broadcast. The number of them mush
        :type a: torch.Tensor
        :type b: torch.Tensor
        :rtype: torch.Tensor
        """
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        return res.reshape(siz0 + siz1)

    def get_const(self):

        input_shape = np.array([3, 4096])
        max_scale = input_shape[-1] // (np.sqrt(2) * 2)
        if max_scale <= 1:
            max_scale = input_shape[-1] // 2
        max_scale = np.floor(self.nv * np.log2(max_scale))

        scales = 2 * (2 ** (1 / self.nv)) ** np.arange(0, max_scale + 1)

        frequencies = self.sr * (6 / (2 * np.pi)) / scales
        frequencies = frequencies[frequencies >= self.flow]  # remove low frequencies
        scales = scales[0 : len(frequencies)]
        frequencies = frequencies[frequencies <= self.fhigh]  # remove high frequencies
        scales = scales[len(scales) - len(frequencies) : len(scales)]
        # # wavft
        padvalue = input_shape[-1] // 2
        n = padvalue * 2 + input_shape[-1]
        omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float64)
        omega = omega * (2 * np.pi) / n
        omega = np.concatenate(
            (np.array([0]), omega, -omega[np.arange(math.floor((n - 1) / 2), 0, -1, dtype=int) - 1])
        )
        _wft = np.zeros([scales.size, omega.size])
        for jj, scale in enumerate(scales):
            expnt = -((scale * omega - 6) ** 2) / 2 * (omega > 0)
            _wft[jj,] = (
                2 * np.exp(expnt) * (omega > 0)
            )

        # # parameters we want to use during call():
        self.wft = torch.tensor(_wft).cuda()
        self.padvalue = padvalue
        self.num_scales = scales.shape[-1]
        self.cron_ones_tensor = torch.ones(self.num_scales, 1).type(torch.complex64).cuda()

    def forward(self, x):

        bs, wave_len = x.shape
        x_left_pad = torch.flip(x[:, 0 : self.padvalue], dims=[1])
        x_right_pad = torch.flip(x[:, -self.padvalue :], dims=[1])
        x = torch.cat([x_left_pad, x, x_right_pad], dim=1)

        x = x.type(torch.complex128)
        f = torch.fft.fft(x, dim=1)

        kron_prod = self.kron(self.cron_ones_tensor.unsqueeze(0).repeat(bs, 1, 1), f.unsqueeze(1))
        cwtcfs = torch.fft.ifft(kron_prod * self.wft.unsqueeze(0).repeat(bs, 1, 1), dim=2)

        logcwt = torch.log(torch.abs(cwtcfs[:, :, self.padvalue : self.padvalue + wave_len]))
        return logcwt
