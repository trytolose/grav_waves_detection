import numpy as np
import timm
import torch
import torch.nn as nn
from nnAudio.Spectrogram import CQT1992v2


class CustomModel_v0(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.model = timm.create_model(cfg.MODEL.ENCODER, pretrained=pretrained, in_chans=1, num_classes=1)
        self.cqt = CQT1992v2(**cfg.qtransform_params)
        self.h, self.w = cfg.MODEL.IMG_SIZE

    def forward(self, x):
        x = x.unsqueeze(1)
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.cqt(x).unsqueeze(1)
        x = nn.functional.interpolate(x, (self.h, self.w))
        output = self.model(x)
        return output


class CustomModel_v1(nn.Module):
    def __init__(
        self,
        encoder="efficientnet_b0",
        pretrained=True,
        sr=2048,
        fmin=20,
        fmax=1024,
        hop_length=32,
        bins_per_octave=8,
        filter_scale=1,
        img_h=256,
        img_w=256,
    ):
        super().__init__()
        self.model = timm.create_model(encoder, pretrained=pretrained, in_chans=3, num_classes=1)
        self.cqt = CQT1992v2(
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            bins_per_octave=bins_per_octave,
            filter_scale=filter_scale,
        )
        self.h, self.w = img_h, img_w

        self.scaler = None

    def set_scaler(self, scaler):
        self.scaler = scaler

    def forward(self, x):
        x = self.spec(x)
        if self.scaler is not None:
            x = self.scaler(x)
        output = self.model(x)
        return output

    def spec(self, x):
        bs, ch, sig_len = x.shape
        x = x.view(-1, sig_len).unsqueeze(1)
        x = self.cqt(x).unsqueeze(1)
        x = nn.functional.interpolate(x, (self.h, self.w))
        _, _, h, w = x.shape
        x = x.view(bs, 3, h, w)
        return x
