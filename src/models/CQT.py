import numpy as np
import timm
import torch
import torch.nn as nn
from nnAudio.Spectrogram import CQT1992v2

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
        self.max_val = (
            torch.tensor(
                [
                    1.4916698,
                    1.3290482,
                    4.407391,
                ]
            )
            .cuda()
            .view(1, 3, 1, 1)
        )

    def set_scaler(self, scaler):
        self.scaler = scaler

    def forward(self, x):
        x = self.spec(x)
        x /= self.max_val
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


class CustomModel_v2(nn.Module):
    def __init__(
        self,
        cqt_params,
        encoder="efficientnet_b0",
        pretrained=True,
        img_h=256,
        img_w=256,
    ):
        super().__init__()
        self.cqts = nn.ModuleList()
        for param in cqt_params:
            self.cqts.append(CQT1992v2(**param))
        self.model = timm.create_model(encoder, pretrained=pretrained, in_chans=3*len(cqt_params), num_classes=1)
        self.h, self.w = img_h, img_w
        self.scaler = None

    def forward(self, x):
        x = torch.cat([self.spec(cqt, x) for cqt in self.cqts], dim=1)
        output = self.model(x)
        return output

    def spec(self, cqt, x):
        bs, ch, sig_len = x.shape
        x = x.view(-1, sig_len).unsqueeze(1)
        x = cqt(x).unsqueeze(1)
        x = nn.functional.interpolate(x, (self.h, self.w))
        _, _, h, w = x.shape
        x = x.view(bs, 3, h, w)
        return x