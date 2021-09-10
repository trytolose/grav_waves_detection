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
        img_h=256,
        img_w=256,
        scale_cqt=False
    ):
        super().__init__()
        self.model = timm.create_model(encoder, pretrained=pretrained, in_chans=3, num_classes=1)
        self.cqt = CQT1992v2(sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, bins_per_octave=bins_per_octave)
        self.h, self.w = img_h, img_w
        self.scale_cqt = scale_cqt
        self.max_vals = torch.Tensor([1.4916698, 1.3290482,  4.407391 ]).cuda()

    def forward(self, x):
        bs, ch, sig_len = x.shape
        x = x.view(-1, sig_len)
        x = self.cqt(x).unsqueeze(1)
        x = nn.functional.interpolate(x, (self.h, self.w))
        _, _, h, w = x.shape
        x = x.view(bs, ch, h, w)

        # sub
        if self.scale_cqt:
            x = torch.transpose(x, 1, 3)/self.max_vals
            x = torch.transpose(x, 3, 1)

        output = self.model(x)
        return output

class CustomModel_v2(nn.Module):
    def __init__(
        self,
        encoder="efficientnet_b0",
        pretrained=True,
        img_h=256,
        img_w=256,
        in_chans=3,
        scale_cqt=False,

        cqts_params = [
            {
            'sr':2048,
            'fmin':20,
            'fmax':1024,
            'hop_length':32,
            'bins_per_octave':8
            }
        ]  
    ):
        super().__init__()
        self.n_cqt = len(cqts_params)

        self.in_chans = in_chans*self.n_cqt 
        self.model = timm.create_model(encoder, pretrained=pretrained, in_chans=self.in_chans, num_classes=1)
        self.h, self.w = img_h, img_w
        self.scale_cqt = scale_cqt
        
        self.cqt0 = CQT1992v2(**cqts_params[0])
        self.cqt1 = CQT1992v2(**cqts_params[1])
        #self.cqt2 = CQT1992v2(**cqts_params[2])

    def forward(self, x):
        h, w = self.h, self.w
        bs, ch, sig_len = x.shape
        x = x.view(-1, sig_len).unsqueeze(1)

        x0 = self.cqt0(x).unsqueeze(1)
        x0 = nn.functional.interpolate(x0, (h, w))

        x1 = self.cqt1(x).unsqueeze(1)
        x1 = nn.functional.interpolate(x1, (h, w))
        
        x = torch.cat([x0, x1], dim=1)
        ch = self.n_cqt*ch
        x = x.view(bs, ch, h, w)
        output = self.model(x)
        return output


class CustomModel_v3(nn.Module):
    def __init__(
        self,
        encoder="efficientnet_b0",
        pretrained=True,
        img_h=256,
        img_w=256,
        in_chans=3,
        scale_cqt=False,

        cqts_params = [
            {
            'sr':2048,
            'fmin':20,
            'fmax':1024,
            'hop_length':32,
            'bins_per_octave':8
            }
        ]  
    ):
        super().__init__()
        self.n_cqt = len(cqts_params)

        self.in_chans = in_chans*self.n_cqt 
        self.model = timm.create_model(encoder, pretrained=pretrained, in_chans=self.in_chans, num_classes=1)
        self.h, self.w = img_h, img_w
        self.scale_cqt = scale_cqt
        
        self.cqt0 = CQT1992v2(**cqts_params[0])
        self.cqt1 = CQT1992v2(**cqts_params[1])
        self.cqt2 = CQT1992v2(**cqts_params[2])

        self.max_vals = torch.Tensor([1.4916698, 1.3290482,  4.407391]).repeat(self.n_cqt).cuda()

    def forward(self, x):
        h, w = self.h, self.w
        bs, ch, sig_len = x.shape
        x = x.view(-1, sig_len).unsqueeze(1)

        x0 = self.cqt0(x).unsqueeze(1)
        x0 = nn.functional.interpolate(x0, (h, w))

        x1 = self.cqt1(x).unsqueeze(1)
        x1 = nn.functional.interpolate(x1, (h, w))

        x2 = self.cqt2(x).unsqueeze(1)
        x2 = nn.functional.interpolate(x2, (h, w))
        
        x = torch.cat([x0, x1, x2], dim=1)
        ch = self.n_cqt*ch
        x = x.view(bs, ch, h, w)

        if self.scale_cqt:
            x = torch.transpose(x, 1, 3)/self.max_vals
            x = torch.transpose(x, 3, 1)
        output = self.model(x)
        return output


class CustomModel_v5(nn.Module):
    def __init__(
        self,
        encoder="efficientnet_b0",
        pretrained=True,
        img_h=256,
        img_w=256,
        in_chans=3,
        scale_cqt=False,

        cqts_params = [
            {
            'sr':2048,
            'fmin':20,
            'fmax':1024,
            'hop_length':32,
            'bins_per_octave':8
            }
        ]  
    ):
        super().__init__()
        self.n_cqt = len(cqts_params)

        self.in_chans = in_chans*self.n_cqt 
        self.model = timm.create_model(encoder, pretrained=pretrained, in_chans=self.in_chans, num_classes=1)
        self.h, self.w = img_h, img_w
        self.scale_cqt = scale_cqt
        
        self.cqt0 = CQT1992v2(**cqts_params[0])
        self.cqt1 = CQT1992v2(**cqts_params[1])
        self.cqt2 = CQT1992v2(**cqts_params[2])
        self.cqt3 = CQT1992v2(**cqts_params[3])
        self.cqt4 = CQT1992v2(**cqts_params[4])

        self.max_vals = torch.Tensor([1.4916698, 1.3290482,  4.407391]).repeat(self.n_cqt).cuda()

    def forward(self, x):
        h, w = self.h, self.w
        bs, ch, sig_len = x.shape
        x = x.view(-1, sig_len).unsqueeze(1)

        x0 = self.cqt0(x).unsqueeze(1)
        x0 = nn.functional.interpolate(x0, (h, w))

        x1 = self.cqt1(x).unsqueeze(1)
        x1 = nn.functional.interpolate(x1, (h, w))

        x2 = self.cqt2(x).unsqueeze(1)
        x2 = nn.functional.interpolate(x2, (h, w))

        x3 = self.cqt3(x).unsqueeze(1)
        x3 = nn.functional.interpolate(x3, (h, w))

        x4 = self.cqt4(x).unsqueeze(1)
        x4 = nn.functional.interpolate(x4, (h, w))
        
        x = torch.cat([x0, x1, x2, x3, x4], dim=1)
        ch = self.n_cqt*ch
        x = x.view(bs, ch, h, w)

        if self.scale_cqt:
            x = torch.transpose(x, 1, 3)/self.max_vals
            x = torch.transpose(x, 3, 1)
        output = self.model(x)
        return output