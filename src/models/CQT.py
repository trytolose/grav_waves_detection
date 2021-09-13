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


class CustomModel_v2_Scaler(nn.Module):
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
        self.ch_num = 3*len(cqt_params)
        self.model = timm.create_model(encoder, pretrained=pretrained, in_chans=self.ch_num, num_classes=1)
        self.h, self.w = img_h, img_w
        self.min = None
        self.max = None
        self.mean = None
        
        self.first_ep = False
    
    def update_stat(self, x):
        spec_maxes = x.amax(dim=(0, 2, 3))
        spec_mines = x.amin(dim=(0, 2, 3))

        if self.min is not None:
            dif_mask_max = spec_maxes > self.max
            dif_mask_min = spec_mines < self.min
            if dif_mask_max.sum() > 0:
                self.max[dif_mask_max] = spec_maxes[dif_mask_max]
                # print("maxes:", self.max)
            if dif_mask_min.sum() > 0:
                self.min[dif_mask_min] = spec_mines[dif_mask_min]
                # print("mines:", self.min)
        else:
            self.max = spec_maxes
            self.min = spec_mines
    
    # def update_mean_std(self, x):
    #     if self.mean is not None:
    #          self.mean = self.mean + x.mean(dim=(0, 2, 3))
            

    def forward(self, x):
        x = torch.cat([self.spec(cqt, x) for cqt in self.cqts], dim=1)
        if self.first_ep:
            self.update_stat(x)
        #scaling at [0, 1]
        x = (x - self.min.reshape(1, self.ch_num, 1, 1)) / (self.max - self.min).reshape(1, self.ch_num, 1, 1)

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
    
    
    
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)

        return x


class CustomModelWavegram(nn.Module):
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
        self.model = timm.create_model(encoder, pretrained=pretrained, in_chans=3 * len(cqt_params)+1, num_classes=1)
        self.h, self.w = img_h, img_w
        self.scaler = None

        self.pre_conv0 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)

    def forward(self, x):
        bs, ch, sig_len = x.shape
        x0 = F.relu_(self.pre_bn0(self.pre_conv0(x)))
        x0 = self.pre_block1(x0, pool_size=4)
        x0 = self.pre_block2(x0, pool_size=4)
        x0 = self.pre_block3(x0, pool_size=4)
        x0 = x0[:, None, :]
        x0 = nn.functional.interpolate(x0, (self.h, self.w))

        x = torch.cat([self.spec(cqt, x) for cqt in self.cqts], dim=1)
        x = x.view(bs, 3*len(self.cqts), self.h, self.w)
        x = torch.cat((x, x0), dim=1)
        output = self.model(x)
        return output

    def spec(self, cqt, x):
        bs, ch, sig_len = x.shape
        x = x.view(-1, sig_len).unsqueeze(1)
        x = cqt(x).unsqueeze(1)
        x = nn.functional.interpolate(x, (self.h, self.w))
        return x
