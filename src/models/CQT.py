import numpy as np
import timm
import torch
import torch.nn as nn
from nnAudio.Spectrogram import CQT1992v2
import torch.nn.functional as F


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

    
class GeM(nn.Module):
    '''
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    '''

    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
               ', ' + 'eps=' + str(self.eps) + ')'

    
class CNN1d(nn.Module):
    """1D convolutional neural network. Classifier of the gravitational waves.
    Architecture from there https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.141103
    """

    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=32),
            GeM(kernel_size=8),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=32),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16),
            GeM(kernel_size=6),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=16),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=16),
            GeM(kernel_size=4),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 11, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.SiLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64+20, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.SiLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.rnn = RNN()
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x_rnn = self.rnn(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = torch.cat([x, x_rnn], dim=1)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class RNN(nn.Module):
    def __init__(self):
         super(RNN, self).__init__()

         self.rnn0 = nn.LSTM(input_size=500, hidden_size=10, batch_first=True, bidirectional=True)
         self.fc = nn.Sequential(
            nn.Linear(64 * 20, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.25),
            nn.SiLU(),
        )

    def forward(self, x):
         r_out, (h_n, h_c) = self.rnn0(x, None)
         r_out = torch.reshape(r_out, (-1, 64 * 20))
         out = self.fc(r_out)
         return out
