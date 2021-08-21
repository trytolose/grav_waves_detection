import torch.nn as nn
import timm 
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
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, in_chans=3
        )
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, 1)
        self.cqt = CQT1992v2(
            **cfg.qtransform_params
        )  

    def forward(self, x):
        bs, ch, sig_len = x.shape
        x = x.view(-1, sig_len)
        x = self.cqt(x).unsqueeze(1)
        x = nn.functional.interpolate(x, (256, 256))
        _, _, h, w = x.shape
        x = x.view(bs, 3, h, w)
        output = self.model(x)
        return output