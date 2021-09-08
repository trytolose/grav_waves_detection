from src.models.wavenet import Andrewnet_v3_true, Andrewnet_v4, Andrewnet_v5
from src.models.CQT import CustomModel_v0, CustomModel_v1
from src.models.CWT import CustomModel_CWT
import timm
from pydoc import locate


def get_model(cfg):
    if cfg.MODEL.NAME == "CustomModel_CWT":
        return CustomModel_CWT(**cfg.MODEL.CFG)
    if cfg.MODEL.NAME == "CustomModel_v1":
        model = CustomModel_v1(**cfg.MODEL.CFG)
        return model

    if cfg.MODEL.NAME == "CustomModel_v0":
        return CustomModel_v0(cfg)
    if cfg.MODEL.NAME == "Wavenet":
        return Andrewnet_v3_true(**cfg.MODEL.CFG)
    if cfg.MODEL.NAME == "Wavenet_v2":
        return Andrewnet_v4(**cfg.MODEL.CFG)
    if cfg.MODEL.NAME == "Wavenet_v3":
        return Andrewnet_v5(**cfg.MODEL.CFG)
    if cfg.MODEL.NAME == "timm":
        arch = cfg.MODEL.CFG.encoder
        return timm.create_model(arch, pretrained=True, in_chans=3, num_classes=1)
