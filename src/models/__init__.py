from src.models.wavenet import Andrewnet_v3_true
from src.models.CQT import CustomModel_v0 
from src.models.CQT import CustomModel_v1
from src.models.CQT import CustomModel_v2
from src.models.CQT import CustomModel_v3
from src.models.CWT import CustomModel_CWT

def get_model(cfg):
    if cfg.MODEL.NAME == "CustomModel_CWT":
        return CustomModel_CWT(cfg)
    if cfg.MODEL.NAME == "CustomModel_v3":
        return CustomModel_v3(**cfg.MODEL.CFG)
    if cfg.MODEL.NAME == "CustomModel_v2":
        return CustomModel_v2(**cfg.MODEL.CFG)
    if cfg.MODEL.NAME == "CustomModel_v1":
        return CustomModel_v1(**cfg.MODEL.CFG)
    if cfg.MODEL.NAME == "CustomModel_v0":
        return CustomModel_v0(cfg)
    if cfg.MODEL.NAME == "Wavenet":
        return Andrewnet_v3_true(**cfg.MODEL.CFG)
