import torch
import os
from pathlib import Path

class ModelCheckpoint:
    def __init__(self, dirname, n_saved, filename_prefix="cp"):
        self.dirname = Path(dirname)
        self.n_saved = n_saved
        self.filename_prefix = filename_prefix
        self.dirname.mkdir(parents=True, exist_ok=True)

    def __remove_old(self):
        existing_cps = list(self.dirname.glob("*.pth"))
        existing_cps = sorted(existing_cps, reverse=True, key=lambda x: x.stem.split("score")[-1])
        for p in existing_cps[self.n_saved:]:
            os.remove(p)
    
    def __call__(self, epoch, score, model):
        self.__remove_old()
        checkpoint = model.state_dict()
        filename = self.dirname / f"{self.filename_prefix}_epoch{epoch:02d}_score{score:.5f}.pth"
        torch.save(checkpoint, filename)