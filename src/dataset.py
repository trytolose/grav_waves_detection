import numpy as np
from torch.utils.data import  Dataset

class TrainDataset(Dataset):
    def __init__(self, df, transform=None, steps_per_epoch=150, mode="train", bs=64):
        self.df = df
        self.file_names = df["path"].values
        self.labels = df["target"].values
        self.transform = transform
        self.steps_per_epoch = steps_per_epoch * bs
        self.mode = mode

    def __len__(self):
        # if self.mode == "train":
        #     return self.steps_per_epoch
        # else:
        return len(self.df)

    def __getitem__(self, idx):
        # if self.mode == "train":
            # rand_id = np.random.randint(len(self.df), size=1)[0]
            # file_path = self.file_names[rand_id]
            # label = float(self.labels[rand_id])
        # else:
        file_path = self.file_names[idx]
        
        waves = np.load(file_path)
        if self.transform is not None:
            waves = self.transform(waves)
        if self.mode=='train':
            label = float(self.labels[idx])
            return waves, label
        if self.mode=='test':
            return waves
