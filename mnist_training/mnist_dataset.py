import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .mnist_settings import NPZ_FILE

class MovingMNISTDataset(Dataset):
    def __init__(self, data_dir, nt, split="train", train_ratio=0.9):
        self.nt = nt

        # ---- Load .npz correctly ----
        # npz_path = os.path.join(data_dir, "TRAIN_ID.npz")
        npz_path = os.path.join(data_dir, NPZ_FILE)
        # TRAIN_ID
        data = np.load(npz_path)["sequences"]  # (N, T, H, W)

        N = data.shape[0]
        split_idx = int(train_ratio * N)

        if split == "train":
            self.data = data[:split_idx]
        elif split == "val":
            self.data = data[split_idx:]
        elif split == "all":  # Add this option
            self.data = data
        else:
            raise ValueError("split must be 'train' or 'val'")
        
        print(f"{split.upper()} dataset loaded: {len(self.data)} sequences")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data[idx]  # (T, H, W)

        seq = seq.astype(np.float32) / 255.0
        seq = np.expand_dims(seq, axis=1)  # (T, 1, H, W)

        return torch.from_numpy(seq)
    # def __getitem__(self, idx):
    #     seq = self.data[idx]
    #     seq = seq.astype(np.float32) / 255.0
    #     seq = 1.0 - seq  # INVERT: black becomes white, white becomes black
    #     seq = np.expand_dims(seq, axis=1)
    #     return torch.from_numpy(seq)
