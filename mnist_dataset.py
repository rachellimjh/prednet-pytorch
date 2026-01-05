import torch
from torch.utils.data import Dataset
import numpy as np

class MovingMNIST(Dataset):
    def __init__(self, data_array, nt, is_train=True, train_split=0.9, is_custom=False):
        """
        Args:
            data_array: The loaded numpy array.
            nt: Number of time steps (sequence length).
            is_train: Boolean, whether to load training or validation split.
            train_split: Percentage of data to use for training (default 0.9).
            is_custom: Set True if using the generated datasets (Goal 1-5). 
                       Set False if using the downloaded 'mnist_test_seq.npy'.
        """
        super(MovingMNIST, self).__init__()
        self.nt = nt

        # --- FIX 1: SHAPE STANDARDIZATION ---
        # Goal: Everyone must become (N_samples, Time, Height, Width)
        
        if not is_custom:
            # The Toronto file is (Time, N, H, W) -> e.g. (20, 10000, 64, 64)
            # We must swap Time and N to match standard PyTorch format
            data_array = data_array.transpose(1, 0, 2, 3) 
            
        # Now both are (N, T, H, W) -> (10000, 20, 64, 64)

        # --- FIX 2: NORMALIZATION CHECK ---
        # The Toronto file is 0-255 (integers).
        # The Custom Generator is 0.0-1.0 (floats).
        # We normalize ONLY if the data looks like integers (max > 1).
        self.needs_norm = (data_array.max() > 1.0)
        
        # --- SPLIT LOGIC ---
        total_samples = data_array.shape[0] # N is now at index 0
        split_idx = int(total_samples * train_split)

        if is_train:
            self.data = data_array[:split_idx]
        else:
            self.data = data_array[split_idx:]

        self.N, self.T, self.H, self.W = self.data.shape
        print(f"Dataset Loaded. Mode: {'Train' if is_train else 'Val'}. Shape: {self.data.shape}")
        print(f"Source: {'Custom Generator' if is_custom else 'Toronto URL'}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Select sample: (Time, Height, Width)
        seq = self.data[idx] 
        
        # Add Channel Dimension: (Time, 1, Height, Width)
        seq = seq[:, np.newaxis, :, :]
        
        # Handle Sequence Length
        if self.nt <= self.T:
            seq = seq[:self.nt]
            
        # Convert to Float32
        seq = seq.astype(np.float32)

        # Apply Normalization only if needed
        if self.needs_norm:
            seq = seq / 255.0
            
        return torch.from_numpy(seq)