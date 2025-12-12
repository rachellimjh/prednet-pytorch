import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import hickle as hkl

class KittiDataset(Dataset):
    def __init__(self, data_file, source_file, nt, sequence_start_mode='all', N_seq=None, output_mode='error'):
        self.nt = nt
        self.output_mode = output_mode
        
        # Load Data
        # Try Loading with h5py first (faster), fallback to hickle
        try:
            with h5py.File(data_file, 'r') as f:
                self.X = f['data_0'][:]
        except:
             self.X = hkl.load(data_file)
             
        try:
            with h5py.File(source_file, 'r') as f:
                self.sources = f['data_0'][:]
        except:
            self.sources = hkl.load(source_file)

        # X is usually (N, H, W, C) in Keras files. PyTorch needs (N, C, H, W)
        # We will permute in __getitem__ to save memory here
        
        self.im_shape = self.X[0].shape

        # Calculate possible start indices
        if sequence_start_mode == 'all':
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) 
                                             if self.sources[i] == self.sources[i + self.nt - 1]])
        elif sequence_start_mode == 'unique':
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if N_seq is not None and len(self.possible_starts) > N_seq:
            self.possible_starts = self.possible_starts[:N_seq]

    def __len__(self):
        return len(self.possible_starts)

    def __getitem__(self, idx):
        start_idx = self.possible_starts[idx]
        end_idx = start_idx + self.nt
        
        # Get sequence: (Time, Height, Width, Channels)
        seq = self.X[start_idx:end_idx]
        
        # Preprocess: Normalize to [0, 1] and Float32
        seq = seq.astype(np.float32) / 255.0
        
        # Convert to PyTorch format: (Time, Channels, Height, Width)
        # Assuming input is (T, H, W, C), we need (T, C, H, W)
        seq = np.transpose(seq, (0, 3, 1, 2))
        
        # If output_mode is 'prediction', the target is the sequence itself (shifted later)
        # If output_mode is 'error', the target is a dummy tensor (loss is calculated internally)
        return torch.from_numpy(seq)