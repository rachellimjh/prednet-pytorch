import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import hickle as hkl
import h5py
from models.prednet import PredNet  # your PyTorch version
import time  # For tracking training time

# ----------------------------
# Dataset (Hickle loader)
# ----------------------------
class KITTIDataset(Dataset):
    def __init__(self, X_file, sources_file, nt=10):
        print("Loading dataset...")
        # LOAD X USING H5PY
        with h5py.File(X_file, "r") as f:
            key = list(f.keys())[0]
            self.X = f[key][:]  # shape (num_frames, H, W, C)

        with h5py.File(sources_file, "r") as f:
            key = list(f.keys())[0]
            self.sources = f[key][:]  # shape (num_frames,)

        self.nt = nt

        # Compute all valid starting positions
        self.starts = []
        for i in range(len(self.X) - nt):
            if self.sources[i] == self.sources[i + nt - 1]:
                self.starts.append(i)

        print(f"Dataset loaded with {len(self.starts)} valid sequences.")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        seq = self.X[start:start+self.nt]  # (nt, H, W, C)

        seq = torch.tensor(seq, dtype=torch.float32) / 255.0
        seq = seq.permute(0, 3, 1, 2)  # (nt, C, H, W)

        # For PredNet: y = x when output_mode="prediction"
        return seq, seq


# ----------------------------
# Training loop
# ----------------------------
def train_epoch(model, loader, optimizer, device, epoch_num):
    model.train()
    total_loss = 0
    criterion = nn.L1Loss()
    num_batches = len(loader)
    
    print(f"Training epoch {epoch_num + 1} - Number of batches: {num_batches}")

    start_time = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        errors = model(x)              # (B, nt, layers)
        errors = errors.mean(dim=2)    # mean over layers → (B, nt)
        errors = errors.mean(dim=1)    # mean over time → (B)

        loss = criterion(errors, torch.zeros_like(errors))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} - Loss: {loss.item():.5f}")

    epoch_duration = time.time() - start_time
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch_num + 1} - Average Loss: {avg_loss:.5f}")
    print(f"Epoch {epoch_num + 1} took {epoch_duration:.2f} seconds.")

    return avg_loss

# ----------------------------
# Main
# ----------------------------
def main():
    DATA_DIR = "kitti_data"

    train_X = os.path.join(DATA_DIR, "X_train.hkl")
    train_sources = os.path.join(DATA_DIR, "sources_train.hkl")

    val_X = os.path.join(DATA_DIR, "X_val.hkl")
    val_sources = os.path.join(DATA_DIR, "sources_val.hkl")

    nt = 10
    batch_size = 2
    epochs = 50

    print("Loading datasets...")
    train_data = KITTIDataset(train_X, train_sources, nt=nt)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    print("Building model...")
    model = PredNet(
        stack_sizes=(3, 48, 96, 192),
        R_stack_sizes=(3, 48, 96, 192),
        A_filt_sizes=(3, 3, 3),
        Ahat_filt_sizes=(3, 3, 3, 3),
        R_filt_sizes=(3, 3, 3, 3),
        output_mode="error",
        return_sequences=True
    )

    # Use CPU as we don't have GPU
    device = torch.device("cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...\n")
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.5f}\n")

    torch.save(model.state_dict(), "prednet_kitti.pth")
    print("\nModel saved to prednet_kitti.pth")

if __name__ == "__main__":
    main()
