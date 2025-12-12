import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Remove tqdm import
# from tqdm import tqdm 

from prednet import PredNet
from data_utils import KittiDataset
from kitti_settings import *

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Ensure WEIGHTS_DIR exists ---
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)


# Parameters
nt = 10
batch_size = 4
nb_epoch = 150
lr = 0.001
decay_epoch = 75
num_workers = 2
print_interval = 100  # Print progress every 100 batches

# Data Files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Model Parameters
stack_sizes = (3, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)

# Weighting for Loss
layer_loss_weights = torch.FloatTensor([1.0, 0.0, 0.0, 0.0]).to(device)
time_loss_weights = torch.ones(nt).to(device)
time_loss_weights[0] = 0
time_loss_weights /= (nt - 1)

# --- Initialize Model ---
model = PredNet(stack_sizes, R_stack_sizes,
                A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                output_mode='error')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Load previous model weights if they exist ---
checkpoint_path = os.path.join(WEIGHTS_DIR, 'prednet_kitti_best.pth')

if os.path.exists(checkpoint_path):
    print(f"Loading previous weights from {checkpoint_path} ...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Weights loaded. Training will resume from the previous checkpoint.")
else:
    print("No previous checkpoint found. Training from scratch.")

# --- Data Loaders ---
print("Loading Training Data...")
train_dataset = KittiDataset(train_file, train_sources, nt, sequence_start_mode='all')
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True
)

print("Loading Validation Data...")
val_dataset = KittiDataset(val_file, val_sources, nt, sequence_start_mode='unique', N_seq=100)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    drop_last=True, 
    num_workers=num_workers, 
    pin_memory=True
)

# --- Custom Loss Function ---
def prednet_loss(errors, layer_weights, time_weights):
    batch_size, nt, nb_layers = errors.shape
    weighted_layer_errors = torch.sum(errors * layer_weights.view(1, 1, -1), dim=2)
    weighted_time_errors = torch.sum(weighted_layer_errors * time_weights.view(1, -1), dim=1)
    loss = torch.mean(weighted_time_errors)
    return loss

# --- Training Loop ---
best_val_loss = float('inf')
num_train_batches = len(train_loader)

print(f"Starting training for {nb_epoch} epochs...")

for epoch in range(nb_epoch):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    
    # LR Scheduler
    if epoch == decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        print("Learning rate dropped to 0.0001")

    print(f"--- Epoch {epoch+1}/{nb_epoch} ---")
    
    # Enumerate to get batch index (i)
    for i, inputs in enumerate(train_loader):
        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        errors = model(inputs)
        
        # Calculate Loss
        loss = prednet_loss(errors, layer_loss_weights, time_loss_weights)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Manual Print Logging
        if (i + 1) % print_interval == 0:
            print(f"Epoch {epoch+1} | Batch {i+1}/{num_train_batches} | Loss: {loss.item():.6f}")
        
    avg_train_loss = epoch_loss / len(train_loader)
    
    # --- Validation ---
    print("Running Validation...")
    model.eval()
    val_loss_sum = 0
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            errors = model(inputs)
            loss = prednet_loss(errors, layer_loss_weights, time_loss_weights)
            val_loss_sum += loss.item()
            
    avg_val_loss = val_loss_sum / len(val_loader)
    elapsed = time.time() - start_time
    
    print(f"Epoch {epoch+1} Completed in {elapsed:.1f}s")
    print(f"Result: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'prednet_kitti_best.pth'))
        print("Saved Best Model!")
    
    print("-" * 50)

print("Training Complete.")