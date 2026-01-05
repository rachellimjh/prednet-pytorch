import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming these are available in your directory
from prednet import PredNet
from mnist_dataset import MovingMNISTDataset # Import the new class
from mnist_settings import * 

# --------------------
# Loss Function
# --------------------
def prednet_loss(errors, layer_weights, time_weights):
    batch_size, nt, nb_layers = errors.shape
    weighted_layer_errors = torch.sum(errors * layer_weights.view(1, 1, -1), dim=2)
    weighted_time_errors = torch.sum(weighted_layer_errors * time_weights.view(1, -1), dim=1)
    loss = torch.mean(weighted_time_errors)
    return loss

# --------------------
# Main Execution Block
# --------------------
if __name__ == '__main__':
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

    # --- Hyperparameters ---
    nt = 10               # Sequence length to train on (Data has 20, we use 10)
    batch_size = 16       # Can often use larger batches with MNIST than KITTI
    nb_epoch = 100
    lr = 0.001
    decay_epoch = 50
    num_workers = 2
    
    # --- Model Parameters (CRITICAL FOR MNIST) ---
    # Input channels is 1 (Greyscale), not 3 (RGB)
    # (Input_Channels, Layer1, Layer2, Layer3)
    stack_sizes = (1, 48, 96, 192) 
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)

    # --- Load Data File ---
    print(f"Loading data from {DATA_DIR}...")
    # This expects the file to be a .npy containing the (20, 10000, 64, 64) array
    # If your data comes from a script, run that function here to get the numpy array.
    try:
        full_data = np.load(DATA_DIR)
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_DIR}")
        pass

    # --- Initialize Datasets ---
    print("Initializing Datasets...")
    train_dataset = MovingMNISTDataset(full_data, nt=nt, is_train=True)
    val_dataset   = MovingMNISTDataset(full_data, nt=nt, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    # --- Initialize Model ---
    # We define pixel_max=1.0 because we normalized data to [0,1]
    model = PredNet(stack_sizes, R_stack_sizes,
                    A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                    output_mode='error', pixel_max=1.0) 
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Weighting for Loss ---
    # Weight errors at different layers (usually layer 0 - pixel error - is highest)
    layer_loss_weights = torch.FloatTensor([1.0, 0.0, 0.0, 0.0]).to(device)
    
    # Time weighting: usually 0 weight for the first timestep (cannot predict t0)
    time_loss_weights = torch.ones(nt).to(device)
    time_loss_weights[0] = 0
    time_loss_weights /= (nt - 1)

    # --- Training Loop ---
    best_val_loss = float('inf')
    
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

        for i, inputs in enumerate(train_loader):
            inputs = inputs.to(device) # Shape: (Batch, Time, 1, 64, 64)
            
            optimizer.zero_grad()
            
            # Forward pass (returns errors)
            errors = model(inputs) 
            
            # Calculate Loss
            loss = prednet_loss(errors, layer_loss_weights, time_loss_weights)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i+1} | Loss: {loss.item():.6f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Validation ---
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
        
        print(f"Epoch {epoch+1} Done | {elapsed:.1f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'prednet_mnist_best.pth'))
            print("Saved Best Model!")

    print("Training Complete.")