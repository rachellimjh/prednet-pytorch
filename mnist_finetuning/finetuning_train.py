import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from prednet import PredNet
from mnist_training.mnist_dataset import MovingMNISTDataset
from mnist_training.mnist_settings import *

# --------------------
# Loss Function
# --------------------
def prednet_loss(errors, layer_weights, time_weights):
    weighted_layer = torch.sum(errors * layer_weights.view(1, 1, -1), dim=2)
    weighted_time = torch.sum(weighted_layer * time_weights.view(1, -1), dim=1)
    return torch.mean(weighted_time)

# --------------------
# Setup
# --------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # --------------------
    # Configuration
    # --------------------
    nt = 20
    batch_size = 8
    nb_epoch = 80
    lr = 1e-4
    num_workers = 2

    # Filenames for saving/loading
    kitti_weights_path = os.path.join(KITTI_WEIGHTS, "prednet_kitti_best.pth")
    checkpoint_path = os.path.join(WEIGHTS_DIR, "checkpoint_finetune.pth") # Saves progress every epoch
    best_model_path = os.path.join(WEIGHTS_DIR, "prednet_kitti_to_mmnist_best.pth") # Saves only best validation

    # --------------------
    # Model Init
    # --------------------
    stack_sizes = (3, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)

    layer_loss_weights = torch.FloatTensor([1.0, 0.1, 0.1, 0.1]).to(device)
    time_loss_weights = torch.ones(nt).to(device)
    time_loss_weights[0] = 0
    time_loss_weights /= time_loss_weights.sum()

    model = PredNet(
        stack_sizes,
        R_stack_sizes,
        A_filt_sizes,
        Ahat_filt_sizes,
        R_filt_sizes,
        output_mode="error",
    ).to(device)

    # --------------------
    # Freeze Layers Logic
    # --------------------
    # We apply this first. If we resume, the optimizer will already know, 
    # but it's safe to set the flags correctly on the model structure anyway.
    print("\n--- Configuring Layer Freezing ---")
    for name, param in model.named_parameters():
        if name.startswith("conv_layers.ahat"):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # --------------------
    # Optimizer Init
    # --------------------
    # Initialize optimizer with currently trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # --------------------
    # RESUME LOGIC (The Fix)
    # --------------------
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(checkpoint_path):
        print(f"\n>> Found checkpoint at {checkpoint_path}. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 1. Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        # 2. Load optimizer (learning rate, momentum, etc.)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 3. Set epoch and best score
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        
        print(f">> Resuming from Epoch {start_epoch+1}")

    else:
        print("\n>> No checkpoint found. Starting fresh from KITTI weights.")
        if os.path.exists(kitti_weights_path):
            # Load KITTI weights into the model structure
            # strict=False is sometimes needed if slight version differences exist, 
            # but usually strict=True is better to catch errors.
            model.load_state_dict(torch.load(kitti_weights_path, map_location=device))
            print(">> Loaded pretrained KITTI weights.")
        else:
            print(f"!! WARNING: KITTI weights not found at {kitti_weights_path} !!")

    # --------------------
    # Dataset
    # --------------------
    train_dataset = MovingMNISTDataset(data_dir=DATA_DIR, nt=nt, split="train")
    val_dataset = MovingMNISTDataset(data_dir=DATA_DIR, nt=nt, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # --------------------
    # Training Loop
    # --------------------
    print(f"\nStarting training loop from Epoch {start_epoch+1} to {nb_epoch}...")

    for epoch in range(start_epoch, nb_epoch):
        start_time = time.time()
        model.train()
        train_loss = 0

        for i, inputs in enumerate(train_loader):
            inputs = inputs.to(device)

            # 1. Handle Channels (1 -> 3)
            inputs = inputs.repeat(1, 1, 3, 1, 1)

            # 2. [SAFETY] Normalize (Resume scripts often crash without this)
            if inputs.max() > 1.0:
                inputs = inputs.float() / 255.0

            optimizer.zero_grad()
            errors = model(inputs)
            loss = prednet_loss(errors, layer_loss_weights, time_loss_weights)
            loss.backward()

            # 3. [SAFETY] Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)
                inputs = inputs.repeat(1, 1, 3, 1, 1)
                if inputs.max() > 1.0: inputs = inputs.float() / 255.0

                errors = model(inputs)
                loss = prednet_loss(errors, layer_loss_weights, time_loss_weights)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{nb_epoch} | "
            f"Train {avg_train_loss:.6f} | "
            f"Val {avg_val_loss:.6f} | "
            f"{elapsed:.1f}s"
        )

        # --------------------
        # SAVE CHECKPOINT (Every Epoch)
        # --------------------
        # This allows you to resume exactly from here if the GPU crashes
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint_dict, checkpoint_path)

        # --------------------
        # SAVE BEST MODEL
        # --------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"   >>> Improved! Saved best model (Val: {best_val_loss:.6f})")

    print("Finetuning complete.")