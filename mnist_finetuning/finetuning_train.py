# pretrain on KITTI -> finetune on Moving MNIST
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from prednet import PredNet
from mnist_training.mnist_dataset import MovingMNISTDataset
from mnist_training.mnist_settings import *

# --------------------
# Loss
# --------------------
def prednet_loss(errors, layer_weights, time_weights):
    weighted_layer = torch.sum(
        errors * layer_weights.view(1, 1, -1), dim=2
    )
    weighted_time = torch.sum(
        weighted_layer * time_weights.view(1, -1), dim=1
    )
    return torch.mean(weighted_time)

# --------------------
# Setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# --------------------
# Training params
# --------------------
nt = 20
batch_size = 8
nb_epoch = 80
lr = 1e-4
num_workers = 2

# --------------------
# Model params (MATCH KITTI)
# --------------------
stack_sizes = (3, 48, 96, 192)   # MUST be 3-channel
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)

# --------------------
# Loss weights
# --------------------
layer_loss_weights = torch.FloatTensor([1.0, 0.1, 0.1, 0.1]).to(device)

time_loss_weights = torch.ones(nt).to(device)
time_loss_weights[0] = 0
time_loss_weights /= time_loss_weights.sum()

# --------------------
# Load model (error mode)
# --------------------
model = PredNet(
    stack_sizes,
    R_stack_sizes,
    A_filt_sizes,
    Ahat_filt_sizes,
    R_filt_sizes,
    output_mode="error",
).to(device)

# ---- Load KITTI pretrained weights ----
kitti_weights = os.path.join(KITTI_WEIGHTS, "prednet_kitti_best.pth")
assert os.path.exists(kitti_weights), "KITTI weights not found"

model.load_state_dict(torch.load(kitti_weights, map_location=device))
print("Loaded pretrained KITTI weights")

# --------------------
# Freeze lower layers (VERY IMPORTANT)
# --------------------
print("\n--- Freezing layers ---")
for name, param in model.named_parameters():
    if name.startswith("conv_layers.ahat"):
        param.requires_grad = True
        print(f"Trainable: {name}")
    else:
        param.requires_grad = False

# --------------------
# Optimizer
# --------------------
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
)

# --------------------
# Dataset
# --------------------
train_dataset = MovingMNISTDataset(
    data_dir=DATA_DIR,
    nt=nt,
    split="train",
)

val_dataset = MovingMNISTDataset(
    data_dir=DATA_DIR,
    nt=nt,
    split="val",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

# --------------------
# Training loop
# --------------------
best_val = float("inf")
print("Starting finetuning on Moving MNIST...")

for epoch in range(nb_epoch):
    start = time.time()
    model.train()
    train_loss = 0

    for inputs in train_loader:
        inputs = inputs.to(device)

        # ---- Convert MNIST (1ch) → 3ch ----
        inputs = inputs.repeat(1, 1, 3, 1, 1)

        optimizer.zero_grad()
        errors = model(inputs)
        loss = prednet_loss(errors, layer_loss_weights, time_loss_weights)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            inputs = inputs.repeat(1, 1, 3, 1, 1)

            errors = model(inputs)
            loss = prednet_loss(errors, layer_loss_weights, time_loss_weights)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    elapsed = time.time() - start

    print(
        f"Epoch {epoch+1}/{nb_epoch} | "
        f"Train {train_loss:.6f} | "
        f"Val {val_loss:.6f} | "
        f"{elapsed:.1f}s"
    )

    # ---- Save best ----
    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            model.state_dict(),
            os.path.join(WEIGHTS_DIR, "prednet_kitti_to_mmnist_best.pth"),
        )
        print("Saved best finetuned model")

print("Finetuning complete.")
