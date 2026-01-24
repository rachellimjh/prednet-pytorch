import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from prednet import PredNet
from .mnist_dataset import MovingMNISTDataset
from .mnist_settings import *

# --------------------
# Loss Function
# --------------------
def prednet_loss(errors, layer_weights, time_weights):
    batch_size, nt, nb_layers = errors.shape
    weighted_layer_errors = torch.sum(
        errors * layer_weights.view(1, 1, -1), dim=2
    )
    weighted_time_errors = torch.sum(
        weighted_layer_errors * time_weights.view(1, -1), dim=1
    )
    return torch.mean(weighted_time_errors)


# --------------------
# Main
# --------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ---- Training params ----
    nt = 20
    batch_size = 8
    nb_epoch = 150
    lr = 0.001
    decay_epoch = 75
    num_workers = 2
    print_interval = 100

    # ---- Steps per epoch ----
    SAMPLES_PER_EPOCH = 500
    MAX_BATCHES = SAMPLES_PER_EPOCH // batch_size

    # ---- Model params ----
    stack_sizes = (1, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)

    # ---- Loss weights (all layers + normalized time) ----
    # layer_loss_weights = torch.FloatTensor([1.0, 0.0, 0.0, 0.0]).to(device)
    layer_loss_weights = torch.FloatTensor([1.0, 0.1, 0.1, 0.1]).to(device)
    time_loss_weights = torch.ones(nt).to(device)
    time_loss_weights[0] = 0
    time_loss_weights /= time_loss_weights.sum()  # normalize

    # ---- Model ----
    model = PredNet(
        stack_sizes,
        R_stack_sizes,
        A_filt_sizes,
        Ahat_filt_sizes,
        R_filt_sizes,
        output_mode="error",
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---- Dataset ----
    train_dataset = MovingMNISTDataset(
        data_dir=DATA_DIR,
        nt=nt,
        split="train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataset = MovingMNISTDataset(
        data_dir=DATA_DIR,
        nt=nt,
        split="val",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,  # <-- include all sequences
        num_workers=num_workers,
        pin_memory=True,
    )

    best_val_loss = float("inf")

    print("Starting Moving MNIST training...")

    for epoch in range(nb_epoch):
        start_time = time.time()
        model.train()
        epoch_loss = 0

        # ---- Learning rate decay ----
        if epoch == decay_epoch:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.0001
            print("Learning rate dropped to 1e-4")

        print(f"--- Epoch {epoch+1}/{nb_epoch} ---")

        # ---- Training Loop with limited steps ----
        for i, inputs in enumerate(train_loader):
            if i >= MAX_BATCHES:
                break  # stop after steps per epoch

            inputs = inputs.to(device)

            optimizer.zero_grad()
            errors = model(inputs)
            loss = prednet_loss(errors, layer_loss_weights, time_loss_weights)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % print_interval == 0:
                print(f"Batch {i+1} | Loss {loss.item():.6f}")

        avg_train_loss = epoch_loss / MAX_BATCHES

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0
        val_batches = 0

        with torch.no_grad():
            sample_input = next(iter(val_loader)).to(device)
            
            # Get predictions (you'll need to set output_mode='prediction')
            model_temp = PredNet(
                stack_sizes, R_stack_sizes, A_filt_sizes, 
                Ahat_filt_sizes, R_filt_sizes,
                output_mode='prediction'
            ).to(device)
            model_temp.load_state_dict(model.state_dict())
            
            predictions = model_temp(sample_input)
            
            print(f"Input range: [{sample_input.min():.3f}, {sample_input.max():.3f}]")
            print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
            print(f"Prediction mean: {predictions.mean():.3f}")
            print(f"Prediction std: {predictions.std():.3f}")
            
            for inputs in val_loader:
                inputs = inputs.to(device)
                errors = model(inputs)
                loss = prednet_loss(errors, layer_loss_weights, time_loss_weights)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1} done in {elapsed:.1f}s | "
            f"Train {avg_train_loss:.6f} | Val {avg_val_loss:.6f}"
        )

        # ---- Save Best Model ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(WEIGHTS_DIR, "prednet_mmnist_best.pth"),
            )
            print("Saved best model")

        print("-" * 50)

    print("Training complete.")
