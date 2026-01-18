import os
import numpy as np
from generator import generate_sequence
from visualize import visualize_sequence, save_gif
from config import *
from load_mnist import load_mnist_digits  # you need a function to load digits into dict

# =====================
# Parameters
# =====================
OUTPUT_DIR = "data"
GIF_DIR = os.path.join(OUTPUT_DIR, "gifs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

SEQUENCES_PER_CONFIG = 5  # how many sequences to generate for each config

# =====================
# Load MNIST digits
# =====================
# mnist_by_digit: dict[int] -> list of digit images
mnist_by_digit = load_mnist_digits()  # implement this as needed

# =====================
# All configs
# =====================
ALL_CONFIGS = {
    "TRAIN_ID": TRAIN_ID,
    "TRAIN_ID_VERTICAL": TRAIN_ID_VERTICAL,
    "ID_APPEAR": ID_APPEAR,
    "ID_DISAPPEAR": ID_DISAPPEAR,
    "ID_STICK": ID_STICK,
    "ID_APPEAR_VERTICAL": ID_APPEAR_VERTICAL,
    "ID_DISAPPEAR_VERTICAL": ID_DISAPPEAR_VERTICAL,
    "ID_STICK_VERTICAL": ID_STICK_VERTICAL,
    "OOD_APPEAR": OOD_APPEAR,
    "OOD_DISAPPEAR": OOD_DISAPPEAR,
    "OOD_STICK": OOD_STICK,
    "ID_NORMAL": ID_NORMAL,
    "OOD_NORMAL": OOD_NORMAL
}

# =====================
# Generate all sequences
# =====================
for name, config in ALL_CONFIGS.items():
    print(f"Generating sequences for {name} ...")
    sequences = []

    for i in range(SEQUENCES_PER_CONFIG):
        seq = generate_sequence(mnist_by_digit, config)
        sequences.append(seq)

        # Save a GIF for the first sequence as sanity check
        if i == 0:
            gif_path = os.path.join(GIF_DIR, f"{name}.gif")
            save_gif(seq, gif_path)

    sequences = np.stack(sequences)
    # Save all sequences to npz
    save_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    np.savez_compressed(save_path, sequences=sequences)

print("All datasets generated and GIFs saved successfully!")
