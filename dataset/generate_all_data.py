import os
import numpy as np
from generator import generate_sequence
from visualize import save_gif
from config import *
from load_mnist import load_mnist_by_digit

# -------------------------
# PARAMETERS
# -------------------------
IMG_SIZE = 64       # frame size
DIGIT_SIZE = 14     # size of MNIST digits
SEQ_LEN = 20        # sequence length

OUTPUT_DIR = "data/custom"
GIF_DIR = os.path.join(OUTPUT_DIR, "gifs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

# -------------------------
# LOAD CONFIGS
# -------------------------
ALL_CONFIGS = {
    # =====================
    # TRAINING
    # =====================
    "TRAIN_ID": TRAIN_ID,
    "TRAIN_ID_VERTICAL": TRAIN_ID_VERTICAL,

    # =====================
    # ID ANOMALIES (0–4)
    # =====================
    "ID_APPEAR": ID_APPEAR,
    "ID_DISAPPEAR": ID_DISAPPEAR,
    "ID_STICK": ID_STICK,

    "ID_APPEAR_VERTICAL": ID_APPEAR_VERTICAL,
    "ID_DISAPPEAR_VERTICAL": ID_DISAPPEAR_VERTICAL,
    "ID_STICK_VERTICAL": ID_STICK_VERTICAL,

    # =====================
    # OOD ANOMALIES (5–9)
    # =====================
    "OOD_APPEAR": OOD_APPEAR,
    "OOD_DISAPPEAR": OOD_DISAPPEAR,
    "OOD_STICK": OOD_STICK,

    "OOD_APPEAR_HORIZONTAL": OOD_APPEAR_HORIZONTAL,
    "OOD_DISAPPEAR_HORIZONTAL": OOD_DISAPPEAR_HORIZONTAL,
    "OOD_STICK_HORIZONTAL": OOD_STICK_HORIZONTAL,

    # =====================
    # NO ANOMALY (TEST)
    # =====================
    "ID_NORMAL": ID_NORMAL,
    "ID_NORMAL_VERTICAL": ID_NORMAL_VERTICAL,
    "OOD_NORMAL": OOD_NORMAL,
    "OOD_NORMAL_HORIZONTAL": OOD_NORMAL_HORIZONTAL,
}

# -------------------------
# SEQUENCE COUNTS
# -------------------------
SEQUENCES_COUNTS = {}
for name in ALL_CONFIGS.keys():
    if name in ["TRAIN_ID", "TRAIN_ID_VERTICAL"]:
        SEQUENCES_COUNTS[name] = 10000  # Training
    else:
        SEQUENCES_COUNTS[name] = 1000   # Testing / anomalies

# -------------------------
# GENERATE DATA
# -------------------------
mnist_by_digit = load_mnist_by_digit()

for name, config in ALL_CONFIGS.items():
    n_sequences = SEQUENCES_COUNTS[name]
    print(f"Generating {n_sequences} sequences for {name} ...")
    
    sequences = []
    for i in range(n_sequences):
        seq = generate_sequence(mnist_by_digit, config)
        sequences.append(seq)

        # Save a GIF for the first 3 sequences
        if i < 3:
            gif_path = os.path.join(GIF_DIR, f"{name}_{i+1}.gif")
            save_gif(seq, gif_path)

        # Optional: Print progress every 1000 sequences
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1}/{n_sequences}")

    sequences = np.stack(sequences)
    save_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    np.savez_compressed(save_path, sequences=sequences)

print("All datasets generated successfully!")
