import os
import numpy as np
from generator import generate_sequence
from visualize import visualize_sequence, save_gif
from config import *
from load_mnist import load_mnist_by_digit  # you need a function to load digits into dict

# -------------------------
# PARAMETERS
# -------------------------
IMG_SIZE = 64       # frame size
DIGIT_SIZE = 14     # size of MNIST digits
SEQ_LEN = 20        # sequence length
SEQUENCES_PER_CONFIG = 20  # how many sequences per config (number of test data)

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
}



# -------------------------
# GENERATE ALL DATA
# -------------------------
mnist_by_digit = load_mnist_by_digit()
for name, config in ALL_CONFIGS.items():
    print(f"Generating sequences for {name} ...")
    sequences = []
    for i in range(SEQUENCES_PER_CONFIG):
        seq = generate_sequence(mnist_by_digit, config)
        sequences.append(seq)

        # Save a GIF for the first sequence
        if i == 0:
            gif_path = os.path.join(GIF_DIR, f"{name}.gif")
            save_gif(seq, gif_path)

    sequences = np.stack(sequences)
    save_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    np.savez_compressed(save_path, sequences=sequences)

print("All datasets generated successfully!")
