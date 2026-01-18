import os
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# --------------------
# Configuration
# --------------------
OUTPUT_DIR = "data/mnist_data/custom"
SEQ_LEN = 20
IMG_SIZE = 64
DIGIT_SIZE = 28
NUM_TRAIN = 10000  # Number of sequences for Training sets
NUM_TEST = 2000    # Number of sequences for Test sets
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Load MNIST
# --------------------
mnist = datasets.MNIST(
    root="data/mnist_data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Group MNIST by digit
digit_to_images = {i: [] for i in range(10)}
for img, label in mnist:
    digit_to_images[label].append(img.squeeze(0).numpy())

# --------------------
# Helper Functions
# --------------------
def sample_digit(digit_pool):
    d = np.random.choice(digit_pool)
    img = digit_to_images[d][np.random.randint(len(digit_to_images[d]))]
    return img, d

def generate_sequence(digit_pool, speed_range, motion_type="any", spawn_region="any", sudden_speed=False, curved_motion=False):
    canvas = np.zeros((SEQ_LEN, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    digit_img, _ = sample_digit(digit_pool)

    # Spawn
    if spawn_region == "center":
        x = np.random.uniform(IMG_SIZE//2 - 10, IMG_SIZE//2 + 10)
        y = np.random.uniform(IMG_SIZE//2 - 10, IMG_SIZE//2 + 10)
    elif spawn_region == "edge":
        if np.random.rand() > 0.5:
            x = np.random.randint(0, IMG_SIZE - DIGIT_SIZE)
            y = np.random.choice([0, IMG_SIZE - DIGIT_SIZE - 1])
        else:
            x = np.random.choice([0, IMG_SIZE - DIGIT_SIZE - 1])
            y = np.random.randint(0, IMG_SIZE - DIGIT_SIZE)
    else:
        x = np.random.randint(0, IMG_SIZE - DIGIT_SIZE)
        y = np.random.randint(0, IMG_SIZE - DIGIT_SIZE)

    speed = np.random.uniform(*speed_range)

    if motion_type == "straight":
        direction = np.random.choice(["h", "v"])
        dx, dy = (speed, 0) if direction == "h" else (0, speed)
    else:
        angle = np.random.uniform(0, 2 * np.pi)
        dx = speed * np.cos(angle)
        dy = speed * np.sin(angle)

    if sudden_speed:
        dx_fast, dy_fast = dx * 3, dy * 3
    if curved_motion:
        angle = np.arctan2(dy, dx)
        d_angle = np.random.uniform(-0.2, 0.2)

    x, y, dx, dy = float(x), float(y), float(dx), float(dy)

    for t in range(SEQ_LEN):
        if x <= 0 or x >= IMG_SIZE - DIGIT_SIZE: dx = -dx
        if y <= 0 or y >= IMG_SIZE - DIGIT_SIZE: dy = -dy
        x = np.clip(x, 0, IMG_SIZE - DIGIT_SIZE)
        y = np.clip(y, 0, IMG_SIZE - DIGIT_SIZE)

        xi, yi = int(x), int(y)
        canvas[t, 0, yi:yi + DIGIT_SIZE, xi:xi + DIGIT_SIZE] = digit_img

        if sudden_speed and t >= SEQ_LEN//2:
            x += dx_fast
            y += dy_fast
        elif curved_motion:
            angle += d_angle
            dx = speed * np.cos(angle)
            dy = speed * np.sin(angle)
            x += dx
            y += dy
        else:
            x += dx
            y += dy

    return canvas

def generate_dataset(name, n_sequences, digit_pool, speed_range, motion_type, spawn_region,
                     curved_motion=False, sudden_speed=False):
    sequences = []
    print(f"Generating {name} ({n_sequences} seqs)...")
    for _ in tqdm(range(n_sequences)):
        seq = generate_sequence(digit_pool, speed_range, motion_type, spawn_region, sudden_speed, curved_motion)
        sequences.append(seq)
    sequences = np.stack(sequences)
    save_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    np.savez_compressed(save_path, data=sequences)
    print(f"Saved to {save_path}")
    return sequences

# --------------------
# MAIN EXECUTION
# --------------------
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("--- Generating TRAINING Sets (ID) ---")
    generate_dataset("train_id_digit", NUM_TRAIN, list(range(9)), (1.0, 2.0), "any", "any")
    generate_dataset("train_id_trajectory", NUM_TRAIN, list(range(10)), (1.0, 2.0), "straight", "any")
    generate_dataset("train_id_spatial", NUM_TRAIN, list(range(10)), (1.0, 2.0), "any", "center")
    generate_dataset("train_id_speed", NUM_TRAIN, list(range(10)), (1.0, 1.2), "any", "any")

    print("\n--- Generating TEST Sets ---")
    test_sets = {}

    # --- ID test sets ---
    test_sets["id_digit"] = generate_dataset("test_id_digit", NUM_TEST, list(range(9)), (1.0, 2.0), "any", "any")
    test_sets["id_trajectory"] = generate_dataset("test_id_trajectory", NUM_TEST, list(range(10)), (1.0, 2.0), "straight", "any")
    test_sets["id_spatial"] = generate_dataset("test_id_spatial", NUM_TEST, list(range(10)), (1.0, 2.0), "any", "center")
    test_sets["id_speed"] = generate_dataset("test_id_speed", NUM_TEST, list(range(10)), (1.0, 1.2), "any", "any")

    # --- OOD test sets ---
    test_sets["ood_digit"] = generate_dataset("test_ood_digit", NUM_TEST, [9], (1.0, 2.0), "any", "any")
    test_sets["ood_trajectory"] = generate_dataset("test_ood_trajectory", NUM_TEST, list(range(10)), (1.0, 2.0), "any", "any", curved_motion=True)
    test_sets["ood_spatial"] = generate_dataset("test_ood_spatial", NUM_TEST, list(range(10)), (1.0, 2.0), "any", "edge")
    test_sets["ood_speed"] = generate_dataset("test_ood_speed", NUM_TEST, list(range(10)), (1.0, 1.2), "any", "any", sudden_speed=True)

    # --- Mixed test sets (single file per type) ---
    print("\n--- Generating MIXED test sets ---")
    for anomaly in ["digit", "trajectory", "spatial", "speed"]:
        id_data = test_sets[f"id_{anomaly}"]
        ood_data = test_sets[f"ood_{anomaly}"]
        mixed_data = np.concatenate([id_data[:NUM_TEST//2], ood_data[:NUM_TEST//2]], axis=0)
        np.random.shuffle(mixed_data)
        save_path = os.path.join(OUTPUT_DIR, f"test_mixed_{anomaly}.npz")
        np.savez_compressed(save_path, data=mixed_data)
        print(f"Saved mixed test set: {save_path}")

    print("\nAll datasets generated successfully!")
