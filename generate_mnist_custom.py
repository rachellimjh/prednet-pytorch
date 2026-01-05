import os
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# --------------------
# Configuration
# --------------------
OUTPUT_DIR = ".data/mnist_data/custom"
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
# We disable the transform here because we want raw numpy access for manual placement
mnist = datasets.MNIST(
    root=".data/mnist_data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Group MNIST by digit
digit_to_images = {i: [] for i in range(10)}
for img, label in mnist:
    # img is (1, 28, 28). We keep it as (28, 28) for simple slicing
    digit_to_images[label].append(img.squeeze(0).numpy())

# --------------------
# Helper Functions
# --------------------
def sample_digit(digit_pool):
    d = np.random.choice(digit_pool)
    # Randomly pick an example of that digit
    img = digit_to_images[d][np.random.randint(len(digit_to_images[d]))]
    return img, d

def generate_sequence(digit_pool, speed_range, motion_type="any", spawn_region="any", sudden_speed=False, curved_motion=False):
    """
    Generates a single sequence with shape (SEQ_LEN, 1, H, W).
    Includes logic for bouncing, restricted spawn regions, 
    optional sudden speed change or curved/diagonal motion.
    """
    canvas = np.zeros((SEQ_LEN, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    
    digit_img, _ = sample_digit(digit_pool)
    
    # --- Spawn Logic ---
    if spawn_region == "center":
        x = np.random.uniform(IMG_SIZE//2 - 10, IMG_SIZE//2 + 10)
        y = np.random.uniform(IMG_SIZE//2 - 10, IMG_SIZE//2 + 10)
    elif spawn_region == "edge":
        if np.random.rand() > 0.5: # Top/Bottom strip
            x = np.random.randint(0, IMG_SIZE - DIGIT_SIZE)
            y = np.random.choice([0, IMG_SIZE - DIGIT_SIZE - 1])
        else: # Left/Right strip
            x = np.random.choice([0, IMG_SIZE - DIGIT_SIZE - 1])
            y = np.random.randint(0, IMG_SIZE - DIGIT_SIZE)
    else:
        x = np.random.randint(0, IMG_SIZE - DIGIT_SIZE)
        y = np.random.randint(0, IMG_SIZE - DIGIT_SIZE)
    
    # --- Velocity ---
    speed = np.random.uniform(*speed_range)
    
    # Base direction
    if motion_type == "straight":
        direction = np.random.choice(["h", "v"])
        dx, dy = (speed, 0) if direction == "h" else (0, speed)
    else:
        angle = np.random.uniform(0, 2 * np.pi)
        dx = speed * np.cos(angle)
        dy = speed * np.sin(angle)
    
    # For sudden speed change
    if sudden_speed:
        speed_fast = speed * 3  # e.g., triple speed after half sequence
        dx_fast, dy_fast = dx * 3, dy * 3

    # For curved/diagonal motion
    if curved_motion:
        # Small angle change per frame
        angle = np.arctan2(dy, dx)
        d_angle = np.random.uniform(-0.2, 0.2)  # radians per frame

    x, y, dx, dy = float(x), float(y), float(dx), float(dy)
    
    # --- Generate frames ---
    for t in range(SEQ_LEN):
        # Bounce off walls
        if x <= 0 or x >= IMG_SIZE - DIGIT_SIZE:
            dx = -dx
        if y <= 0 or y >= IMG_SIZE - DIGIT_SIZE:
            dy = -dy

        x = np.clip(x, 0, IMG_SIZE - DIGIT_SIZE)
        y = np.clip(y, 0, IMG_SIZE - DIGIT_SIZE)
        
        xi, yi = int(x), int(y)
        canvas[t, 0, yi:yi + DIGIT_SIZE, xi:xi + DIGIT_SIZE] = digit_img
        
        # Update position
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
    """
    Generate a dataset of sequences and save as .npz.
    Supports optional curved motion and sudden speed changes.
    """
    sequences = []
    print(f"Generating {name} ({n_sequences} seqs)...")
    
    for _ in tqdm(range(n_sequences)):
        seq = generate_sequence(
            digit_pool=digit_pool,
            speed_range=speed_range,
            motion_type=motion_type,
            spawn_region=spawn_region,
            curved_motion=curved_motion,
            sudden_speed=sudden_speed
        )
        sequences.append(seq)
    
    sequences = np.stack(sequences)
    save_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    np.savez_compressed(save_path, data=sequences)
    print(f"Saved to {save_path}")


# --------------------
# MAIN EXECUTION
# --------------------
if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("--- 1. Generating TRAINING Sets (In-Distribution) ---")
    
    # A. Unseen digit anomaly (Training: 0-8)
    generate_dataset("train_id_digit", NUM_TRAIN, 
                     digit_pool=list(range(9)), speed_range=(1.0, 2.0), motion_type="any", spawn_region="any")

    # B. Trajectory anomaly (Training: Straight only)
    generate_dataset("train_id_trajectory", NUM_TRAIN, 
                     digit_pool=list(range(10)), speed_range=(1.0, 2.0), motion_type="straight", spawn_region="any")

    # C. Spatial anomaly (Training: Center only)
    generate_dataset("train_id_spatial", NUM_TRAIN, 
                     digit_pool=list(range(10)), speed_range=(1.0, 2.0), motion_type="any", spawn_region="center")

    # D. Speed anomaly (Training: Slow only)
    generate_dataset("train_id_speed", NUM_TRAIN, 
                     digit_pool=list(range(10)), speed_range=(1.0, 1.2), motion_type="any", spawn_region="any")


    print("\n--- 2. Generating TEST Sets (Out-of-Distribution) ---")

    # --- Test Sets with Stronger Anomalies ---

    # A. Unseen digit (same as before)
    generate_dataset("test_ood_digit", NUM_TEST, 
                    digit_pool=[9], speed_range=(1.0, 2.0), motion_type="any", spawn_region="any")

    # B. Trajectory anomaly (curved/diagonal)
    generate_dataset("test_ood_trajectory", NUM_TEST, 
                    digit_pool=list(range(10)), speed_range=(1.0, 2.0), motion_type="any", 
                    spawn_region="any", curved_motion=True)

    # C. Spatial anomaly (edges)
    generate_dataset("test_ood_spatial", NUM_TEST, 
                    digit_pool=list(range(10)), speed_range=(1.0, 2.0), motion_type="any", spawn_region="edge")

    # D. Speed anomaly (sudden speed change)
    generate_dataset("test_ood_speed", NUM_TEST, 
                    digit_pool=list(range(10)), speed_range=(1.0, 1.2), motion_type="any", 
                    spawn_region="any", sudden_speed=True)


    print("\n All datasets generated successfully!")