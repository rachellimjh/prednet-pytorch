# # video style plotting (vs frame by frame)
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse

# def load_sequence(npz_path, index=0):
#     data = np.load(npz_path)["data"]   # (N, T, C, H, W)
#     return data[index]                 # (T, C, H, W)

# def visualize_sequence(seq, pause=0.2, title=""):
#     """
#     seq: (T, C, H, W)
#     """
#     plt.figure()
#     # video style plotting
#     # for t in range(seq.shape[0]):
#     #     plt.clf()
#     #     plt.imshow(seq[t, 0], cmap="gray")
#     #     plt.title(f"{title} | Frame {t}")
#     #     plt.axis("off")
#     #     plt.pause(pause)
#     # plt.show()
#     # frame by frame
#     fig, axes = plt.subplots(4, 5, figsize=(10, 8))
#     for i, ax in enumerate(axes.flat):
#         ax.imshow(seq[i, 0], cmap="gray")
#         ax.set_title(f"Frame {i}")
#         ax.axis("off")
#     plt.show()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path", type=str, required=True,
#                         help="Path to .npz file")
#     parser.add_argument("--index", type=int, default=0,
#                         help="Sequence index")
#     parser.add_argument("--pause", type=float, default=0.15,
#                         help="Pause between frames")
#     args = parser.parse_args()

#     seq = load_sequence(args.path, args.index)
#     visualize_sequence(seq, pause=args.pause, title=args.path)

# import numpy as np

# path = "data/custom/TRAIN_ID.npz"

# with np.load(path) as data:
#     sequences = data["sequences"]
#     print("Number of sequences:", sequences.shape[0])
#     print("Sequence shape:", sequences.shape)

import h5py

train_file = "data/kitti_data/X_train.hkl"

with h5py.File(train_file, 'r') as f:
    print("Keys in the file:", list(f.keys()))
    data0 = f['data_0']
    print("Shape of data_0:", data0.shape)
    print("Datatype:", data0.dtype)

import hickle as hkl
import h5py
import numpy as np

# -----------------------------
# Path to your training data
# -----------------------------
data_file = "data/kitti_data/X_train.hkl"
source_file = "data/kitti_data/sources_train.hkl"

# -----------------------------
# Load data safely
# -----------------------------
try:
    with h5py.File(data_file, 'r') as f:
        X = f['data_0'][:]
except:
    X = hkl.load(data_file)

try:
    with h5py.File(source_file, 'r') as f:
        sources = f['data_0'][:]
except:
    sources = hkl.load(source_file)

# -----------------------------
# Parameters
# -----------------------------
nt = 10  # sequence length

# -----------------------------
# Compute possible start indices (mode='all')
# -----------------------------
possible_starts = [i for i in range(len(X) - nt + 1) if sources[i] == sources[i + nt - 1]]
num_sequences = len(possible_starts)

print(f"Total frames in dataset: {len(X)}")
print(f"Sequence length (nt): {nt}")
print(f"Number of sequences in 'all' mode: {num_sequences}")

