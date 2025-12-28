import numpy as np
import matplotlib.pyplot as plt

# Load the file
data = np.load('mnist_data/mnist.npy', allow_pickle=True)

# Inspect the type and shape
print(type(data))
print(data.shape) # (20,10000,64,64) 
# 20 frames per video, 10000 videos, 64x64 frame res

# to plot 20 frames of 1 video
# select the first video
video = data[:, 0, :, :]  # shape: (20, 64, 64)

# Plot all 20 frames in a 4x5 grid
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
for i in range(20):
    ax = axes[i // 5, i % 5]
    ax.imshow(video[i], cmap='gray')
    ax.set_title(f'Frame {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()