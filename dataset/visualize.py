import matplotlib.pyplot as plt
import imageio

# frame by frame
def visualize_sequence(sequence, title=None):
    """
    sequence: (T, H, W)
    """
    T = sequence.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(T * 1.2, 1.5))

    for i in range(T):
        axes[i].imshow(sequence[i], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"t={i}")

    if title:
        plt.suptitle(title)

    plt.show()

# video format
def save_gif(sequence, path):
    imageio.mimsave(
        path,
        sequence,
        duration=0.1
    )
