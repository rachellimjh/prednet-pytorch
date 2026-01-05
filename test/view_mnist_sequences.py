# video style plotting (vs frame by frame)
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_sequence(npz_path, index=0):
    data = np.load(npz_path)["data"]   # (N, T, C, H, W)
    return data[index]                 # (T, C, H, W)

def visualize_sequence(seq, pause=0.2, title=""):
    """
    seq: (T, C, H, W)
    """
    plt.figure()
    # video style plotting
    # for t in range(seq.shape[0]):
    #     plt.clf()
    #     plt.imshow(seq[t, 0], cmap="gray")
    #     plt.title(f"{title} | Frame {t}")
    #     plt.axis("off")
    #     plt.pause(pause)
    # plt.show()
    # frame by frame
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(seq[i, 0], cmap="gray")
        ax.set_title(f"Frame {i}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="Path to .npz file")
    parser.add_argument("--index", type=int, default=0,
                        help="Sequence index")
    parser.add_argument("--pause", type=float, default=0.15,
                        help="Pause between frames")
    args = parser.parse_args()

    seq = load_sequence(args.path, args.index)
    visualize_sequence(seq, pause=args.pause, title=args.path)
