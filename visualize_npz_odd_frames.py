"""
For each NPZ file in DATA_DIR, plots a grid of frames:
  - rows  = sequences (up to N_SEQUENCES_TO_SHOW)
  - cols  = odd-numbered frames only (frame 1, 3, 5, … i.e. indices 0, 2, 4, …)

One PNG is saved per NPZ file into OUTPUT_DIR.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data/custom"
OUTPUT_DIR = "results/data_visualizations"
N_SEQUENCES_TO_SHOW = 5   # rows per plot; set to None to show all sequences
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

npz_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".npz"))

if not npz_files:
    print(f"No .npz files found in {DATA_DIR}")
    exit(0)

for npz_name in npz_files:
    npz_path = os.path.join(DATA_DIR, npz_name)

    try:
        data = np.load(npz_path)["sequences"]   # (N, T, H, W)
    except KeyError:
        print(f"SKIP {npz_name}: no 'sequences' key found")
        continue

    N, T, H, W = data.shape

    # Odd-numbered frames: 1st, 3rd, 5th, … → indices 0, 2, 4, …
    frame_indices = list(range(0, T, 2))
    n_cols = len(frame_indices)

    n_rows = N if N_SEQUENCES_TO_SHOW is None else min(N_SEQUENCES_TO_SHOW, N)

    # --- Figure sizing: each cell ≈ 1.8 × 1.8 inches, plus space for labels ---
    cell_w = 1.8
    cell_h = 1.8
    fig_w = n_cols * cell_w + 0.6          # +0.6 for row labels
    fig_h = n_rows * cell_h + 0.5          # +0.5 for column headers

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        wspace=0.04,
        hspace=0.06,
        left=0.06, right=0.99,
        top=0.93,  bottom=0.02,
    )

    # Track axes so we can compute the separator position afterwards
    axes_grid = {}

    for row in range(n_rows):
        seq = data[row].astype(np.float32)
        if seq.max() > 1.0:
            seq = seq / 255.0

        for col, t in enumerate(frame_indices):
            ax = fig.add_subplot(gs[row, col])
            axes_grid[(row, col)] = ax
            ax.imshow(seq[t], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")

            # Column header: frame number (1-indexed, odd)
            if row == 0:
                ax.set_title(f"Frame {t + 1}", fontsize=9, pad=3)

            # Row label: sequence index
            if col == 0:
                ax.set_ylabel(f"Seq {row}", fontsize=9, rotation=0,
                              labelpad=28, va="center")

    # Draw red separator line between frame 9 and frame 11 for anomaly files.
    # Odd-frame columns: 0→F1, 1→F3, 2→F5, 3→F7, 4→F9 | 5→F11, …
    # The anomaly is inserted at frame 10, so the line sits between col 4 and col 5.
    is_anomaly = any(kw in npz_name.lower() for kw in ("appear", "disappear", "stick"))
    SEP_COL = 4  # right edge of frame-9 column

    if is_anomaly and SEP_COL + 1 < n_cols:
        # Force layout so get_position() returns accurate figure coordinates
        fig.canvas.draw()

        # x: midpoint of the gap between col SEP_COL and col SEP_COL+1
        x_right = axes_grid[(0, SEP_COL)].get_position().x1
        x_left  = axes_grid[(0, SEP_COL + 1)].get_position().x0
        x_mid   = (x_right + x_left) / 2

        # y: from bottom of last row to top of first row
        y_bottom = axes_grid[(n_rows - 1, SEP_COL)].get_position().y0
        y_top    = axes_grid[(0, SEP_COL)].get_position().y1

        line = plt.Line2D(
            [x_mid, x_mid], [y_bottom, y_top],
            transform=fig.transFigure,
            color="red",
            linewidth=2.5,
            linestyle="--",
            zorder=10,
        )
        fig.add_artist(line)

    fig.suptitle(os.path.splitext(npz_name)[0], fontsize=11, y=0.98)

    out_name = os.path.splitext(npz_name)[0] + "_odd_frames.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path}  [{N} sequences total, showing {n_rows}]")

print("\nDone.")
