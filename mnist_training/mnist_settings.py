DATA_DIR = "data/custom" 
# 1. ID (digits)
# 2. ID (vertical motion)
# 3. both?
NPZ_FILE = "OOD_NORMAL_HORIZONTAL.npz"

KITTI_WEIGHTS= "models"
WEIGHTS_DIR = "models/mnist_id_vertical"
MNIST_MODEL="prednet_mmnist_best.pth"
# MNIST_MODEL="prednet_kitti_to_mmnist_best.pth"

RESULTS_SAVE_DIR="results/ood_horizontal_normal"
EVAL_ANOMALY_ONLY = False # True for anomaly datasets j