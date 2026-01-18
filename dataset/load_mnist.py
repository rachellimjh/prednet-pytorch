import torch
from torchvision import datasets, transforms
from collections import defaultdict
import cv2
import numpy as np

DIGIT_SIZE = 14  # size of each digit after resizing

def load_mnist_by_digit():
    """
    Returns:
        dict[int] -> list of 2D numpy arrays of size DIGIT_SIZE x DIGIT_SIZE
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # 1x28x28, values 0-1
    ])

    dataset = datasets.MNIST(
        root="data/mnist_data",
        train=True,
        download=True,
        transform=transform
    )

    mnist_by_digit = defaultdict(list)

    for img, label in dataset:
        img_np = img.squeeze(0).numpy()           # remove channel dim
        img_resized = cv2.resize(img_np, (DIGIT_SIZE, DIGIT_SIZE))
        img_resized = (img_resized * 255).astype(np.uint8)
        mnist_by_digit[label].append(img_resized)

    return mnist_by_digit
