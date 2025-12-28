#!/bin/bash

# Define Directory and URL
TARGET_DIR="mnist_data"
BASE_URL="https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
OUTPUT_FILE="$TARGET_DIR/mnist.npy"

# Create the directory if it doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating directory: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

echo "Downloading Moving MNIST dataset into '$TARGET_DIR'..."

# Check if wget is installed, otherwise use curl
if command -v wget &> /dev/null; then
    wget -O "$OUTPUT_FILE" "$BASE_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$OUTPUT_FILE" "$BASE_URL"
else
    echo "Error: Neither wget nor curl is installed. Please install one to proceed."
    exit 1
fi

echo "Download complete! File saved at: $OUTPUT_FILE"