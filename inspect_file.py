import h5py
import os

# CHANGE THIS to your actual file path
filename = 'kitti_data/sources_train.hkl' 

if not os.path.exists(filename):
    print(f"Error: File not found at {filename}")
else:
    try:
        with h5py.File(filename, 'r') as f:
            print(f"\n--- Inspecting {filename} ---")
            keys = list(f.keys())
            print(f"Keys found: {keys}")
            
            # Print the shape of the first dataset found
            if len(keys) > 0:
                first_key = keys[0]
                print(f"Shape of '{first_key}': {f[first_key].shape}")
                
    except Exception as e:
        print(f"Could not open file: {e}")