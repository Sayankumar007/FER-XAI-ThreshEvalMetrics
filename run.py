"""
Runner script for training/evaluating different CNN models.
Allows specifying model, dataset path, and folder name via command line.
Supports default hyperparameters which can be overridden using argparse.
"""

import argparse
import os
from PIL import Image

from main import func_vgg, func_resnet, func_googlenet, func_mobilenet, func_efficientnet, func_densenet



# ==================== Dataset Integrity Check ====================
def check_dataset_integrity(root_dir):
    """
    Checks the dataset integrity for the given root directory.

    Checks performed:
    1. Presence of 'train', 'val', and 'test' folders.
    2. Each folder contains at least one class subfolder.
    3. Each class folder contains at least one valid image file.
    4. Detects corrupted images that cannot be opened with PIL.

    Args:
        root_dir (str): Root directory of dataset.
    
    Raises:
        ValueError: If any check fails.
    """
    required_splits = ['train', 'val', 'test']

    for split in required_splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Missing required dataset split folder: '{split}' in '{root_dir}'")
        
        class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        if len(class_dirs) == 0:
            raise ValueError(f"No class folders found in '{split_dir}'")
        
        for cls in class_dirs:
            cls_dir = os.path.join(split_dir, cls)
            image_files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
            if len(image_files) == 0:
                raise ValueError(f"No images found in class folder '{cls_dir}'")
            
            for img_file in image_files:
                img_path = os.path.join(cls_dir, img_file)
                try:
                    img = Image.open(img_path)
                    img.verify()  # Verify image integrity
                except Exception as e:
                    raise ValueError(f"Corrupted image detected: '{img_path}' ({e})")
    
    print("✅ Dataset integrity check passed successfully.")


if __name__ == "__main__":
    # ---------------------------
    # Argument parser setup
    # ---------------------------
    parser = argparse.ArgumentParser(description="Run specified CNN model on given dataset.")

    # Required arguments
    parser.add_argument("folder_name", type=str, help="Output folder name to save results, PR curves, and visualizations")
    parser.add_argument("model_name", type=str, choices=["vgg", "resnet", "googlenet", "mobilenet", "efficientnet", "densenet"], help="CNN model to run")
    parser.add_argument("root_dir", type=str, help="Root directory path of the dataset (should contain 'val' and 'train' subfolders)")

    # Optional hyperparameters with default values
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate (default: 0.00001)")
    parser.add_argument("--n", type=int, default=1, help="Number of samples to visualize and plotting (default: 1)")
    parser.add_argument("--visualize", action="store_true", help="Enable saving algorithm's intermediate visualizations for each sample")

    args = parser.parse_args()

    folder_name = args.folder_name
    model_name = args.model_name.lower()
    root_dir = args.root_dir
    epochs = args.epochs
    lr = args.lr
    n = args.n
    visualize = args.visualize

    # ---------------------------
    # Model function mapping
    # ---------------------------
    model_funcs = {
        "vgg": func_vgg,
        "resnet": func_resnet,
        "googlenet": func_googlenet,
        "mobilenet": func_mobilenet,
        "efficientnet": func_efficientnet,
        "densenet": func_densenet
    }
    
    
    # ---------------------------
    # Check dataset integrity
    # ---------------------------
    print(f"Checking integrity of the dataset at '{root_dir}'.. ")
    check_dataset_integrity(root_dir)
    

    # ---------------------------
    # Run the selected model
    # ---------------------------
    print(f"Running {model_name} on dataset at '{root_dir}'")
    print(f"Results will be saved in folder '{folder_name}'")
    print(f"Epochs: {epochs}, Learning rate: {lr}, Visualize {n} samples: {visualize}\n\n\n")

    try:
        model_funcs[model_name](folder_name, root_dir, epochs=epochs, lr=lr, n=n, visualize=visualize)
        print(f"\n✅ {model_name} finished successfully!")
    except Exception as e:
        print(f"\n❌ Error running {model_name}: {e}")