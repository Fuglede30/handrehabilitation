import os
import shutil
import csv
from typing import List, Tuple
import yaml
from pathlib import Path
from ultralytics import YOLO

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset2_tiled"  # Tiled dataset - 4 tiles per image!
OUTPUT_DIR = PROJECT_ROOT / "yolo_results"
DATASET_YAML = OUTPUT_DIR / "dataset.yaml"

# Training parameters
MODEL_NAME = "yolov8s"  # Small model - better for tiny objects than nano
EPOCHS = 150  # Increased for better convergence
IMGSZ = 416
BATCH_SIZE = 16  # CRITICAL: was 1, now 16 for stable gradients
DEVICE = -1  # GPU device (0 for first GPU, -1 for CPU)
LR = 0.01  # Learning rate
SEED = 42  # Random seed for reproducibility


def create_dataset_yaml():
    """Create YOLO dataset.yaml configuration file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    dataset_config = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,  # Number of classes (plate detection = 1 class)
        "names": ["plate"]  # Class names
    }
    
    with open(DATASET_YAML, "w") as f:
        yaml.dump(dataset_config, f)
    
    print(f"Created dataset.yaml: {DATASET_YAML}")
    return DATASET_YAML


def _read_loss_columns(csv_path: Path) -> Tuple[List[int], List[float], List[float]]:
    """Read epochs and train/val loss columns from results.csv."""
    epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return epochs, train_losses, val_losses

        # Prefer box_loss; fall back to total loss if present
        train_key = "train/box_loss" if "train/box_loss" in reader.fieldnames else "train/loss"
        val_key = "val/box_loss" if "val/box_loss" in reader.fieldnames else "val/loss"

        if train_key not in reader.fieldnames or val_key not in reader.fieldnames:
            return epochs, train_losses, val_losses

        for row in reader:
            try:
                epochs.append(int(float(row.get("epoch", "0"))))
                train_losses.append(float(row[train_key]))
                val_losses.append(float(row[val_key]))
            except (TypeError, ValueError):
                continue

    return epochs, train_losses, val_losses


def plot_loss_curves(csv_path: Path, output_path: Path) -> None:
    """Plot train/val loss curves from results.csv."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed; skipping loss plot.")
        return

    epochs, train_losses, val_losses = _read_loss_columns(csv_path)
    if not epochs or not train_losses or not val_losses:
        print(f"Warning: could not find loss columns in {csv_path}")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to: {output_path}")


def train_yolo():
    """Train YOLO model."""
    print(f"Starting YOLO training...")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMGSZ}")
    print()
    
    # Load model
    model = YOLO(f"{MODEL_NAME}.pt")
    
    # Train
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="train_tiled",
        lr0=LR
    )
    
    # Persist per-epoch metrics for plotting (train/val loss in results.csv)
    run_dir = Path(getattr(results, "save_dir", OUTPUT_DIR / "train_tiled"))
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        saved_metrics = OUTPUT_DIR / "train_tiled_results.csv"
        shutil.copyfile(results_csv, saved_metrics)
        print(f"Saved per-epoch metrics to: {saved_metrics}")

        plot_path = OUTPUT_DIR / "train_tiled_loss.png"
        plot_loss_curves(saved_metrics, plot_path)
    else:
        print(f"Warning: results.csv not found at {results_csv}")

    return results


def main():
    """Main training function."""
    # Check dataset exists
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        print("Please run preprocess.py first to create the dataset.")
        return

    # Ensure labels folders exist (YOLO expects labels beside images)
    for split in ["train", "val", "test"]:
        labels_dir = DATASET_DIR / split / "labels"
        if not labels_dir.exists():
            print(f"Error: Labels folder missing: {labels_dir}")
            print("Please run preprocess.py to generate YOLO .txt labels.")
            return
    
    train_count = len(list((DATASET_DIR / "train" / "images").glob("*")))
    val_count = len(list((DATASET_DIR / "val" / "images").glob("*")))
    test_count = len(list((DATASET_DIR / "test" / "images").glob("*")))
    train_label_count = len(list((DATASET_DIR / "train" / "labels").glob("*.txt")))
    val_label_count = len(list((DATASET_DIR / "val" / "labels").glob("*.txt")))
    test_label_count = len(list((DATASET_DIR / "test" / "labels").glob("*.txt")))
    
    print(f"Dataset found:")
    print(f"  Train: {train_count} images")
    print(f"  Train labels: {train_label_count} txt")
    print(f"  Val: {val_count} images")
    print(f"  Val labels: {val_label_count} txt")
    print(f"  Test: {test_count} images")
    print(f"  Test labels: {test_label_count} txt")
    print()
    
    # Create YAML config
    create_dataset_yaml()
    print()
    
    # Train model
    try:
        train_yolo()
        print()
        print("Training complete!")
        print(f"Results saved to: {OUTPUT_DIR / 'train_4'}")
        print(f"Best model: {OUTPUT_DIR / 'train_4' / 'weights' / 'best.pt'}")
        print(f"Last model: {OUTPUT_DIR / 'train_4' / 'weights' / 'last.pt'}")
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
