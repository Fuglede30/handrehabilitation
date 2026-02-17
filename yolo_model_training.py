import os
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
PATIENCE = 25  # Early stopping patience
LR0 = 0.01  # Initial learning rate
LRF = 0.01  # Final learning rate


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
        patience=PATIENCE,
        project=str(OUTPUT_DIR),
        name="train_tiled",  # Distinguish from non-tiled training
        exist_ok=True,
        save=True,
        verbose=True,
        lr0=LR0,
        lrf=LRF,
        augment=True,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=15
    )
    
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
        print(f"Results saved to: {OUTPUT_DIR / 'train3'}")
        print(f"Best model: {OUTPUT_DIR / 'train3' / 'weights' / 'best.pt'}")
        print(f"Last model: {OUTPUT_DIR / 'train3' / 'weights' / 'last.pt'}")
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
