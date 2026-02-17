import os
from pathlib import Path
from ultralytics import YOLO

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset2_class"
MODEL_PATH = PROJECT_ROOT / "yolo_results" / "train_class" / "weights" / "best.pt"
TEST_IMAGES_DIR = DATASET_DIR / "test" / "images"


def test_yolo_model():
    """Test YOLO model on test dataset."""
    # Check model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using yolo_model_train_class.py")
        return
    
    # Check test dataset exists
    if not TEST_IMAGES_DIR.exists():
        print(f"Error: Test images not found at {TEST_IMAGES_DIR}")
        print("Please run preprocess_by_class.py first to create the dataset.")
        return
    
    # Load model
    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    # Get test images count
    test_images = list(TEST_IMAGES_DIR.glob("*.*"))
    print(f"Testing on {len(test_images)} images\n")
    
    # Run validation on test set
    results = model.val(
        data=str(PROJECT_ROOT / "yolo_results" / "dataset.yaml"),
        split="test"
    )
    
    # Print results
    print("\nTest Results:")
    print(f"Precision: {results.box.p[0]:.4f}")
    print(f"Recall: {results.box.r[0]:.4f}")


def main():
    """Main testing function."""
    try:
        test_yolo_model()
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    main()
