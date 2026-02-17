import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "dataset2"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

plates = [
    "3020", "3021", "3022", "3023", "3024",
    "3031", "3032", "3034", "3035",
    "3710",
    "2420", "2429", "2430", "2431", "2432",
    "2445", "2450", "2454", "2456",
    "24866", "27925", "33909", "35480",
    "32013", "32028", "32054", "32062", "32064",
    "32073", "32123", "32140", "32184",
    "32278", "32316",
    "3832", "4162", "4274", "4519",
    "4740", "50950", "54200", "85080"
]


def create_split_folders():
    """Create dataset folder structure."""
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure in {OUTPUT_DIR}")


def convert_xml_to_yolo(xml_file, img_width, img_height):
    """
    Convert XML annotation to YOLO format, filtering only plate bricks.
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
    
    Returns:
        List of YOLO format strings (only for bricks in plates list)
    """
    yolo_lines = []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None or name_elem.text is None:
                continue
            
            class_name = name_elem.text.strip()
            
            # Only process bricks that are in the plates list
            if class_name not in plates:
                continue  # Skip non-plate bricks (treat as background)
            
            class_id = 0  # Single class: plate
            
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            yolo_lines.append(yolo_line)
    
    except Exception as e:
        print(f"Error converting {xml_file}: {e}")
    
    return yolo_lines


def get_image_dimensions(img_file):
    """Get image dimensions using PIL or OpenCV."""
    try:
        from PIL import Image
        img = Image.open(img_file)
        return img.width, img.height
    except:
        try:
            import cv2
            img = cv2.imread(str(img_file))
            if img is not None:
                return img.shape[1], img.shape[0]  # width, height
        except:
            pass
    
    print(f"Warning: Could not determine dimensions for {img_file}")
    return 416, 416  # Default size


def get_file_pairs():
    """Get list of (image, annotation) file pairs."""
    image_files = sorted([f.stem for f in IMAGES_DIR.glob("*.*") if f.is_file()])
    pairs = []
    
    for img_name in image_files:
        # Find corresponding annotation
        ann_file = ANNOTATIONS_DIR / f"{img_name}.xml"
        if ann_file.exists():
            pairs.append((img_name, ann_file.suffix))
        else:
            print(f"Warning: No annotation found for {img_name}")
    
    return pairs


def split_and_copy(pairs):
    """Split file pairs into train/val/test and copy them."""
    random.shuffle(pairs)
    total = len(pairs)
    
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:train_count + val_count]
    test_pairs = pairs[train_count + val_count:]
    
    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }
    
    for split_name, file_pairs in splits.items():
        for img_name, ann_ext in file_pairs:
            # Find image file (could be .png, .jpg, etc.)
            img_file = None
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                candidates = list(IMAGES_DIR.glob(f"{img_name}{ext}"))
                if candidates:
                    img_file = candidates[0]
                    break
            
            if img_file:
                # Copy image
                dst_img = OUTPUT_DIR / split_name / "images" / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Convert XML to YOLO format and save as .txt
                src_ann = ANNOTATIONS_DIR / f"{img_name}.xml"
                dst_label = OUTPUT_DIR / split_name / "labels" / f"{img_name}.txt"
                
                if src_ann.exists():
                    img_w, img_h = get_image_dimensions(img_file)
                    yolo_lines = convert_xml_to_yolo(src_ann, img_w, img_h)
                    
                    if yolo_lines:
                        with open(dst_label, 'w') as f:
                            f.writelines(yolo_lines)
        
        print(f"{split_name}: {len(file_pairs)} samples")


def main():
    """Main preprocessing function."""
    print("Starting dataset preprocessing...")
    print(f"Images: {IMAGES_DIR}")
    print(f"Annotations: {ANNOTATIONS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Create folders
    create_split_folders()
    
    # Get file pairs
    pairs = get_file_pairs()
    total = len(pairs)
    print(f"Found {total} image-annotation pairs")
    print()
    
    # Split and copy
    print("Splitting dataset:")
    split_and_copy(pairs)
    
    print()
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
