"""
Preprocessing with image tiling - splits each image into 4 quadrants.
This makes small objects appear larger, improving detection accuracy.
"""
import xml.etree.ElementTree as ET
import random
from pathlib import Path
from PIL import Image
import shutil
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "dataset2_tiled"

# Configuration
MIN_PLATE_SIZE = 0.001  # Can be smaller now since tiles are smaller
RANDOM_SEED = 42
TRAIN_COUNT = 350
VAL_COUNT = 75
TEST_COUNT = 75

# LEGO plate part numbers to detect (add your plate part numbers here)
# Example plate parts - UPDATE THIS LIST with your actual plate part numbers
PLATE_PART_NUMBERS = {
    "3020", "3021", "3022", "3023", "3024","3031", "3032", "3034", "3035","3710","2420", "2429", "2430", "2431", "2432",
    "2445", "2450", "2454", "2456","24866", "27925", "33909", "35480","32013", "32028", "32054", "32062", "32064",
    "32073", "32123", "32140", "32184","32278", "32316","3832", "4162", "4274", "4519","4740", "50950", "54200", "85080", "3958"
}

PLATE_PART_NUMBERS_2 = {
    "11212", "3020", "3021", "3022", "3031", "3032", "3034", "3035", "3795", "3832", "3958"
}

# Tile configuration (2x2 grid)
TILES = [
    ('tl', 0, 0),  # top-left
    ('tr', 1, 0),  # top-right
    ('bl', 0, 1),  # bottom-left
    ('br', 1, 1),  # bottom-right
]


def get_bbox_area(bbox, img_width, img_height):
    """Calculate bounding box area as fraction of image."""
    xmin, ymin, xmax, ymax = bbox
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return width * height


def bbox_in_tile(bbox, tile_x, tile_y, img_width, img_height):
    """
    Check if bounding box intersects with tile and return adjusted bbox.
    
    Args:
        bbox: (xmin, ymin, xmax, ymax) in original image coordinates
        tile_x: 0 (left) or 1 (right)
        tile_y: 0 (top) or 1 (bottom)
        img_width, img_height: Original image dimensions
    
    Returns:
        Adjusted bbox in tile coordinates, or None if no intersection
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate tile boundaries in original image
    tile_width = img_width // 2
    tile_height = img_height // 2
    tile_xmin = tile_x * tile_width
    tile_ymin = tile_y * tile_height
    tile_xmax = tile_xmin + tile_width
    tile_ymax = tile_ymin + tile_height
    
    # Check if bbox intersects with tile
    if xmax < tile_xmin or xmin > tile_xmax:
        return None
    if ymax < tile_ymin or ymin > tile_ymax:
        return None
    
    # Clip bbox to tile boundaries
    clipped_xmin = max(xmin, tile_xmin)
    clipped_ymin = max(ymin, tile_ymin)
    clipped_xmax = min(xmax, tile_xmax)
    clipped_ymax = min(ymax, tile_ymax)
    
    # Convert to tile-relative coordinates
    tile_rel_xmin = clipped_xmin - tile_xmin
    tile_rel_ymin = clipped_ymin - tile_ymin
    tile_rel_xmax = clipped_xmax - tile_xmin
    tile_rel_ymax = clipped_ymax - tile_ymin
    
    # Check if clipped box is large enough
    clipped_area = ((clipped_xmax - clipped_xmin) * (clipped_ymax - clipped_ymin)) / (tile_width * tile_height)
    if clipped_area < MIN_PLATE_SIZE:
        return None
    
    return (tile_rel_xmin, tile_rel_ymin, tile_rel_xmax, tile_rel_ymax)


def parse_xml_for_tiles(xml_file, img_file):
    """Parse XML and organize objects by tile."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions from actual image file
        img = Image.open(img_file)
        img_width, img_height = img.size
        
        # Organize objects by tile
        tile_objects = {tile_name: [] for tile_name, _, _ in TILES}
        filtered_count = 0  # Track non-plate objects
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            
            # Skip if not a plate part number
            if name not in PLATE_PART_NUMBERS_2:
                filtered_count += 1
                continue
            
            bndbox = obj.find('bndbox')
            
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            original_bbox = (xmin, ymin, xmax, ymax)
            
            # Check which tiles this object appears in
            for tile_name, tile_x, tile_y in TILES:
                adjusted_bbox = bbox_in_tile(original_bbox, tile_x, tile_y, img_width, img_height)
                
                if adjusted_bbox:
                    tile_width = img_width // 2
                    tile_height = img_height // 2
                    area = get_bbox_area(adjusted_bbox, tile_width, tile_height)
                    
                    tile_objects[tile_name].append({
                        'name': name,
                        'bbox': adjusted_bbox,
                        'area': area
                    })
        
        return tile_objects, img_width, img_height, filtered_count
        
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return None, 0, 0, 0


def convert_to_yolo_format(bbox, tile_width, tile_height):
    """Convert bounding box to YOLO format (normalized to tile size)."""
    xmin, ymin, xmax, ymax = bbox
    
    # Convert to YOLO format: center_x, center_y, width, height (normalized)
    x_center = ((xmin + xmax) / 2) / tile_width
    y_center = ((ymin + ymax) / 2) / tile_height
    width = (xmax - xmin) / tile_width
    height = (ymax - ymin) / tile_height
    
    return x_center, y_center, width, height


def split_image_into_tiles(img_file, output_folder, img_name):
    """
    Split image into 4 tiles and save all of them.
    
    Args:
        img_file: Path to source image
        output_folder: Where to save tiles
        img_name: Base name for tiles
    
    Returns:
        Dict mapping tile names to paths, tile dimensions
    """
    img = Image.open(img_file)
    img_width, img_height = img.size
    
    tile_width = img_width // 2
    tile_height = img_height // 2
    
    tile_files = {}
    
    for tile_name, tile_x, tile_y in TILES:
        # Calculate crop box
        left = tile_x * tile_width
        top = tile_y * tile_height
        right = left + tile_width
        bottom = top + tile_height
        
        # Crop and save tile
        tile_img = img.crop((left, top, right, bottom))
        tile_filename = f"{img_name}_{tile_name}{img_file.suffix}"
        tile_path = output_folder / tile_filename
        tile_img.save(tile_path)
        
        tile_files[tile_name] = tile_path
    
    return tile_files, tile_width, tile_height


def create_tiled_dataset():
    """Create tiled dataset with image splitting."""
    print("=" * 60)
    print("CREATING TILED DATASET (4 tiles per image)")
    print("=" * 60)
    print(f"Each image split into 2x2 grid (4 tiles)")
    print(f"Min plate size: {MIN_PLATE_SIZE * 100:.2f}% of tile")
    print(f"Sampling with seed: {RANDOM_SEED}")
    print(f"Filtering for {len(PLATE_PART_NUMBERS)} plate part numbers only")
    print(f"Plate parts: {', '.join(sorted(PLATE_PART_NUMBERS)[:10])}...")
    print()
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get all annotation files
    xml_files = sorted(list(ANNOTATIONS_DIR.glob("*.xml")))
    total_files = len(xml_files)
    
    # Randomly sample fixed counts for each split
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(xml_files)
    
    requested_total = TRAIN_COUNT + VAL_COUNT + TEST_COUNT
    if requested_total > total_files:
        print(
            f"Warning: Requested {requested_total} files, but only {total_files} are available."
        )
        print("Reducing counts in order: train -> val -> test")
    
    train_count = min(TRAIN_COUNT, total_files)
    remaining = total_files - train_count
    val_count = min(VAL_COUNT, remaining)
    remaining -= val_count
    test_count = min(TEST_COUNT, remaining)
    
    train_files = xml_files[:train_count]
    val_files = xml_files[train_count:train_count + val_count]
    test_files = xml_files[train_count + val_count:train_count + val_count + test_count]
    
    print(
        f"Using {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files"
    )
    
    stats = {
        'train': {'images': 0, 'tiles': 0, 'total_plates': 0, 'skipped': 0, 'filtered_objects': 0},
        'val': {'images': 0, 'tiles': 0, 'total_plates': 0, 'skipped': 0, 'filtered_objects': 0},
        'test': {'images': 0, 'tiles': 0, 'total_plates': 0, 'skipped': 0, 'filtered_objects': 0}
    }
    
    def process_split(xml_list, split_name):
        """Process one data split."""
        output_images = OUTPUT_DIR / split_name / 'images'
        output_labels = OUTPUT_DIR / split_name / 'labels'
        
        for xml_file in xml_list:
            # Find corresponding image
            img_name = xml_file.stem
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_img = IMAGES_DIR / f"{img_name}{ext}"
                if potential_img.exists():
                    img_file = potential_img
                    break
            
            if not img_file or not img_file.exists():
                stats[split_name]['skipped'] += 1
                continue
            
            # Parse XML and organize by tiles
            tile_objects, img_width, img_height, filtered_count = parse_xml_for_tiles(xml_file, img_file)
            
            if tile_objects is None:
                stats[split_name]['skipped'] += 1
                continue
            
            # Track filtered objects
            stats[split_name]['filtered_objects'] += filtered_count
            
            # Split image into all 4 tiles
            tile_files, tile_width, tile_height = split_image_into_tiles(
                img_file, output_images, img_name
            )
            
            stats[split_name]['images'] += 1
            
            # Process each tile - create labels only for tiles with objects
            for tile_name, tile_x, tile_y in TILES:
                objects = tile_objects[tile_name]
                
                if not objects:
                    # No plates in this tile, skip label creation
                    continue
                
                # Create YOLO label file for this tile
                label_file = output_labels / f"{img_name}_{tile_name}.txt"
                with open(label_file, 'w') as f:
                    for obj in objects:
                        x_center, y_center, width, height = convert_to_yolo_format(
                            obj['bbox'], tile_width, tile_height
                        )
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                stats[split_name]['tiles'] += 1
                stats[split_name]['total_plates'] += len(objects)
    
    print("Processing train split...")
    process_split(train_files, 'train')
    
    print("Processing val split...")
    process_split(val_files, 'val')
    
    print("Processing test split...")
    process_split(test_files, 'test')
    
    # Clean up: Remove tile images that don't have corresponding labels
    print("\nCleaning up tiles without labels...")
    removed_count = {'train': 0, 'val': 0, 'test': 0}
    
    for split in ['train', 'val', 'test']:
        images_dir = OUTPUT_DIR / split / 'images'
        labels_dir = OUTPUT_DIR / split / 'labels'
        
        for img_file in images_dir.glob('*'):
            # Check if corresponding label exists
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                # No label for this image, remove it
                img_file.unlink()
                removed_count[split] += 1
    
    print(f"  Removed {removed_count['train']} train tiles without labels")
    print(f"  Removed {removed_count['val']} val tiles without labels")
    print(f"  Removed {removed_count['test']} test tiles without labels")

    # Final image/label counts after cleanup
    final_counts = {}
    for split in ['train', 'val', 'test']:
        images_dir = OUTPUT_DIR / split / 'images'
        labels_dir = OUTPUT_DIR / split / 'labels'
        final_counts[split] = {
            'images': len(list(images_dir.glob('*'))),
            'labels': len(list(labels_dir.glob('*.txt')))
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("TILED DATASET CREATION COMPLETE")
    print("=" * 60)
    for split in ['train', 'val', 'test']:
        s = stats[split]
        avg_plates = s['total_plates'] / s['tiles'] if s['tiles'] > 0 else 0
        print(f"\n{split.upper()}:")
        print(f"  Original images: {s['images']}")
        print(f"  Tiles created: {s['tiles']}")
        print(f"  Final tile images: {final_counts[split]['images']}")
        print(f"  Final label files: {final_counts[split]['labels']}")
        print(f"  Total plates: {s['total_plates']}")
        print(f"  Avg plates per tile: {avg_plates:.1f}")
        print(f"  Non-plate objects filtered: {s['filtered_objects']}")
        print(f"  Skipped: {s['skipped']}")
    
    print(f"\n✓ Dataset saved to: {OUTPUT_DIR}")
    print("\nProcessing approach:")
    print("  • All tiles are initially saved from each image")
    print("  • Labels created only for tiles containing plates")
    print("  • Tiles without labels are automatically removed")
    print("  • Final dataset has perfect 1:1 image-label correspondence")
    print("\nBenefits of tiling:")
    print("  • Objects are 4x larger relative to tile size")
    print("  • Model can focus on smaller regions")
    print("  • Better detection of tiny objects")
    print("\nNext steps:")
    print("1. Update yolo_model_training.py to use 'dataset2_tiled'")
    print("2. YOLOv8s should work well with this approach")


if __name__ == "__main__":
    create_tiled_dataset()
