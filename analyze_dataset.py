"""
Diagnostic script to analyze the dataset and identify issues.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
DATASET_DIR = PROJECT_ROOT / "dataset2_tiled"

def analyze_annotations():
    """Analyze XML annotations to understand dataset composition."""
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    all_plate_names = Counter()
    objects_per_image = []
    
    xml_files = list(ANNOTATIONS_DIR.glob("*.xml"))
    print(f"\nTotal annotation files: {len(xml_files)}")
    
    # Sample first few files
    for xml_file in xml_files[:10]:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            objects = root.findall('object')
            
            objects_per_image.append(len(objects))
            
            for obj in objects:
                name = obj.find('name').text
                all_plate_names[name] += 1
                
        except Exception as e:
            print(f"Error reading {xml_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"CRITICAL FINDING:")
    print(f"{'='*60}")
    print(f"Average objects per image: {sum(objects_per_image) / len(objects_per_image):.1f}")
    print(f"Min objects: {min(objects_per_image)}")
    print(f"Max objects: {max(objects_per_image)}")
    
    print(f"\n{'='*60}")
    print(f"UNIQUE PLATE TYPES: {len(all_plate_names)}")
    print(f"{'='*60}")
    print(f"Top 20 most common plates:")
    for name, count in all_plate_names.most_common(20):
        print(f"  {name}: {count}")
    
    print(f"\n{'='*60}")
    print(f"PROBLEM DIAGNOSIS:")
    print(f"{'='*60}")
    
    if sum(objects_per_image) / len(objects_per_image) > 50:
        print("❌ Too many objects per image (>50)")
        print("   This makes training extremely difficult.")
        print("   YOLO works best with 1-20 objects per image.")
    
    if len(all_plate_names) > 10:
        print(f"❌ Too many unique classes ({len(all_plate_names)})")
        print("   But you're training with only 1 class!")
        print("   This creates massive confusion.")
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS:")
    print(f"{'='*60}")
    print("1. If you need to detect specific plates:")
    print("   → Use multi-class training with top 10-20 plate types")
    print("   → Filter out rare plates")
    print("")
    print("2. If you only need to detect 'any plate':")
    print("   → Filter to keep only 5-10 largest plates per image")
    print("   → Current 80+ plates per image is too dense")
    print("")
    print("3. If this is for hand rehabilitation:")
    print("   → Focus only on the target interaction plates")
    print("   → Ignore background/irrelevant plates")
    

def check_label_files():
    """Check generated label files."""
    print(f"\n{'='*60}")
    print(f"LABEL FILE ANALYSIS:")
    print(f"{'='*60}")
    
    train_labels = list((DATASET_DIR / "train" / "labels").glob("*.txt"))
    
    if train_labels:
        sample_file = train_labels[0]
        with open(sample_file) as f:
            lines = f.readlines()
        
        print(f"\nSample label file: {sample_file.name}")
        print(f"Number of labels: {len(lines)}")
        print(f"First 5 labels:")
        for line in lines[:5]:
            parts = line.strip().split()
            class_id, x, y, w, h = parts
            print(f"  class={class_id}, center=({float(x):.3f}, {float(y):.3f}), size=({float(w):.3f}, {float(h):.3f})")
        
        # Calculate avg size
        sizes = []
        for line in lines:
            parts = line.strip().split()
            w, h = float(parts[3]), float(parts[4])
            sizes.append(w * h)
        
        avg_size = sum(sizes) / len(sizes)
        print(f"\nAverage object size (normalized): {avg_size:.4f}")
        
        if avg_size < 0.01:
            print("⚠️  Objects are VERY small (< 1% of image)")
            print("   YOLOv8n might struggle with such tiny objects")
            print("   Consider using YOLOv8m or YOLOv8x for small objects")


if __name__ == "__main__":
    analyze_annotations()
    check_label_files()
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
