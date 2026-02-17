import xml.etree.ElementTree as ET
from pathlib import Path

def print_unique_brick_names(annotations_folder="../annotations"):
    """Print unique brick names found in annotations."""
    annotations_path = Path(__file__).parent.parent / "annotations"
    
    if not annotations_path.exists():
        print(f"Error: {annotations_path} not found")
        return set()
    
    brick_names = set()
    brick_colors = set()
    xml_files = list(annotations_path.glob("*.xml"))
    
    if not xml_files:
        print(f"No XML files found in {annotations_path}")
        return set()
    
    print(f"Processing {len(xml_files)} XML files...")
    
    # Parse each XML file
    for xml_file in xml_files:
        try:
            # Read the XML content
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse XML
            root = ET.fromstring(content)
            
            # Extract object names and colors
            for obj in root.findall('.//object'):
                name_elem = obj.find('name')
                color_elem = obj.find('color')
                
                if name_elem is not None and name_elem.text:
                    brick_names.add(name_elem.text)
                
                if color_elem is not None and color_elem.text:
                    brick_colors.add(color_elem.text)
        
        except Exception as e:
            print(f"Error processing {xml_file.name}: {e}")
    
    # Print results
    print(f"\nFound {len(brick_names)} unique brick names:")
    for name in sorted(brick_names):
        print(f"  - {name}")
    
    print(f"\nFound {len(brick_colors)} unique brick colors:")
    for color in sorted(brick_colors):
        print(f"  - {color}")
    
    return brick_names, brick_colors

if __name__ == "__main__":
    print_unique_brick_names()