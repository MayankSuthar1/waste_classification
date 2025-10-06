"""
Dataset Setup Script
Prepares the WasteClassificationNeuralNetwork dataset for training
"""
import os
import shutil
from pathlib import Path
import config

def check_dataset():
    """Check if the dataset has been cloned"""
    dataset_path = os.path.join(config.DATA_DIR, 'WasteClassificationNeuralNetwork', 'WasteImagesDataset')
    
    print("=" * 60)
    print("DATASET SETUP CHECK")
    print("=" * 60)
    
    if not os.path.exists(dataset_path):
        print("\n❌ Dataset not found!")
        print(f"\nExpected location: {dataset_path}")
        print("\nTo download the dataset:")
        print("1. Open terminal in the 'data' folder")
        print("2. Run: git clone https://github.com/cardstdani/WasteClassificationNeuralNetwork.git")
        print("\nOr download manually from:")
        print("https://github.com/cardstdani/WasteClassificationNeuralNetwork")
        return False
    
    print(f"\n✓ Dataset found at: {dataset_path}")
    
    # Count images per class
    print("\nDataset Statistics:")
    print("-" * 60)
    
    total_images = 0
    class_counts = {}
    
    for class_name in config.WASTE_CATEGORIES:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            # Count image files
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            count = len(images)
            class_counts[class_name] = count
            total_images += count
            print(f"  {class_name:<25}: {count:>5} images")
        else:
            print(f"  {class_name:<25}: NOT FOUND")
            class_counts[class_name] = 0
    
    print("-" * 60)
    print(f"  {'TOTAL':<25}: {total_images:>5} images")
    print("-" * 60)
    
    if total_images == 0:
        print("\n❌ No images found in dataset!")
        return False
    
    print(f"\n✓ Dataset ready with {total_images} images across {len(config.WASTE_CATEGORIES)} classes")
    return True


def verify_dataset_structure():
    """Verify the dataset structure matches expectations"""
    dataset_path = os.path.join(config.DATA_DIR, 'WasteClassificationNeuralNetwork', 'WasteImagesDataset')
    
    print("\n" + "=" * 60)
    print("VERIFYING DATASET STRUCTURE")
    print("=" * 60)
    
    missing_classes = []
    empty_classes = []
    
    for class_name in config.WASTE_CATEGORIES:
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            missing_classes.append(class_name)
            print(f"❌ Missing folder: {class_name}")
        else:
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if len(images) == 0:
                empty_classes.append(class_name)
                print(f"⚠  Empty folder: {class_name}")
            else:
                print(f"✓  {class_name}: OK")
    
    if missing_classes:
        print(f"\n⚠ Missing {len(missing_classes)} class folders")
    if empty_classes:
        print(f"\n⚠ Found {len(empty_classes)} empty class folders")
    
    if not missing_classes and not empty_classes:
        print("\n✓ All class folders present and contain images")
        return True
    
    return False


def show_sample_images():
    """Display information about sample images from each class"""
    dataset_path = os.path.join(config.DATA_DIR, 'WasteClassificationNeuralNetwork', 'WasteImagesDataset')
    
    print("\n" + "=" * 60)
    print("SAMPLE IMAGES")
    print("=" * 60)
    
    for class_name in config.WASTE_CATEGORIES[:3]:  # Show first 3 classes
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                print(f"\n{class_name}:")
                for img in images[:3]:  # Show first 3 images
                    img_path = os.path.join(class_path, img)
                    size = os.path.getsize(img_path) / 1024  # KB
                    print(f"  - {img} ({size:.1f} KB)")


def get_dataset_info():
    """Get detailed dataset information"""
    dataset_path = os.path.join(config.DATA_DIR, 'WasteClassificationNeuralNetwork', 'WasteImagesDataset')
    
    info = {
        'total_images': 0,
        'total_size_mb': 0,
        'classes': {},
        'dataset_path': dataset_path
    }
    
    for class_name in config.WASTE_CATEGORIES:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            class_size = sum(os.path.getsize(os.path.join(class_path, img)) 
                           for img in images) / (1024 * 1024)  # MB
            
            info['classes'][class_name] = {
                'count': len(images),
                'size_mb': class_size
            }
            info['total_images'] += len(images)
            info['total_size_mb'] += class_size
    
    return info


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         WASTE CLASSIFICATION DATASET SETUP                           ║")
    print("║         WasteClassificationNeuralNetwork Dataset                     ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    # Check if dataset exists
    if check_dataset():
        # Verify structure
        verify_dataset_structure()
        
        # Show samples
        show_sample_images()
        
        # Get detailed info
        info = get_dataset_info()
        
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total Images: {info['total_images']}")
        print(f"Total Size: {info['total_size_mb']:.2f} MB")
        print(f"Number of Classes: {len(config.WASTE_CATEGORIES)}")
        print(f"Dataset Path: {info['dataset_path']}")
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("✓ Dataset is ready!")
        print("\n1. Train the model:")
        print("   python train_model.py")
        print("\n2. The training will automatically:")
        print("   - Use all 9 waste categories")
        print("   - Split data into train/validation (90/10)")
        print("   - Extract features with SqueezeNet")
        print("   - Train XGBoost classifier")
        print("\n3. After training, start the web app:")
        print("   python app.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TO DOWNLOAD THE DATASET")
        print("=" * 60)
        print("\nOption 1: Using Git")
        print("  cd data")
        print("  git clone https://github.com/cardstdani/WasteClassificationNeuralNetwork.git")
        print("\nOption 2: Manual Download")
        print("  1. Visit: https://github.com/cardstdani/WasteClassificationNeuralNetwork")
        print("  2. Click 'Code' > 'Download ZIP'")
        print("  3. Extract to: data/WasteClassificationNeuralNetwork/")
        print("\nThen run this script again:")
        print("  python setup_dataset.py")
        print("=" * 60)
