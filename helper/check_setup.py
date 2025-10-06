"""
Quick start script to check if everything is set up correctly
"""
import os
import sys

def check_setup():
    """Check if the project is set up correctly"""
    print("=" * 60)
    print("WASTE CLASSIFICATION - SETUP CHECK")
    print("=" * 60)
    
    issues = []
    
    # Check Python version
    print("\n1. Checking Python version...")
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
        print("   ❌ Python version too old")
    else:
        print(f"   ✓ Python {sys.version.split()[0]}")
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    dependencies = [
        'torch', 'torchvision', 'xgboost', 'numpy', 
        'cv2', 'PIL', 'sklearn', 'matplotlib', 
        'seaborn', 'flask', 'tqdm', 'joblib'
    ]
    
    for dep in dependencies:
        try:
            if dep == 'cv2':
                __import__('cv2')
            elif dep == 'PIL':
                __import__('PIL')
            elif dep == 'sklearn':
                __import__('sklearn')
            else:
                __import__(dep)
            print(f"   ✓ {dep}")
        except ImportError:
            issues.append(f"Missing dependency: {dep}")
            print(f"   ❌ {dep} not installed")
    
    # Check directories
    print("\n3. Checking directories...")
    dirs = ['data', 'data/train', 'models', 'results', 'templates', 'static/uploads']
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"   ✓ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ (will be created)")
    
    # Check for training data
    print("\n4. Checking training data...")
    train_dir = 'data/train'
    if os.path.exists(train_dir):
        categories = [d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))]
        if categories:
            print(f"   ✓ Found {len(categories)} categories")
            for cat in categories:
                cat_path = os.path.join(train_dir, cat)
                num_images = len([f for f in os.listdir(cat_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                print(f"     - {cat}: {num_images} images")
        else:
            issues.append("No training data found")
            print("   ⚠ No training data found")
    
    # Check for trained models
    print("\n5. Checking trained models...")
    model_files = [
        'models/squeezenet_embeddings.pth',
        'models/xgboost_classifier.json',
        'models/label_encoder.pkl'
    ]
    
    models_exist = all(os.path.exists(f) for f in model_files)
    if models_exist:
        print("   ✓ Trained models found")
    else:
        print("   ⚠ Models not trained yet")
        print("     Run: python train_model.py")
    
    # Summary
    print("\n" + "=" * 60)
    if issues:
        print("⚠ SETUP INCOMPLETE")
        print("=" * 60)
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nTo fix:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Add training data to data/train/")
        print("  3. Run training: python train_model.py")
    else:
        print("✓ SETUP COMPLETE")
        print("=" * 60)
        print("\nYou're ready to go!")
        if not models_exist:
            print("\nNext steps:")
            print("  1. Add training data to data/train/")
            print("  2. Train model: python train_model.py")
            print("  3. Start web app: python app.py")
        else:
            print("\nStart the web app:")
            print("  python app.py")
    
    print("=" * 60)


if __name__ == "__main__":
    check_setup()
