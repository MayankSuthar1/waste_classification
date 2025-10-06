# 🗑️ Waste Classification Dataset Guide
## WasteClassificationNeuralNetwork Dataset (9 Classes)

This project uses a comprehensive waste classification dataset with **5,078 images** across **9 waste categories**.

---

## 📦 Dataset Information

**Source:** https://github.com/cardstdani/WasteClassificationNeuralNetwork

**Statistics:**
- **Total Images:** 5,078
- **Number of Classes:** 9
- **Size:** ~200 MB
- **Format:** JPG images
- **Resolution:** Variable (will be resized to 224×224 for SqueezeNet)

**Dataset Split:**
- Training: 90% (4,571 images)
- Validation: 10% (507 images)

---

## 🏷️ Waste Categories

1. **Metal** - Metal cans, foils, and containers (aluminum, steel, etc.)
2. **Carton** - Cardboard boxes and carton packaging
3. **Glass** - Glass bottles, jars, and containers
4. **Organic Waste** - Food scraps, biodegradable materials
5. **Other Plastics** - Mixed plastic items
6. **Paper and Cardboard** - Paper products, cardboard
7. **Plastic** - Plastic bottles and containers
8. **Textiles** - Fabric, clothing, cloth materials
9. **Wood** - Wood pieces, wooden items

---

## 🚀 Quick Setup (Choose One Method)

### Method 1: Automated Setup (Recommended)

Run the batch script:
```bash
setup_and_train.bat
```

This will:
1. ✓ Check if dataset exists
2. ✓ Download dataset if needed
3. ✓ Verify dataset structure
4. ✓ Offer to train the model
5. ✓ Launch the web app after training

### Method 2: Manual Setup

1. **Download Dataset:**
   ```bash
   cd data
   git clone https://github.com/cardstdani/WasteClassificationNeuralNetwork.git
   cd ..
   ```

2. **Verify Setup:**
   ```bash
   python setup_dataset.py
   ```

3. **Train Model:**
   ```bash
   python train_model.py
   ```

4. **Launch Web App:**
   ```bash
   python app.py
   ```

---

## 📁 Dataset Structure

After cloning, your data folder will look like this:

```
data/
└── WasteClassificationNeuralNetwork/
    └── WasteImagesDataset/
        ├── Metal/
        │   ├── image001.jpg
        │   ├── image002.jpg
        │   └── ...
        ├── Carton/
        ├── Glass/
        ├── Organic Waste/
        ├── Other Plastics/
        ├── Paper and Cardboard/
        ├── Plastic/
        ├── Textiles/
        └── Wood/
```

---

## 🎓 Training Process

### Expected Timeline

| Step | Duration | Description |
|------|----------|-------------|
| Data Loading | ~30 sec | Loading 5,078 images |
| Feature Extraction | 5-15 min | SqueezeNet processes all images |
| XGBoost Training | 2-5 min | Trains the classifier |
| Evaluation | ~1 min | Tests on validation set |
| **Total** | **10-25 min** | Depends on CPU/GPU |

**Note:** GPU significantly speeds up feature extraction (5-10x faster)

### Expected Performance

With this dataset, you should achieve:
- **Training Accuracy:** 95-98%
- **Validation Accuracy:** 85-93%
- **Per-Class Accuracy:** 80-95% (varies by category)

### Common Confusions

The model may occasionally confuse:
- **Plastic ↔ Other Plastics** (similar materials)
- **Carton ↔ Paper and Cardboard** (similar texture)
- **Metal ↔ Glass** (reflective surfaces)

These are normal and can be improved with more training data.

---

## 💻 Training Commands

### Standard Training
```bash
python train_model.py
```

### Check Setup Before Training
```bash
python setup_dataset.py
```

### Evaluate After Training
```bash
python evaluate_model.py
```

### Test Single Image
```bash
python predict.py path/to/image.jpg
```

---

## 🎯 Using the Trained Model

### 1. Web Application

Start the Flask server:
```bash
python app.py
```

Then open: http://localhost:5000

**Features:**
- Drag & drop image upload
- Real-time classification
- Confidence scores for all 9 classes
- Beautiful visual interface

### 2. Python API

```python
from predict import WasteClassifier

# Initialize classifier
classifier = WasteClassifier()

# Predict single image
result = classifier.predict('test_image.jpg', return_probabilities=True)

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Show all class probabilities
for class_name, prob in result['probabilities'].items():
    print(f"  {class_name}: {prob:.2%}")
```

### 3. Command Line

```bash
python predict.py my_waste_image.jpg
```

---

## 📊 Model Architecture

```
INPUT IMAGE (any size)
    ↓
PREPROCESSING (resize to 224×224, normalize)
    ↓
SQUEEZENET (pre-trained CNN)
    ↓
FEATURE VECTOR (512 dimensions)
    ↓
XGBOOST CLASSIFIER
    ↓
PREDICTION (9 classes with probabilities)
```

**Why SqueezeNet + XGBoost?**
- **SqueezeNet:** Lightweight CNN (~5MB) with good accuracy
- **XGBoost:** Fast, accurate gradient boosting for classification
- **Combined:** Best of both worlds - deep features + powerful classifier

---

## 🔧 Configuration

All settings are in `config.py`:

```python
# Dataset (automatically configured)
WASTE_CATEGORIES = [
    'Metal', 'Carton', 'Glass', 
    'Organic Waste', 'Other Plastics',
    'Paper and Cardboard', 'Plastic', 
    'Textiles', 'Wood'
]

# Training
BATCH_SIZE = 32          # Reduce to 16 if memory issues
IMAGE_SIZE = 224         # Required for SqueezeNet
VAL_SIZE = 0.1          # 10% validation split

# XGBoost
XGBOOST_PARAMS = {
    'n_estimators': 200,     # More = better (slower)
    'max_depth': 6,          # Tree depth
    'learning_rate': 0.1,    # Step size
}
```

---

## 📈 Improving Performance

### 1. More Training Time
Increase XGBoost iterations:
```python
XGBOOST_PARAMS = {
    'n_estimators': 500,  # Instead of 200
}
```

### 2. Deeper Trees
```python
XGBOOST_PARAMS = {
    'max_depth': 8,  # Instead of 6
}
```

### 3. Data Augmentation
Already included in training:
- Random rotation (±15°)
- Horizontal flipping
- Color jittering
- Random translation

### 4. Balance Dataset
Check class distribution:
```bash
python setup_dataset.py
```

If some classes have many more images, consider:
- Adding more images to underrepresented classes
- Using class weights in XGBoost

---

## 🐛 Troubleshooting

### Dataset Not Found
```
Error: No training data found!
```
**Solution:** Run `python setup_dataset.py` to verify, or re-download:
```bash
cd data
git clone https://github.com/cardstdani/WasteClassificationNeuralNetwork.git
```

### Out of Memory
```
CUDA out of memory / MemoryError
```
**Solution:** Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # or even 8
```

### Slow Training
**Problem:** Feature extraction taking too long

**Solutions:**
- Use GPU if available (10x faster)
- Reduce batch size won't help speed (only memory)
- Be patient - 5,078 images takes time on CPU

### Low Accuracy on Specific Class
**Problem:** One class performs poorly

**Solutions:**
1. Check confusion matrix: `python evaluate_model.py`
2. Look at misclassified examples
3. Verify images are correctly labeled in dataset
4. Consider if class is inherently difficult

---

## 📊 Monitoring Training

### During Training, You'll See:

```
[Step 3/6] Extracting features from training data...
Extracting features: 100%|████████| 143/143 [05:23<00:00,  2.27s/it]
Extracted features shape: (4571, 512)

[Step 5/6] Training XGBoost classifier...
[0]     validation_0-mlogloss:2.04567
[50]    validation_0-mlogloss:0.32145
[100]   validation_0-mlogloss:0.21234
[150]   validation_0-mlogloss:0.18456
[199]   validation_0-mlogloss:0.17234

Validation Accuracy: 0.9053 (90.53%)
```

### Output Files

After training, check:
- `models/` - Trained model files
- `results/confusion_matrix.png` - Visual confusion matrix
- `results/feature_importance.png` - Important features
- `results/metadata.json` - Training statistics

---

## 🌐 Deploying the Web App

### Local Network Access

1. Find your IP address:
   ```bash
   ipconfig
   ```

2. Look for "IPv4 Address" (e.g., 192.168.1.100)

3. Start the app:
   ```bash
   python app.py
   ```

4. Access from other devices:
   ```
   http://192.168.1.100:5000
   ```

### Cloud Deployment

Consider these platforms:
- **Heroku** (free tier available)
- **Google Cloud Run**
- **AWS Elastic Beanstalk**
- **Azure App Service**
- **Render** (free tier)

---

## 📚 Dataset Citation

If you use this dataset, please credit:

```
WasteClassificationNeuralNetwork Dataset
GitHub: cardstdani/WasteClassificationNeuralNetwork
URL: https://github.com/cardstdani/WasteClassificationNeuralNetwork
```

---

## ✅ Pre-Flight Checklist

Before starting, ensure:

- [ ] Python 3.8+ installed
- [ ] Git installed (for cloning dataset)
- [ ] At least 2GB free disk space
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Dataset cloned to: `data/WasteClassificationNeuralNetwork/`
- [ ] Dataset verified: `python setup_dataset.py` shows all 9 classes

Then you're ready to train! 🚀

---

## 🎉 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup dataset (automatic)
python setup_dataset.py

# 3. Train model (10-25 minutes)
python train_model.py

# 4. Launch web app
python app.py

# 5. Open browser
# http://localhost:5000
```

**That's it!** You now have a fully functional waste classification system with 9 categories.

---

## 💡 Tips for Success

1. **Be Patient:** First training takes time (10-25 min)
2. **Use GPU:** If available, training is 10x faster
3. **Check Results:** Review confusion matrix to understand model
4. **Test Thoroughly:** Try different waste items in web app
5. **Monitor Performance:** Check accuracy on validation set
6. **Iterate:** Adjust hyperparameters if needed

---

**Happy Classifying!** ♻️🌍
