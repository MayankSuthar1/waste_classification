# 🚀 QUICK START GUIDE - WASTE CLASSIFICATION

## Welcome! 👋

This guide will help you get your waste classification system up and running in minutes.

## 📦 Step 1: Install Dependencies

Open your terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (Deep Learning)
- XGBoost (Machine Learning)
- Flask (Web Framework)
- OpenCV, Pillow (Image Processing)
- And other required libraries

**Note:** Installation may take 5-10 minutes depending on your internet speed.

## 📁 Step 2: Prepare Your Dataset

### Option A: Use an Existing Dataset

Download a waste classification dataset such as:
- [TrashNet Dataset](https://github.com/garythung/trashnet)
- [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- Any similar waste/recycling dataset

### Option B: Use Your Own Images

Create your own dataset by taking photos of waste items.

### Organize Your Data

Place images in this structure:

```
data/
└── train/
    ├── cardboard/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

**Requirements:**
- Minimum: 50-100 images per category
- Recommended: 200+ images per category for better accuracy
- Supported formats: JPG, PNG, JPEG, BMP

**Tips:**
- Use diverse images (different angles, lighting, backgrounds)
- Include clear, focused images
- Ensure images are properly labeled in correct folders

## 🔧 Step 3: Configure Settings (Optional)

Edit `config.py` if needed:

```python
# Update waste categories if different
WASTE_CATEGORIES = [
    'cardboard', 'glass', 'metal',
    'paper', 'plastic', 'trash'
]

# Adjust training parameters
BATCH_SIZE = 32  # Reduce if you have memory issues
IMAGE_SIZE = 224  # Keep at 224 for SqueezeNet

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 200,  # More trees = better accuracy (slower)
    'max_depth': 6,       # Deeper trees = more complex model
    'learning_rate': 0.1  # Lower = slower learning (more stable)
}
```

## 🎓 Step 4: Train Your Model

Run the training script:

```bash
python train_model.py
```

### What Happens During Training:

1. **Data Loading** (~30 sec)
   - Loads images from `data/train/`
   - Splits into training and validation sets
   
2. **Feature Extraction** (2-10 min depending on dataset size)
   - Uses SqueezeNet to extract features from images
   - GPU accelerated if CUDA available
   
3. **Model Training** (1-5 min)
   - XGBoost learns to classify based on features
   - Shows progress and evaluation metrics
   
4. **Evaluation** (~1 min)
   - Tests model on validation set
   - Generates confusion matrix and metrics
   
5. **Saving Models**
   - Saves trained models to `models/` directory
   - Saves results to `results/` directory

### Expected Output:

```
[Step 6/6] Evaluating model...

Training Accuracy: 0.9850
Validation Accuracy: 0.9200

Classification Report:
              precision    recall  f1-score   support

   cardboard       0.91      0.94      0.93        35
       glass       0.93      0.89      0.91        28
       metal       0.95      0.92      0.93        25
       paper       0.90      0.93      0.91        30
     plastic       0.92      0.91      0.92        33
       trash       0.91      0.93      0.92        29

TRAINING COMPLETED SUCCESSFULLY!
```

## 🔮 Step 5: Test Predictions (Optional)

Test on a single image:

```bash
python predict.py path/to/your/test_image.jpg
```

Example output:
```
Predicted Class: plastic
Confidence: 0.9567

Class Probabilities:
  plastic        : 0.9567 (95.67%)
  cardboard      : 0.0231 (2.31%)
  paper          : 0.0102 (1.02%)
  glass          : 0.0065 (0.65%)
  metal          : 0.0025 (0.25%)
  trash          : 0.0010 (0.10%)
```

## 🌐 Step 6: Launch Web Application

Start the Flask server:

```bash
python app.py
```

You should see:
```
======================================================
WASTE CLASSIFICATION WEB APPLICATION
======================================================

Starting server...
Access the application at: http://localhost:5000
======================================================
```

### Using the Web App:

1. **Open Browser**
   - Navigate to `http://localhost:5000`

2. **Upload Image**
   - Click the upload area or drag & drop an image
   - Supported formats: JPG, PNG, BMP (up to 16MB)

3. **View Results**
   - See predicted waste category
   - View confidence score
   - Check probability distribution for all classes

4. **Classify More**
   - Click "Classify Another Image" to test more images

## 📊 Step 7: Evaluate Model (Optional)

If you have a separate test dataset:

1. Add test images to `data/test/` (same structure as train)

2. Run evaluation:
```bash
python evaluate_model.py
```

This generates:
- Overall accuracy metrics
- Per-class performance
- Confusion matrix
- Detailed evaluation report

## 🐛 Troubleshooting

### "No module named 'torch'"
```bash
pip install torch torchvision
```

### "CUDA out of memory"
In `config.py`, reduce:
```python
BATCH_SIZE = 16  # or even 8
```

### "No training data found"
- Check that images are in `data/train/class_name/`
- Verify folder names match `WASTE_CATEGORIES` in config.py

### Low Accuracy
- Add more training images (200+ per class recommended)
- Ensure images are clear and properly labeled
- Try adjusting XGBoost parameters in config.py

### Web app shows "Model not loaded"
- Make sure you've run `train_model.py` first
- Check that model files exist in `models/` directory

## 🎯 Quick Test Checklist

Before deploying, verify:

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Training data organized in `data/train/`
- [ ] Model trained successfully (`python train_model.py`)
- [ ] Test prediction works (`python predict.py test_image.jpg`)
- [ ] Web app launches (`python app.py`)
- [ ] Can upload and classify images in browser

## 📈 Improving Model Performance

### Get More Data
- Aim for 500+ images per class
- Include various lighting conditions
- Add images with different backgrounds

### Data Augmentation
Already included in training:
- Random rotation (±15°)
- Horizontal flipping
- Color jittering
- Random translation

### Hyperparameter Tuning
In `config.py`, experiment with:
```python
XGBOOST_PARAMS = {
    'n_estimators': 300,    # Try 200-500
    'max_depth': 8,         # Try 4-10
    'learning_rate': 0.05,  # Try 0.01-0.2
}
```

### Use Better Hardware
- GPU significantly speeds up feature extraction
- More RAM allows larger batch sizes

## 🎨 Customization Ideas

### Add New Waste Categories
1. Create new folder in `data/train/new_category/`
2. Add images to the folder
3. Update `WASTE_CATEGORIES` in `config.py`
4. Retrain: `python train_model.py`

### Change Web UI Theme
Edit `templates/index.html`:
- Modify color gradients in CSS
- Change layout structure
- Add custom logos/images

### Add Features
- Save prediction history
- Batch upload multiple images
- Export results to CSV
- Add user authentication
- Create mobile app version

## 🚀 Deployment

### Local Network Access
Run with:
```bash
python app.py
```
Access from other devices using your computer's IP: `http://192.168.1.X:5000`

### Cloud Deployment
Consider platforms:
- Heroku (free tier available)
- Google Cloud Run
- AWS Elastic Beanstalk
- Azure App Service

## 📚 Additional Resources

- [SqueezeNet Paper](https://arxiv.org/abs/1602.07360)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## 💡 Pro Tips

1. **Start Small**: Test with a few images per class first
2. **Monitor Performance**: Check confusion matrix to see which classes are confused
3. **Balance Dataset**: Try to have similar number of images per class
4. **Clean Data**: Remove blurry or mislabeled images
5. **Test Regularly**: Use evaluation script to track improvements

## 🎉 You're All Set!

Your waste classification system is ready to use. Start classifying waste and help the environment! ♻️

---

**Need Help?**
- Check README.md for detailed documentation
- Run `python check_setup.py` to verify installation
- Review error messages carefully - they usually indicate the issue

**Happy Classifying!** 🌍
