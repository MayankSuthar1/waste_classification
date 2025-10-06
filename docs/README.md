# Waste Classification using SqueezeNet + XGBoost

An AI-powered waste classification system that uses SqueezeNet for feature extraction and XGBoost for classification. Includes a Flask web application for easy image upload and prediction.

## 🎯 Features

- **Deep Learning**: Uses pre-trained SqueezeNet for efficient feature extraction
- **Machine Learning**: XGBoost classifier for accurate multi-class classification
- **Web Interface**: User-friendly Flask web application
- **Real-time Predictions**: Upload images and get instant classification results
- **Detailed Analytics**: View confidence scores and probability distributions
- **Visualization**: Confusion matrix and feature importance plots

## 🏗️ Project Structure

```
waste_classification/
├── config.py                  # Configuration settings
├── data_preprocessing.py      # Image preprocessing and data loading
├── feature_extraction.py      # SqueezeNet feature extractor
├── train_model.py            # Model training script
├── predict.py                # Prediction module
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
├── data/                     # Dataset directory
│   ├── train/               # Training images
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   ├── test/                # Test images
│   └── val/                 # Validation images
├── models/                   # Saved models
├── results/                  # Training results and plots
├── templates/               # HTML templates
│   ├── index.html
│   └── about.html
└── static/                  # Static files
    └── uploads/            # Uploaded images
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- XGBoost
- Flask
- OpenCV
- NumPy
- Pillow
- scikit-learn

## 🚀 Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Prepare Your Dataset

Organize your dataset in the following structure:

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

Update the `WASTE_CATEGORIES` in `config.py` to match your classes.

## 🎓 Training the Model

1. **Configure settings** (optional)
   Edit `config.py` to adjust:
   - Image size
   - Batch size
   - XGBoost parameters
   - Waste categories

2. **Run training**
```bash
python train_model.py
```

The training process will:
- Load and preprocess images
- Extract features using SqueezeNet
- Train XGBoost classifier
- Generate evaluation metrics
- Save trained models to `models/` directory
- Create visualizations in `results/` directory

## 🔮 Making Predictions

### Command Line
```bash
python predict.py path/to/image.jpg
```

### Python API
```python
from predict import WasteClassifier

classifier = WasteClassifier()
result = classifier.predict('image.jpg', return_probabilities=True)

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}")
print(f"Probabilities: {result['probabilities']}")
```

## 🌐 Web Application

1. **Start the Flask server**
```bash
python app.py
```

2. **Access the application**
   Open your browser and navigate to: `http://localhost:5000`

3. **Use the interface**
   - Upload an image by clicking or dragging
   - View prediction results with confidence scores
   - See probability distribution across all classes

## 📱 API Endpoints

- `GET /` - Home page
- `POST /upload` - Upload image for classification
- `GET /about` - About page
- `GET /api/health` - Health check and model status
- `GET /api/classes` - Get available waste classes

## 🧪 Model Architecture

### Feature Extraction (SqueezeNet)
- Pre-trained on ImageNet
- Fire modules with squeeze and expand layers
- 512-dimensional feature embeddings
- Lightweight: ~5MB model size

### Classification (XGBoost)
- Gradient boosting decision trees
- Multi-class classification
- Hyperparameter optimization
- Feature importance analysis

## 📈 Performance Metrics

The model provides:
- Accuracy scores
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Feature importance plots
- Per-class probability distributions

## 🔧 Configuration

Key parameters in `config.py`:

```python
# Image settings
IMAGE_SIZE = 224
BATCH_SIZE = 32

# SqueezeNet settings
SQUEEZENET_VERSION = '1_1'
PRETRAINED = True

# XGBoost settings
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    ...
}

# Waste categories
WASTE_CATEGORIES = [
    'cardboard', 'glass', 'metal',
    'paper', 'plastic', 'trash'
]
```

## 🎨 Customization

### Adding New Waste Categories
1. Add images to `data/train/new_category/`
2. Update `WASTE_CATEGORIES` in `config.py`
3. Retrain the model

### Adjusting Model Performance
- Increase `n_estimators` for better accuracy
- Adjust `max_depth` to control overfitting
- Modify `learning_rate` for training stability

## 🐛 Troubleshooting

**Model not found error:**
- Make sure you've trained the model first with `python train_model.py`

**Low accuracy:**
- Ensure sufficient training data (100+ images per class)
- Check image quality and variety
- Adjust XGBoost hyperparameters

**Memory issues:**
- Reduce `BATCH_SIZE` in config.py
- Use smaller images (reduce `IMAGE_SIZE`)

## 📚 References

- SqueezeNet paper: [arXiv:1602.07360](https://arxiv.org/abs/1602.07360)
- XGBoost documentation: [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)
- Based on research methodology from Paper 37

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

CV Semester 7 Project - Waste Classification

---

**Happy Classifying! ♻️**
