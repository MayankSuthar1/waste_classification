"""
Configuration file for waste classification project
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
# Using WasteClassificationNeuralNetwork dataset
TRAIN_DIR = os.path.join(DATA_DIR, 'WasteImagesDataset')
TEST_DIR = os.path.join(DATA_DIR, 'WasteImagesDataset')  # Will split this
VAL_DIR = os.path.join(DATA_DIR, 'val')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
EMBEDDING_LAYER = 'features.12'  # SqueezeNet layer for feature extraction
NUM_WORKERS = 2

# SqueezeNet parameters
SQUEEZENET_VERSION = '1_1'  # or '1_0'
PRETRAINED = True

# XGBoost parameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

# Training parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Waste categories (update based on your dataset)
# Updated to match WasteClassificationNeuralNetwork dataset
WASTE_CATEGORIES = [
    'Metal',
    'Carton',
    'Glass',
    'Organic Waste',
    'Other Plastics',
    'Paper and Cardboard',
    'Plastic',
    'Textiles',
    'Wood'
]

# Model paths
SQUEEZENET_MODEL_PATH = os.path.join(MODEL_DIR, 'squeezenet_embeddings.pth')
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_classifier.json')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
