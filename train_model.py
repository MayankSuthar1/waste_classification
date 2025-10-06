"""
Main training script for waste classification
Uses SqueezeNet for feature extraction and XGBoost for classification
"""
import os
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

import config
from data_preprocessing import load_dataset_from_directory, create_data_loader
from feature_extraction import SqueezeNetFeatureExtractor, extract_features_from_loader


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_feature_importance(model, save_path, top_n=20):
    """Plot feature importance from XGBoost"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Feature importance plot saved to {save_path}")
    plt.close()


def train_model():
    """Main training function"""
    print("=" * 60)
    print("WASTE CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Step 1: Load dataset
    print("\n[Step 1/6] Loading dataset...")
    train_images, train_labels_str = load_dataset_from_directory(config.TRAIN_DIR)
    
    if len(train_images) == 0:
        print("\n" + "="*60)
        print("WARNING: No training data found!")
        print("="*60)
        print(f"\nPlease organize your dataset in the following structure:")
        print(f"{config.TRAIN_DIR}/")
        print(f"  ├── cardboard/")
        print(f"  │   ├── image1.jpg")
        print(f"  │   └── image2.jpg")
        print(f"  ├── glass/")
        print(f"  ├── metal/")
        print(f"  ├── paper/")
        print(f"  ├── plastic/")
        print(f"  └── trash/")
        print("\nUpdate the WASTE_CATEGORIES in config.py to match your classes.")
        return
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels_str)
    
    print(f"Classes found: {label_encoder.classes_}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Split into train and validation
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        train_images, train_labels, 
        test_size=config.VAL_SIZE, 
        random_state=config.RANDOM_SEED,
        stratify=train_labels
    )
    
    print(f"Training samples: {len(X_train_paths)}")
    print(f"Validation samples: {len(X_val_paths)}")
    
    # Step 2: Initialize SqueezeNet feature extractor
    print("\n[Step 2/6] Initializing SqueezeNet feature extractor...")
    feature_extractor = SqueezeNetFeatureExtractor(
        pretrained=config.PRETRAINED,
        version=config.SQUEEZENET_VERSION
    )
    feature_extractor.to(device)
    
    # Step 3: Extract features from training data
    print("\n[Step 3/6] Extracting features from training data...")
    train_loader = create_data_loader(X_train_paths, y_train, 
                                     batch_size=config.BATCH_SIZE,
                                     shuffle=False, train=False)
    X_train_features, y_train = extract_features_from_loader(
        feature_extractor, train_loader, device
    )
    
    # Step 4: Extract features from validation data
    print("\n[Step 4/6] Extracting features from validation data...")
    val_loader = create_data_loader(X_val_paths, y_val,
                                   batch_size=config.BATCH_SIZE,
                                   shuffle=False, train=False)
    X_val_features, y_val = extract_features_from_loader(
        feature_extractor, val_loader, device
    )
    
    # Step 5: Train XGBoost classifier
    print("\n[Step 5/6] Training XGBoost classifier...")
    print(f"XGBoost parameters: {config.XGBOOST_PARAMS}")
    
    xgb_classifier = xgb.XGBClassifier(**config.XGBOOST_PARAMS)
    
    # Train with evaluation set
    xgb_classifier.fit(
        X_train_features, y_train,
        eval_set=[(X_val_features, y_val)],
        verbose=True
    )
    
    # Step 6: Evaluate model
    print("\n[Step 6/6] Evaluating model...")
    
    # Training accuracy
    y_train_pred = xgb_classifier.predict(X_train_features)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    
    # Validation accuracy
    y_val_pred = xgb_classifier.predict(X_val_features)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, 
                               target_names=label_encoder.classes_))
    
    # Save confusion matrix
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_val, y_val_pred, label_encoder.classes_, cm_path)
    
    # Save feature importance
    fi_path = os.path.join(config.RESULTS_DIR, 'feature_importance.png')
    plot_feature_importance(xgb_classifier, fi_path)
    
    # Step 7: Save models
    print("\n[Step 7/7] Saving models...")
    
    # Save SqueezeNet feature extractor
    torch.save(feature_extractor.state_dict(), config.SQUEEZENET_MODEL_PATH)
    print(f"SqueezeNet saved to {config.SQUEEZENET_MODEL_PATH}")
    
    # Save XGBoost classifier
    xgb_classifier.save_model(config.XGBOOST_MODEL_PATH)
    print(f"XGBoost saved to {config.XGBOOST_MODEL_PATH}")
    
    # Save label encoder
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)
    print(f"Label encoder saved to {config.LABEL_ENCODER_PATH}")
    
    # Save training metadata
    metadata = {
        'classes': label_encoder.classes_.tolist(),
        'num_classes': len(label_encoder.classes_),
        'train_samples': len(X_train_paths),
        'val_samples': len(X_val_paths),
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'squeezenet_version': config.SQUEEZENET_VERSION,
        'embedding_dim': feature_extractor.get_embedding_dim(),
        'xgboost_params': config.XGBOOST_PARAMS
    }
    
    metadata_path = os.path.join(config.MODEL_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nModels saved in: {config.MODEL_DIR}")
    print(f"Results saved in: {config.RESULTS_DIR}")
    print(f"\nFinal Validation Accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    train_model()
