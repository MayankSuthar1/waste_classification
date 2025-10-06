"""
Prediction module for waste classification
"""
import os
import torch
import numpy as np
import xgboost as xgb
import joblib
from PIL import Image
import json

import config
from data_preprocessing import preprocess_single_image
from feature_extraction import SqueezeNetFeatureExtractor, extract_features_from_image


class WasteClassifier:
    """
    Complete waste classification pipeline
    """
    
    def __init__(self, 
                 squeezenet_path: str = config.SQUEEZENET_MODEL_PATH,
                 xgboost_path: str = config.XGBOOST_MODEL_PATH,
                 label_encoder_path: str = config.LABEL_ENCODER_PATH):
        """
        Initialize the classifier
        
        Args:
            squeezenet_path: Path to SqueezeNet model
            xgboost_path: Path to XGBoost model
            label_encoder_path: Path to label encoder
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load SqueezeNet feature extractor
        print("Loading SqueezeNet feature extractor...")
        self.feature_extractor = SqueezeNetFeatureExtractor(
            pretrained=False,
            version=config.SQUEEZENET_VERSION
        )
        self.feature_extractor.load_state_dict(
            torch.load(squeezenet_path, map_location=self.device)
        )
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Load XGBoost classifier
        print("Loading XGBoost classifier...")
        self.xgb_classifier = xgb.XGBClassifier()
        self.xgb_classifier.load_model(xgboost_path)
        
        # Load label encoder
        print("Loading label encoder...")
        self.label_encoder = joblib.load(label_encoder_path)
        
        # Load metadata if available
        metadata_path = os.path.join(config.MODEL_DIR, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Model trained with {self.metadata['val_accuracy']:.4f} validation accuracy")
        
        print("Classifier ready!")
        print(f"Classes: {self.label_encoder.classes_}")
    
    def predict(self, image_path: str, return_probabilities: bool = False):
        """
        Predict the class of a single image
        
        Args:
            image_path: Path to the image file
            return_probabilities: If True, return class probabilities
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        image_tensor = preprocess_single_image(image_path)
        
        # Extract features using SqueezeNet
        features = extract_features_from_image(
            self.feature_extractor,
            image_tensor,
            self.device
        )
        
        # Predict using XGBoost
        prediction = self.xgb_classifier.predict(features)[0]
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': int(prediction)
        }
        
        # Get probabilities if requested
        if return_probabilities:
            probabilities = self.xgb_classifier.predict_proba(features)[0]
            class_probabilities = {
                class_name: float(prob)
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            }
            result['probabilities'] = class_probabilities
            result['confidence'] = float(max(probabilities))
        
        return result
    
    def predict_batch(self, image_paths: list, return_probabilities: bool = False):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image file paths
            return_probabilities: If True, return class probabilities
        
        Returns:
            List of prediction results
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, return_probabilities)
            result['image_path'] = img_path
            results.append(result)
        
        return results


def predict_image(image_path: str):
    """
    Convenient function to predict a single image
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Prediction result dictionary
    """
    classifier = WasteClassifier()
    result = classifier.predict(image_path, return_probabilities=True)
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("\nExample:")
        print("  python predict.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"Classifying image: {image_path}")
    print("-" * 60)
    
    result = predict_image(image_path)
    
    print(f"\nPredicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nClass Probabilities:")
    for class_name, prob in sorted(result['probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True):
        print(f"  {class_name:15s}: {prob:.4f} ({prob*100:.2f}%)")
