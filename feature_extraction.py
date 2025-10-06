"""
Feature extraction module using SqueezeNet
Extracts embeddings from images using pre-trained SqueezeNet
"""
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple
import config


class SqueezeNetFeatureExtractor(nn.Module):
    """
    SqueezeNet-based feature extractor for image embeddings
    """
    
    def __init__(self, pretrained: bool = True, version: str = '1_1'):
        """
        Initialize SqueezeNet feature extractor
        
        Args:
            pretrained: Whether to use pre-trained weights
            version: SqueezeNet version ('1_0' or '1_1')
        """
        super(SqueezeNetFeatureExtractor, self).__init__()
        
        # Load SqueezeNet model
        if version == '1_0':
            self.model = models.squeezenet1_0(pretrained=pretrained)
        else:
            self.model = models.squeezenet1_1(pretrained=pretrained)
        
        # Remove the classifier to use as feature extractor
        self.features = self.model.features
        
        # Add adaptive pooling to get fixed-size output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set to evaluation mode
        self.eval()
        
        print(f"SqueezeNet {version} feature extractor initialized")
        print(f"Pretrained: {pretrained}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features
        
        Args:
            x: Input image tensor (batch_size, 3, H, W)
        
        Returns:
            Feature embeddings (batch_size, feature_dim)
        """
        # Extract features
        x = self.features(x)
        
        # Apply adaptive pooling
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the output embeddings"""
        # SqueezeNet 1.0 final conv: 512 channels
        # SqueezeNet 1.1 final conv: 512 channels
        return 512


def extract_features_from_loader(model: SqueezeNetFeatureExtractor, 
                                 data_loader: DataLoader,
                                 device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from all images in a DataLoader
    
    Args:
        model: SqueezeNet feature extractor
        data_loader: DataLoader containing images
        device: Device to run the model on
    
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    model.eval()
    model.to(device)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting features"):
            # Move to device
            images = images.to(device)
            
            # Extract features
            features = model(images)
            
            # Move to CPU and convert to numpy
            features = features.cpu().numpy()
            labels = labels.numpy()
            
            all_features.append(features)
            all_labels.append(labels)
    
    # Concatenate all batches
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    print(f"Extracted features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    return all_features, all_labels


def extract_features_from_image(model: SqueezeNetFeatureExtractor,
                                image_tensor: torch.Tensor,
                                device: torch.device) -> np.ndarray:
    """
    Extract features from a single image
    
    Args:
        model: SqueezeNet feature extractor
        image_tensor: Preprocessed image tensor (1, 3, H, W)
        device: Device to run the model on
    
    Returns:
        Feature vector as numpy array
    """
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        features = model(image_tensor)
        features = features.cpu().numpy()
    
    return features


def save_feature_extractor(model: SqueezeNetFeatureExtractor, path: str):
    """Save the feature extractor model"""
    torch.save(model.state_dict(), path)
    print(f"Feature extractor saved to {path}")


def load_feature_extractor(path: str, version: str = '1_1', device: torch.device = None) -> SqueezeNetFeatureExtractor:
    """Load the feature extractor model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SqueezeNetFeatureExtractor(pretrained=False, version=version)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Feature extractor loaded from {path}")
    
    return model


if __name__ == "__main__":
    # Test the feature extraction module
    print("Testing feature extraction...")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SqueezeNetFeatureExtractor(
        pretrained=config.PRETRAINED,
        version=config.SQUEEZENET_VERSION
    )
    
    # Test with dummy data
    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    features = extract_features_from_image(model, dummy_input, device)
    print(f"Feature shape: {features.shape}")
    print(f"Embedding dimension: {model.get_embedding_dim()}")
