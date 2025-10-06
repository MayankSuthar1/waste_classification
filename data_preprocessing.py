"""
Data preprocessing module for waste classification
Handles image loading, preprocessing, and augmentation
"""
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import config


class WasteDataset(Dataset):
    """Custom dataset for waste images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def get_transforms(train: bool = True):
    """
    Get image preprocessing transforms
    
    Args:
        train: If True, includes data augmentation
    
    Returns:
        torchvision transforms
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def load_dataset_from_directory(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load dataset from directory structure
    Expected structure: data_dir/class_name/image.jpg
    
    Args:
        data_dir: Root directory containing class folders
    
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} does not exist")
        return image_paths, labels
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(class_name)
    
    print(f"Loaded {len(image_paths)} images from {data_dir}")
    return image_paths, labels


def preprocess_single_image(image_path: str) -> torch.Tensor:
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed image tensor
    """
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension


def create_data_loader(image_paths: List[str], 
                       labels: List[int], 
                       batch_size: int = config.BATCH_SIZE,
                       shuffle: bool = True,
                       train: bool = True) -> DataLoader:
    """
    Create a DataLoader for the dataset
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        train: Whether this is training data (affects augmentation)
    
    Returns:
        PyTorch DataLoader
    """
    transform = get_transforms(train=train)
    dataset = WasteDataset(image_paths, labels, transform=transform)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return data_loader


if __name__ == "__main__":
    # Test the preprocessing module
    print("Testing data preprocessing...")
    
    # Test loading dataset
    train_images, train_labels = load_dataset_from_directory(config.TRAIN_DIR)
    print(f"Found {len(train_images)} training images")
    
    if len(train_images) > 0:
        # Test transforms
        transform = get_transforms(train=True)
        sample_image = Image.open(train_images[0]).convert('RGB')
        transformed = transform(sample_image)
        print(f"Transformed image shape: {transformed.shape}")
