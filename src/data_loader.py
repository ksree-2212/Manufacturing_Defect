"""
Data loading and preprocessing for the MVTec AD defect detection dataset.
Handles dataset loading, splitting, augmentation, and batching.
"""

import os
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class MVTecADDataset(Dataset):
    """
    Custom PyTorch Dataset for MVTec AD anomaly detection dataset.
    
    Structure expected:
    data/
    ├── category_name/
    │   ├── train/
    │   │   ├── good/
    │   │   └── defective/
    │   └── test/
    │       ├── good/
    │       └── various_defect_types/
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, img_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            image_paths: List of paths to images
            labels: List of binary labels (0 = normal, 1 = defective)
            transform: Optional torchvision transforms
            img_size: Target image size (H, W)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a single image and its label.
        
        Returns:
            (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.img_size, Image.Resampling.LANCZOS)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image)
        
        return image, label


class DataManager:
    """
    Manages MVTec AD dataset loading, preprocessing, and DataLoader creation.
    """
    
    def __init__(self, data_dir: str = "data", img_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            data_dir: Root directory containing MVTec AD dataset
            img_size: Target image size
        """
        self.data_dir = data_dir
        self.img_size = img_size
    
    def load_category(self, category: str) -> Tuple[List[str], List[int]]:
        """
        Load all images and labels for a specific MVTec category.
        
        Args:
            category: Category name (e.g., 'bottle', 'cable', 'carpet')
        
        Returns:
            (image_paths, labels) where labels are binary (0=normal, 1=defect)
        """
        category_dir = os.path.join(self.data_dir, category)
        
        if not os.path.exists(category_dir):
            raise FileNotFoundError(f"Category '{category}' not found at {category_dir}")
        
        image_paths = []
        labels = []
        
        # Load normal images (label=0)
        normal_dir = os.path.join(category_dir, 'train', 'good')
        if os.path.exists(normal_dir):
            for fname in os.listdir(normal_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(normal_dir, fname))
                    labels.append(0)
        
        # Load defective images (label=1)
        test_dir = os.path.join(category_dir, 'test')
        if os.path.exists(test_dir):
            for defect_type in os.listdir(test_dir):
                defect_path = os.path.join(test_dir, defect_type)
                if os.path.isdir(defect_path) and defect_type != 'good':
                    for fname in os.listdir(defect_path):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(os.path.join(defect_path, fname))
                            labels.append(1)
        
        return image_paths, labels
    
    def split_data(self, image_paths: List[str], labels: List[int], 
                   val_ratio: float = 0.1, test_ratio: float = 0.1, 
                   random_state: int = 42) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys, each containing (paths, labels)
        """
        # First, split into train and temp (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels,
            test_size=val_ratio + test_ratio,
            random_state=random_state,
            stratify=labels
        )
        
        # Then split temp into val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=1 - val_size,
            random_state=random_state,
            stratify=temp_labels
        )
        
        return {
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels),
            'test': (test_paths, test_labels)
        }
    
    def get_dataloaders(self, category: str, batch_size: int = 32, 
                        num_workers: int = 0, augment: bool = True) -> Dict[str, DataLoader]:
        """
        Create DataLoader objects for train, validation, and test sets.
        
        Args:
            category: MVTec category name
            batch_size: Batch size for DataLoader
            num_workers: Number of worker threads
            augment: Whether to apply data augmentation to training set
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoader objects
        """
        # Load and split data
        image_paths, labels = self.load_category(category)
        splits = self.split_data(image_paths, labels)
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) if augment else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Create datasets
        train_paths, train_labels = splits['train']
        val_paths, val_labels = splits['val']
        test_paths, test_labels = splits['test']
        
        train_dataset = MVTecADDataset(train_paths, train_labels, train_transform, self.img_size)
        val_dataset = MVTecADDataset(val_paths, val_labels, test_transform, self.img_size)
        test_dataset = MVTecADDataset(test_paths, test_labels, test_transform, self.img_size)
        
        # Create dataloaders
        return {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }


def get_mvtec_categories() -> List[str]:
    """
    Return list of available MVTec AD categories.
    """
    return [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
