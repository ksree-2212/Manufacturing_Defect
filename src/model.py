"""
Model architecture for binary defect classification.
Uses transfer learning with pretrained ResNet18 backbone.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


class DefectDetectionModel(nn.Module):
    """
    Binary classification model for defect detection.
    Uses ResNet18 pretrained on ImageNet with a custom classification head.
    
    Architecture:
    - ResNet18 backbone (pretrained on ImageNet)
    - Global Average Pooling
    - Fully connected layers with dropout
    - Binary output (0=pass, 1=defect)
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.5):
        """
        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to load ImageNet pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super(DefectDetectionModel, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the original classification head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone (before classification head).
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor from backbone
        """
        return self.backbone(x)
    
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (for transfer learning)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone_except_last_layer(self) -> None:
        """Freeze backbone except the last ResNet layer."""
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False


def load_model(checkpoint_path: str, device: str = 'cpu') -> DefectDetectionModel:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved checkpoint
        device: Device to load model to ('cpu' or 'cuda')
    
    Returns:
        Loaded model in evaluation mode
    """
    model = DefectDetectionModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full model saves and state_dict saves
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def save_model(model: nn.Module, checkpoint_path: str, 
               optimizer: torch.optim.Optimizer = None, 
               epoch: int = None, metrics: dict = None) -> None:
    """
    Save model checkpoint with optional training metadata.
    
    Args:
        model: Model to save
        checkpoint_path: Path where to save the checkpoint
        optimizer: Optional optimizer to save state
        epoch: Optional current epoch number
        metrics: Optional dictionary of training metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, checkpoint_path)


def create_model(num_classes: int = 2, pretrained: bool = True) -> DefectDetectionModel:
    """
    Factory function to create a new model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        DefectDetectionModel instance
    """
    return DefectDetectionModel(num_classes=num_classes, pretrained=pretrained)
