"""
Training script for the defect detection model.
Includes training loop, validation, and checkpoint saving.

Usage:
    python train.py --category bottle --epochs 50 --batch-size 32
"""

import os
import argparse
import logging
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import DataManager, get_mvtec_categories
from model import DefectDetectionModel, save_model, load_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for model training and validation.
    """
    
    def __init__(self, model: DefectDetectionModel, device: str = 'cpu', 
                 checkpoint_dir: str = 'checkpoints'):
        """
        Args:
            model: Model to train
            device: Device to train on ('cpu' or 'cuda')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training set
            criterion: Loss function
            optimizer: Optimizer
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation set
            criterion: Loss function
        
        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             num_epochs: int = 50, learning_rate: float = 1e-3,
             freeze_backbone: bool = True, early_stopping_patience: int = 10) -> Dict:
        """
        Full training loop with validation and early stopping.
        
        Args:
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            freeze_backbone: Whether to freeze backbone initially
            early_stopping_patience: Epochs to wait before stopping
        
        Returns:
            Dictionary with training history
        """
        # Setup loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Freeze backbone if specified (transfer learning strategy)
        if freeze_backbone:
            self.model.freeze_backbone()
            logger.info("Backbone frozen for transfer learning")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'best_epoch': 0,
            'best_val_accuracy': 0.0
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader, criterion)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                history['best_val_accuracy'] = best_val_accuracy
                history['best_epoch'] = epoch + 1
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                save_model(self.model, checkpoint_path, optimizer, epoch+1, history)
                logger.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Adjust learning rate
            scheduler.step(val_loss)
            
            # Optionally unfreeze backbone after first few epochs
            if freeze_backbone and epoch == 5:
                self.model.unfreeze_backbone()
                logger.info("Backbone unfrozen for fine-tuning")
        
        logger.info(f"\nBest validation accuracy: {history['best_val_accuracy']:.2f}% (epoch {history['best_epoch']})")
        
        return history


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train defect detection model")
    parser.add_argument('--category', type=str, default='bottle',
                       choices=get_mvtec_categories(),
                       help='MVTec category to train on')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Root data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint save directory')
    
    args = parser.parse_args()
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Training on category: {args.category}")
    
    # Load data
    logger.info("Loading data...")
    data_manager = DataManager(data_dir=args.data_dir)
    dataloaders = data_manager.get_dataloaders(
        args.category,
        batch_size=args.batch_size,
        augment=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = DefectDetectionModel(num_classes=2, pretrained=True)
    
    # Train
    trainer = Trainer(model, device=args.device, checkpoint_dir=args.checkpoint_dir)
    history = trainer.train(
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
