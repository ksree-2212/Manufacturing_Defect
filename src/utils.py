"""
Utility functions for visualization, image preprocessing, and helper operations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from PIL import Image
import torch


def load_image(image_path: str, size: tuple = (224, 224)) -> np.ndarray:
    """
    Load and preprocess an image for model inference.
    
    Args:
        image_path: Path to the image file
        size: Target size for resizing (H, W)
    
    Returns:
        Preprocessed numpy array normalized to [0, 1]
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.Resampling.LANCZOS)
    return np.array(image) / 255.0


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy image array to PyTorch tensor with normalization.
    
    Args:
        image: numpy array of shape (H, W, 3)
    
    Returns:
        Normalized tensor of shape (1, 3, H, W)
    """
    # ImageNet normalization constants
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    # Normalize
    normalized = (image - IMAGENET_MEAN) / IMAGENET_STD
    
    # Convert to tensor and rearrange to (C, H, W)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
    
    # Add batch dimension
    return tensor.unsqueeze(0)


def generate_heatmap(image: np.ndarray, grad_cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        image: Original image as numpy array (H, W, 3) in [0, 1] range
        grad_cam: Grad-CAM activation map (H, W)
        alpha: Transparency of the overlay
    
    Returns:
        Overlayed image as numpy array
    """
    # Normalize grad_cam to [0, 1]
    grad_cam_normalized = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
    
    # Apply colormap (Jet)
    colormap = get_cmap('jet')
    heatmap = colormap(grad_cam_normalized)[:, :, :3]  # Remove alpha channel
    
    # Blend with original image
    overlay = alpha * heatmap + (1 - alpha) * image
    return np.clip(overlay, 0, 1)


def plot_result(image: np.ndarray, heatmap: np.ndarray, prediction: float, 
                confidence: float, title: str = "Defect Detection Result") -> plt.Figure:
    """
    Create a visualization plot of the image, heatmap, and prediction.
    
    Args:
        image: Original image (H, W, 3)
        heatmap: Grad-CAM heatmap (H, W)
        prediction: Binary prediction (0 = pass, 1 = defect)
        confidence: Confidence score (0-1)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmap overlay
    overlay = generate_heatmap(image, heatmap)
    axes[1].imshow(overlay)
    result_text = "DEFECT" if prediction > 0.5 else "PASS"
    axes[1].set_title(f"{result_text} (Confidence: {confidence:.2%})")
    axes[1].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    return fig


def save_figure(fig: plt.Figure, output_path: str) -> None:
    """Save a matplotlib figure to disk."""
    fig.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
