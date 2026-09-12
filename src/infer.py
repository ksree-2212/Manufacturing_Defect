"""
Inference module with Grad-CAM visualization for explainability.
Provides functions to run predictions and generate visual explanations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from PIL import Image

from model import load_model, DefectDetectionModel
from utils import load_image, image_to_tensor, generate_heatmap, plot_result


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for visualization.
    Highlights regions in the image that influence the model's predictions.
    """
    
    def __init__(self, model: DefectDetectionModel, target_layer: str = 'layer4'):
        """
        Args:
            model: The model to visualize
            target_layer: Name of the layer to get gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get the target layer
        target = dict(self.model.backbone.named_modules())[self.target_layer]
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input.
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            class_idx: Target class index (if None, uses predicted class)
        
        Returns:
            Grad-CAM heatmap of shape (H, W)
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Determine target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted sum of activations
        cam = torch.sum(weights[:, None, None] * activations, dim=0)
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        input_height, input_width = input_tensor.shape[-2:]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(input_height, input_width),
            mode='bilinear',
            align_corners=False
        )[0, 0]
        
        return cam.cpu().numpy()


class Predictor:
    """
    Main inference class for defect detection.
    Combines model prediction with Grad-CAM visualization.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.model = load_model(checkpoint_path, device=device)
        self.grad_cam = GradCAM(self.model)
    
    def predict(self, image_path: str, return_heatmap: bool = True) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to image file
            return_heatmap: Whether to generate Grad-CAM heatmap
        
        Returns:
            (prediction, confidence, original_image, heatmap)
            - prediction: 0 (pass) or 1 (defect)
            - confidence: Probability of the predicted class (0-1)
            - original_image: Original image as numpy array
            - heatmap: Grad-CAM heatmap (or None if return_heatmap=False)
        """
        # Load and preprocess image
        original_image = load_image(image_path)
        input_tensor = image_to_tensor(original_image)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
        
        # Get prediction and confidence
        pred_class = output.argmax(dim=1).item()
        confidence = probabilities[0, pred_class].item()
        
        # Generate Grad-CAM if requested
        heatmap = None
        if return_heatmap:
            heatmap = self.grad_cam.generate(input_tensor, class_idx=pred_class)
        
        return pred_class, confidence, original_image, heatmap
    
    def batch_predict(self, image_paths: list) -> list:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            List of (prediction, confidence, image, heatmap) tuples
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results


def run_inference_demo(checkpoint_path: str, image_path: str, device: str = 'cpu'):
    """
    Demonstration function for running inference and displaying results.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        image_path: Path to image to test
        device: Device to use
    """
    print(f"Loading model from {checkpoint_path}...")
    predictor = Predictor(checkpoint_path, device=device)
    
    print(f"Running inference on {image_path}...")
    pred_class, confidence, image, heatmap = predictor.predict(image_path)
    
    # Display results
    result_text = "DEFECT DETECTED" if pred_class == 1 else "PASS (No defect)"
    print(f"\nResult: {result_text}")
    print(f"Confidence: {confidence:.2%}")
    
    # Create and display visualization
    if heatmap is not None:
        fig = plot_result(image, heatmap, pred_class, confidence, 
                         title=f"Defect Detection - {result_text}")
        fig.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    
    args = parser.parse_args()
    run_inference_demo(args.checkpoint, args.image, args.device)
