"""
Streamlit web application for manufacturing defect detection.
Upload an image of a manufactured part, and the model will detect defects
and highlight suspicious regions using Grad-CAM visualization.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from infer import Predictor
from utils import plot_result, generate_heatmap
import matplotlib.pyplot as plt

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Manufacturing Defect Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Manufacturing Defect Detection MVP")
st.markdown("""
**AI-powered visual inspection for small manufacturers**

Upload a photo of your manufactured part, and our deep learning model will:
1. Analyze the image for surface defects
2. Return a **PASS** or **DEFECT** result
3. Highlight suspicious regions using AI explainability techniques

This MVP demonstrates high-accuracy defect detection without expensive enterprise hardware.
""")

# ============================================================================
# Sidebar Configuration
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# Model selection
checkpoint_file = "checkpoints/best_model.pt"
model_available = os.path.exists(checkpoint_file)

if not model_available:
    st.sidebar.warning(
        "⚠️ No trained model found at `checkpoints/best_model.pt`\n\n"
        "Please run the training script first:\n"
        "`python src/train.py --category bottle --epochs 50`"
    )
    st.stop()

# Device selection
device_option = st.sidebar.radio(
    "Select device for inference:",
    options=["CPU", "GPU (CUDA)"],
    index=0
)
device = "cuda" if device_option == "GPU (CUDA)" else "cpu"

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.5,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Predictions below this threshold will be less conclusive"
)

# Load model (cached for efficiency)
@st.cache_resource
def load_predictor(checkpoint, device):
    """Load predictor model (cached)."""
    return Predictor(checkpoint, device=device)

try:
    predictor = load_predictor(checkpoint_file, device)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ============================================================================
# Main App Interface
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("📊 About")
st.sidebar.markdown("""
**Technology Stack:**
- **Model**: ResNet18 (Transfer Learning)
- **Framework**: PyTorch
- **Explainability**: Grad-CAM
- **UI**: Streamlit
- **Dataset**: MVTec AD

**Use Cases:**
- Surface scratch detection
- Crack and dent identification
- Misalignment detection
- Quality assurance automation
""")

# ============================================================================
# File Upload & Inference
# ============================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image of a manufactured part",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_image_path = "temp_upload.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with col2:
        st.subheader("🖼️ Uploaded Image")
        st.image(uploaded_file, use_column_width=True, caption="Input Image")
    
    # ====================================================================
    # Run Inference
    # ====================================================================
    st.subheader("🚀 Inference Results")
    
    with st.spinner("Analyzing image..."):
        try:
            pred_class, confidence, image, heatmap = predictor.predict(
                temp_image_path,
                return_heatmap=True
            )
        except Exception as e:
            st.error(f"Error during inference: {e}")
            st.stop()
    
    # Display Results
    st.markdown("---")
    
    # Result Badge
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if pred_class == 1:
            st.error("🚨 DEFECT DETECTED")
        else:
            st.success("✅ PASS - No Defect")
    
    with col2:
        st.metric("Confidence Score", f"{confidence:.1%}")
    
    with col3:
        if confidence < confidence_threshold and pred_class == 1:
            st.warning("⚠️ Low confidence - Manual review recommended")
    
    # Visualization
    st.markdown("---")
    st.subheader("📊 AI Explanation (Grad-CAM Heatmap)")
    st.markdown("""
    The heatmap below shows which regions of the image influenced the model's decision.
    Brighter areas indicate higher influence on the defect prediction.
    """)
    
    if heatmap is not None:
        # Generate visualization
        overlay = generate_heatmap(image, heatmap, alpha=0.4)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, use_column_width=True, caption="Original Image")
        
        with col2:
            st.image(overlay, use_column_width=True, caption="Defect Heatmap Overlay")
    
    # Cleanup
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

# ============================================================================
# Demo & Instructions
# ============================================================================

else:
    st.info("""
    **👈 Upload an image to get started!**
    
    This MVP detects manufacturing defects using a deep learning model trained on the 
    MVTec AD dataset. The model can identify:
    - Surface scratches and abrasions
    - Dents and deformations
    - Cracks and fractures
    - Misalignment and assembly issues
    """)
    
    st.subheader("📚 How to Use")
    st.markdown("""
    1. **Upload an Image**: Click the file uploader on the left to select an image
    2. **View Results**: The model will analyze and show:
       - **Pass/Defect**: Binary classification result
       - **Confidence**: Probability score (0-100%)
       - **Heatmap**: Visual explanation of what the model detected
    3. **Interpret Heatmap**: Brighter regions = areas the model focused on for its decision
    """)
    
    st.subheader("💡 Tips for Best Results")
    st.markdown("""
    - **Image Quality**: Use clear, well-lit photos
    - **Framing**: Capture the entire part in the frame
    - **Angle**: Photograph from directly above when possible
    - **Consistency**: Use similar lighting conditions to training data
    """)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<small>
**Manufacturing Defect Detection MVP** | Built with PyTorch + Streamlit | 
Zero-cost deployment demo | For pilot evaluation only
</small>
""", unsafe_allow_html=True)
