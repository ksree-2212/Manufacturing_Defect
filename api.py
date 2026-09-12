import os
import sys
import base64
import io
import uuid
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS

# Add src to path so we can import from infer and utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from infer import Predictor
from utils import generate_heatmap

app = Flask(__name__, static_folder='static')
CORS(app)

# Global predictor instance
predictor = None
CHECKPOINT_PATH = "checkpoints/best_model.pt"

def load_predictor():
    global predictor
    if os.path.exists(CHECKPOINT_PATH):
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            predictor = Predictor(CHECKPOINT_PATH, device=device)
            print(f"Model loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")

def numpy_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array (0-1 float) to base64 data URI."""
    if img_array is None:
        return None
    img_uint8 = (img_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.route('/api/predict', methods=['POST'])
def predict_images():
    if predictor is None:
        return jsonify({"error": "Model is not loaded. Please train a model first."}), 500
    
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
        
    files = request.files.getlist('images')
    results = []
    
    for file in files:
        if file.filename == '':
            continue
            
        temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
        try:
            file.save(temp_path)
            
            # Predict
            pred_class, confidence, original_image, heatmap = predictor.predict(
                temp_path, return_heatmap=True
            )
            
            # Generate overlay heatmap
            overlay = None
            if heatmap is not None:
                overlay = generate_heatmap(original_image, heatmap, alpha=0.5)
            
            orig_b64 = numpy_to_base64(original_image)
            heatmap_b64 = numpy_to_base64(overlay)
            
            results.append({
                "filename": file.filename,
                "is_defect": bool(pred_class == 1),
                "confidence": float(confidence),
                "original_image": orig_b64,
                "heatmap_image": heatmap_b64
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    return jsonify({"results": results})

@app.route('/')
def read_root():
    return redirect('/static/index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    load_predictor()
    app.run(host='127.0.0.1', port=8000, debug=False)
