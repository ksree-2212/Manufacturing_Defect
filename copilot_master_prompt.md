Manufacturing Defect Detection MVP

You are acting as a senior ML engineer and full-stack developer. Your job is to help me build a working MVP for a computer-vision-based manufacturing defect detection tool, starting from zero budget. Read the full spec below, then propose an implementation plan broken into phases, and execute it phase by phase — writing code, creating files, and explaining each step as you go. Ask me clarifying questions only if something blocks progress; otherwise make reasonable assumptions and proceed.

### 1. Project Overview

**Product:** A lightweight web app where a user uploads a photo (or streams a webcam feed) of a manufactured part, and the system uses a trained deep learning model to detect and highlight surface defects (e.g., scratches, dents, cracks, misalignment) — returning a pass/fail result and a visual overlay showing where defects were found.

**Target user:** Small and mid-size manufacturers who currently rely on manual visual inspection and cannot afford expensive enterprise machine-vision systems.

**Goal for this phase:** Build a free, working, demoable prototype using public data — no paid infrastructure, no real customer data required yet. This will be used to pitch a free pilot to a real small manufacturer.

### 2. Constraints

- Zero budget: use only free-tier tools and services.
- Use Python for the ML pipeline and a lightweight Python web framework for the demo UI.
- Must run on a free-tier GPU (Google Colab or Kaggle Notebooks) for training, and be deployable as a simple local or free-hosted web demo for inference.
- Code should be modular and well-commented so I can explain it to non-technical potential customers and investors later.

### 3. Architecture

**Data layer:**
- Use a public defect-detection dataset to start. Recommended: MVTec AD (anomaly detection, multiple object/texture categories) or the NEU Surface Defect Database (steel surface defects). Pick MVTec AD by default unless I say otherwise, since it has both normal and defective samples across multiple categories.
- Structure: `/data/train`, `/data/val`, `/data/test` folders split by class (normal vs. defect types).

**Model layer:**
- Use transfer learning: a pretrained CNN backbone (ResNet18 or EfficientNet-B0) fine-tuned for binary classification (defect / no defect) as the first milestone.
- Stretch goal (phase 2): move to a segmentation or anomaly-localization approach (e.g., PatchCore, or a simple autoencoder-based anomaly detector) so the app can highlight *where* the defect is, not just flag that one exists.
- Framework: PyTorch (preferred) or TensorFlow/Keras — default to PyTorch unless I specify otherwise.
- Training environment: Google Colab notebook using free GPU runtime.

**Application layer:**
- Build a simple web interface using Streamlit (fastest path to a working demo).
- Features:
  1. Upload an image.
  2. Run inference using the trained model.
  3. Display: pass/fail result, confidence score, and a visual overlay (heatmap or bounding box) showing the defect location if applicable.
- Keep the UI simple enough to demo on a laptop to a factory owner with no technical background.

**Deployment layer (for demo purposes only, still free):**
- Package the Streamlit app so it can run locally via `streamlit run app.py`.
- Also prepare it for free hosting on Streamlit Community Cloud or Hugging Face Spaces so I can share a live link without needing the viewer to install anything.

### 4. Project Structure to Create

```
defect-detector/
├── data/                  # dataset (gitignored, downloaded via script)
├── notebooks/
│   └── train_model.ipynb  # Colab-compatible training notebook
├── src/
│   ├── data_loader.py     # dataset loading & preprocessing
│   ├── model.py           # model architecture & loading
│   ├── train.py           # training loop (script version of notebook)
│   ├── infer.py           # inference function used by the app
│   └── utils.py           # helper functions (visualization, overlays, etc.)
├── app.py                 # Streamlit web app
├── requirements.txt
├── README.md              # explains setup, usage, and project goal
└── .gitignore
```

### 5. Implementation Plan I Want You to Follow

Please propose this as a phased plan, confirm it with me briefly, then execute step by step:

**Phase 1 — Setup**
- Initialize the project structure above.
- Write `requirements.txt` (torch, torchvision, streamlit, pillow, numpy, matplotlib, opencv-python, scikit-learn).
- Write a data download/setup script or clear instructions for downloading MVTec AD (note: some datasets require manual download due to licensing — flag this clearly if so).

**Phase 2 — Data pipeline**
- Write `data_loader.py` to load and preprocess images (resizing, normalization, augmentation for training).
- Split into train/val/test.

**Phase 3 — Model training**
- Write `model.py` with a ResNet18-based binary classifier (transfer learning, pretrained ImageNet weights, replace final layer).
- Write `train.py` / `train_model.ipynb` with a training loop, validation accuracy tracking, and model checkpoint saving.
- Optimize for Colab's free GPU (batch size, mixed precision if useful).

**Phase 4 — Inference & visualization**
- Write `infer.py` to load a trained checkpoint and run predictions on a new image.
- Add a simple visualization method (e.g., Grad-CAM) to highlight which region of the image influenced the "defect" prediction — this is important for the demo, since factory owners need to see *why*, not just a yes/no.

**Phase 5 — Web app**
- Build `app.py` using Streamlit: image upload, run inference, display result + confidence + heatmap overlay.
- Keep it visually clean and simple.

**Phase 6 — Packaging & deployment prep**
- Write a clear `README.md`: what the project does, setup instructions, how to run locally, and how to deploy free on Streamlit Community Cloud or Hugging Face Spaces.
- Add `.gitignore` for data/checkpoints/venv.

### 6. How I Want You to Work

- After reading this, first output a short implementation plan confirming the phases above (or improving them if you see a better approach).
- Then start executing Phase 1 immediately — create the folder structure and files.
- After each phase, briefly summarize what you did and what I need to do next (e.g., "download the dataset from this link and place it in /data").
- Write code with comments explaining what each part does, since I'll need to explain this to non-technical people later.
- If a step requires something outside VS Code (e.g., running a Colab notebook, downloading a dataset manually due to licensing), tell me clearly and give exact steps.

