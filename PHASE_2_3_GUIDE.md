# Phase 1 & 2: Local Setup + Phase 3 Colab Training Guide

## ✅ Status Summary

| Phase | Task | Status |
|-------|------|--------|
| 1 | Project Setup (structure, files, code) | ✅ COMPLETE |
| 2 | Data Pipeline (data_loader.py ready) | ✅ READY |
| 3 | Model Training (Colab notebook created) | ✅ READY |
| 4 | Inference & Visualization | ⏳ After Phase 3 |
| 5 | Web App (Streamlit) | ⏳ After Phase 3 |

---

## 🎯 Complete Workflow

### OPTION A: Quick Start (Recommended) - Use Colab GPU
**Time: ~30 minutes total**

```
1. Go to Google Colab
2. Upload train_model.ipynb
3. Run all cells
4. Download checkpoint
5. Run web app locally
Total: 30 minutes ✓
```

### OPTION B: Local Training (CPU/GPU)
**Time: 1-2 hours (CPU) or 20 mins (GPU)**

```
1. Setup venv locally
2. Download MVTec dataset
3. Run src/train.py
4. Run web app
Total: 1-2 hours
```

### OPTION C: Local Setup + Colab Training (Hybrid)
**Time: 20 minutes total**

```
1. Setup local environment (5 mins)
2. Train in Colab (20 mins)
3. Download checkpoint (2 mins)
4. Run web app locally (instant)
Total: 20 minutes ✓
```

---

## 📋 DETAILED STEPS

### PART 1: Local Environment Setup (5 minutes)

#### Step 1.1: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 1.2: Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- PyTorch 2.0.0
- Streamlit 1.28.0
- OpenCV, Pillow, matplotlib
- scikit-learn, tqdm
- grad-cam (for Grad-CAM visualization)

**Installation time:** ~3 minutes

#### Step 1.3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch OK: {torch.__version__}'); import streamlit; print('Streamlit OK')"
```

---

### PART 2: Download Dataset (Choose One)

#### Option A: Use Mock Dataset (QUICKEST)
**Time: 2 minutes**

The Colab notebook includes mock dataset generation. Recommended for testing.

```bash
# The notebook will create it automatically when you run it
```

#### Option B: Download Real MVTec AD
**Time: ~15 minutes**

1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Click "Download" button
3. Fill registration form (free, takes 2 minutes)
4. Wait for email with download link
5. Download `mvtec_ad_v1.0.tar.xz` (~5GB)
6. Extract:

**Windows:**
```bash
# Install 7-Zip if needed, then:
# Right-click mvtec_ad_v1.0.tar.xz > 7-Zip > Extract
# Then extract the resulting .tar file

# OR use Python:
python -c "import tarfile; tarfile.open('mvtec_ad_v1.0.tar.xz').extractall('data/')"
```

**macOS/Linux:**
```bash
tar -xf mvtec_ad_v1.0.tar.xz -C data/
```

#### Verify Dataset

```bash
python -c "
import os
import glob

categories = ['bottle', 'cable', 'carpet']
for cat in categories:
    train = len(glob.glob(f'data/{cat}/train/good/*'))
    test = len(glob.glob(f'data/{cat}/test/*/*.png'))
    if train > 0:
        print(f'✓ {cat}: {train} train, {test} test samples')
    else:
        print(f'✗ {cat}: Not found')
"
```

---

### PART 3: Train Model (Choose Path)

#### PATH A: Train in Google Colab (⭐ RECOMMENDED)

**Advantages:**
- ✅ Free GPU (T4/P100)
- ✅ No local GPU needed
- ✅ Only 30 minutes
- ✅ Can run overnight
- ✅ Automatic checkpoint download

**Steps:**

1. **Go to Google Colab:**
   - Open: https://colab.research.google.com

2. **Upload Notebook:**
   - File → Upload notebook
   - Select: `notebooks/train_model.ipynb`
   - Or drag-drop the file

3. **Enable GPU (IMPORTANT!):**
   - Runtime → Change runtime type
   - GPU → Select (should say "T4" or better)
   - Click Save

4. **Run All Cells:**
   - Runtime → Run all
   - Or press Ctrl+F9
   - Wait ~30 minutes

5. **Download Checkpoint:**
   - Cell will automatically trigger download
   - Save as: `checkpoints/best_model.pt`

6. **Local Setup:**
   ```bash
   # Copy downloaded file to local project
   # Place in: c:\Users\K Sreeshanth\Downloads\p1\checkpoints\best_model.pt
   ```

#### PATH B: Train Locally (CPU - Slow)

```bash
python src/train.py \
    --category bottle \
    --epochs 50 \
    --batch-size 16 \
    --device cpu
```

**Expected time:** 1-2 hours on modern CPU

#### PATH C: Train Locally (GPU - Fast)

```bash
python src/train.py \
    --category bottle \
    --epochs 50 \
    --batch-size 32 \
    --device cuda
```

**Expected time:** ~20 minutes on modern GPU (RTX 3060+)

---

### PART 4: Run Web App

#### Step 4.1: Verify Checkpoint

```bash
# Check if model is in place
python -c "import os; print('✓ Model ready' if os.path.exists('checkpoints/best_model.pt') else '✗ Model missing')"
```

#### Step 4.2: Start Streamlit App

```bash
streamlit run app.py
```

**Output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

#### Step 4.3: Test the App

1. Open browser: http://localhost:8501
2. Upload an image (JPG, PNG, BMP)
3. See results:
   - **PASS** or **DEFECT** classification
   - Confidence percentage
   - Grad-CAM heatmap showing which regions influenced prediction

---

## 📊 Timeline Comparison

| Path | Setup | Dataset | Training | Total |
|------|-------|---------|----------|-------|
| **Colab** | 5 min | 0 min* | 30 min | **35 min** |
| **Local GPU** | 5 min | 10 min | 20 min | **35 min** |
| **Local CPU** | 5 min | 10 min | 120 min | **135 min** |

*Colab notebook creates mock dataset automatically; can use real data if uploaded

---

## 🚀 Quickest Path (Recommended)

**Total Time: ~35 minutes**

```bash
# 1. Local setup (5 min)
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Go to Colab (25 min)
# - Open https://colab.research.google.com
# - Upload notebooks/train_model.ipynb
# - Enable GPU: Runtime > Change runtime type > GPU
# - Run all: Runtime > Run all
# - Download best_model.pt when done

# 3. Local - move checkpoint (1 min)
# Move downloaded file to: checkpoints/best_model.pt

# 4. Run app (1 min)
streamlit run app.py

# 5. Test! (3 min)
# Upload images, see predictions
```

---

## 📁 Project Files Ready

| File | Purpose | Status |
|------|---------|--------|
| `src/model.py` | ResNet18 architecture | ✅ Ready |
| `src/data_loader.py` | Dataset loading | ✅ Ready |
| `src/train.py` | Training script (local) | ✅ Ready |
| `src/infer.py` | Inference + Grad-CAM | ✅ Ready |
| `src/utils.py` | Utility functions | ✅ Ready |
| `notebooks/train_model.ipynb` | Colab training | ✅ Ready |
| `app.py` | Streamlit web UI | ✅ Ready |
| `requirements.txt` | Dependencies | ✅ Ready |
| `README.md` | Full documentation | ✅ Ready |
| `SETUP_LOCAL.md` | Local setup guide | ✅ Ready |

---

## ⚡ Quick Decision Matrix

**Choose Colab if you:**
- ✅ Don't have a GPU locally
- ✅ Want fastest results (30 min)
- ✅ Want to try before committing compute
- ✅ Want guaranteed GPU availability

**Choose Local if you:**
- ✅ Have a powerful GPU (RTX 3060+)
- ✅ Want full control over training
- ✅ Plan to iterate/modify code
- ✅ Have limited internet

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install -r requirements.txt
```

### Colab GPU not available
- Runtime → Change runtime type
- Select GPU (should show "T4" or better)
- Click Save

### Checkpoint not found in app
```bash
# Verify file exists:
python -c "import os; print(os.path.exists('checkpoints/best_model.pt'))"
# Should print: True
```

### Training too slow
```bash
# If on CPU, reduce batch size:
python src/train.py --batch-size 8
# Or reduce epochs:
python src/train.py --epochs 20
```

### Out of memory
```bash
# Reduce batch size:
python src/train.py --batch-size 16
```

---

## 📚 Resources

- **Colab**: https://colab.research.google.com
- **MVTec Dataset**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **Streamlit Docs**: https://docs.streamlit.io/

---

## ✅ Final Checklist

Before running the app:

- [ ] Python venv created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Either:
  - [ ] Dataset downloaded (real MVTec) OR
  - [ ] Will use mock in Colab
- [ ] Either:
  - [ ] Trained locally (checkpoint in `checkpoints/best_model.pt`) OR
  - [ ] Will train in Colab
- [ ] Checkpoint file exists: `checkpoints/best_model.pt`
- [ ] Ready to run: `streamlit run app.py`

---

## 🎯 Recommended Next Step

**I recommend: Use Colab (fastest & easiest)**

1. Go to: https://colab.research.google.com
2. File → Upload notebook
3. Select: `notebooks/train_model.ipynb` from your local project
4. Enable GPU before running
5. Run all cells
6. Download checkpoint
7. Copy to `checkpoints/best_model.pt` locally
8. Run `streamlit run app.py`

This takes ~35 minutes total and requires no local GPU!

---

**Ready to proceed? Let me know which path you choose!**
