# Local Setup Guide: Phase 1 & 2

## ✅ Phase 1: Project Setup (COMPLETE)

Your project structure is ready:
```
defect-detector/
├── src/                    # ML pipeline code
│   ├── data_loader.py      # Dataset management
│   ├── model.py            # ResNet18 model
│   ├── train.py            # Training script
│   ├── infer.py            # Inference + Grad-CAM
│   ├── utils.py            # Utilities
│   └── __init__.py
├── notebooks/
│   └── train_model.ipynb   # Colab training notebook (ready to upload)
├── app.py                  # Streamlit web UI
├── requirements.txt        # Dependencies
├── README.md               # Full documentation
├── data/                   # Dataset (to be downloaded)
├── checkpoints/            # Model weights
└── .gitignore
```

---

## 🔄 Phase 2: Data Pipeline Setup (LOCAL)

### Step 1: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download MVTec AD Dataset

**Option A: Manual Download (Recommended for first time)**

1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Click "Download" button
3. Fill out registration form (free, no credit card needed)
   - Name, Email, Organization
   - Agree to terms
4. Check email for download link (arrives within minutes)
5. Download `mvtec_ad_v1.0.tar.xz` (~5GB)
6. Extract to this project:

```bash
# Windows (use 7-Zip or WinRAR if you have them)
# Or use Python:
python -c "import tarfile; tarfile.open('mvtec_ad_v1.0.tar.xz').extractall('data/')"

# macOS/Linux
tar -xf mvtec_ad_v1.0.tar.xz -C data/
```

**Option B: Quick Test with Mock Dataset**

If you don't want to download ~5GB yet, create a tiny mock dataset:

```bash
python -c "
import os
from PIL import Image
import numpy as np

# Create structure
os.makedirs('data/bottle/train/good', exist_ok=True)
os.makedirs('data/bottle/test/crack', exist_ok=True)

# Create 20 random 'good' images
for i in range(20):
    img = Image.fromarray(np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8))
    img.save(f'data/bottle/train/good/normal_{i:03d}.png')

# Create 10 random 'defect' images
for i in range(10):
    img = Image.fromarray(np.random.randint(50, 100, (224, 224, 3), dtype=np.uint8))
    img.save(f'data/bottle/test/crack/crack_{i:03d}.png')

print('✓ Mock dataset created')
"
```

### Step 4: Verify Dataset Structure

After extraction, you should have:

```
data/
├── bottle/
│   ├── train/
│   │   └── good/           # Normal samples
│   └── test/
│       ├── good/           # Good test samples
│       ├── crack/          # Defective samples
│       ├── contamination/  # Defective samples
│       └── ...
├── cable/
├── carpet/
└── ... (12 more categories)
```

**Check with Python:**

```bash
python -c "
import os
import glob

for category in ['bottle', 'cable']:
    train_good = len(glob.glob(f'data/{category}/train/good/*'))
    if train_good > 0:
        print(f'{category}: {train_good} normal samples')
    else:
        print(f'{category}: NOT FOUND (dataset not downloaded yet)')
"
```

---

## 🚀 Phase 3: Training (Choose Your Path)

### Path A: Train Locally

**Fast (GPU):**
```bash
python src/train.py --category bottle --epochs 50 --batch-size 32 --device cuda
```

**CPU (slower but works):**
```bash
python src/train.py --category bottle --epochs 50 --batch-size 16 --device cpu
```

**With logging:**
```bash
python src/train.py --category bottle --epochs 50 --checkpoint-dir checkpoints
```

Training will:
- Load data automatically
- Show progress bars per epoch
- Save best model to `checkpoints/best_model.pt`
- Print validation accuracy after each epoch

**Expected results:**
- Accuracy: 95%+ (bottle category)
- Time: 1-2 hours CPU, ~20 mins GPU
- Checkpoint size: ~45 MB

### Path B: Train in Google Colab (Recommended)

1. **Open Colab**: https://colab.research.google.com
2. **Upload Notebook**:
   - File → Upload notebook
   - Select `notebooks/train_model.ipynb`
3. **Run all cells** (Runtime → Run all)
4. **Download checkpoint** when complete
5. **Copy to local**:
   ```bash
   # Move downloaded best_model.pt to:
   checkpoints/best_model.pt
   ```

**Colab advantages:**
- ✅ Free GPU (30 min training)
- ✅ No local setup needed
- ✅ Can run overnight
- ✅ Automatic checkpoint download

---

## ✅ After Training

Once you have `checkpoints/best_model.pt`:

### Run Web App

```bash
streamlit run app.py
```

Then open: http://localhost:8501

### Run Inference on Single Image

```bash
python -m src.infer --checkpoint checkpoints/best_model.pt --image path/to/image.jpg
```

---

## 📋 Quick Reference

| Step | Command | Time |
|------|---------|------|
| Create venv | `python -m venv venv && venv\Scripts\activate` | 1 min |
| Install deps | `pip install -r requirements.txt` | 3 mins |
| Download data | Visit mvtec.com + extract | 10 mins |
| Test setup | `python src/train.py --help` | instant |
| Train locally | `python src/train.py ...` | 1-2 hours |
| Train in Colab | Upload notebook → Run | 30 mins |
| Run web app | `streamlit run app.py` | instant |

---

## 🐛 Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'torch'"**
```bash
pip install -r requirements.txt
```

**Issue: Dataset not found**
```bash
# Check if data exists:
python -c "import os; print(os.listdir('data/'))"
# Should show: ['bottle', 'cable', 'carpet', ...]
```

**Issue: GPU not detected**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, use CPU: --device cpu
```

**Issue: Out of memory (CUDA)**
```bash
# Reduce batch size:
python src/train.py --category bottle --batch-size 16
```

---

## 🎯 Recommended Workflow

```
1. Create venv + install deps    (5 mins)
   ↓
2. Download MVTec dataset         (10 mins, or use mock)
   ↓
3. Option A: Train locally (GPU)  (20 mins) 
   Option B: Upload to Colab      (30 mins, recommended)
   ↓
4. Download/move checkpoint
   ↓
5. Run: streamlit run app.py
   ↓
6. Upload images → see results!
```

---

## 📚 Next Steps

- ✅ Complete Phase 1 setup (DONE - project ready)
- ⏳ Complete Phase 2 (download/setup data)
- ⏳ Complete Phase 3 (train model)
- ⏳ Phase 4: Run web app
- ⏳ Phase 5: Deploy to cloud

**Ready to proceed? Choose one:**
1. Download real MVTec dataset
2. Use mock dataset for quick testing
3. Jump straight to Colab training
