import os
import urllib.request
import tarfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_mvtec_bottle():
    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe7374faea6c767592d/download/420937370-1629951468/bottle.tar.xz"
    output_path = "bottle.tar.xz"
    data_dir = "data"
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading MVTec AD 'bottle' dataset (~120MB)...")
    print("This contains REAL images of normal and defective bottles.")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        
    print("\nExtracting dataset to data/ folder...")
    with tarfile.open(output_path) as tar:
        tar.extractall(path=data_dir)
        
    print(f"Extraction complete! Real images are now in: {os.path.abspath(data_dir)}/bottle")
    
    # Cleanup
    if os.path.exists(output_path):
        os.remove(output_path)
        
    print("\nSUCCESS! You can now re-train your model using these real images.")
    print("To train locally, run:")
    print("  .\\.venv\\Scripts\\python.exe src\\train.py --category bottle --epochs 20")
    print("Note: If you train on Colab instead, you will need to upload these images to Colab.")

if __name__ == "__main__":
    download_mvtec_bottle()
