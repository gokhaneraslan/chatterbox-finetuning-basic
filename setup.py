import os
import requests
import sys
from tqdm import tqdm

# Configuration
DEST_DIR = "pretrained_models"

# Map: Target Filename -> Source URL
FILES_TO_DOWNLOAD = {
    "ve.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.safetensors?download=true",
    "t3_cfg.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_cfg.safetensors?download=true",
    "s3gen.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.safetensors?download=true",
    "conds.pt": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/conds.pt?download=true",
    "tokenizer.json": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/grapheme_mtl_merged_expanded_v1.json?download=true"
}

def download_file(url, dest_path):
    """Downloads a file from a URL to a specific destination with a progress bar."""
    
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading: {os.path.basename(dest_path)}...")
    
    try:
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            
            for data in response.iter_content(block_size):
                
                size = file.write(data)
                bar.update(size)
                
        print(f"Download complete: {dest_path}\n")
        
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)


def main():
    
    print("--- Chatterbox Pretrained Model Setup ---\n")
    
    # 1. Create the directory if it doesn't exist
    if not os.path.exists(DEST_DIR):
        
        print(f"Creating directory: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)
        
    else:
        print(f"Directory found: {DEST_DIR}")

    # 2. Download files
    for filename, url in FILES_TO_DOWNLOAD.items():
        dest_path = os.path.join(DEST_DIR, filename)
        download_file(url, dest_path)

    print("All models are set up in 'pretrained_models/' folder.")
    print(f"Note: 'grapheme_mtl_merged_expanded_v1.json' was saved as 'tokenizer.json' for the new vocabulary.")


if __name__ == "__main__":
    main()