"""
scripts/upload_model_to_huggingface.py
=======================================
Run this ONCE after training completes on Kaggle.
It uploads your best.pt checkpoint to Hugging Face Hub so that:
  - The trained weights are stored permanently for free
  - The inference server can download them automatically on startup
  - You never need to retrain unless you want to improve the model

HOW TO USE (for beginners — step by step):
  1. Make sure training is complete and best.pt exists
  2. Open your .env file and fill in HUGGINGFACE_TOKEN and HF_MODEL_REPO
  3. Run:  python scripts/upload_model_to_huggingface.py
  4. Done — your model is now live on Hugging Face
  
On Kaggle, run this in a notebook cell after training:
  !python scripts/upload_model_to_huggingface.py \
      --token YOUR_HF_TOKEN \
      --repo  YOUR_HF_USERNAME/rrin-retina-restoration \
      --checkpoint /kaggle/working/checkpoints/best.pt
"""

import argparse
import os
import sys


def load_env_file(env_path: str = ".env") -> None:
    """Read .env file and set environment variables from it."""
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def upload_model(token: str, repo_id: str, checkpoint_path: str) -> None:
    """
    Upload the trained model checkpoint and a model card to Hugging Face Hub.

    params:
        token           — your Hugging Face write-access token
        repo_id         — "YourUsername/repo-name"
        checkpoint_path — path to best.pt on disk
    """
    # Import here so the script still gives a useful error message if
    # huggingface_hub is not installed
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface-hub package not installed.")
        print("Run:  pip install huggingface-hub")
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at:  {checkpoint_path}")
        print("Make sure training has completed and best.pt exists.")
        sys.exit(1)

    file_size_mb = os.path.getsize(checkpoint_path) / 1_000_000
    print(f"\nCheckpoint found:  {checkpoint_path}  ({file_size_mb:.0f} MB)")
    print(f"Uploading to Hugging Face repository:  {repo_id}")
    print("(This is a one-time upload — the model is stored permanently)\n")

    api = HfApi(token=token)

    # Create the repository on HF if it does not exist yet
    print("Step 1/3 — Creating repository on Hugging Face (if not already there)...")
    create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="model",
        exist_ok=True,       # Don't fail if it already exists
        private=False,       # Public by default — set True for private
    )
    print(f"  Repository ready: https://huggingface.co/{repo_id}")

    # Upload the checkpoint file
    print("\nStep 2/3 — Uploading model weights (best.pt)...")
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="best.pt",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Upload trained RRIN retina restoration checkpoint",
    )
    print("  best.pt uploaded successfully.")

    # Upload a README / model card to describe what this model does
    print("\nStep 3/3 — Uploading model card (README.md)...")
    model_card = f"""---
language: en
tags:
  - retinal-image-restoration
  - medical-imaging
  - GAN
  - PyTorch
  - fundus-photography
license: mit
---

# RRIN — Restorative Retinal Imaging Network

This model restores degraded retinal fundus photographs.

## What it does
Given a blurry, low-contrast, or reflection-affected retina image,
the AI produces a cleaner, sharper version suitable for clinical review.

## How to use

```python
from huggingface_hub import hf_hub_download
import torch

# Download the checkpoint (only downloads once, then cached)
checkpoint_path = hf_hub_download(repo_id="{repo_id}", filename="best.pt")
print(f"Model downloaded to: {{checkpoint_path}}")
```

## Architecture
- **Generator**: U-Net with residual bottleneck and attention gates (4→3 channels)
- **Discriminator**: PatchGAN with spectral normalisation (conditional)
- **Losses**: L1 (weight 100) + SSIM (weight 10) + VGG perceptual (weight 10) + LSGAN adversarial (weight 1)
- **Training data**: EyePACS, APTOS 2019, RFMiD, ODIR-5K (pseudo-clean pool, top 25% by quality score)

## Inference
Input: 4-channel tensor (3 RGB channels + 1 green channel copy), values in [-1, 1]  
Output: 3-channel RGB tensor, values in [-1, 1]
"""

    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Add model card",
    )

    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nYour model is now live at:")
    print(f"  https://huggingface.co/{repo_id}")
    print(f"\nTo load it in any Python environment:")
    print(f'  from huggingface_hub import hf_hub_download')
    print(f'  path = hf_hub_download(repo_id="{repo_id}", filename="best.pt")')
    print(f"\nUpdate your .env file:")
    print(f"  HF_MODEL_REPO={repo_id}")
    print("\nThe inference server will now download this model automatically")
    print("on first startup — no retraining needed ever again.")


def main():
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Upload trained RRIN model to Hugging Face Hub"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HUGGINGFACE_TOKEN", ""),
        help="Hugging Face write-access token (from .env or command line)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("HF_MODEL_REPO", ""),
        help="Repository ID in format  YourUsername/repo-name"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to the trained model checkpoint (default: checkpoints/best.pt)"
    )
    args = parser.parse_args()

    # Validate inputs
    missing = []
    if not args.token or "PASTE_YOUR" in args.token:
        missing.append("--token  (your Hugging Face token from huggingface.co/settings/tokens)")
    if not args.repo or "PASTE_YOUR" in args.repo:
        missing.append("--repo   (e.g.  yourname/rrin-retina-restoration)")

    if missing:
        print("\nERROR: Missing required arguments:")
        for m in missing:
            print(f"  {m}")
        print("\nEither:")
        print("  1. Fill in HUGGINGFACE_TOKEN and HF_MODEL_REPO in your .env file, OR")
        print("  2. Pass them on the command line:")
        print("     python scripts/upload_model_to_huggingface.py \\")
        print("         --token hf_abc123... \\")
        print("         --repo  yourname/rrin-retina-restoration \\")
        print("         --checkpoint checkpoints/best.pt")
        sys.exit(1)

    upload_model(args.token, args.repo, args.checkpoint)


if __name__ == "__main__":
    main()
