"""
Amazon ML Challenge 2026 - Meta DINOv2 Visual Foundation Model Extractor
=======================================================================
Extracts rich, self-supervised semantic & geometric representations using Meta's DINOv2 (Vision Transformer).

Why DINOv2 is a Competitive ML Secret Weapon:
1. Self-Supervised on 142M images: Learns fine-grained object boundaries, packaging textures, and physical geometry.
2. Orthogonal to Supervised CNNs: Unlike standard ImageNet classifiers that only see labels, DINOv2 captures
   spatial depth, contours, and physical product characteristics.
3. Linear Probing Power: DINOv2 frozen features fed into LightGBM or MLP achieve top-tier performance with zero fine-tuning overhead!
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T

# Standard ImageNet normalization for DINOv2
DINO_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DINOv2Extractor:
    def __init__(self, model_size="small", device=None):
        """
        model_size: 'small' (vits14 - 384 dim), 'base' (vitb14 - 768 dim), or 'large' (vitl14 - 1024 dim)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_names = {
            "small": "dinov2_vits14",
            "base": "dinov2_vitb14",
            "large": "dinov2_vitl14"
        }
        self.repo_model = model_names.get(model_size, "dinov2_vits14")
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"[*] Loading Meta DINOv2 ({self.repo_model}) onto {self.device.upper()}...")
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", self.repo_model)
            self.model.to(self.device)
            self.model.eval()
            print("[+] DINOv2 loaded successfully!")
        except Exception as e:
            print(f"[!] DINOv2 torch hub load error: {e}. Can fallback to timm or HuggingFace.")

    def extract_single_image(self, image_path):
        if self.model is None or not os.path.exists(image_path):
            return None

        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                tensor = DINO_TRANSFORM(img_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # DINOv2 output is class token [B, feature_dim]
                features = self.model(tensor)
                return features.squeeze(0).cpu().numpy()
        except Exception:
            return None

    def extract_batch_to_numpy(self, image_dir, image_filenames, output_npy="dinov2_features.npy", batch_size=32):
        if self.model is None:
            print("[!] DINOv2 model not initialized.")
            return None

        print(f"[*] Extracting DINOv2 features for {len(image_filenames):,} images...")
        all_features = []
        valid_indices = []

        for i in tqdm(range(0, len(image_filenames), batch_size), desc="DINOv2 Feature Extraction"):
            batch_filenames = image_filenames[i:i + batch_size]
            batch_tensors = []
            batch_idx_tracker = []

            for sub_i, fname in enumerate(batch_filenames):
                img_path = os.path.join(image_dir, str(fname))
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as img:
                            img_rgb = img.convert("RGB")
                            batch_tensors.append(DINO_TRANSFORM(img_rgb))
                            batch_idx_tracker.append(i + sub_i)
                    except Exception:
                        pass

            if batch_tensors:
                batch_stack = torch.stack(batch_tensors).to(self.device)
                with torch.no_grad():
                    feats = self.model(batch_stack).cpu().numpy()
                    all_features.append(feats)
                    valid_indices.extend(batch_idx_tracker)

        if all_features:
            final_features = np.vstack(all_features)
            np.save(output_npy, final_features)
            print(f"[+] Successfully extracted and saved features: {final_features.shape} to {output_npy}")
            return final_features
        return None

if __name__ == "__main__":
    print("=" * 65)
    print("  AMAZON ML CHALLENGE - META DINOv2 VISUAL BACKBONE")
    print("=" * 65)
    extractor = DINOv2Extractor(model_size="small")
    print("[*] DINOv2 module ready for feature extraction.")
