"""
Amazon ML Challenge 2026 - Kaggle Notebook Exporter & Packager
=============================================================
Bundles all scripts into a single, beautiful, production-ready Kaggle Notebook (.ipynb)
with GPU acceleration, progress tracking, and 1-click execution.

Why this is convenient:
- During the 48-hour competition, your team can upload this single notebook to Kaggle,
  turn ON the T4 x 2 GPU accelerator, and click 'Run All'.
- No copy-pasting code, no missing files, no dependency conflicts!
"""

import os
import sys
import json
import argparse

def create_kaggle_notebook(output_ipynb="kaggle_amazon_ml_solution.ipynb"):
    print("=" * 65)
    print("  PACKAGING ENTIRE PIPELINE INTO SELF-CONTAINED KAGGLE NOTEBOOK")
    print("=" * 65)

    cells = []

    # Cell 1: Markdown Title & Intro
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Amazon ML Challenge 2026 - Grandmaster Solution\n",
            "### SOTA Multimodal Pipeline: Florence-2 VLM + DINOv2 + ConvNeXt + DeBERTa-v3 + 5-Fold OOF Stacking\n",
            "---\n",
            "This notebook runs the complete end-to-end competition pipeline on Kaggle GPU."
        ]
    })

    # Cell 2: Hardware & Environment Verification
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Check GPU\n",
            "!nvidia-smi\n",
            "!pip install -q timm albumentations paddleocr paddlepaddle-gpu lightgbm catboost open_clip_torch"
        ]
    })

    # Read key local scripts and embed them into executable cells
    scripts_to_embed = [
        ("1. Fast Parallel Downloader", "download_images.py"),
        ("2. Batch OCR & Unit Normalizer", "ocr_extractor.py"),
        ("3. Competition Losses", "losses.py"),
        ("4. Multimodal PyTorch Pipeline", "train_multimodal.py"),
        ("5. Submission Integrity Shield", "submission_validator.py")
    ]

    for title, fname in scripts_to_embed:
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                code_content = f.read()

            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## 🛠️ {title} (`{fname}`)"]
            })
            
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [f"%%writefile {fname}\n" + code_content]
            })

    # Final Execution Cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## ⚡ Execute Complete Pipeline"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python download_images.py --csv_path /kaggle/input/amazon-ml-challenge-2026/train.csv --output_dir images/train --workers 80\n",
            "!python ocr_extractor.py --image_dir images/train --output_csv ocr_train.csv --gpu\n",
            "!python train_multimodal.py --epochs 5 --batch_size 32\n",
            "!python submission_validator.py --submission submission.csv --test_csv /kaggle/input/amazon-ml-challenge-2026/test.csv"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "gpuClass": "standard",
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(output_ipynb, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"[SUCCESS] Kaggle Master Notebook generated: {os.path.abspath(output_ipynb)}")
    print(f"[*] Total Cells: {len(cells)}")
    print("=" * 65)

if __name__ == "__main__":
    create_kaggle_notebook()
