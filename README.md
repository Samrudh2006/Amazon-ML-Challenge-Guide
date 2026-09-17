# 🚀 Amazon ML Challenge 2026 - Top 50 Blueprint

Welcome to the official workspace for the **Amazon ML Challenge 2026**.
This repository contains our battle-tested utilities, templates, and scripts designed to secure a **Top 50 Rank** and earn an Amazon Direct Interview.

---

## 🛠️ Project Structure
```text
amazon-ml-challenge-2026/
├── auto_adapt.py               # 🔮 100% Zero-Touch Auto-Adaptive Schema & Task Detector
├── run_pipeline.py             # 🎯 Master 1-Click End-to-End Pipeline Controller
├── export_to_kaggle.py         # 📦 Packages entire solution into 1-Click Kaggle Notebook
├── kaggle_amazon_ml_solution.ipynb # Standalone self-contained Kaggle GPU Notebook
├── bootcamp_tracker.md         # 🗓️ 13-Day Interactive Preparation Roadmap (Sept 13-25)
├── losses.py                   # ⚖️ Competition Metric Losses (FocalLoss, SMAPE, LogCosh)
├── download_images.py          # High-speed multi-threaded image downloader (50-100 workers)
├── test_downloader_demo.py     # Verification script to test image downloading
├── ocr_extractor.py            # Batch OCR & Amazon Unit Normalization pipeline
├── test_ocr_regex.py           # Unit tests verifying extraction accuracy
├── submission_validator.py     # Disqualification shield & leaderboard validator
├── test_validator_demo.py      # Validator tests on valid vs corrupted submissions
├── train_multimodal.py         # PyTorch Late-Fusion Multimodal network (TIMM + Transformers)
├── florence2_vlm_extractor.py  # SOTA Microsoft Florence-2 Vision-Language Model (VLM)
├── dinov2_feature_extractor.py # Meta DINOv2 Self-Supervised Physical & Geometric Backbone
├── tabular_booster.py          # Fast GBDT (LightGBM/CatBoost) on TF-IDF + OCR Candidates
├── tta_inference.py            # Test-Time Augmentation Engine (Multi-View Test Boost)
├── adversarial_validation.py   # Private Leaderboard Shakeup & Distribution Drift Shield
├── oof_stacking_ensemble.py    # Kaggle Grandmaster Out-Of-Fold Stacking & Meta-Ensembler
├── pseudo_labeler.py           # Semi-Supervised Pseudo-Labeling Engine (Final +3% Boost)
├── requirements.txt            # Competition dependencies (PyTorch, TIMM, Transformers, PaddleOCR)
└── sample_images/              # Test downloads folder
```

---

## ⚡ 1. Ultra-Fast Parallel Image Downloader

### Why this matters:
Amazon provides 100,000+ image URLs in CSV format rather than a `.zip` archive. Standard single-threaded downloads take 15–24 hours. Our multi-threaded pipeline completes this in **15 to 25 minutes**.

### How to Run:
```bash
python download_images.py \
  --csv_path train.csv \
  --image_col image_link \
  --id_col index \
  --output_dir images/train \
  --workers 60
```

### Key Features:
- **Resumable**: If your connection drops, re-running skips existing valid images instantly.
- **Corrupt Check**: Automatically validates non-zero and non-corrupt image bytes via PIL.
- **Failed Logging**: Saves any 404 or broken links to `failed_images.csv` for targeted handling.

---

## 👥 Team Roles & Responsibilities (4-Person Setup)

| Member | Focus | Tools / Models |
| :--- | :--- | :--- |
| **Member 1 (Data & OCR)** | Dataset cleaning, parallel download, image OCR extraction, regex normalization. | `paddleocr`, `re`, `pandas` |
| **Member 2 (Vision)** | Visual feature extraction, image classification / regression. | `timm` (`convnext`, `swin`), `albumentations` |
| **Member 3 (NLP / Multimodal)** | Text modeling, title/description tokenization, Late-Fusion network. | HuggingFace `transformers` (`deberta-v3`), `open_clip` |
| **Member 4 (MLOps & Ensembling)** | 5-Fold Stratified CV, blending/stacking, submission formatting & validation. | `scikit-learn`, `lightgbm`, `scipy.optimize` |

---

## 📅 Roadmap to Sept 25:
1. **Sept 12 - 15:** Environment setup (Kaggle/Colab accounts verified for GPU).
2. **Sept 16 - 19:** PaddleOCR pipeline testing on sample images.
3. **Sept 20 - 22:** PyTorch Multimodal Late-Fusion model template built.
4. **Sept 23 - 24:** 5-Fold CV & Submission validator dry run.
5. **Sept 25 (Competition Kickoff):** Download in 20 mins -> Baseline submitted within 2 hours!
