# IIT / NIT CHAMPION PLAYBOOK — AMAZON ML CHALLENGE 2026
**Target: Top 10 National Rank | SMAPE 35%–38% | Zero Disqualification Protocol**

---

## 1. The 48-Hour Battle Workflow (IIT/NIT Blueprint)

```
[Day 1: Hours 00 - 06]  Data Ingestion -> Auto Schema Detect -> 5-Fold Local Baseline (LightGBM + CatBoost)
[Day 1: Hours 06 - 18]  Image Batch Download on Kaggle T4 GPU -> PaddleOCR -> Text Extraction
[Day 1: Hours 18 - 30]  Hugging Face Zero-Shot Embeddings (MiniLM / DeBERTa) + OpenCLIP Vision
[Day 2: Hours 30 - 40]  Multimodal Late Fusion Network (PyTorch SMAPE Loss) + MRP Anchoring
[Day 2: Hours 40 - 46]  Convex Weight Ensembling + Nelder-Mead Multiplier Calibration (alpha ~ 0.94)
[Day 2: Hours 46 - 48]  Anti-Disqualification Verification -> Final CSV Lock & Submission
```

---

## 2. Hugging Face Toolchain Ready in Workspace

| Module | What It Does | Why Top Teams Use It |
|---|---|---|
| [`hf_embedding_extractor.py`](file:///C:/Users/HP/.gemini/antigravity-ide/scratch/amazon-ml-challenge-2026/hf_embedding_extractor.py) | Mean-pooled 384-d semantic vectors from product titles via `all-MiniLM-L6-v2` or `BGE-M3`. | Converts unstructured text into mathematical coordinates for CatBoost and FAISS similarity. |
| [`hf_multimodal_fusion.py`](file:///C:/Users/HP/.gemini/antigravity-ide/scratch/amazon-ml-challenge-2026/hf_multimodal_fusion.py) | PyTorch Multimodal Late Fusion Neural Network with differentiable `SMAPELoss`. | Blends text embeddings + vision embeddings + domain tabular features directly into final price. |
| [`hf_deberta_trainer.py`](file:///C:/Users/HP/.gemini/antigravity-ide/scratch/amazon-ml-challenge-2026/hf_deberta_trainer.py) | Fine-tunes `microsoft/deberta-v3-small` end-to-end on `catalog_content` with evaluation callbacks. | Captures subtle e-commerce context (brand tiers, product models, bundle deals). |
| [`hf_entity_attribute_extractor.py`](file:///C:/Users/HP/.gemini/antigravity-ide/scratch/amazon-ml-challenge-2026/hf_entity_attribute_extractor.py) | High-precision regex pattern anchoring + token extractor for weight, volume, voltage, wattage. | Handles entity extraction format challenges (e.g., 2023/2024 Amazon problem statements). |
| [`hf_offline_cache_prep.py`](file:///C:/Users/HP/.gemini/antigravity-ide/scratch/amazon-ml-challenge-2026/hf_offline_cache_prep.py) | Pre-caches model weights & tokenizers to `./models_cache/`. | **Survival Guarantee**: Prevents test evaluation crash in no-internet evaluation environments. |

---

## 3. How to Run During Competition (Sept 25–27, 2026)

### Step 1: Automatic Ingestion & Feature Engineering (Local CPU)
Place the official `train.csv` and `test.csv` in this directory, then run:
```bash
.venv\Scripts\python.exe run_pipeline.py
```
This automatically runs schema detection, tabular feature extraction, 5-fold CatBoost & LightGBM, and generates your first valid baseline submission.

### Step 2: GPU Embedding Extraction & Multimodal Training (Kaggle T4 x 2)
Upload [`kaggle_master_notebook.ipynb`](file:///C:/Users/HP/.gemini/antigravity-ide/scratch/amazon-ml-challenge-2026/kaggle_master_notebook.ipynb) to Kaggle with GPU enabled:
- Runs parallel image downloading.
- Extracts OpenCLIP ViT image vectors.
- Generates Hugging Face DeBERTa/MiniLM text embeddings.
- Download the generated `test_preds_hf_multimodal.csv`.

### Step 3: Master Ensembling & Post-Processing (Local CPU)
Combine your local GBDT predictions with Kaggle deep learning predictions:
```bash
.venv\Scripts\python.exe ensemble_stacker.py
```
This solves the convex optimization problem to find the lowest possible SMAPE blend weights and applies Nelder-Mead post-processing.

### Step 4: Submission Integrity Check
Always audit your final file before uploading to the portal:
```bash
.venv\Scripts\python.exe submission_verifier.py
```
Ensures exact row count, matching IDs, strictly positive prices, no NaNs, and distribution alignment.
