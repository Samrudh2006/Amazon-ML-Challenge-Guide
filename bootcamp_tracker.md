# 🗓️ 13-Day Master Battle Bootcamp (Sept 13 - Sept 25)
### Target: Amazon ML Challenge 2026 - TOP 10 GRAND FINALE PODIUM

Use this interactive tracker to maintain championship momentum every day leading up to the competition.

---

## 🎯 Week 1: Infrastructure, Accounts & Baseline Familiarity

- [ ] **Day 1 (Sept 13 - Tomorrow): Team Setup & GPU Verification**
  - [ ] Send the WhatsApp message to your group.
  - [ ] Register all 4 members on the official Unstop page.
  - [ ] All 4 members sign up on Kaggle.com and verify mobile phone numbers (Unlocks 4 × 30 = 120 hours free GPU!).
  - [ ] Run `python test_downloader_demo.py` on your local laptop to verify image downloads.

- [ ] **Day 2 (Sept 14): OCR & Regex Mastery**
  - [ ] Run `python test_ocr_regex.py` to see unit standardization in action.
  - [ ] Member 1 explores PaddleOCR documentation/tutorial for 30 minutes.
  - [ ] Test OCR on 5-10 product images found in your kitchen/room.

- [ ] **Day 3 (Sept 15): Kaggle Notebook Test Run**
  - [ ] Run `python export_to_kaggle.py` to generate `kaggle_amazon_ml_solution.ipynb`.
  - [ ] Upload `kaggle_amazon_ml_solution.ipynb` to Kaggle, attach a free T4 GPU, and test running the initial cells.
  - [ ] Confirm PyTorch and CUDA work on Kaggle.

- [ ] **Day 4 (Sept 16): Multimodal Deep Learning Understanding**
  - [ ] Member 2 & 3 review `train_multimodal.py`.
  - [ ] Understand how ConvNeXt visual embeddings are combined with DeBERTa-v3 text embeddings.
  - [ ] Review `losses.py` (Focal Loss and SMAPE).

- [ ] **Day 5 (Sept 17): Tabular Gradient Boosting Practice**
  - [ ] Member 4 reviews `tabular_booster.py`.
  - [ ] Understand how LightGBM and CatBoost run 5-Fold Stratified Cross-Validation on text TF-IDF + OCR features.

---

## 🚀 Week 2: Advanced SOTA Models & Mock Drills

- [ ] **Day 6 (Sept 18): Vision-Language Models (VLM)**
  - [ ] Review `florence2_vlm_extractor.py`.
  - [ ] Test running Florence-2 on Kaggle GPU on a sample product image to see visual reasoning live.

- [ ] **Day 7 (Sept 19): Meta DINOv2 Exploration**
  - [ ] Review `dinov2_feature_extractor.py`.
  - [ ] Understand why self-supervised features capture product depth, geometry, and shape.

- [ ] **Day 8 (Sept 20): Test-Time Augmentation (TTA)**
  - [ ] Review `tta_inference.py`.
  - [ ] Observe how multi-scale and horizontal flip averaging eliminates borderline prediction errors.

- [ ] **Day 9 (Sept 21): Kaggle Grandmaster Out-Of-Fold Ensembling**
  - [ ] Review `oof_stacking_ensemble.py`.
  - [ ] Understand how Nelder-Mead optimization calculates the best weights to blend VLM + PyTorch + Tabular predictions.

- [ ] **Day 10 (Sept 22): Pseudo-Labeling Strategy**
  - [ ] Review `pseudo_labeler.py`.
  - [ ] Understand how 100% agreement on test predictions boosts training data for the final +3% score jump.

- [ ] **Day 11 (Sept 23): Adversarial Validation & Drift Detection**
  - [ ] Review `adversarial_validation.py`.
  - [ ] Understand how sample importance weighting prevents the Public-to-Private Leaderboard shakeup.

- [ ] **Day 12 (Sept 24 - The Eve of Battle): Mock Hackathon Simulation**
  - [ ] Conduct a 2-hour dry-run with your team.
  - [ ] Test `python run_pipeline.py --stage all` end-to-end.
  - [ ] Verify `submission_validator.py` outputs green light on a test CSV.
  - [ ] Rest well and get 8 hours of sleep.

---

## 💥 Sept 25 (Competition Kickoff Day - 48-Hour Sprint)

- [ ] **Hour 0 - 1:** Download `train.csv` and `test.csv` from Unstop.
- [ ] **Hour 1 - 2:** Launch `download_images.py` (Finish downloading 100,000 images in 20 mins!).
- [ ] **Hour 2 - 3:** Submit first valid baseline submission using OCR/Regex -> **Take initial Top 50 spot on Leaderboard!**
- [ ] **Day 2:** Train Florence-2 VLM + ConvNeXt + DeBERTa-v3 5-Fold Models in parallel across all 4 team Kaggle accounts.
- [ ] **Day 2 Night:** Run `oof_stacking_ensemble.py` + `pseudo_labeler.py` + `tta_inference.py`.
- [ ] **Final Hours:** Run `submission_validator.py` -> Upload final submission -> **Secure Top 10 Grand Finale Rank!**
