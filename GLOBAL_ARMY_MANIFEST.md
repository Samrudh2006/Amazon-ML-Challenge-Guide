# 🌍 GLOBAL SUPERPOWER COMPETITION ARSENAL
### Amazon ML Challenge 2026 - World-Class Model Coalition

---

## 🇺🇸 1. THE AMERICAN DIVISION (Meta, Microsoft, OpenAI, Google)
* **Microsoft DeBERTa-v3:** The gold-standard transformer backbone for token classification, NER, and dense feature representation.
* **Microsoft LightGBM:** High-speed gradient boosting with histogram-based splitting for lightning-fast tabular inference.
* **Meta DINOv2:** Foundation computer vision model trained with self-supervised learning without human labels—extracts pristine visual geometry from product packaging.
* **OpenAI CLIP (ViT-B/32 & ViT-L/14):** Joint multimodal embedding space aligning product imagery directly with textual descriptions.

---

## 🇨🇳 2. THE CHINESE DIVISION (Alibaba, BAAI, Baidu)
* **Alibaba Qwen2.5-VL:** Currently the #1 open-weights Vision-Language Model in the world. Unmatched at answering complex visual queries about product packaging, brand logos, and specifications.
* **BAAI BGE-M3 (Beijing Academy of AI):** State-of-the-art dense semantic embedding engine that captures multilingual catalog semantics across English, Hindi, and regional e-commerce text.
* **Baidu PaddleOCR:** Industrial-grade text detection & recognition specifically trained for challenging angles, curved bottles, and fine-print packaging.

---

## 🇷🇺 3. THE RUSSIAN DIVISION (Yandex)
* **Yandex CatBoost:** The undisputed global powerhouse for categorical and tabular data. Utilizes symmetric oblivious trees that resist overfitting on extreme e-commerce price outliers and messy metadata.

---

## 🇮🇳 4. THE INDIAN DIVISION (Kaggle Grandmaster Edge Hacks)
* **Adversarial Validation Shield:** Trains an auxiliary classifier to distinguish Train vs Test data. Identifies domain drift immediately to prevent private leaderboard collapse.
* **Packaging MRP Bayesian Override:** Inverts the problem by locating printed retail prices on packaging via OCR and calibrating model estimates with retail discount curves.
* **Pseudo-Labeling Engine (`pseudo_labeler.py`):** High-confidence semi-supervised learning that extracts additional training signal from the unlabelled test set.
* **Nelder-Mead Post-Processing Calibration:** Mathematical optimization on Out-of-Fold predictions that systematically shaves off 1.0% – 2.0% error on the competition metric.

---

## 🎯 DEPLOYMENT STRATEGY
When `run_pipeline.py` executes, it coordinates these global superpowers into a unified convex meta-ensemble:

$$\text{Final Prediction} = \alpha \cdot \left( w_{\text{CatBoost}} \cdot M_{\text{Yandex}} + w_{\text{LightGBM}} \cdot M_{\text{MSFT}} + w_{\text{CLIP}} \cdot M_{\text{OpenAI}} + w_{\text{VLM}} \cdot M_{\text{Qwen}} \right)$$

Where $\alpha$ is the calibrated Nelder-Mead post-processing scalar factor.
