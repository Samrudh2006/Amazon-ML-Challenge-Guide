"""
=================================================================================
GLOBAL TOP 20 OPEN-SOURCE MODEL ZOO REGISTRY
Interactive Python Catalog for instant loading and reference of the world's
best free, open-source competitive AI models across USA, China, Russia, and Europe.
=================================================================================
"""

MODELS_ZOO = {
    # Vision-Language Models (VLMs)
    1: {"name": "Qwen2.5-VL-7B", "country": "China (Alibaba)", "type": "VLM", "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct", "ram_gpu": "16GB GPU (4-bit / FP16)"},
    2: {"name": "Florence-2-large", "country": "USA (Microsoft)", "type": "VLM", "hf_id": "microsoft/Florence-2-large", "ram_gpu": "4GB GPU / CPU Friendly"},
    3: {"name": "PaliGemma-2-3B", "country": "USA (Google)", "type": "VLM", "hf_id": "google/paligemma2-3b-pt-224", "ram_gpu": "8GB GPU"},
    4: {"name": "Molmo-7B", "country": "USA (Allen AI)", "type": "VLM", "hf_id": "allenai/Molmo-7B-D-0924", "ram_gpu": "16GB GPU"},
    5: {"name": "Pixtral-12B", "country": "France (Mistral)", "type": "VLM", "hf_id": "mistralai/Pixtral-12B-2409", "ram_gpu": "16GB+ GPU"},

    # NLP & Dense Embeddings
    6: {"name": "DeBERTa-v3-large", "country": "USA (Microsoft)", "type": "NLP Encoder", "hf_id": "microsoft/deberta-v3-large", "ram_gpu": "4GB GPU / CPU"},
    7: {"name": "ModernBERT-large", "country": "USA/France", "type": "NLP Encoder", "hf_id": "answerdotai/ModernBERT-large", "ram_gpu": "4GB GPU / Fast"},
    8: {"name": "BGE-M3", "country": "China (BAAI)", "type": "Embedding", "hf_id": "BAAI/bge-m3", "ram_gpu": "4GB GPU / CPU"},
    9: {"name": "Llama-3.2-3B-Instruct", "country": "USA (Meta)", "type": "Text LLM", "hf_id": "meta-llama/Llama-3.2-3B-Instruct", "ram_gpu": "8GB GPU"},
    10: {"name": "Gemma-2-2B-IT", "country": "USA (Google)", "type": "Text LLM", "hf_id": "google/gemma-2-2b-it", "ram_gpu": "6GB GPU / CPU"},
    11: {"name": "E5-Mistral-7B", "country": "USA (Microsoft)", "type": "Embedding", "hf_id": "intfloat/e5-mistral-7b-instruct", "ram_gpu": "16GB GPU"},
    12: {"name": "XLM-RoBERTa-large", "country": "USA (Meta)", "type": "Multilingual", "hf_id": "xlm-roberta-large", "ram_gpu": "4GB GPU / CPU"},

    # Computer Vision & OCR
    13: {"name": "DINOv2 (ViT-L/14)", "country": "USA/France (Meta)", "type": "Vision Backbone", "hf_id": "facebook/dinov2-large", "ram_gpu": "4GB GPU"},
    14: {"name": "OpenCLIP (ViT-B/32)", "country": "Global Open Source", "type": "Multimodal Dual", "hf_id": "open_clip", "ram_gpu": "2GB GPU / CPU"},
    15: {"name": "SigLIP (SO400M)", "country": "USA (Google)", "type": "Multimodal Dual", "hf_id": "google/siglip-so400m-patch14-384", "ram_gpu": "4GB GPU"},
    16: {"name": "PaddleOCR v4", "country": "China (Baidu)", "type": "Packaging OCR", "hf_id": "paddleocr", "ram_gpu": "1GB GPU / CPU"},
    17: {"name": "ConvNeXt-V2-large", "country": "USA (Meta)", "type": "Pure CNN", "hf_id": "facebook/convnextv2-large-22k-224", "ram_gpu": "4GB GPU"},

    # Tabular & GBDT Engines
    18: {"name": "CatBoost", "country": "Russia (Yandex)", "type": "GBDT (Symmetric)", "hf_id": "pip install catboost", "ram_gpu": "CPU Native / Fast"},
    19: {"name": "LightGBM", "country": "USA (Microsoft)", "type": "GBDT (Histogram)", "hf_id": "pip install lightgbm", "ram_gpu": "CPU Native / Ultra Fast"},
    20: {"name": "FAISS / NearestNeighbors", "country": "USA (Meta)", "type": "Vector Search", "hf_id": "scikit-learn / faiss-cpu", "ram_gpu": "CPU Native / Milliseconds"},

    # Official Amazon Science & AWS Models
    21: {"name": "Amazon Chronos (T5)", "country": "USA (Amazon Science)", "type": "Probabilistic Foundation", "hf_id": "amazon/chronos-t5-small", "ram_gpu": "2GB GPU / CPU"},
    22: {"name": "Amazon AutoGluon", "country": "USA (Amazon AWS)", "type": "Multi-Layer Stacking", "hf_id": "autogluon.tabular", "ram_gpu": "CPU Native / Fast"},

    # Frontier Open-Source LLMs & VLMs (DeepSeek, GLM, Yi / 01.AI)
    23: {"name": "DeepSeek-R1-Distill-1.5B", "country": "China (DeepSeek)", "type": "Reasoning LLM", "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ram_gpu": "3GB RAM / CPU Friendly"},
    24: {"name": "GLM-4V-9B", "country": "China (Zhipu AI / Tsinghua)", "type": "Multimodal VLM", "hf_id": "THUDM/glm-4v-9b", "ram_gpu": "16GB GPU (Kaggle T4)"},
    25: {"name": "Yi-1.5-6B", "country": "China (01.AI)", "type": "Dense Text LLM", "hf_id": "01-ai/Yi-1.5-6B", "ram_gpu": "12GB GPU / 4-bit CPU"}
}

def get_global_model_zoo():
    return MODELS_ZOO

def verify_model_zoo_status():
    return len(MODELS_ZOO) >= 20

def display_model_zoo():
    print("\n" + "="*85)
    print("   GLOBAL TOP 20 FREE & OPEN-SOURCE AI MODELS FOR COMPETITIVE MACHINE LEARNING")
    print("="*85)
    print(f"{'#':<3} | {'Model Name':<24} | {'Origin':<22} | {'Category':<16} | {'Compute Specs'}")
    print("-" * 85)
    for idx, data in MODELS_ZOO.items():
        print(f"{idx:<3} | {data['name']:<24} | {data['country']:<22} | {data['type']:<16} | {data['ram_gpu']}")
    print("="*85 + "\n")

if __name__ == "__main__":
    display_model_zoo()
