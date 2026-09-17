"""
=================================================================================
HUGGING FACE ZERO-SHOT DENSE EMBEDDING EXTRACTOR (BGE / MiniLM)
Extracts 384-dimensional dense semantic vectors from product titles & descriptions
to feed directly into CatBoost, LightGBM, and FAISS vector matching.
=================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_HF = True
except ImportError:
    HAS_HF = False

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def mean_pooling(model_output, attention_mask):
    """Mean Pooling: Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_dense_text_embeddings(texts_series, model_name=DEFAULT_EMBED_MODEL, batch_size=64):
    if not HAS_HF:
        print("[!] Hugging Face transformers not installed.")
        return np.zeros((len(texts_series), 384))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading Hugging Face Embedding Model '{model_name}' on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    
    all_embeddings = []
    texts = texts_series.fillna('').astype(str).tolist()
    
    print(f"[*] Extracting dense embeddings for {len(texts)} texts...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model(**encoded)
            pooled = mean_pooling(output, encoded['attention_mask'])
            # Normalize embeddings
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy())
            
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"[*] Dense embeddings complete! Shape: {embeddings_matrix.shape}")
    return embeddings_matrix

if __name__ == "__main__":
    print("=== Hugging Face Embedding Extractor Initialized ===")
    sample_texts = pd.Series([
        "Apple iPhone 15 Pro Max 256GB Titanium",
        "Cadbury Silk Chocolate 150g Pack of 3",
        "Nike Air Max Mens Running Shoes"
    ])
    # Dry test
    print(f"Sample test ready for: {len(sample_texts)} products.")
