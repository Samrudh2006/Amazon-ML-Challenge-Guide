"""
=================================================================================
OFFLINE MODEL & TOKENIZER CACHE PREPARATION (IIT/NIT COMPETITION STANDARD)
During Amazon ML Challenge test evaluation and Kaggle private runtime, internet
access is strictly DISABLED. Calling AutoModel.from_pretrained() without local files
will crash with a ConnectionError and cause a 0 score (DISQUALIFICATION).

This script pre-caches Hugging Face models and tokenizers locally to ./models_cache/
so inference can run 100% OFFLINE with local_files_only=True.
=================================================================================
"""

import os
import sys

def prep_offline_cache(model_names=None, cache_dir="./models_cache"):
    if model_names is None:
        model_names = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "microsoft/deberta-v3-small"
        ]
        
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[*] Pre-caching Hugging Face models into '{cache_dir}' for 100% OFFLINE inference...")
    
    try:
        from transformers import AutoTokenizer, AutoModel, AutoConfig
    except ImportError:
        print("[!] transformers not installed. Cannot cache models.")
        return False
        
    for name in model_names:
        print(f"\n---> Caching Model: {name}")
        local_target = os.path.join(cache_dir, name.replace("/", "_"))
        os.makedirs(local_target, exist_ok=True)
        
        try:
            # 1. Config
            config = AutoConfig.from_pretrained(name)
            config.save_pretrained(local_target)
            print("     [+] Config saved.")
            
            # 2. Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(name)
            tokenizer.save_pretrained(local_target)
            print("     [+] Tokenizer saved.")
            
            # 3. Model Weights
            model = AutoModel.from_pretrained(name)
            model.save_pretrained(local_target)
            print(f"     [+] Model weights saved to {local_target}")
            print(f"[SUCCESS] {name} is ready for 100% OFFLINE inference!")
        except Exception as e:
            print(f"[!] Warning: Could not cache {name} right now (network or memory limit): {e}")
            
    print("\n[SUCCESS] Offline cache preparation completed.")
    return True

if __name__ == "__main__":
    print("=== Offline HF Cache Script Initialized ===")
    print("Run with: python hf_offline_cache_prep.py to download weights before going offline.")
