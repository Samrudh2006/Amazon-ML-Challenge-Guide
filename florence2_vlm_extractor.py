"""
Amazon ML Challenge 2026 - State-of-the-Art Vision-Language Model (VLM) Pipeline
================================================================================
Uses Microsoft Florence-2 (SOTA Lightweight Vision-Language Foundation Model)
for Direct Visual Question Answering, Dense OCR, and Zero-Shot Entity Extraction.

Why Florence-2 Beats Traditional OCR + CNN:
1. End-to-End Multimodal: It "sees" the image and understands natural language prompts simultaneously.
2. Dense Packaging OCR: Extracts grounded text, text coordinates, and product specs with high precision.
3. Fast & Kaggle-Ready: 0.2B (base) or 0.7B (large) parameters fits easily on a single free Kaggle T4 GPU (16GB VRAM)
   with FP16 / FlashAttention.
"""

import os
import sys
import argparse
import re
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Standard Amazon unit mapping
UNIT_MAP = {
    "g": "gram", "gm": "gram", "gms": "gram", "gram": "gram", "grams": "gram",
    "kg": "kilogram", "kgs": "kilogram", "kilo": "kilogram", "kilogram": "kilogram",
    "mg": "milligram", "mgs": "milligram", "milligram": "milligram",
    "oz": "ounce", "ounce": "ounce", "ounces": "ounce",
    "lb": "pound", "lbs": "pound", "pound": "pound",
    "ml": "millilitre", "mls": "millilitre", "millilitre": "millilitre",
    "l": "litre", "ltr": "litre", "liter": "litre", "liters": "litre", "litre": "litre",
    "cl": "centilitre", "fl oz": "fluid ounce", "floz": "fluid ounce", "gallon": "gallon",
    "cm": "centimetre", "mm": "millimetre", "m": "metre", "inch": "inch", "foot": "foot",
    "v": "volt", "volt": "volt", "w": "watt", "watt": "watt", "kw": "kilowatt"
}

def parse_vlm_response_to_entity(vlm_text, target_entity=None):
    """
    Parses VLM natural language answers into strict Amazon format: '<number> <allowed_unit>'
    """
    if not isinstance(vlm_text, str):
        return ""
        
    text = vlm_text.lower().strip()
    all_units_sorted = sorted(UNIT_MAP.keys(), key=len, reverse=True)
    units_pattern = "|".join([re.escape(u) for u in all_units_sorted])
    
    matches = re.findall(rf'(\d+(?:\.\d+)?)\s*({units_pattern})\b', text)
    if matches:
        val_str, unit_raw = matches[0]
        canonical_unit = UNIT_MAP.get(unit_raw, unit_raw)
        try:
            val_float = float(val_str)
            return f"{val_float:g} {canonical_unit}"
        except ValueError:
            pass
    return ""

class Florence2VLM:
    def __init__(self, model_id="microsoft/Florence-2-base", device=None, fp16=True):
        self.model_id = model_id
        self.device = device or ("cuda" if (torch.cuda.is_available()) else "cpu")
        self.fp16 = fp16 and ("cuda" in self.device)
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        if not HAS_TRANSFORMERS:
            print("[!] Transformers library is required to run Florence-2 VLM.")
            return

        print(f"[*] Loading Florence-2 Model: {self.model_id} onto {self.device.upper()} (FP16={self.fp16})...")
        dtype = torch.float16 if self.fp16 else torch.float32
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model.eval()
            print("[+] Florence-2 VLM loaded successfully!")
        except Exception as e:
            print(f"[!] Error loading Florence-2: {e}")

    def query_image(self, image_path, task_prompt="<OCR_WITH_REGION>", custom_question=None):
        """
        Queries the VLM with either predefined Florence-2 tasks:
        - '<OCR_WITH_REGION>' (Dense visual text)
        - '<DETAILED_CAPTION>' (Deep product description)
        Or a custom question prompt.
        """
        if self.model is None or self.processor is None:
            return ""

        if not os.path.exists(image_path):
            return ""

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            if custom_question:
                prompt = f"<DocVQA> {custom_question}"
            else:
                prompt = task_prompt

            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.fp16:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=3,
                    do_sample=False
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=prompt.split()[0],
                image_size=(image.width, image.height)
            )
            return str(parsed_answer)
        except Exception as e:
            return f"Error: {e}"

def run_vlm_batch(image_dir, output_csv="vlm_predictions.csv", model_id="microsoft/Florence-2-base", limit=None):
    vlm = Florence2VLM(model_id=model_id)
    if vlm.model is None:
        print("[!] Florence-2 engine not loaded. Please ensure GPU environment has transformers installed.")
        return

    import glob
    image_paths = glob.glob(os.path.join(image_dir, "*.*"))
    image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]]
    if limit:
        image_paths = image_paths[:limit]

    print(f"[*] Processing {len(image_paths)} images with Florence-2 VLM...")
    results = []

    for img_path in tqdm(image_paths, desc="Florence-2 Inference"):
        filename = os.path.basename(img_path)
        # Ask targeted packaging questions
        ocr_response = vlm.query_image(img_path, task_prompt="<OCR>")
        weight_answer = vlm.query_image(img_path, custom_question="What is the net weight, volume, or quantity printed on this product?")
        
        parsed_entity = parse_vlm_response_to_entity(weight_answer) or parse_vlm_response_to_entity(ocr_response)
        
        results.append({
            "image_filename": filename,
            "vlm_raw_answer": weight_answer,
            "vlm_ocr": ocr_response,
            "vlm_prediction": parsed_entity
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"[+] Saved Florence-2 VLM predictions to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Florence-2 SOTA VLM Extractor")
    parser.add_argument("--image_dir", type=str, default="sample_images")
    parser.add_argument("--output_csv", type=str, default="vlm_predictions.csv")
    parser.add_argument("--model", type=str, default="microsoft/Florence-2-base")
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    print("=" * 65)
    print("  AMAZON ML CHALLENGE - SOTA VISION-LANGUAGE MODEL ENGINE")
    print("=" * 65)
    print(f"[*] Model: {args.model}")
    print(f"[*] PyTorch CUDA Available: {torch.cuda.is_available()}")
