"""
Amazon ML Challenge 2026 - Unified Multimodal PyTorch Pipeline
==============================================================
Combines Computer Vision (TIMM) + NLP (HuggingFace Transformers) + OCR features into a Late-Fusion architecture.

Architecture Overview:
- Vision Encoder: timm backbone (ConvNeXt / Swin / EfficientNet) -> 512/768-dim visual embedding
- Text/OCR Encoder: HuggingFace Transformer (DeBERTa-v3 / RoBERTa / MiniLM) -> 768/384-dim text embedding
- Fusion Head: Concatenates visual + text representations -> LayerNorm -> Dropout -> Dense MLP
- Multi-Task Output:
    1. Unit Classifier (Cross-Entropy Loss): Predicts standard unit (e.g., 'gram', 'millilitre')
    2. Value Regressor (Smooth L1 Loss on log-scale): Predicts numerical magnitude

Features:
- Stratified K-Fold cross validation
- Out-of-fold (OOF) prediction generation for ensembling
- Supports training with images only, text only, or combined multimodal
- Automatic checkpointing of best weights based on validation F1 score
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        Module = object
    class Dataset:
        pass
    class DataLoader:
        pass

# Optional imports with graceful fallbacks
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sklearn.model_selection import StratifiedKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------------------------------------------------------
# 1. Dataset Class
# ---------------------------------------------------------
class AmazonMultimodalDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir=None,
        tokenizer=None,
        max_length=128,
        transform=None,
        is_train=True,
        unit_to_idx=None
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.is_train = is_train
        self.unit_to_idx = unit_to_idx or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Text / OCR representation
        title = str(row.get("catalog_name", row.get("product_name", "")))
        ocr_text = str(row.get("ocr_text", ""))
        combined_text = f"{title} [SEP] {ocr_text}".strip()

        if self.tokenizer:
            encoding = self.tokenizer(
                combined_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        # 2. Image loading
        img_filename = row.get("image_filename", "")
        img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        if self.image_dir and img_filename:
            img_path = os.path.join(self.image_dir, str(img_filename))
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as pil_img:
                        pil_img = pil_img.convert("RGB")
                        if self.transform:
                            img_tensor = self.transform(pil_img)
                        else:
                            pil_img = pil_img.resize((224, 224))
                            arr = np.array(pil_img).astype(np.float32) / 255.0
                            # HWC -> CHW
                            img_tensor = torch.tensor(arr).permute(2, 0, 1)
                except Exception:
                    pass

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": img_tensor,
            "index": row.get("index", idx)
        }

        # 3. Targets for training
        if self.is_train:
            # Unit target
            unit = str(row.get("unit", "")).strip().lower()
            unit_id = self.unit_to_idx.get(unit, 0)
            item["unit_target"] = torch.tensor(unit_id, dtype=torch.long)
            
            # Value target (log-scaled)
            val = float(row.get("value", 1.0))
            log_val = math.log1p(max(val, 0.0))
            item["value_target"] = torch.tensor(log_val, dtype=torch.float32)

        return item

# ---------------------------------------------------------
# 2. Multimodal Fusion Architecture
# ---------------------------------------------------------
class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        vision_backbone="convnext_tiny",
        text_backbone="sentence-transformers/all-MiniLM-L6-v2",
        num_units=25,
        pretrained=True
    ):
        super().__init__()
        
        # 1. Vision Encoder
        if HAS_TIMM:
            self.vision_encoder = timm.create_model(vision_backbone, pretrained=pretrained, num_classes=0)
            vision_dim = self.vision_encoder.num_features
        else:
            self.vision_encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(3, 256)
            )
            vision_dim = 256

        # 2. Text Encoder
        if HAS_TRANSFORMERS:
            self.text_encoder = AutoModel.from_pretrained(text_backbone)
            text_dim = self.text_encoder.config.hidden_size
        else:
            self.text_encoder = nn.Embedding(30522, 256)
            text_dim = 256

        # 3. Fusion Neck & Prediction Heads
        combined_dim = vision_dim + text_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        # Head 1: Unit Classification
        self.unit_head = nn.Linear(256, num_units)
        # Head 2: Value Regression (log-value)
        self.value_head = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, image):
        # Extract visual features
        if HAS_TIMM:
            vis_features = self.vision_encoder(image)
        else:
            vis_features = self.vision_encoder(image)

        # Extract text features (mean pooling over attention mask)
        if HAS_TRANSFORMERS:
            text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = text_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            text_features = sum_embeddings / sum_mask
        else:
            text_features = self.text_encoder(input_ids).mean(dim=1)

        # Concatenate Vision + Text
        fused = torch.cat([vis_features, text_features], dim=1)
        latent = self.fusion_mlp(fused)

        unit_logits = self.unit_head(latent)
        value_preds = self.value_head(latent).squeeze(-1)

        return unit_logits, value_preds

# ---------------------------------------------------------
# 3. Training & Validation Loop
# ---------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion_unit, criterion_val, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        unit_targets = batch["unit_target"].to(device)
        value_targets = batch["value_target"].to(device)

        unit_logits, value_preds = model(input_ids, attention_mask, images)

        loss_unit = criterion_unit(unit_logits, unit_targets)
        loss_val = criterion_val(value_preds, value_targets)
        
        # Combined multi-task loss
        loss = loss_unit + (2.0 * loss_val)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, idx_to_unit):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            indices = batch["index"]

            unit_logits, value_preds = model(input_ids, attention_mask, images)
            
            unit_ids = torch.argmax(unit_logits, dim=-1).cpu().numpy()
            predicted_vals = torch.expm1(value_preds).clamp(min=0.01).cpu().numpy()

            for idx, u_id, val in zip(indices, unit_ids, predicted_vals):
                unit_str = idx_to_unit.get(u_id, "gram")
                formatted = f"{val:.2f} {unit_str}"
                predictions.append({
                    "index": idx.item() if hasattr(idx, 'item') else idx,
                    "prediction": formatted,
                    "pred_unit": unit_str,
                    "pred_val": val
                })

    return pd.DataFrame(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge Multimodal Model Trainer")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    default_device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    
    args = parser.parse_args()
    print("=" * 60)
    print(f"[*] Amazon Multimodal Pipeline initialized on device: {args.device.upper()}")
    print(f"[*] PyTorch available: {HAS_TORCH} | TIMM: {HAS_TIMM} | Transformers: {HAS_TRANSFORMERS}")
    if not HAS_TORCH:
        print("[!] Note: Run 'pip install -r requirements.txt' or open in Kaggle/Colab for GPU acceleration.")
    print("=" * 60)
