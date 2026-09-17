"""
=================================================================================
IIT/NIT CHAMPION ARCHITECTURE: MULTIMODAL LATE FUSION NEURAL NETWORK
Combines:
  1. Dense Text Semantic Embeddings (DeBERTa / MiniLM / BGE)
  2. Vision Semantic Embeddings (CLIP / SigLIP / ViT)
  3. Domain Tabular Features (Regex Units, Cluster IDs, Out-of-fold target encodings)
Loss Function: Differentiable PyTorch SMAPE Loss + Log-Price Regression Head
=================================================================================
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DifferentiableSMAPELoss(nn.Module):
    """
    Differentiable SMAPE Loss for PyTorch:
    SMAPE = 200% * |y_pred - y_true| / (|y_pred| + |y_true| + eps)
    Since model predicts in log1p space, we exponentiate predictions back to price space.
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, log_preds, log_targets):
        preds = torch.expm1(torch.clamp(log_preds, min=0.0, max=15.0))
        targets = torch.expm1(torch.clamp(log_targets, min=0.0, max=15.0))
        
        numerator = torch.abs(preds - targets)
        denominator = (torch.abs(preds) + torch.abs(targets)) / 2.0 + self.eps
        smape = torch.mean(numerator / denominator) * 100.0
        return smape

class MultimodalDataset(Dataset):
    def __init__(self, tabular_feats, text_embeds=None, vision_embeds=None, targets=None):
        self.tabular = torch.tensor(tabular_feats, dtype=torch.float32)
        
        if text_embeds is not None:
            self.text = torch.tensor(text_embeds, dtype=torch.float32)
        else:
            self.text = None
            
        if vision_embeds is not None:
            self.vision = torch.tensor(vision_embeds, dtype=torch.float32)
        else:
            self.vision = None
            
        if targets is not None:
            # targets should already be log1p(price)
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        else:
            self.targets = None

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        item = {'tabular': self.tabular[idx]}
        if self.text is not None:
            item['text'] = self.text[idx]
        if self.vision is not None:
            item['vision'] = self.vision[idx]
        if self.targets is not None:
            item['target'] = self.targets[idx]
        return item

class MultimodalLateFusionNet(nn.Module):
    """
    IIT/NIT Style Multimodal Late Fusion Network.
    Projects text, vision, and tabular representations into a unified latent space,
    concatenates with residual projection, passes through deep GELU MLP blocks with Dropout.
    """
    def __init__(self, tabular_dim, text_dim=384, vision_dim=512, hidden_dim=256, dropout=0.2):
        super().__init__()
        
        # 1. Tabular Projection Sub-network
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Text Projection Sub-network (if available)
        self.has_text = text_dim > 0
        if self.has_text:
            self.text_net = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
        # 3. Vision Projection Sub-network (if available)
        self.has_vision = vision_dim > 0
        if self.has_vision:
            self.vision_net = nn.Sequential(
                nn.Linear(vision_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
        # 4. Fusion Layers
        combined_dim = hidden_dim * (1 + int(self.has_text) + int(self.has_vision))
        
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 1) # Predicts log1p(price)
        )

    def forward(self, tabular, text=None, vision=None):
        streams = [self.tabular_net(tabular)]
        
        if self.has_text and text is not None:
            streams.append(self.text_net(text))
            
        if self.has_vision and vision is not None:
            streams.append(self.vision_net(vision))
            
        fused = torch.cat(streams, dim=1)
        output = self.fusion_head(fused)
        return output

def train_multimodal_model(
    tabular_train, y_train_log, 
    text_train=None, vision_train=None,
    epochs=5, batch_size=32, lr=1e-3, device=None
):
    """
    Trains the Multimodal Late Fusion model using PyTorch AdamW and SMAPE Loss.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"[*] Initializing Multimodal Fusion Net on {device}...")
    
    tab_dim = tabular_train.shape[1]
    txt_dim = text_train.shape[1] if text_train is not None else 0
    vis_dim = vision_train.shape[1] if vision_train is not None else 0
    
    dataset = MultimodalDataset(tabular_train, text_train, vision_train, y_train_log)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MultimodalLateFusionNet(
        tabular_dim=tab_dim, 
        text_dim=txt_dim, 
        vision_dim=vis_dim
    ).to(device)
    
    criterion = DifferentiableSMAPELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            tab = batch['tabular'].to(device)
            txt = batch.get('text', None)
            vis = batch.get('vision', None)
            if txt is not None: txt = txt.to(device)
            if vis is not None: vis = vis.to(device)
            target = batch['target'].to(device)
            
            preds = model(tab, txt, vis)
            loss = criterion(preds, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(tab)
            
        scheduler.step()
        epoch_smape = total_loss / len(dataset)
        print(f"    Epoch {epoch+1:02d}/{epochs:02d} | Train SMAPE: {epoch_smape:.2f}%")
        
    print("[SUCCESS] Multimodal Late Fusion model training complete!")
    return model

if __name__ == "__main__":
    print("=== Testing IIT/NIT Multimodal Fusion Pipeline ===")
    np.random.seed(42)
    torch.manual_seed(42)
    
    N = 64
    mock_tabular = np.random.randn(N, 15).astype(np.float32)
    mock_text = np.random.randn(N, 384).astype(np.float32) # MiniLM embedding size
    mock_vision = np.random.randn(N, 512).astype(np.float32) # CLIP embedding size
    mock_prices = np.random.uniform(50, 2000, size=N)
    mock_log_prices = np.log1p(mock_prices).astype(np.float32)
    
    # Train 3 quick epochs on CPU
    model = train_multimodal_model(
        mock_tabular, mock_log_prices,
        text_train=mock_text, vision_train=mock_vision,
        epochs=3, batch_size=16
    )
    print("[SUCCESS] Multimodal Late Fusion verified and ready.")
