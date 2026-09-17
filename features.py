import re
import pandas as pd
import numpy as np

def extract_ipq(text):
    """
    Extract Item Pack Quantity (IPQ) e.g., 'Pack of 3', 'Set of 12', '4-pack', 'Count 10'
    """
    if not isinstance(text, str) or not text.strip():
        return 1.0
        
    patterns = [
        r'\bpack\s*of\s*(\d+)\b',
        r'\bset\s*of\s*(\d+)\b',
        r'\b(\d+)\s*(?:pack|pk|pcs|pieces|count|ct|units)\b',
        r'\bipq\s*[:=]?\s*(\d+)\b',
        r'\((\d+)\s*pack\)',
        r'(\d+)\s*in\s*1\b'
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if 1 <= val <= 200:  # Sanity check against arbitrary numbers
                return val
    return 1.0

def extract_weight_in_grams(text):
    """
    Extract weight and normalize to grams (e.g. 2.5 kg -> 2500.0 g)
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
        
    # Check for KG
    match_kg = re.search(r'(\d+(?:\.\d+)?)\s*(?:kg|kilogram|kilo|kgs)\b', text, re.IGNORECASE)
    if match_kg:
        return float(match_kg.group(1)) * 1000.0
        
    # Check for Grams
    match_g = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|gm|gram|grams|gms)\b', text, re.IGNORECASE)
    if match_g:
        return float(match_g.group(1))
        
    # Check for Pounds / Oz
    match_lb = re.search(r'(\d+(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)\b', text, re.IGNORECASE)
    if match_lb:
        return float(match_lb.group(1)) * 453.592
        
    match_oz = re.search(r'(\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces)\b', text, re.IGNORECASE)
    if match_oz:
        return float(match_oz.group(1)) * 28.3495
        
    return 0.0

def extract_volume_in_ml(text):
    """
    Extract volume and normalize to millilitres (e.g. 1.5 L -> 1500.0 ml)
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
        
    match_l = re.search(r'(\d+(?:\.\d+)?)\s*(?:l|ltr|litre|litres|liter|liters)\b', text, re.IGNORECASE)
    if match_l:
        return float(match_l.group(1)) * 1000.0
        
    match_ml = re.search(r'(\d+(?:\.\d+)?)\s*(?:ml|millilitre|milliliter|millilitres)\b', text, re.IGNORECASE)
    if match_ml:
        return float(match_ml.group(1))
        
    match_floz = re.search(r'(\d+(?:\.\d+)?)\s*(?:fl\s*oz|fluid\s*ounce)\b', text, re.IGNORECASE)
    if match_floz:
        return float(match_floz.group(1)) * 29.5735
        
    return 0.0

def extract_dimensions_in_cm(text):
    """
    Extract dimension (cm, mm, inch) normalized to cm
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
        
    match_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre|meters)\b', text, re.IGNORECASE)
    if match_m:
        return float(match_m.group(1)) * 100.0
        
    match_cm = re.search(r'(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimetre)\b', text, re.IGNORECASE)
    if match_cm:
        return float(match_cm.group(1))
        
    match_inch = re.search(r'(\d+(?:\.\d+)?)\s*(?:inch|inches|in|\")\b', text, re.IGNORECASE)
    if match_inch:
        return float(match_inch.group(1)) * 2.54
        
    return 0.0

def build_tabular_features(df, text_column='catalog_content'):
    """
    Constructs comprehensive numerical & boolean signals from product text
    """
    texts = df[text_column].fillna('').astype(str)
    
    feats = pd.DataFrame(index=df.index)
    
    # Text length stats
    feats['char_len'] = texts.str.len()
    feats['word_count'] = texts.apply(lambda x: len(x.split()))
    feats['num_digits'] = texts.apply(lambda x: sum(c.isdigit() for c in x))
    feats['num_uppercase'] = texts.apply(lambda x: sum(c.isupper() for c in x))
    feats['upper_ratio'] = feats['num_uppercase'] / (feats['char_len'] + 1.0)
    feats['digit_ratio'] = feats['num_digits'] / (feats['char_len'] + 1.0)
    
    # Entity Extraction
    feats['ipq'] = texts.apply(extract_ipq)
    feats['weight_g'] = texts.apply(extract_weight_in_grams)
    feats['volume_ml'] = texts.apply(extract_volume_in_ml)
    feats['dimension_cm'] = texts.apply(extract_dimensions_in_cm)
    
    # Log transforms for scale features
    feats['log_ipq'] = np.log1p(feats['ipq'])
    feats['log_weight'] = np.log1p(feats['weight_g'])
    feats['log_volume'] = np.log1p(feats['volume_ml'])
    
    # Premium keywords (correlated with higher price)
    keywords = [
        'premium', 'luxury', 'pro', 'ultra', 'organic', 'authentic',
        'leather', 'gold', 'silver', 'wireless', 'bluetooth', 'combo',
        'imported', 'stainless steel', 'heavy duty', 'edition'
    ]
    for kw in keywords:
        feats[f'kw_{kw.replace(" ", "_")}'] = texts.str.contains(rf'\b{kw}\b', case=False, regex=True).astype(np.int8)
        
    return feats
