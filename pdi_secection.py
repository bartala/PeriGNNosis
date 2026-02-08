import json
import numpy as np
import os
from collections import Counter

PTH ='...'

json_path = os.path.join(PTH, 'graphRAG/data/full_results.json')

def analyze_feature_importance(json_path):
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    subsets = results.get('selected_subsets', [])
    num_folds = len(subsets)
    
    # Flatten the list of all selected indices
    all_selected = [item for sub in subsets for item in sub]
    counts = Counter(all_selected)
    
    print("=== PDI FEATURE ROBUSTNESS ANALYSIS ===")
    print(f"Total Folds Analyzed: {num_folds}")
    print("-" * 40)
    print(f"{'PDI Index':<12} | {'Selection Frequency':<20} | {'Stability %'}")
    print("-" * 40)
    
    # Sort by frequency (highest first)
    for pdi, freq in counts.most_common():
        stability = (freq / num_folds) * 100
        print(f"{pdi:<12} | {freq:<20} | {stability:>10.1f}%")

    # Identify the "Core" features (those appearing in > 60% of folds)
    core_features = [pdi for pdi, freq in counts.items() if (freq / num_folds) >= 0.6]
    print("-" * 40)
    print(f"Core Predictors (>= 60% stability): {core_features}")

analyze_feature_importance(json_path)
