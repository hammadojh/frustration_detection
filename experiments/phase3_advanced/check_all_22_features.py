#!/usr/bin/env python3
"""
Check all 22 features to understand which are actually constant
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
import sys

sys.path.append('/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase2_features')
from fast_feature_extractor import FastFrustrationFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load processed examples"""
    processed_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/data/subset_processed.json"
    with open(processed_path, 'r') as f:
        examples = json.load(f)
    
    texts = [ex['text'] for ex in examples]
    labels = [ex['label'] for ex in examples]
    
    n = len(texts)
    train_end = int(0.7 * n)
    
    return texts[:train_end], labels[:train_end]

def extract_all_features_with_names(texts):
    """Extract all features and return with names"""
    extractor = FastFrustrationFeatureExtractor()
    
    # All 22 feature names in order
    feature_names = [
        'sentiment_slope', 'sentiment_volatility', 'avg_politeness', 
        'politeness_decline', 'avg_confusion', 'avg_negation', 
        'avg_caps', 'avg_exclamation',  # linguistic (8)
        'total_turns', 'avg_turn_length', 'repeated_turns', 'corrections',  # dialogue (4)
        'escalation_requests', 'negative_feedback',  # behavioral (2)
        'avg_urgency', 'urgency_increase', 'task_complexity',  # contextual (3)
        'emotion_drift', 'emotion_volatility',  # emotion_dynamics (2)
        'response_clarity', 'response_relevance',  # system (2)
        'trust_decline'  # user_model (1)
    ]
    
    all_bundles = [
        'linguistic_bundle', 'dialogue_bundle', 'behavioral_bundle',
        'contextual_bundle', 'emotion_dynamics_bundle', 'system_bundle', 'user_model_bundle'
    ]
    
    all_features = []
    
    for text in texts:
        text_features = []
        for bundle in all_bundles:
            bundle_features = extractor.extract_bundle_features([text], bundle)
            text_features.extend(list(bundle_features.values()))
        
        all_features.append(text_features)
    
    return np.array(all_features), feature_names

def main():
    logger.info("=" * 80)
    logger.info("CHECKING ALL 22 FEATURES FOR VARIANCE")
    logger.info("=" * 80)
    
    # Load data
    texts, labels = load_data()
    
    # Extract features
    logger.info("Extracting features...")
    features, feature_names = extract_all_features_with_names(texts)
    
    logger.info(f"Extracted {features.shape[1]} features from {features.shape[0]} examples")
    
    # Check raw variance
    logger.info("\nRAW FEATURE STATISTICS:")
    logger.info("=" * 60)
    
    results = []
    for i, name in enumerate(feature_names):
        col = features[:, i]
        variance = np.var(col)
        mean = np.mean(col)
        std = np.std(col)
        unique_vals = len(np.unique(col))
        min_val = np.min(col)
        max_val = np.max(col)
        
        results.append({
            'index': i,
            'feature_name': name,
            'variance': variance,
            'mean': mean,
            'std': std,
            'unique_values': unique_vals,
            'min': min_val,
            'max': max_val,
            'is_constant': variance < 1e-10
        })
        
        logger.info(f"{i:2d}. {name:<25} | Var: {variance:.6f} | Mean: {mean:.4f} | Unique: {unique_vals:3d} | Range: [{min_val:.4f}, {max_val:.4f}]")
    
    # Scale features and check variance again
    logger.info("\n" + "=" * 80)
    logger.info("SCALED FEATURE VARIANCE CHECK")
    logger.info("=" * 80)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    constant_features = []
    nonconstant_features = []
    
    for i, name in enumerate(feature_names):
        col_scaled = features_scaled[:, i]
        variance_scaled = np.var(col_scaled)
        
        is_constant = variance_scaled < 1e-6
        if is_constant:
            constant_features.append((i, name, variance_scaled))
            status = "CONSTANT"
        else:
            nonconstant_features.append((i, name, variance_scaled))
            status = "VARIABLE"
        
        logger.info(f"{i:2d}. {name:<25} | Scaled Var: {variance_scaled:.8f} | {status}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"Total features: {len(feature_names)}")
    logger.info(f"Constant features: {len(constant_features)}")
    logger.info(f"Non-constant features: {len(nonconstant_features)}")
    
    logger.info(f"\nCONSTANT FEATURES ({len(constant_features)}):")
    for i, (idx, name, var) in enumerate(constant_features, 1):
        logger.info(f"  {i:2d}. [{idx:2d}] {name} (var: {var:.8f})")
    
    logger.info(f"\nNON-CONSTANT FEATURES ({len(nonconstant_features)}):")
    for i, (idx, name, var) in enumerate(nonconstant_features, 1):
        logger.info(f"  {i:2d}. [{idx:2d}] {name} (var: {var:.8f})")
    
    # Cross-check with our 12 features
    our_12_indices = [2, 4, 5, 7, 9, 11, 12, 14, 16, 19, 20, 21]
    our_12_names = [
        'avg_politeness', 'avg_confusion', 'avg_negation', 'avg_exclamation', 
        'avg_turn_length', 'corrections', 'escalation_requests', 'avg_urgency', 
        'task_complexity', 'response_clarity', 'response_relevance', 'trust_decline'
    ]
    
    logger.info(f"\nCROSS-CHECK WITH OUR 12 NON-CONSTANT FEATURES:")
    logger.info("=" * 60)
    
    nonconstant_indices = [idx for idx, _, _ in nonconstant_features]
    
    matches = 0
    for i, idx in enumerate(our_12_indices):
        name = our_12_names[i]
        actual_name = feature_names[idx]
        is_nonconstant = idx in nonconstant_indices
        
        if is_nonconstant:
            matches += 1
            status = "✅ CORRECT"
        else:
            status = "❌ MISMATCH"
        
        logger.info(f"  [{idx:2d}] {name:<20} vs {actual_name:<20} | {status}")
    
    logger.info(f"\nMatch rate: {matches}/{len(our_12_indices)} = {matches/len(our_12_indices)*100:.1f}%")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("/Users/omarhammad/Documents/code_local/frustration_researcher/results/all_22_features_variance_check.csv", index=False)
    
    logger.info(f"\nResults saved to: all_22_features_variance_check.csv")
    
    return constant_features, nonconstant_features

if __name__ == "__main__":
    main()