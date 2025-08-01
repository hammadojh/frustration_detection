#!/usr/bin/env python3
"""
Quick Feature Importance Analysis
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
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
    
    feature_name_mapping = {
        'linguistic_bundle': [
            'sentiment_slope', 'sentiment_volatility', 'avg_politeness', 
            'politeness_decline', 'avg_confusion', 'avg_negation', 
            'avg_caps', 'avg_exclamation'
        ],
        'dialogue_bundle': [
            'total_turns', 'avg_turn_length', 'repeated_turns', 'corrections'
        ],
        'behavioral_bundle': [
            'escalation_requests', 'negative_feedback'
        ],
        'contextual_bundle': [
            'avg_urgency', 'urgency_increase', 'task_complexity'
        ],
        'emotion_dynamics_bundle': [
            'emotion_drift', 'emotion_volatility'
        ],
        'system_bundle': [
            'response_clarity', 'response_relevance'
        ],
        'user_model_bundle': [
            'trust_decline'
        ]
    }
    
    all_bundles = [
        'linguistic_bundle', 'dialogue_bundle', 'behavioral_bundle',
        'contextual_bundle', 'emotion_dynamics_bundle', 'system_bundle', 'user_model_bundle'
    ]
    
    all_features = []
    feature_names_flat = []
    
    for text in texts:
        text_features = []
        for bundle in all_bundles:
            bundle_features = extractor.extract_bundle_features([text], bundle)
            text_features.extend(list(bundle_features.values()))
            
            # Add feature names for first text only
            if len(feature_names_flat) < 22:
                for feat_name in feature_name_mapping[bundle]:
                    feature_names_flat.append(f"{bundle}_{feat_name}")
        
        all_features.append(text_features)
    
    return np.array(all_features), feature_names_flat

def main():
    logger.info("=" * 80)
    logger.info("QUICK FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 80)
    
    # Load data
    texts, labels = load_data()
    
    # Extract features
    logger.info("Extracting features...")
    features, feature_names = extract_all_features_with_names(texts)
    
    logger.info(f"Extracted {features.shape[1]} features from {features.shape[0]} examples")
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Remove constant features for analysis
    non_constant_mask = np.var(features_scaled, axis=0) > 1e-6
    features_clean = features_scaled[:, non_constant_mask]
    feature_names_clean = [feature_names[i] for i in range(len(feature_names)) if non_constant_mask[i]]
    
    logger.info(f"After removing constant features: {features_clean.shape[1]} features")
    
    # Compute feature importance
    logger.info("Computing feature importance...")
    
    # F-statistic
    try:
        f_scores, f_pvalues = f_classif(features_clean, labels)
    except:
        f_scores = np.zeros(features_clean.shape[1])
        f_pvalues = np.ones(features_clean.shape[1])
    
    # Mutual Information
    try:
        mi_scores = mutual_info_classif(features_clean, labels, random_state=42)
    except:
        mi_scores = np.zeros(features_clean.shape[1])
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'feature_name': feature_names_clean,
        'f_score': f_scores,
        'f_pvalue': f_pvalues,
        'mutual_info': mi_scores,
    })
    
    # Add bundle information
    results_df['bundle'] = results_df['feature_name'].apply(
        lambda x: '_'.join(x.split('_')[:2])
    )
    
    # Sort by mutual information
    results_df = results_df.sort_values('mutual_info', ascending=False)
    
    # Display results
    logger.info("\n" + "=" * 90)
    logger.info("TOP FEATURES BY MUTUAL INFORMATION")
    logger.info("=" * 90)
    
    for i, row in results_df.head(15).iterrows():
        logger.info(f"{row['feature_name']:<40} | MI: {row['mutual_info']:.4f} | F: {row['f_score']:.2f} | Bundle: {row['bundle']}")
    
    # Bundle-level analysis
    logger.info("\n" + "=" * 90)
    logger.info("BUNDLE-LEVEL IMPORTANCE")
    logger.info("=" * 90)
    
    bundle_importance = results_df.groupby('bundle').agg({
        'mutual_info': ['mean', 'max', 'sum'],
        'f_score': ['mean', 'max']
    }).round(4)
    
    bundle_importance.columns = ['MI_mean', 'MI_max', 'MI_sum', 'F_mean', 'F_max']
    bundle_importance = bundle_importance.sort_values('MI_sum', ascending=False)
    
    for bundle, row in bundle_importance.iterrows():
        logger.info(f"{bundle:<25} | MI_sum: {row['MI_sum']:.4f} | MI_max: {row['MI_max']:.4f} | MI_mean: {row['MI_mean']:.4f}")
    
    # Key insights
    logger.info("\n" + "=" * 90)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 90)
    
    top_5_features = results_df.head(5)['feature_name'].tolist()
    logger.info(f"Top 5 most important features:")
    for i, feat in enumerate(top_5_features, 1):
        bundle = feat.split('_')[0] + '_' + feat.split('_')[1]
        logger.info(f"  {i}. {feat} ({bundle})")
    
    # Linguistic vs non-linguistic
    linguistic_features = results_df[results_df['bundle'] == 'linguistic_bundle']
    non_linguistic_features = results_df[results_df['bundle'] != 'linguistic_bundle']
    
    logger.info(f"\nLinguistic features average MI: {linguistic_features['mutual_info'].mean():.4f}")
    logger.info(f"Non-linguistic features average MI: {non_linguistic_features['mutual_info'].mean():.4f}")
    
    # Previously "harmful" bundles
    dialogue_features = results_df[results_df['bundle'] == 'dialogue_bundle']
    contextual_features = results_df[results_df['bundle'] == 'contextual_bundle']
    
    logger.info(f"\nDialogue bundle (previously harmful) average MI: {dialogue_features['mutual_info'].mean():.4f}")
    logger.info(f"Contextual bundle (previously harmful) average MI: {contextual_features['mutual_info'].mean():.4f}")
    
    # Save results
    results_df.to_csv("/Users/omarhammad/Documents/code_local/frustration_researcher/results/feature_importance_analysis.csv", index=False)
    bundle_importance.to_csv("/Users/omarhammad/Documents/code_local/frustration_researcher/results/bundle_importance_analysis.csv")
    
    logger.info(f"\nResults saved to CSV files")
    
    return results_df, bundle_importance

if __name__ == "__main__":
    main()