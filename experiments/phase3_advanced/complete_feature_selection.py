#!/usr/bin/env python3
"""
Complete Feature Selection Testing
Test all remaining feature selection methods systematically
"""

import json
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import mutual_info_classif, RFE
import logging
import sys

sys.path.append('/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase2_features')
from fast_feature_extractor import FastFrustrationFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to avoid reloading models
tokenizer = None
roberta = None
device = None

def initialize_models():
    global tokenizer, roberta, device
    if tokenizer is None:
        tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
        roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        roberta.to(device)
        roberta.eval()

def load_data():
    """Load processed examples"""
    processed_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/data/subset_processed.json"
    with open(processed_path, 'r') as f:
        examples = json.load(f)
    
    texts = [ex['text'] for ex in examples]
    labels = [ex['label'] for ex in examples]
    
    n = len(texts)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    return {
        'train_texts': texts[:train_end],
        'train_labels': labels[:train_end],
        'test_texts': texts[val_end:],
        'test_labels': labels[val_end:]
    }

def extract_all_features(texts):
    """Extract all 22 features"""
    extractor = FastFrustrationFeatureExtractor()
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
    
    return np.array(all_features)

def test_feature_subset(data, feature_indices, method_name):
    """Test early fusion with specific feature subset"""
    global tokenizer, roberta, device
    
    logger.info(f"Testing {method_name} with {len(feature_indices)} features...")
    
    # Extract and select features
    train_features_full = extract_all_features(data['train_texts'])
    test_features_full = extract_all_features(data['test_texts'])
    
    train_features = train_features_full[:, feature_indices]
    test_features = test_features_full[:, feature_indices]
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Convert to tokens
    def features_to_tokens(features_scaled):
        feature_tokens = []
        for feat_vec in features_scaled:
            tokens = []
            for i, feat_val in enumerate(feat_vec):
                level = int(np.clip((feat_val + 3) / 6 * 10, 0, 9))
                tokens.append(f"[FEAT{i}_{level}]")
            feature_tokens.append(" ".join(tokens))
        return feature_tokens
    
    train_feat_tokens = features_to_tokens(train_features_scaled)
    test_feat_tokens = features_to_tokens(test_features_scaled)
    
    # Create enhanced texts
    train_texts_enhanced = [f"{feat_tokens} {text}" for feat_tokens, text in zip(train_feat_tokens, data['train_texts'])]
    test_texts_enhanced = [f"{feat_tokens} {text}" for feat_tokens, text in zip(test_feat_tokens, data['test_texts'])]
    
    # Get embeddings
    def get_embeddings(texts):
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, truncation=True, padding="max_length", 
                                 max_length=512, return_tensors="pt").to(device)
                outputs = roberta(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(embeddings)
    
    train_embeddings = get_embeddings(train_texts_enhanced)
    test_embeddings = get_embeddings(test_texts_enhanced)
    
    # Train and predict
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train_embeddings, data['train_labels'])
    predictions = model.predict(test_embeddings)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        data['test_labels'], predictions, average='binary'
    )
    
    logger.info(f"{method_name}: F1={f1:.4f}")
    return {
        'method': method_name,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'num_features': len(feature_indices)
    }

def main():
    logger.info("=" * 80)
    logger.info("COMPLETE FEATURE SELECTION TESTING")
    logger.info("=" * 80)
    
    # Initialize models once
    initialize_models()
    
    # Load data
    data = load_data()
    baseline_f1 = 0.8831
    
    logger.info(f"Baseline (all 22 features): F1 = {baseline_f1:.4f}")
    
    # Extract features for selection algorithms
    train_features = extract_all_features(data['train_texts'])
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    
    # Remove constant features
    non_constant_mask = np.var(train_features_scaled, axis=0) > 1e-6
    train_features_clean = train_features_scaled[:, non_constant_mask]
    clean_indices = np.where(non_constant_mask)[0]
    
    logger.info(f"Non-constant features: {len(clean_indices)} out of 22")
    
    results = []
    
    # PHASE 1: Complete Mutual Information Testing
    logger.info("\n=== MUTUAL INFORMATION SELECTION (COMPLETE) ===")
    mi_scores = mutual_info_classif(train_features_clean, data['train_labels'], random_state=42)
    
    # Test all reasonable k values
    for k in [3, 5, 8, 10, 12]:
        if k <= len(mi_scores):
            top_k_clean = np.argsort(mi_scores)[-k:]
            top_k_original = clean_indices[top_k_clean]
            
            result = test_feature_subset(data, top_k_original, f"MI_top_{k}")
            results.append(result)
    
    # PHASE 2: LASSO Testing
    logger.info("\n=== LASSO REGULARIZATION TESTING ===")
    
    # Test different alpha values for different sparsity levels
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        lasso.fit(train_features_clean, data['train_labels'])
        
        selected_mask = np.abs(lasso.coef_) > 1e-6
        num_selected = np.sum(selected_mask)
        
        if num_selected > 0:
            selected_clean = np.where(selected_mask)[0]
            selected_original = clean_indices[selected_clean]
            
            result = test_feature_subset(data, selected_original, f"LASSO_a{alpha}")
            if result:
                results.append(result)
        else:
            logger.info(f"LASSO α={alpha}: No features selected")
    
    # PHASE 3: Recursive Feature Elimination
    logger.info("\n=== RECURSIVE FEATURE ELIMINATION ===")
    
    for n_features in [3, 5, 8, 10]:
        if n_features <= len(clean_indices):
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            
            try:
                rfe.fit(train_features_clean, data['train_labels'])
                selected_clean = np.where(rfe.support_)[0]
                selected_original = clean_indices[selected_clean]
                
                result = test_feature_subset(data, selected_original, f"RFE_{n_features}")
                results.append(result)
                
            except Exception as e:
                logger.error(f"RFE with {n_features} features failed: {e}")
    
    # PHASE 4: Hybrid Method (MI + LASSO)
    logger.info("\n=== HYBRID MI + LASSO ===")
    
    # Step 1: Get top 10 by MI
    top_10_clean = np.argsort(mi_scores)[-10:]
    top_10_features = train_features_clean[:, top_10_clean]
    
    # Step 2: Apply LASSO to top 10
    for alpha in [0.01, 0.05, 0.1]:
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        lasso.fit(top_10_features, data['train_labels'])
        
        lasso_mask = np.abs(lasso.coef_) > 1e-6
        if np.sum(lasso_mask) > 0:
            final_clean = top_10_clean[lasso_mask]
            final_original = clean_indices[final_clean]
            
            result = test_feature_subset(data, final_original, f"Hybrid_MI10_LASSO{alpha}")
            results.append(result)
    
    # COMPREHENSIVE SUMMARY
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE FEATURE SELECTION RESULTS")
    logger.info("=" * 80)
    
    if results:
        # Sort by F1 score
        sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
        
        logger.info(f"Baseline (22 features): F1 = {baseline_f1:.4f}")
        logger.info("")
        logger.info("ALL RESULTS (sorted by F1 score):")
        logger.info("-" * 80)
        
        for i, result in enumerate(sorted_results):
            improvement = result['f1_score'] - baseline_f1
            efficiency = result['f1_score'] / result['num_features']
            
            if i == 0:
                status = "🏆"  # Best
            elif improvement > 0:
                status = "✅"  # Better than baseline
            elif improvement > -0.01:
                status = "≈"  # Close to baseline
            else:
                status = "❌"  # Significantly worse
            
            logger.info(f"{status} {result['method']:<20} | F1: {result['f1_score']:.4f} "
                       f"(Δ={improvement:+.4f}) | Features: {result['num_features']:2d} | "
                       f"Efficiency: {efficiency:.4f}")
        
        # Analysis
        best = sorted_results[0]
        logger.info(f"\n🎯 BEST METHOD: {best['method']}")
        logger.info(f"F1 Score: {best['f1_score']:.4f}")
        logger.info(f"Improvement: {best['f1_score'] - baseline_f1:+.4f}")
        logger.info(f"Features: {best['num_features']} (vs 22 baseline)")
        logger.info(f"Feature reduction: {(22 - best['num_features'])/22*100:.1f}%")
        
        # Find most efficient method
        most_efficient = max(sorted_results, key=lambda x: x['f1_score'] / x['num_features'])
        logger.info(f"\n⚡ MOST EFFICIENT: {most_efficient['method']}")
        logger.info(f"Efficiency: {most_efficient['f1_score'] / most_efficient['num_features']:.4f} F1/feature")
        
        # Save results
        df = pd.DataFrame(sorted_results)
        results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/complete_feature_selection_results.csv"
        df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to {results_path}")
        
        return sorted_results
    else:
        logger.info("No results generated")
        return []

if __name__ == "__main__":
    main()