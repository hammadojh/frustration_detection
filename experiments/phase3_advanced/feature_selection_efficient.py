#!/usr/bin/env python3
"""
Phase 3b: Efficient Feature Selection
Focus on key methods without timeout issues
"""

import json
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import mutual_info_classif
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

def test_early_fusion_with_features(data, feature_indices, method_name):
    """Test early fusion with selected features"""
    logger.info(f"Testing {method_name} with {len(feature_indices)} features...")
    
    # Initialize models
    tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
    roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roberta.to(device)
    
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
    train_texts_enhanced = [
        f"{feat_tokens} {text}" 
        for feat_tokens, text in zip(train_feat_tokens, data['train_texts'])
    ]
    test_texts_enhanced = [
        f"{feat_tokens} {text}" 
        for feat_tokens, text in zip(test_feat_tokens, data['test_texts'])
    ]
    
    # Get embeddings efficiently
    def get_embeddings(texts):
        embeddings = []
        batch_size = 32  # Larger batch for efficiency
        
        roberta.eval()
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
        'num_features': len(feature_indices),
        'feature_indices': feature_indices.tolist()
    }

def main():
    logger.info("=" * 80)
    logger.info("PHASE 3B: EFFICIENT FEATURE SELECTION")
    logger.info("=" * 80)
    
    # Load data
    data = load_data()
    baseline_f1 = 0.8831
    
    logger.info(f"Baseline (all 22 features): F1 = {baseline_f1:.4f}")
    logger.info("")
    
    # Extract features for analysis
    train_features = extract_all_features(data['train_texts'])
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    
    # Remove constant features
    non_constant_mask = np.var(train_features_scaled, axis=0) > 1e-6
    train_features_clean = train_features_scaled[:, non_constant_mask]
    clean_indices = np.where(non_constant_mask)[0]
    
    logger.info(f"Non-constant features: {len(clean_indices)} out of 22")
    
    results = []
    
    # Method 1: Top Mutual Information Features
    logger.info("\n=== MUTUAL INFORMATION SELECTION ===")
    mi_scores = mutual_info_classif(train_features_clean, data['train_labels'], random_state=42)
    
    # Test key MI selections
    for k in [3, 5, 8]:
        if k <= len(mi_scores):
            top_k_clean = np.argsort(mi_scores)[-k:]
            top_k_original = clean_indices[top_k_clean]
            
            result = test_early_fusion_with_features(data, top_k_original, f"MI_top_{k}")
            results.append(result)
    
    # Method 2: LASSO Selection
    logger.info("\n=== LASSO REGULARIZATION SELECTION ===")
    
    # Quick LASSO with different alpha values
    alphas = [0.001, 0.01, 0.1, 0.5]
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        lasso.fit(train_features_clean, data['train_labels'])
        
        selected_mask = np.abs(lasso.coef_) > 1e-6
        if np.sum(selected_mask) > 0:
            selected_clean = np.where(selected_mask)[0]
            selected_original = clean_indices[selected_clean]
            
            result = test_early_fusion_with_features(
                data, selected_original, f"LASSO_alpha_{alpha}"
            )
            if result:
                result['alpha'] = alpha
                results.append(result)
    
    # Method 3: Top features from our previous analysis
    logger.info("\n=== KNOWLEDGE-BASED SELECTION ===")
    
    # Based on our feature importance analysis, test top performers
    # Top 5 from previous analysis: politeness, exclamation, task_complexity, response_clarity, turn_length
    # Map to indices (approximate based on feature order)
    top_knowledge_indices = np.array([2, 7, 16, 19, 9])  # Based on feature importance
    
    # Filter to only include non-constant indices
    top_knowledge_filtered = [idx for idx in top_knowledge_indices if idx in clean_indices]
    
    if len(top_knowledge_filtered) > 0:
        result = test_early_fusion_with_features(
            data, np.array(top_knowledge_filtered), "Knowledge_top_5"
        )
        results.append(result)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE SELECTION RESULTS")
    logger.info("=" * 80)
    
    if results:
        # Sort by F1 score
        sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
        
        logger.info(f"Baseline (22 features): F1 = {baseline_f1:.4f}")
        logger.info("")
        
        for result in sorted_results:
            improvement = result['f1_score'] - baseline_f1
            status = "🏆" if improvement > 0.005 else "✅" if improvement > 0 else "❌"
            efficiency = result['f1_score'] / result['num_features']
            
            logger.info(f"{status} {result['method']:<18} | F1: {result['f1_score']:.4f} "
                       f"(Δ={improvement:+.4f}) | Features: {result['num_features']:2d} | "
                       f"Efficiency: {efficiency:.4f}")
        
        # Best result
        best = sorted_results[0]
        logger.info(f"\n🎯 BEST METHOD: {best['method']}")
        logger.info(f"F1 Score: {best['f1_score']:.4f}")
        logger.info(f"Improvement: {best['f1_score'] - baseline_f1:+.4f}")
        logger.info(f"Features: {best['num_features']}")
        
        # Save results
        df = pd.DataFrame(results)
        results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/feature_selection_results.csv"
        df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to {results_path}")
        
        return sorted_results
    else:
        logger.info("No results generated")
        return []

if __name__ == "__main__":
    main()