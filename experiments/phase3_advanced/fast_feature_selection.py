#!/usr/bin/env python3
"""
Fast Feature Selection Testing
Pre-compute features and embeddings, then test selections quickly
"""

import json
import numpy as np
import pandas as pd
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

def test_feature_selection_directly(train_features, train_labels, test_features, test_labels, 
                                   feature_indices, method_name):
    """Test feature selection directly on feature space (without early fusion)"""
    
    # Select features
    train_selected = train_features[:, feature_indices]
    test_selected = test_features[:, feature_indices]
    
    # Scale
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_selected)
    test_scaled = scaler.transform(test_selected)
    
    # Train classifier
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train_scaled, train_labels)
    predictions = model.predict(test_scaled)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='binary'
    )
    
    return {
        'method': method_name,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'num_features': len(feature_indices)
    }

def main():
    logger.info("=" * 80)
    logger.info("FAST FEATURE SELECTION TESTING")
    logger.info("=" * 80)
    logger.info("NOTE: Testing feature selection directly on feature space (without early fusion)")
    logger.info("This gives us insights into feature importance for comparison with early fusion results")
    logger.info("")
    
    # Load data and extract features
    data = load_data()
    
    logger.info("Extracting all features...")
    train_features = extract_all_features(data['train_texts'])
    test_features = extract_all_features(data['test_texts'])
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Remove constant features
    non_constant_mask = np.var(train_features_scaled, axis=0) > 1e-6
    train_clean = train_features_scaled[:, non_constant_mask]
    test_clean = test_features_scaled[:, non_constant_mask]
    clean_indices = np.where(non_constant_mask)[0]
    
    logger.info(f"Non-constant features: {len(clean_indices)} out of 22")
    
    # Feature names for interpretation
    feature_names = [
        'sentiment_slope', 'sentiment_volatility', 'avg_politeness', 'politeness_decline',
        'avg_confusion', 'avg_negation', 'avg_caps', 'avg_exclamation',
        'total_turns', 'avg_turn_length', 'repeated_turns', 'corrections',
        'escalation_requests', 'negative_feedback', 'avg_urgency', 'urgency_increase',
        'task_complexity', 'emotion_drift', 'emotion_volatility', 'response_clarity',
        'response_relevance', 'trust_decline'
    ]
    
    clean_feature_names = [feature_names[i] for i in clean_indices]
    logger.info(f"Non-constant features: {clean_feature_names}")
    
    results = []
    
    # Baseline: All features
    baseline_result = test_feature_selection_directly(
        train_features_scaled, data['train_labels'], 
        test_features_scaled, data['test_labels'],
        np.arange(22), "All_22_features"
    )
    results.append(baseline_result)
    baseline_f1 = baseline_result['f1_score']
    
    logger.info(f"Baseline (all 22 features): F1 = {baseline_f1:.4f}")
    
    # Baseline: Only non-constant features
    clean_baseline = test_feature_selection_directly(
        train_clean, data['train_labels'], test_clean, data['test_labels'],
        np.arange(len(clean_indices)), "Clean_12_features"
    )
    results.append(clean_baseline)
    
    # METHOD 1: Mutual Information Selection
    logger.info("\n=== MUTUAL INFORMATION SELECTION ===")
    mi_scores = mutual_info_classif(train_clean, data['train_labels'], random_state=42)
    
    # Show top features by MI
    mi_ranking = np.argsort(mi_scores)[::-1]
    logger.info("Top features by Mutual Information:")
    for i in range(min(8, len(mi_ranking))):
        idx = mi_ranking[i]
        logger.info(f"  {i+1}. {clean_feature_names[idx]}: MI = {mi_scores[idx]:.4f}")
    
    # Test different k values
    for k in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
        if k <= len(mi_scores):
            top_k_indices = mi_ranking[:k]
            original_indices = clean_indices[top_k_indices]
            
            result = test_feature_selection_directly(
                train_features_scaled, data['train_labels'],
                test_features_scaled, data['test_labels'],
                original_indices, f"MI_top_{k}"
            )
            results.append(result)
    
    # METHOD 2: LASSO Selection
    logger.info("\n=== LASSO REGULARIZATION ===")
    
    # Test different alpha values
    alphas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        lasso.fit(train_clean, data['train_labels'])
        
        selected_mask = np.abs(lasso.coef_) > 1e-6
        num_selected = np.sum(selected_mask)
        
        if num_selected > 0:
            selected_clean_indices = np.where(selected_mask)[0]
            selected_original_indices = clean_indices[selected_clean_indices]
            
            result = test_feature_selection_directly(
                train_features_scaled, data['train_labels'],
                test_features_scaled, data['test_labels'],
                selected_original_indices, f"LASSO_a{alpha}"
            )
            results.append(result)
            
            # Show selected features
            selected_names = [clean_feature_names[i] for i in selected_clean_indices]
            logger.info(f"LASSO α={alpha}: {num_selected} features - {selected_names}")
        else:
            logger.info(f"LASSO α={alpha}: No features selected")
    
    # METHOD 3: Recursive Feature Elimination
    logger.info("\n=== RECURSIVE FEATURE ELIMINATION ===")
    
    for n_features in [1, 2, 3, 4, 5, 8, 10]:
        if n_features <= len(clean_indices):
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            
            try:
                rfe.fit(train_clean, data['train_labels'])
                selected_clean_indices = np.where(rfe.support_)[0]
                selected_original_indices = clean_indices[selected_clean_indices]
                
                result = test_feature_selection_directly(
                    train_features_scaled, data['train_labels'],
                    test_features_scaled, data['test_labels'],
                    selected_original_indices, f"RFE_{n_features}"
                )
                results.append(result)
                
                # Show selected features
                selected_names = [clean_feature_names[i] for i in selected_clean_indices]
                logger.info(f"RFE {n_features}: {selected_names}")
                
            except Exception as e:
                logger.error(f"RFE with {n_features} features failed: {e}")
    
    # METHOD 4: Hybrid approaches
    logger.info("\n=== HYBRID METHODS ===")
    
    # Top MI + LASSO refinement
    for mi_k in [8, 10]:
        if mi_k <= len(mi_scores):
            top_mi_indices = mi_ranking[:mi_k]
            top_mi_features = train_clean[:, top_mi_indices]
            
            # Apply LASSO to top MI features
            for alpha in [0.01, 0.05, 0.1]:
                lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
                lasso.fit(top_mi_features, data['train_labels'])
                
                lasso_mask = np.abs(lasso.coef_) > 1e-6
                if np.sum(lasso_mask) > 0:
                    final_clean_indices = top_mi_indices[lasso_mask]
                    final_original_indices = clean_indices[final_clean_indices]
                    
                    result = test_feature_selection_directly(
                        train_features_scaled, data['train_labels'],
                        test_features_scaled, data['test_labels'],
                        final_original_indices, f"Hybrid_MI{mi_k}_LASSO{alpha}"
                    )
                    results.append(result)
    
    # COMPREHENSIVE RESULTS
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE FEATURE SELECTION RESULTS")
    logger.info("=" * 80)
    
    # Sort by F1 score
    sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    
    logger.info("ALL RESULTS (sorted by F1 score):")
    logger.info("-" * 80)
    
    for i, result in enumerate(sorted_results):
        improvement = result['f1_score'] - baseline_f1
        efficiency = result['f1_score'] / result['num_features']
        
        if i == 0:
            status = "🏆"
        elif improvement > 0.005:
            status = "✅"
        elif improvement > -0.005:
            status = "≈"
        else:
            status = "❌"
        
        logger.info(f"{status} {result['method']:<25} | F1: {result['f1_score']:.4f} "
                   f"(Δ={improvement:+.4f}) | Features: {result['num_features']:2d} | "
                   f"Efficiency: {efficiency:.4f}")
    
    # Key insights
    best = sorted_results[0]
    most_efficient = max(sorted_results, key=lambda x: x['f1_score'] / x['num_features'])
    
    logger.info(f"\n🎯 BEST METHOD: {best['method']}")
    logger.info(f"   F1: {best['f1_score']:.4f}, Features: {best['num_features']}")
    
    logger.info(f"\n⚡ MOST EFFICIENT: {most_efficient['method']}")
    logger.info(f"   Efficiency: {most_efficient['f1_score'] / most_efficient['num_features']:.4f} F1/feature")
    
    # Save results
    df = pd.DataFrame(sorted_results)
    results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/fast_feature_selection_results.csv"
    df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to {results_path}")
    
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"   Best F1: {best['f1_score']:.4f} (vs {baseline_f1:.4f} baseline)")
    logger.info(f"   Feature reduction: {(22 - best['num_features'])/22*100:.1f}%")
    logger.info(f"   Methods tested: {len(results)}")
    
    return sorted_results

if __name__ == "__main__":
    main()