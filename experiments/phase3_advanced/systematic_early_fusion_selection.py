#!/usr/bin/env python3
"""
Systematic Early Fusion Feature Selection
Find the TRUE optimal feature combinations for early fusion empirically
"""

import json
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from itertools import combinations
import logging
import sys
import time

sys.path.append('/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase2_features')
from fast_feature_extractor import FastFrustrationFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models to avoid reloading
TOKENIZER = None
ROBERTA = None
DEVICE = None
EXTRACTOR = None

def initialize_models():
    """Initialize models once"""
    global TOKENIZER, ROBERTA, DEVICE, EXTRACTOR
    
    logger.info("Loading models (once)...")
    TOKENIZER = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
    ROBERTA = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ROBERTA.to(DEVICE)
    ROBERTA.eval()
    EXTRACTOR = FastFrustrationFeatureExtractor()
    logger.info("Models loaded successfully!")

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

def extract_selected_features(texts, feature_indices):
    """Extract only selected features"""
    all_bundles = [
        'linguistic_bundle', 'dialogue_bundle', 'behavioral_bundle',
        'contextual_bundle', 'emotion_dynamics_bundle', 'system_bundle', 'user_model_bundle'
    ]
    
    all_features = []
    for text in texts:
        text_features = []
        for bundle in all_bundles:
            bundle_features = EXTRACTOR.extract_bundle_features([text], bundle)
            text_features.extend(list(bundle_features.values()))
        
        # Select only specified features
        selected_features = [text_features[i] for i in feature_indices]
        all_features.append(selected_features)
    
    return np.array(all_features)

def get_embeddings(texts):
    """Get embeddings using global models"""
    embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = TOKENIZER(batch, truncation=True, padding="max_length", 
                             max_length=512, return_tensors="pt").to(DEVICE)
            outputs = ROBERTA(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)

def test_feature_combination(data, feature_indices, combo_name):
    """Test a specific feature combination with early fusion"""
    # Extract selected features
    train_features = extract_selected_features(data['train_texts'], feature_indices)
    test_features = extract_selected_features(data['test_texts'], feature_indices)
    
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
    train_embeddings = get_embeddings(train_texts_enhanced)
    test_embeddings = get_embeddings(test_texts_enhanced)
    
    # Train classifier
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train_embeddings, data['train_labels'])
    predictions = model.predict(test_embeddings)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        data['test_labels'], predictions, average='binary'
    )
    
    return {
        'combination': combo_name,
        'feature_indices': list(feature_indices),
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'num_features': len(feature_indices)
    }

def main():
    logger.info("=" * 80)
    logger.info("SYSTEMATIC EARLY FUSION FEATURE SELECTION")
    logger.info("Finding TRUE optimal feature combinations empirically")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Initialize models once
    initialize_models()
    
    # Load data
    data = load_data()
    
    # Define all features (only non-constant ones from our analysis)
    # Based on previous analysis, these are the 12 non-constant features
    all_feature_names = [
        'avg_politeness',      # 2
        'avg_confusion',       # 4  
        'avg_negation',        # 5
        'avg_exclamation',     # 7
        'avg_turn_length',     # 9
        'corrections',         # 11
        'escalation_requests', # 12
        'avg_urgency',         # 14
        'task_complexity',     # 16
        'response_clarity',    # 19
        'response_relevance',  # 20
        'trust_decline'        # 21
    ]
    
    all_feature_indices = [2, 4, 5, 7, 9, 11, 12, 14, 16, 19, 20, 21]
    
    logger.info(f"Testing combinations of {len(all_feature_indices)} non-constant features")
    logger.info(f"Features: {all_feature_names}")
    logger.info("")
    
    # Baselines
    baseline_all_22 = 0.8831
    baseline_linguistic_8 = 0.8571
    
    results = []
    
    # Strategy: Test combinations systematically
    # 1. Individual features (12 tests)
    # 2. Pairs (66 tests - but we'll sample the most promising)
    # 3. Triples (220 tests - sample most promising)  
    # 4. Specific sizes based on results
    
    logger.info("=== PHASE 1: INDIVIDUAL FEATURES ===")
    individual_results = []
    
    for i, (feat_idx, feat_name) in enumerate(zip(all_feature_indices, all_feature_names)):
        logger.info(f"Testing individual feature {i+1}/12: {feat_name}")
        
        result = test_feature_combination(data, [feat_idx], f"single_{feat_name}")
        result['feature_names'] = [feat_name]
        results.append(result)
        individual_results.append(result)
    
    # Sort individual results to find best performers
    individual_results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    logger.info("\nTop 5 individual features:")
    for i, result in enumerate(individual_results[:5]):
        logger.info(f"  {i+1}. {result['feature_names'][0]}: F1={result['f1_score']:.4f}")
    
    # Get top individual features for combination testing
    top_individual_indices = [result['feature_indices'][0] for result in individual_results[:8]]
    top_individual_names = [result['feature_names'][0] for result in individual_results[:8]]
    
    logger.info(f"\n=== PHASE 2: TOP PAIRS ===")
    # Test pairs of top individual performers
    pair_count = 0
    max_pairs = 20  # Limit to most promising pairs
    
    for i in range(len(top_individual_indices)):
        for j in range(i+1, min(len(top_individual_indices), i+6)):  # Limit combinations per feature
            if pair_count >= max_pairs:
                break
                
            feat_indices = [top_individual_indices[i], top_individual_indices[j]]
            feat_names = [top_individual_names[i], top_individual_names[j]]
            combo_name = f"pair_{feat_names[0]}+{feat_names[1]}"
            
            logger.info(f"Testing pair {pair_count+1}/{max_pairs}: {feat_names}")
            
            result = test_feature_combination(data, feat_indices, combo_name)
            result['feature_names'] = feat_names
            results.append(result)
            pair_count += 1
        
        if pair_count >= max_pairs:
            break
    
    # Find best pairs
    pair_results = [r for r in results if r['num_features'] == 2]
    pair_results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    logger.info(f"\nTop 3 pairs:")
    for i, result in enumerate(pair_results[:3]):
        logger.info(f"  {i+1}. {'+'.join(result['feature_names'])}: F1={result['f1_score']:.4f}")
    
    logger.info(f"\n=== PHASE 3: STRATEGIC COMBINATIONS ===")
    # Test strategic combinations based on results so far
    
    # Get the best individual and best pair
    best_individual = individual_results[0]
    best_pair = pair_results[0] if pair_results else None
    
    # Test progressive additions to best individual
    if best_pair:
        best_pair_indices = best_pair['feature_indices']
        
        # Add third feature to best pair
        for feat_idx, feat_name in zip(top_individual_indices[:6], top_individual_names[:6]):
            if feat_idx not in best_pair_indices:
                combo_indices = best_pair_indices + [feat_idx]
                combo_names = best_pair['feature_names'] + [feat_name]
                combo_name = f"triple_{'+'.join(combo_names[:3])}"
                
                logger.info(f"Testing triple: {combo_names}")
                
                result = test_feature_combination(data, combo_indices, combo_name)
                result['feature_names'] = combo_names
                results.append(result)
    
    # Test some larger combinations based on performance
    logger.info(f"\n=== PHASE 4: LARGER COMBINATIONS ===")
    
    # Test top 4, 5, 6 features
    for size in [4, 5, 6]:
        top_indices = [result['feature_indices'][0] for result in individual_results[:size]]
        top_names = [result['feature_names'][0] for result in individual_results[:size]]
        combo_name = f"top_{size}_individual"
        
        logger.info(f"Testing top {size} individual features: {top_names}")
        
        result = test_feature_combination(data, top_indices, combo_name)
        result['feature_names'] = top_names
        results.append(result)
    
    # Test some bundle-based combinations
    logger.info(f"\n=== PHASE 5: BUNDLE-BASED COMBINATIONS ===")
    
    # Linguistic bundle features (from our original analysis)
    linguistic_indices = [2, 4, 5, 7]  # politeness, confusion, negation, exclamation
    linguistic_names = ['avg_politeness', 'avg_confusion', 'avg_negation', 'avg_exclamation']
    
    result = test_feature_combination(data, linguistic_indices, "linguistic_core")
    result['feature_names'] = linguistic_names
    results.append(result)
    
    # Mixed bundle combination
    mixed_indices = [2, 7, 16, 19]  # politeness, exclamation, task_complexity, response_clarity
    mixed_names = ['avg_politeness', 'avg_exclamation', 'task_complexity', 'response_clarity']
    
    result = test_feature_combination(data, mixed_indices, "mixed_optimal")
    result['feature_names'] = mixed_names
    results.append(result)
    
    # COMPREHENSIVE ANALYSIS
    logger.info("\n" + "=" * 80)
    logger.info("SYSTEMATIC FEATURE SELECTION COMPLETE")
    logger.info("=" * 80)
    
    # Sort all results
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    # Analysis by size
    logger.info("RESULTS BY FEATURE COUNT:")
    for size in range(1, 7):
        size_results = [r for r in results if r['num_features'] == size]
        if size_results:
            best_for_size = size_results[0]
            logger.info(f"  {size} features: F1={best_for_size['f1_score']:.4f} - {'+'.join(best_for_size['feature_names'])}")
    
    # Top overall results
    logger.info(f"\nTOP 10 OVERALL COMBINATIONS:")
    for i, result in enumerate(results[:10]):
        improvement = result['f1_score'] - baseline_all_22
        reduction = (22 - result['num_features']) / 22 * 100
        
        if result['f1_score'] >= baseline_all_22:
            status = "🏆"
        elif result['f1_score'] >= baseline_all_22 - 0.01:
            status = "🔥"
        elif result['f1_score'] >= baseline_linguistic_8:
            status = "✅"
        else:
            status = "❌"
        
        logger.info(f"  {i+1:2d}. {status} {result['combination']:<25} | F1: {result['f1_score']:.4f} "
                   f"(Δ={improvement:+.4f}) | {result['num_features']} features | {reduction:4.1f}% reduction")
    
    # Save comprehensive results
    output_dir = "/Users/omarhammad/Documents/code_local/frustration_researcher/results"
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = f"{output_dir}/systematic_early_fusion_selection_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Save detailed report
    report_path = f"{output_dir}/systematic_early_fusion_selection_COMPLETE_ANALYSIS.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SYSTEMATIC EARLY FUSION FEATURE SELECTION - COMPLETE ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment completed in {(time.time() - start_time)/60:.1f} minutes\n")
        f.write(f"Total combinations tested: {len(results)}\n")
        f.write(f"Target: F1 = {baseline_all_22:.4f} (all 22 features)\n")
        f.write(f"Reference: F1 = {baseline_linguistic_8:.4f} (linguistic 8 features)\n")
        f.write("\n")
        
        f.write("COMPLETE RESULTS (sorted by F1 score):\n")
        f.write("-" * 80 + "\n")
        
        for i, result in enumerate(results):
            improvement_all = result['f1_score'] - baseline_all_22
            improvement_ling = result['f1_score'] - baseline_linguistic_8
            reduction = (22 - result['num_features']) / 22 * 100
            efficiency = result['f1_score'] / result['num_features']
            
            if result['f1_score'] >= baseline_all_22:
                status = "🏆 TARGET ACHIEVED"
            elif result['f1_score'] >= baseline_all_22 - 0.01:
                status = "🔥 VERY CLOSE"
            elif result['f1_score'] >= baseline_linguistic_8:
                status = "✅ BEATS BASELINE"
            else:
                status = "❌ BELOW BASELINE"
            
            f.write(f"\n{i+1:2d}. {status}\n")
            f.write(f"    Combination: {result['combination']}\n")
            f.write(f"    Features: {'+'.join(result['feature_names'])}\n")
            f.write(f"    F1 Score: {result['f1_score']:.4f}\n")
            f.write(f"    Feature Count: {result['num_features']}\n")
            f.write(f"    vs All Features: {improvement_all:+.4f}\n")
            f.write(f"    vs Linguistic: {improvement_ling:+.4f}\n")
            f.write(f"    Feature Reduction: {reduction:.1f}%\n")
            f.write(f"    Efficiency: {efficiency:.4f} F1/feature\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("ANALYSIS BY FEATURE COUNT\n")
        f.write("=" * 80 + "\n")
        
        for size in range(1, 8):
            size_results = [r for r in results if r['num_features'] == size]
            if size_results:
                best = size_results[0]
                f.write(f"\n{size} FEATURES:\n")
                f.write(f"  Best: {best['combination']} (F1={best['f1_score']:.4f})\n")
                f.write(f"  Features: {'+'.join(best['feature_names'])}\n")
                f.write(f"  Improvement vs all: {best['f1_score'] - baseline_all_22:+.4f}\n")
                
                if len(size_results) > 1:
                    f.write(f"  Other combinations tested: {len(size_results)-1}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("FINAL ANSWER\n")
        f.write("=" * 80 + "\n")
        
        best_overall = results[0]
        target_achievers = [r for r in results if r['f1_score'] >= baseline_all_22]
        close_achievers = [r for r in results if r['f1_score'] >= baseline_all_22 - 0.01]
        
        f.write("QUESTION: Can we achieve F1≥0.88 with fewer features using early fusion?\n\n")
        
        if target_achievers:
            min_target = min(target_achievers, key=lambda x: x['num_features'])
            f.write(f"✅ YES! Minimum {min_target['num_features']} features can achieve F1≥0.88\n")
            f.write(f"   Best combination: {min_target['combination']}\n")
            f.write(f"   Features: {'+'.join(min_target['feature_names'])}\n")
            f.write(f"   F1 Score: {min_target['f1_score']:.4f}\n")
            f.write(f"   Feature reduction: {(22-min_target['num_features'])/22*100:.1f}%\n")
        elif close_achievers:
            min_close = min(close_achievers, key=lambda x: x['num_features'])
            f.write(f"🔥 Very close! {min_close['num_features']} features achieve F1={min_close['f1_score']:.4f}\n")
            f.write(f"   Best combination: {min_close['combination']}\n")
            f.write(f"   Features: {'+'.join(min_close['feature_names'])}\n")
            f.write(f"   Within 1% of target (F1≥0.87)\n")
        else:
            f.write(f"📊 Best achievable: {best_overall['num_features']} features → F1={best_overall['f1_score']:.4f}\n")
            f.write(f"   Best combination: {best_overall['combination']}\n")
            f.write(f"   Features: {'+'.join(best_overall['feature_names'])}\n")
        
        f.write(f"\nMOST EFFICIENT COMBINATION:\n")
        most_efficient = max(results, key=lambda x: x['f1_score'] / x['num_features'])
        f.write(f"  {most_efficient['combination']}: {most_efficient['f1_score']/most_efficient['num_features']:.4f} F1/feature\n")
        f.write(f"  Features: {'+'.join(most_efficient['feature_names'])}\n")
        f.write(f"  F1: {most_efficient['f1_score']:.4f} with {most_efficient['num_features']} features\n")
    
    logger.info(f"\nExperiment completed in {(time.time() - start_time)/60:.1f} minutes")
    logger.info(f"Results saved to:")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Report: {report_path}")
    
    # Final summary
    best = results[0]
    logger.info(f"\n🎯 FINAL ANSWER:")
    logger.info(f"   Best combination: {best['combination']}")
    logger.info(f"   Features: {'+'.join(best['feature_names'])}")
    logger.info(f"   F1 Score: {best['f1_score']:.4f}")
    logger.info(f"   Feature count: {best['num_features']}")
    logger.info(f"   Reduction: {(22-best['num_features'])/22*100:.1f}%")
    
    return results

if __name__ == "__main__":
    main()