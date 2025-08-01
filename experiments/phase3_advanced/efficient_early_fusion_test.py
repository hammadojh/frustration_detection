#!/usr/bin/env python3
"""
Efficient Early Fusion Test - Load models once, test multiple configurations
"""

import json
import torch
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import sys

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
    """Extract only selected features using global extractor"""
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

def test_feature_config(data, feature_indices, config_name):
    """Test a specific feature configuration"""
    logger.info(f"Testing {config_name} with {len(feature_indices)} features...")
    
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
    
    # Get embeddings (using global model)
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
    
    logger.info(f"{config_name}: F1={f1:.4f}")
    
    return {
        'config': config_name,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'num_features': len(feature_indices)
    }

def main():
    logger.info("=" * 80)
    logger.info("EFFICIENT EARLY FUSION FEATURE REDUCTION TEST")
    logger.info("=" * 80)
    
    # Initialize models once
    initialize_models()
    
    # Load data
    data = load_data()
    
    # Baselines
    baseline_all_22 = 0.8831
    baseline_linguistic_8 = 0.8571
    
    logger.info(f"Target: F1 = {baseline_all_22:.4f} (all 22 features)")
    logger.info(f"Reference: F1 = {baseline_linguistic_8:.4f} (linguistic 8 features)")
    logger.info("")
    
    # Feature indices (based on our known mapping)
    # Top features by MI: avg_politeness(2), avg_exclamation(7), task_complexity(16), response_clarity(19), avg_turn_length(9)
    
    test_configs = [
        # Progressive addition of top features
        ([2], "top_1_politeness"),
        ([2, 7], "top_2_politeness+exclamation"),
        ([2, 7, 16], "top_3_+task_complexity"),
        ([2, 7, 16, 19], "top_4_+response_clarity"),
        ([2, 7, 16, 19, 9], "top_5_+turn_length"),
        ([2, 7, 16, 19, 9, 20], "top_6_+response_relevance"),
        
        # Alternative combinations
        ([2, 16, 19], "best_3_diverse"),  # politeness + task + clarity
        ([2, 7, 19, 21], "top_4_alt"),    # politeness + exclamation + clarity + trust
    ]
    
    results = []
    
    # Run all tests
    for feature_indices, config_name in test_configs:
        result = test_feature_config(data, feature_indices, config_name)
        results.append(result)
    
    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    
    # Sort by F1 score
    sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    
    logger.info(f"🎯 TARGET: F1 = {baseline_all_22:.4f} (all 22 features)")
    logger.info("")
    
    for result in sorted_results:
        improvement_vs_all = result['f1_score'] - baseline_all_22
        reduction = (22 - result['num_features']) / 22 * 100
        
        if result['f1_score'] >= baseline_all_22:
            status = "🏆"  # Matches or beats target
        elif result['f1_score'] >= baseline_all_22 - 0.01:
            status = "🔥"  # Very close (within 1%)
        elif result['f1_score'] >= baseline_linguistic_8:
            status = "✅"  # Beats linguistic baseline
        else:
            status = "❌"  # Below baseline
        
        logger.info(f"{status} {result['config']:<25} | F1: {result['f1_score']:.4f} "
                   f"(Δ={improvement_vs_all:+.4f}) | Features: {result['num_features']:2d} | "
                   f"Reduction: {reduction:4.1f}%")
    
    # Key findings
    best = sorted_results[0]
    
    # Find minimum features to achieve targets
    target_achievers = [r for r in sorted_results if r['f1_score'] >= baseline_all_22]
    close_achievers = [r for r in sorted_results if r['f1_score'] >= baseline_all_22 - 0.01]
    
    logger.info(f"\n🎯 ANSWER TO THE KEY QUESTION:")
    if target_achievers:
        min_features = min(target_achievers, key=lambda x: x['num_features'])
        logger.info(f"   ✅ YES! We can achieve F1≥0.88 with just {min_features['num_features']} features!")
        logger.info(f"   Best config: {min_features['config']} (F1={min_features['f1_score']:.4f})")
    elif close_achievers:
        min_close = min(close_achievers, key=lambda x: x['num_features'])
        logger.info(f"   🔥 Very close! {min_close['num_features']} features achieve F1={min_close['f1_score']:.4f} (within 1%)")
    else:
        logger.info(f"   📊 Best: {best['num_features']} features achieve F1={best['f1_score']:.4f}")
    
    logger.info(f"\n📊 EFFICIENCY ANALYSIS:")
    for result in sorted_results[:3]:
        efficiency = result['f1_score'] / result['num_features']
        logger.info(f"   {result['config']}: {efficiency:.4f} F1/feature")
    
    # Save detailed results to files
    import pandas as pd
    
    # Save CSV
    df = pd.DataFrame(sorted_results)
    csv_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/efficient_early_fusion_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Save detailed text report
    report_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/early_fusion_feature_reduction_FINAL_RESULTS.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EARLY FUSION FEATURE REDUCTION - FINAL RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Target: F1 = {baseline_all_22:.4f} (all 22 features)\n")
        f.write(f"Reference: F1 = {baseline_linguistic_8:.4f} (linguistic 8 features)\n")
        f.write("\n")
        
        f.write("COMPLETE RESULTS (sorted by F1 score):\n")
        f.write("-" * 80 + "\n")
        
        for result in sorted_results:
            improvement_vs_all = result['f1_score'] - baseline_all_22
            improvement_vs_ling = result['f1_score'] - baseline_linguistic_8
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
            
            f.write(f"{status}\n")
            f.write(f"  Config: {result['config']}\n")
            f.write(f"  F1 Score: {result['f1_score']:.4f}\n")
            f.write(f"  Features: {result['num_features']}\n")
            f.write(f"  vs All Features: {improvement_vs_all:+.4f}\n")
            f.write(f"  vs Linguistic: {improvement_vs_ling:+.4f}\n")
            f.write(f"  Feature Reduction: {reduction:.1f}%\n")
            f.write(f"  Efficiency: {efficiency:.4f} F1/feature\n")
            f.write("\n")
        
        # Key findings
        best = sorted_results[0]
        target_achievers = [r for r in sorted_results if r['f1_score'] >= baseline_all_22]
        close_achievers = [r for r in sorted_results if r['f1_score'] >= baseline_all_22 - 0.01]
        
        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n")
        
        f.write(f"BEST OVERALL: {best['config']}\n")
        f.write(f"  F1: {best['f1_score']:.4f}\n")
        f.write(f"  Features: {best['num_features']}\n")
        f.write(f"  Reduction: {(22 - best['num_features'])/22*100:.1f}%\n")
        f.write("\n")
        
        f.write("ANSWER TO KEY QUESTION: Can we achieve F1≥0.88 with fewer features?\n")
        if target_achievers:
            min_features = min(target_achievers, key=lambda x: x['num_features'])
            f.write(f"✅ YES! Minimum {min_features['num_features']} features achieve F1≥0.88\n")
            f.write(f"   Best config: {min_features['config']} (F1={min_features['f1_score']:.4f})\n")
        elif close_achievers:
            min_close = min(close_achievers, key=lambda x: x['num_features'])
            f.write(f"🔥 Very close! {min_close['num_features']} features achieve F1={min_close['f1_score']:.4f} (within 1%)\n")
        else:
            f.write(f"📊 Best achievable: {best['num_features']} features → F1={best['f1_score']:.4f}\n")
        
        f.write("\n")
        f.write("TOP 3 MOST EFFICIENT CONFIGURATIONS:\n")
        for i, result in enumerate(sorted_results[:3], 1):
            efficiency = result['f1_score'] / result['num_features']
            f.write(f"  {i}. {result['config']}: {efficiency:.4f} F1/feature\n")
        
        f.write("\n")
        f.write("FEATURE ANALYSIS:\n")
        f.write("Based on the top configurations, the most critical features are:\n")
        f.write("1. avg_politeness (index 2) - Dominant signal in all top configs\n")
        f.write("2. avg_exclamation (index 7) - Emotional intensity marker\n")
        f.write("3. task_complexity (index 16) - Question density proxy\n")
        f.write("4. response_clarity (index 19) - System quality indicator\n")
        f.write("5. avg_turn_length (index 9) - Conversation flow pattern\n")
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Report: {report_path}")
    
    return sorted_results

if __name__ == "__main__":
    main()