#!/usr/bin/env python3
"""
Test Early Fusion with Top Selected Features
Can we achieve F1=0.88 with fewer features?
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
        
        # Select only specified features
        selected_features = [text_features[i] for i in feature_indices]
        all_features.append(selected_features)
    
    return np.array(all_features)

def test_early_fusion_with_selected_features(data, feature_indices, method_name, feature_names):
    """Test early fusion with selected features"""
    logger.info(f"Testing {method_name} with {len(feature_indices)} features: {feature_names}")
    
    # Initialize models
    tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
    roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roberta.to(device)
    
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
    def get_embeddings(texts):
        embeddings = []
        batch_size = 32
        
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
    
    # Train classifier
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train_embeddings, data['train_labels'])
    predictions = model.predict(test_embeddings)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        data['test_labels'], predictions, average='binary'
    )
    accuracy = accuracy_score(data['test_labels'], predictions)
    
    logger.info(f"{method_name}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
    
    return {
        'method': method_name,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'num_features': len(feature_indices),
        'feature_names': feature_names
    }

def main():
    logger.info("=" * 80)
    logger.info("EARLY FUSION WITH TOP SELECTED FEATURES")
    logger.info("Testing if we can achieve F1=0.88 with fewer features!")
    logger.info("=" * 80)
    
    # Load data
    data = load_data()
    
    # Baselines for comparison
    baseline_all_22 = 0.8831
    baseline_linguistic_8 = 0.8571
    
    logger.info(f"Target: F1 = {baseline_all_22:.4f} (all 22 features)")
    logger.info(f"Reference: F1 = {baseline_linguistic_8:.4f} (linguistic 8 features)")
    logger.info("")
    
    # Feature mapping (based on our extraction order)
    # From feature selection results, the top features by MI are:
    # 1. avg_politeness (index 2 in linguistic bundle)
    # 2. avg_exclamation (index 7 in linguistic bundle) 
    # 3. task_complexity (index 16 in contextual bundle)
    # 4. response_clarity (index 19 in system bundle)
    # 5. avg_turn_length (index 9 in dialogue bundle)
    
    all_feature_names = [
        'sentiment_slope', 'sentiment_volatility', 'avg_politeness', 'politeness_decline',  # 0-3: linguistic
        'avg_confusion', 'avg_negation', 'avg_caps', 'avg_exclamation',  # 4-7: linguistic
        'total_turns', 'avg_turn_length', 'repeated_turns', 'corrections',  # 8-11: dialogue
        'escalation_requests', 'negative_feedback',  # 12-13: behavioral
        'avg_urgency', 'urgency_increase', 'task_complexity',  # 14-16: contextual
        'emotion_drift', 'emotion_volatility',  # 17-18: emotion_dynamics
        'response_clarity', 'response_relevance',  # 19-20: system
        'trust_decline'  # 21: user_model
    ]
    
    results = []
    
    # Test configurations
    test_configs = [
        # Top 1 feature
        ([2], ["avg_politeness"]),
        
        # Top 2 features  
        ([2, 7], ["avg_politeness", "avg_exclamation"]),
        
        # Top 3 features
        ([2, 7, 16], ["avg_politeness", "avg_exclamation", "task_complexity"]),
        
        # Top 4 features
        ([2, 7, 16, 19], ["avg_politeness", "avg_exclamation", "task_complexity", "response_clarity"]),
        
        # Top 5 features (from feature selection)
        ([2, 7, 16, 19, 9], ["avg_politeness", "avg_exclamation", "task_complexity", "response_clarity", "avg_turn_length"]),
        
        # Top 6 features
        ([2, 7, 16, 19, 9, 20], ["avg_politeness", "avg_exclamation", "task_complexity", "response_clarity", "avg_turn_length", "response_relevance"]),
        
        # Top 8 features (including trust_decline and avg_confusion)  
        ([2, 7, 16, 19, 9, 20, 21, 4], ["avg_politeness", "avg_exclamation", "task_complexity", "response_clarity", "avg_turn_length", "response_relevance", "trust_decline", "avg_confusion"]),
        
        # Linguistic bundle for reference (8 features)
        ([0, 1, 2, 3, 4, 5, 6, 7], ["sentiment_slope", "sentiment_volatility", "avg_politeness", "politeness_decline", "avg_confusion", "avg_negation", "avg_caps", "avg_exclamation"])
    ]
    
    # Run tests
    for feature_indices, feature_names in test_configs:
        result = test_early_fusion_with_selected_features(
            data, feature_indices, f"EarlyFusion_top_{len(feature_indices)}", feature_names
        )
        results.append(result)
    
    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("EARLY FUSION FEATURE REDUCTION RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"🎯 TARGET: F1 = {baseline_all_22:.4f} (all 22 features)")
    logger.info("")
    
    # Sort by F1 score
    sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    
    for result in sorted_results:
        improvement_vs_all = result['f1_score'] - baseline_all_22
        improvement_vs_ling = result['f1_score'] - baseline_linguistic_8
        reduction = (22 - result['num_features']) / 22 * 100
        
        # Status indicators
        if result['f1_score'] >= baseline_all_22:
            status = "🏆"  # Matches or beats all features
        elif result['f1_score'] >= baseline_all_22 - 0.005:
            status = "🔥"  # Very close (within 0.5%)
        elif result['f1_score'] >= baseline_linguistic_8:
            status = "✅"  # Beats linguistic baseline
        else:
            status = "❌"  # Below linguistic baseline
        
        logger.info(f"{status} {result['method']:<22} | F1: {result['f1_score']:.4f} "
                   f"(vs all: {improvement_vs_all:+.4f}) | Features: {result['num_features']:2d} | "
                   f"Reduction: {reduction:4.1f}%")
    
    # Key findings
    best_result = sorted_results[0]
    
    logger.info(f"\n🎯 BEST RESULT: {best_result['method']}")
    logger.info(f"   F1 Score: {best_result['f1_score']:.4f}")
    logger.info(f"   Features: {best_result['num_features']} (vs 22 baseline)")
    logger.info(f"   Reduction: {(22 - best_result['num_features'])/22*100:.1f}%")
    logger.info(f"   vs All Features: {best_result['f1_score'] - baseline_all_22:+.4f}")
    
    # Find minimum features to match/beat targets
    target_matches = []
    for threshold, name in [(baseline_all_22, "all features"), (baseline_linguistic_8, "linguistic baseline")]:
        matches = [r for r in sorted_results if r['f1_score'] >= threshold]
        if matches:
            best_match = min(matches, key=lambda x: x['num_features'])
            target_matches.append((name, best_match))
    
    logger.info(f"\n📊 MINIMUM FEATURES NEEDED:")
    for target_name, result in target_matches:
        logger.info(f"   To match {target_name}: {result['num_features']} features (F1={result['f1_score']:.4f})")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(sorted_results)
    results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/early_fusion_top_features_results.csv"
    df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to {results_path}")
    
    # Answer the key question
    if best_result['f1_score'] >= baseline_all_22:
        logger.info(f"\n🎉 ANSWER: YES! We can achieve F1≥0.88 with just {best_result['num_features']} features!")
    elif any(r['f1_score'] >= baseline_all_22 - 0.01 for r in sorted_results):
        close_result = max([r for r in sorted_results if r['f1_score'] >= baseline_all_22 - 0.01], key=lambda x: x['f1_score'])
        logger.info(f"\n🔥 ANSWER: Very close! {close_result['num_features']} features achieve F1={close_result['f1_score']:.4f} (within 1% of 0.88)")
    else:
        logger.info(f"\n📊 ANSWER: Best we can do is F1={best_result['f1_score']:.4f} with {best_result['num_features']} features")
    
    return sorted_results

if __name__ == "__main__":
    main()