
#!/usr/bin/env python3
"""
Comprehensive Ablation Study for Feature Importance
Test multiple ablation strategies to understand feature contributions
"""

import json
import torch
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import mutual_info_classif
import logging
import sys
import time
import pandas as pd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

#sys.path.append('/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase2_features')
from fast_feature_extractor import FastFrustrationFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model loading to avoid timeout
tokenizer = None
roberta = None
device = None

def initialize_models():
    """Initialize models once to avoid timeout"""
    global tokenizer, roberta, device
    if tokenizer is None:
        logger.info("Initializing RoBERTa models...")
        tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
        roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        roberta.to(device)
        roberta.eval()
        logger.info("Models initialized successfully!")

def load_data():
    """Load processed examples"""
    processed_path = "data/subset_processed.json"
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

def extract_features_by_indices(texts, feature_indices):
    """Extract specific features by their indices"""
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

def test_early_fusion_combination(data, feature_indices, combo_name):
    """Test early fusion with specific feature combination"""
    initialize_models()
    
    # Handle empty feature case (text-only baseline)
    if len(feature_indices) == 0:
        # Text-only case: use original texts without feature tokens
        train_texts_enhanced = data['train_texts']
        test_texts_enhanced = data['test_texts']
    else:
        # Extract selected features
        train_features = extract_features_by_indices(data['train_texts'], feature_indices)
        test_features = extract_features_by_indices(data['test_texts'], feature_indices)
        
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
    
    # Train classifier
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train_embeddings, data['train_labels'])
    
    # Predictions
    test_predictions = model.predict(test_embeddings)
    
    # Calculate metrics
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        data['test_labels'], test_predictions, average='binary'
    )
    test_accuracy = accuracy_score(data['test_labels'], test_predictions)
    
    return {
        'combination': combo_name,
        'feature_indices': feature_indices,
        'num_features': len(feature_indices),
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_accuracy': test_accuracy
    }

def main():
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ABLATION STUDY FOR FEATURE IMPORTANCE")
    logger.info("=" * 80)
    logger.info("This study will conduct ~70 experiments:")
    logger.info("  1. Baseline (text-only): 1 test")
    logger.info("  2. Individual features: 22 tests")
    logger.info("  3. Bundle-level: 7 tests")
    logger.info("  4. Top-K combinations: 6 tests")
    logger.info("  5. Leave-one-out: 22 tests")
    logger.info("  6. Cumulative addition: 10 tests")
    logger.info("  Total: ~68 experiments")
    logger.info("")
    
    # Load data
    data = load_data()
    
    # Feature names for reference
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
    
    results = []
    
    # 1. BASELINE: Text-only (no features)
    logger.info("\n1. Testing BASELINE: Text-only (no features)")
    baseline_result = test_early_fusion_combination(data, [], "text_only_baseline")
    results.append(baseline_result)
    logger.info(f"   Text-only F1: {baseline_result['test_f1']:.4f}")
    
    # 2. INDIVIDUAL FEATURE IMPORTANCE: Test each feature alone
    logger.info("\n2. Testing INDIVIDUAL FEATURES (22 tests)")
    individual_results = []
    for i, name in enumerate(feature_names):
        logger.info(f"   Testing feature {i+1}/22: {name}")
        result = test_early_fusion_combination(data, [i], f"individual_{name}")
        results.append(result)
        individual_results.append((i, name, result['test_f1']))
    
    # Sort individual features by performance
    individual_results.sort(key=lambda x: x[2], reverse=True)
    logger.info("\n   TOP 10 INDIVIDUAL FEATURES:")
    for rank, (idx, name, f1) in enumerate(individual_results[:10], 1):
        logger.info(f"     {rank:2d}. {name:<25} | F1: {f1:.4f} | Idx: {idx}")
    
    # 3. BUNDLE-LEVEL ABLATION: Test each bundle
    logger.info("\n3. Testing BUNDLE-LEVEL PERFORMANCE (7 tests)")
    bundle_indices = {
        'linguistic_bundle': list(range(0, 8)),
        'dialogue_bundle': list(range(8, 12)), 
        'behavioral_bundle': list(range(12, 14)),
        'contextual_bundle': list(range(14, 17)),
        'emotion_dynamics_bundle': list(range(17, 19)),
        'system_bundle': list(range(19, 21)),
        'user_model_bundle': [21]
    }
    
    bundle_results = []
    for bundle_name, indices in bundle_indices.items():
        logger.info(f"   Testing {bundle_name}: {len(indices)} features")
        result = test_early_fusion_combination(data, indices, f"bundle_{bundle_name}")
        results.append(result)
        bundle_results.append((bundle_name, len(indices), result['test_f1']))
    
    # Sort bundles by performance
    bundle_results.sort(key=lambda x: x[2], reverse=True)
    logger.info("\n   BUNDLE PERFORMANCE RANKING:")
    for rank, (name, size, f1) in enumerate(bundle_results, 1):
        logger.info(f"     {rank}. {name:<25} | F1: {f1:.4f} | Size: {size}")
    
    # 4. TOP-K FEATURE COMBINATIONS
    logger.info("\n4. Testing TOP-K FEATURE COMBINATIONS")
    top_features = [x[0] for x in individual_results[:15]]  # Top 15 individual features
    
    for k in [3, 5, 8, 10, 12, 15]:
        if k <= len(top_features):
            logger.info(f"   Testing top {k} features")
            result = test_early_fusion_combination(data, top_features[:k], f"top_{k}_features")
            results.append(result)
            logger.info(f"     Top {k} F1: {result['test_f1']:.4f}")
            
            # Save intermediate progress
            if k % 5 == 0:
                temp_df = pd.DataFrame(results)
                temp_path = f"results/ablation_progress_{len(results)}_tests.csv"
                temp_df.to_csv(temp_path, index=False)
                logger.info(f"     Progress saved: {len(results)} tests completed")
    
    # 5. LEAVE-ONE-OUT ABLATION: Remove each feature from full set
    logger.info("\n5. Testing LEAVE-ONE-OUT ABLATION (22 tests)")
    full_set = list(range(22))
    loo_results = []
    
    for i, name in enumerate(feature_names):
        logger.info(f"   Removing feature {i+1}/22: {name}")
        remaining_features = [x for x in full_set if x != i]
        result = test_early_fusion_combination(data, remaining_features, f"loo_remove_{name}")
        results.append(result)
        
        # Calculate performance drop
        full_f1 = 0.8831  # Known full performance
        drop = full_f1 - result['test_f1']
        loo_results.append((i, name, result['test_f1'], drop))
        
        # Save progress every 10 features
        if (i + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_path = f"results/ablation_progress_{len(results)}_tests.csv"
            temp_df.to_csv(temp_path, index=False)
            logger.info(f"     Progress saved: {len(results)} tests completed")
    
    # Sort by performance drop (most important features cause biggest drops)
    loo_results.sort(key=lambda x: x[3], reverse=True)
    logger.info("\n   MOST IMPORTANT FEATURES (by removal impact):")
    for rank, (idx, name, f1, drop) in enumerate(loo_results[:10], 1):
        logger.info(f"     {rank:2d}. {name:<25} | Drop: {drop:+.4f} | F1: {f1:.4f}")
    
    # 6. CUMULATIVE FEATURE ADDITION
    logger.info("\n6. Testing CUMULATIVE ADDITION")
    cumulative_indices = []
    for i, (idx, name, _) in enumerate(individual_results[:10]):
        cumulative_indices.append(idx)
        logger.info(f"   Adding feature {i+1}: {name}")
        result = test_early_fusion_combination(data, cumulative_indices.copy(), f"cumulative_{i+1}")
        results.append(result)
        logger.info(f"     Cumulative {i+1} F1: {result['test_f1']:.4f}")
    
    # Final progress save
    temp_df = pd.DataFrame(results)
    temp_path = "results/ablation_final_progress.csv"
    temp_df.to_csv(temp_path, index=False)
    logger.info(f"\nFinal progress saved: {len(results)} total tests completed")
    
    # Save all results
    output_dir = "results"
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    csv_path = f"{output_dir}/comprehensive_ablation_study.csv"
    df.to_csv(csv_path, index=False)
    
    # Create detailed report
    report_path = f"{output_dir}/comprehensive_ablation_study_REPORT.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE ABLATION STUDY - FEATURE IMPORTANCE ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total experiments conducted: {len(results)}\n")
        f.write(f"Baseline (text-only) F1: {baseline_result['test_f1']:.4f}\n")
        f.write(f"Full 22 features F1: 0.8831 (reference)\n\n")
        
        f.write("TOP 10 INDIVIDUAL FEATURES (when used alone):\n")
        f.write("-" * 60 + "\n")
        for rank, (idx, name, f1) in enumerate(individual_results[:10], 1):
            f.write(f"{rank:2d}. {name:<25} | F1: {f1:.4f} | Index: {idx:2d}\n")
        
        f.write("\nBUNDLE PERFORMANCE RANKING:\n")
        f.write("-" * 60 + "\n")
        for rank, (name, size, f1) in enumerate(bundle_results, 1):
            f.write(f"{rank}. {name:<25} | F1: {f1:.4f} | Features: {size:2d}\n")
        
        f.write("\nMOST CRITICAL FEATURES (by removal impact):\n")
        f.write("-" * 60 + "\n")
        for rank, (idx, name, f1, drop) in enumerate(loo_results[:10], 1):
            f.write(f"{rank:2d}. {name:<25} | Impact: {drop:+.4f} | F1: {f1:.4f}\n")
        
        f.write("\nKEY INSIGHTS:\n")
        f.write("-" * 60 + "\n")
        
        # Best individual feature
        best_individual = individual_results[0]
        f.write(f"• Best individual feature: {best_individual[1]} (F1: {best_individual[2]:.4f})\n")
        
        # Best bundle
        best_bundle = bundle_results[0]
        f.write(f"• Best bundle: {best_bundle[0]} (F1: {best_bundle[2]:.4f})\n")
        
        # Most critical feature
        most_critical = loo_results[0]
        f.write(f"• Most critical feature: {most_critical[1]} (impact: {most_critical[3]:+.4f})\n")
        
        # Performance ranges
        individual_f1s = [x[2] for x in individual_results]
        f.write(f"• Individual feature F1 range: {min(individual_f1s):.4f} to {max(individual_f1s):.4f}\n")
        
        bundle_f1s = [x[2] for x in bundle_results]
        f.write(f"• Bundle F1 range: {min(bundle_f1s):.4f} to {max(bundle_f1s):.4f}\n")
    
    logger.info(f"\n📊 ABLATION STUDY COMPLETE!")
    logger.info(f"Results saved to:")
    logger.info(f"  Data: {csv_path}")
    logger.info(f"  Report: {report_path}")
    
    # Print key findings
    logger.info(f"\n🔍 KEY FINDINGS:")
    logger.info(f"  Best individual: {individual_results[0][1]} (F1: {individual_results[0][2]:.4f})")
    logger.info(f"  Best bundle: {bundle_results[0][0]} (F1: {bundle_results[0][2]:.4f})")
    logger.info(f"  Most critical: {loo_results[0][1]} (impact: {loo_results[0][3]:+.4f})")
    
    return results

if __name__ == "__main__":
    main()
