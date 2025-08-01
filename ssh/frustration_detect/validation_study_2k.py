#!/usr/bin/env python3
"""
Validation Study with 2000 Balanced Datapoints
Scale up the comprehensive ablation study to validate findings with larger dataset
"""

import json
import torch
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import mutual_info_classif
from datasets import load_dataset
import logging
import sys
import time
import pandas as pd
from itertools import combinations
import warnings
import os
import random
warnings.filterwarnings('ignore')

sys.path.append('/Users/omarhammad/Documents/code_local/frustration_researcher/ssh/frustration_detect')
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

def create_balanced_dataset_2k():
    """Create balanced 2000-sample dataset from EmoWOZ"""
    logger.info("Loading EmoWOZ dataset for 2k validation...")
    dataset = load_dataset("hhu-dsml/emowoz")
    
    # Map EmoWOZ emotion IDs to binary frustration labels
    FRUSTRATED_EMOTION_IDS = {2, 5, 6}  # dissatisfied, abusive, fearful
    
    # Extract turn-level examples with emotions
    frustrated_examples = []
    not_frustrated_examples = []
    
    # Use all splits for maximum data diversity
    for split_name in ['train', 'validation', 'test']:
        split_data = dataset[split_name]
        logger.info(f"Processing {split_name} split with {len(split_data)} dialogues...")
        
        for dialogue in split_data:
            texts = dialogue['log']['text']
            emotions = dialogue['log']['emotion']
            
            for text, emotion_id in zip(texts, emotions):
                if emotion_id == -1:  # Skip system turns
                    continue
                    
                example = {
                    'text': text,
                    'emotion_id': emotion_id,
                    'dialogue_id': dialogue['dialogue_id'],
                    'split': split_name
                }
                
                if emotion_id in FRUSTRATED_EMOTION_IDS:
                    frustrated_examples.append(example)
                else:
                    not_frustrated_examples.append(example)
    
    logger.info(f"Total available: {len(frustrated_examples)} frustrated, {len(not_frustrated_examples)} not frustrated")
    
    # Balance the 2k subset
    target_frustrated = min(1000, len(frustrated_examples))
    target_not_frustrated = min(1000, len(not_frustrated_examples))
    
    # Shuffle examples to ensure good mixing across dialogues and splits
    random.seed(42)  # For reproducibility
    random.shuffle(frustrated_examples)
    random.shuffle(not_frustrated_examples)
    
    subset_examples = (
        frustrated_examples[:target_frustrated] + 
        not_frustrated_examples[:target_not_frustrated]
    )
    
    # Shuffle the final subset to mix classes
    random.shuffle(subset_examples)
    
    logger.info(f"2k Dataset created: {target_frustrated} frustrated, {target_not_frustrated} not frustrated")
    
    # Convert to processed format
    processed_examples = []
    for example in subset_examples:
        label = int(example['emotion_id'] in FRUSTRATED_EMOTION_IDS)
        processed_examples.append({
            'text': example['text'],
            'label': label,
            'emotion_id': example['emotion_id'],
            'dialogue_id': example['dialogue_id'],
            'split': example['split']
        })
    
    return processed_examples

def save_2k_dataset(examples, output_dir="data"):
    """Save 2k dataset for reproducibility"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed examples
    with open(os.path.join(output_dir, 'subset_2k_processed.json'), 'w') as f:
        json.dump(examples, f, indent=2)
    
    # Save statistics
    labels = [ex['label'] for ex in examples]
    emotion_ids = [ex['emotion_id'] for ex in examples]
    splits = [ex['split'] for ex in examples]
    
    stats = {
        'total_examples': len(examples),
        'frustrated_count': sum(labels),
        'not_frustrated_count': len(labels) - sum(labels),
        'emotion_id_distribution': pd.Series(emotion_ids).value_counts().to_dict(),
        'split_distribution': pd.Series(splits).value_counts().to_dict()
    }
    
    with open(os.path.join(output_dir, 'subset_2k_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"2k Dataset saved to {output_dir}")
    logger.info(f"Statistics: {stats}")

def load_data():
    """Load processed 2k examples"""
    processed_path = "data/subset_2k_processed.json"
    
    # Create dataset if it doesn't exist
    if not os.path.exists(processed_path):
        logger.info("2k dataset not found, creating it...")
        examples = create_balanced_dataset_2k()
        save_2k_dataset(examples)
    else:
        logger.info("Loading existing 2k dataset...")
        with open(processed_path, 'r') as f:
            examples = json.load(f)
    
    texts = [ex['text'] for ex in examples]
    labels = [ex['label'] for ex in examples]
    
    n = len(texts)
    train_end = int(0.7 * n)  # 70% train = 1400 samples
    val_end = int(0.85 * n)   # 15% val = 300 samples, 15% test = 300 samples
    
    logger.info(f"Data splits: Train={train_end}, Val={val_end-train_end}, Test={n-val_end}")
    
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
    batch_size = 100  # Process in batches for efficiency
    
    logger.info(f"Extracting features for {len(texts)} texts in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        if (i // batch_size) % 5 == 0:  # Log every 5 batches
            logger.info(f"  Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        for text in batch_texts:
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
    
    logger.info(f"Testing {combo_name} with {len(feature_indices)} features...")
    
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
        
        logger.info(f"  Extracting embeddings for {len(texts)} texts...")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                if (i // batch_size) % 10 == 0:  # Log every 10 batches
                    logger.info(f"    Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, truncation=True, padding="max_length", 
                                 max_length=512, return_tensors="pt").to(device)
                outputs = roberta(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(embeddings)
    
    train_embeddings = get_embeddings(train_texts_enhanced)
    test_embeddings = get_embeddings(test_texts_enhanced)
    
    # Train classifier
    logger.info("  Training classifier...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train_embeddings, data['train_labels'])
    
    # Predictions
    test_predictions = model.predict(test_embeddings)
    
    # Calculate metrics
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        data['test_labels'], test_predictions, average='binary'
    )
    test_accuracy = accuracy_score(data['test_labels'], test_predictions)
    
    logger.info(f"  Result: F1={test_f1:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}")
    
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
    logger.info("VALIDATION STUDY: 2K BALANCED DATAPOINTS")
    logger.info("=" * 80)
    logger.info("This validation study will test key configurations from the 500-sample ablation:")
    logger.info("  1. Baseline (text-only): 1 test")
    logger.info("  2. Top individual features: 5 tests")
    logger.info("  3. Best bundles: 3 tests")
    logger.info("  4. Key combinations: 5 tests")
    logger.info("  5. Leave-one-out (top features): 5 tests")
    logger.info("  Total: ~19 focused experiments on 2k data")
    logger.info("")
    
    # Load 2k data
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
    baseline_result = test_early_fusion_combination(data, [], "text_only_baseline_2k")
    results.append(baseline_result)
    logger.info(f"   2k Text-only F1: {baseline_result['test_f1']:.4f}")
    
    # 2. TOP INDIVIDUAL FEATURES (from 500-sample study)
    logger.info("\n2. Testing TOP INDIVIDUAL FEATURES from 500-sample study")
    # Based on ablation results: sentiment_slope, avg_politeness, avg_confusion, avg_negation were top
    top_individual_indices = [0, 2, 4, 5, 8]  # Top 5 from previous study
    top_individual_names = [feature_names[i] for i in top_individual_indices]
    
    for i, idx in enumerate(top_individual_indices):
        name = feature_names[idx]
        logger.info(f"   Testing individual feature {i+1}/5: {name}")
        result = test_early_fusion_combination(data, [idx], f"individual_{name}_2k")
        results.append(result)
    
    # 3. BEST BUNDLES (from 500-sample study)
    logger.info("\n3. Testing BEST BUNDLES from 500-sample study")
    bundle_indices = {
        'linguistic_bundle': list(range(0, 8)),      # Best bundle
        'behavioral_bundle': list(range(12, 14)),    # High efficiency
        'system_bundle': list(range(19, 21))         # High efficiency
    }
    
    for bundle_name, indices in bundle_indices.items():
        logger.info(f"   Testing {bundle_name}: {len(indices)} features")
        result = test_early_fusion_combination(data, indices, f"bundle_{bundle_name}_2k")
        results.append(result)
    
    # 4. KEY COMBINATIONS (from 500-sample study)
    logger.info("\n4. Testing KEY COMBINATIONS from 500-sample study")
    key_combinations = {
        'top_3_features': [0, 2, 4],                    # Top 3 individual
        'top_8_features': [0, 2, 4, 5, 8, 9, 10, 11],  # Best cumulative from ablation
        'linguistic_8': list(range(0, 8)),              # Linguistic bundle (best individual bundle)
        'all_22_features': list(range(22)),             # Full feature set
        'optimal_21_features': list(range(21))          # N-1 optimal (remove last feature)
    }
    
    for combo_name, indices in key_combinations.items():
        logger.info(f"   Testing {combo_name}: {len(indices)} features")
        result = test_early_fusion_combination(data, indices, f"{combo_name}_2k")
        results.append(result)
        
        # Save progress after each major test
        temp_df = pd.DataFrame(results)
        temp_path = f"results/validation_2k_progress_{len(results)}_tests.csv"
        temp_df.to_csv(temp_path, index=False)
        logger.info(f"     Progress saved: {len(results)} tests completed")
    
    # 5. LEAVE-ONE-OUT for critical features (from 500-sample study)
    logger.info("\n5. Testing LEAVE-ONE-OUT for most critical features")
    # Test removing the most critical features found in 500-sample study
    critical_features_to_remove = [0, 2, 4, 5, 7]  # Top 5 most critical from ablation
    full_set = list(range(22))
    
    for i, feat_idx in enumerate(critical_features_to_remove):
        name = feature_names[feat_idx]
        logger.info(f"   Removing critical feature {i+1}/5: {name}")
        remaining_features = [x for x in full_set if x != feat_idx]
        result = test_early_fusion_combination(data, remaining_features, f"loo_remove_{name}_2k")
        results.append(result)
    
    # Final save
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    csv_path = f"{output_dir}/validation_study_2k.csv"
    df.to_csv(csv_path, index=False)
    
    # Create comparison report
    report_path = f"{output_dir}/validation_study_2k_REPORT.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VALIDATION STUDY: 2K BALANCED DATAPOINTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total validation experiments: {len(results)}\n")
        f.write(f"Dataset size: 2000 balanced examples (1000 frustrated, 1000 not frustrated)\n")
        f.write(f"Train/Val/Test split: 1400/300/300\n\n")
        
        # Baseline comparison
        baseline_f1 = baseline_result['test_f1']
        f.write(f"2K BASELINE PERFORMANCE:\n")
        f.write(f"  Text-only F1: {baseline_f1:.4f}\n\n")
        
        # Sort results by F1 score
        results_sorted = sorted(results, key=lambda x: x['test_f1'], reverse=True)
        
        f.write("TOP PERFORMING CONFIGURATIONS (2K validation):\n")
        f.write("-" * 60 + "\n")
        for rank, result in enumerate(results_sorted[:10], 1):
            f.write(f"{rank:2d}. {result['combination']:<30} | F1: {result['test_f1']:.4f} | Features: {result['num_features']:2d}\n")
        
        f.write("\nKEY VALIDATION INSIGHTS:\n")
        f.write("-" * 60 + "\n")
        
        # Compare with 500-sample results
        f.write("• 500-sample text-only baseline: 0.8378\n")
        f.write(f"• 2k-sample text-only baseline: {baseline_f1:.4f}\n")
        f.write(f"• Baseline difference: {baseline_f1 - 0.8378:+.4f}\n\n")
        
        # Best configuration
        best_result = results_sorted[0]
        f.write(f"• Best 2k configuration: {best_result['combination']}\n")
        f.write(f"• Best 2k F1 score: {best_result['test_f1']:.4f}\n")
        f.write(f"• Improvement over 2k baseline: {best_result['test_f1'] - baseline_f1:+.4f}\n")
    
    logger.info(f"\n🎯 VALIDATION STUDY COMPLETE!")
    logger.info(f"Results saved to:")
    logger.info(f"  Data: {csv_path}")
    logger.info(f"  Report: {report_path}")
    
    # Print key findings
    best_result = sorted(results, key=lambda x: x['test_f1'], reverse=True)[0]
    logger.info(f"\n🔍 KEY VALIDATION FINDINGS:")
    logger.info(f"  2k Text-only baseline: {baseline_result['test_f1']:.4f}")
    logger.info(f"  Best 2k configuration: {best_result['combination']} (F1: {best_result['test_f1']:.4f})")
    logger.info(f"  Improvement: {best_result['test_f1'] - baseline_result['test_f1']:+.4f}")
    
    return results

if __name__ == "__main__":
    main()