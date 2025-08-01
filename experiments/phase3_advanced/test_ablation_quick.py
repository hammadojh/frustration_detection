#!/usr/bin/env python3
"""
Quick test of ablation study - just test a few cases to ensure it works
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
import time
import warnings
warnings.filterwarnings('ignore')

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

def extract_features_by_indices(texts, feature_indices):
    """Extract specific features by their indices"""
    if len(feature_indices) == 0:
        return np.empty((len(texts), 0))
    
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

def test_combination(data, feature_indices, combo_name):
    """Test specific feature combination"""
    logger.info(f"Testing {combo_name}: {len(feature_indices)} features")
    
    # Initialize models
    tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
    roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roberta.to(device)
    roberta.eval()
    
    # Handle empty feature case (text-only baseline)
    if len(feature_indices) == 0:
        train_texts_enhanced = data['train_texts']
        test_texts_enhanced = data['test_texts']
    else:
        # Extract and process features
        train_features = extract_features_by_indices(data['train_texts'], feature_indices)
        test_features = extract_features_by_indices(data['test_texts'], feature_indices)
        
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
    
    logger.info(f"  Result: F1={test_f1:.4f}, Acc={test_accuracy:.4f}")
    
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
    logger.info("=" * 60)
    logger.info("QUICK ABLATION TEST")
    logger.info("=" * 60)
    
    data = load_data()
    
    # Test key cases
    test_cases = [
        ([], "text_only_baseline"),
        ([2], "single_avg_politeness"),
        ([2, 4], "top_2_features"),
        ([0, 1, 2, 3, 4, 5, 6, 7], "linguistic_bundle"),
        (list(range(22)), "all_22_features")
    ]
    
    results = []
    for indices, name in test_cases:
        result = test_combination(data, indices, name)
        results.append(result)
    
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for result in results:
        logger.info(f"{result['combination']:<25} | F1: {result['test_f1']:.4f} | Features: {result['num_features']:2d}")
    
    logger.info(f"\n✅ All tests completed successfully!")
    logger.info("The full ablation study should work correctly.")
    
    return results

if __name__ == "__main__":
    main()