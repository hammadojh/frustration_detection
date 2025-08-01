#!/usr/bin/env python3
"""
Test remaining fusion strategies (3 & 4) efficiently
"""

import json
import torch
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif
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

def get_cached_embeddings_and_features():
    """Get embeddings and features efficiently"""
    tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
    roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
    extractor = FastFrustrationFeatureExtractor()
    
    data = load_data()
    
    logger.info("Extracting embeddings and features...")
    
    # Get embeddings
    def get_embeddings(texts):
        embeddings = []
        batch_size = 16
        
        roberta.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, truncation=True, padding="max_length", 
                                 max_length=512, return_tensors="pt")
                outputs = roberta(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())
        return np.vstack(embeddings)
    
    # Get features
    def get_features(texts):
        all_features = []
        for text in texts:
            features = extractor.extract_linguistic_bundle([text])
            all_features.append(list(features.values()))
        return np.array(all_features)
    
    train_embeddings = get_embeddings(data['train_texts'])
    test_embeddings = get_embeddings(data['test_texts'])
    
    train_features = get_features(data['train_texts'])
    test_features = get_features(data['test_texts'])
    
    return data, train_embeddings, test_embeddings, train_features, test_features

def test_weighted_fusion(data, train_embeddings, test_embeddings, train_features, test_features):
    """Strategy 3: Weighted fusion"""
    logger.info("Testing Strategy 3: Weighted Fusion")
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    best_alpha = 0.5
    best_score = 0
    
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        weighted_train = np.hstack([
            alpha * train_embeddings,
            (1 - alpha) * train_features_scaled
        ])
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(weighted_train, data['train_labels'])
        train_pred = model.predict(weighted_train)
        train_acc = accuracy_score(data['train_labels'], train_pred)
        
        if train_acc > best_score:
            best_score = train_acc
            best_alpha = alpha
    
    # Final model
    weighted_train = np.hstack([
        best_alpha * train_embeddings,
        (1 - best_alpha) * train_features_scaled
    ])
    weighted_test = np.hstack([
        best_alpha * test_embeddings,
        (1 - best_alpha) * test_features_scaled
    ])
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(weighted_train, data['train_labels'])
    predictions = model.predict(weighted_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        data['test_labels'], predictions, average='binary'
    )
    
    logger.info(f"weighted_fusion: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, alpha={best_alpha}")
    return f1, precision, recall, best_alpha

def test_feature_selection_fusion(data, train_embeddings, test_embeddings, train_features, test_features):
    """Strategy 4: Feature selection fusion"""
    logger.info("Testing Strategy 4: Feature Selection Fusion")
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    best_k = train_features.shape[1]
    best_score = 0
    
    for k in [3, 5, train_features.shape[1]]:
        if k > train_features.shape[1]:
            k = train_features.shape[1]
        
        selector = SelectKBest(f_classif, k=k)
        train_feat_selected = selector.fit_transform(train_features_scaled, data['train_labels'])
        test_feat_selected = selector.transform(test_features_scaled)
        
        train_combined = np.hstack([train_embeddings, train_feat_selected])
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_combined, data['train_labels'])
        train_pred = model.predict(train_combined)
        train_acc = accuracy_score(data['train_labels'], train_pred)
        
        if train_acc > best_score:
            best_score = train_acc
            best_k = k
    
    # Final model
    selector = SelectKBest(f_classif, k=best_k)
    train_feat_selected = selector.fit_transform(train_features_scaled, data['train_labels'])
    test_feat_selected = selector.transform(test_features_scaled)
    
    train_combined = np.hstack([train_embeddings, train_feat_selected])
    test_combined = np.hstack([test_embeddings, test_feat_selected])
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train_combined, data['train_labels'])
    predictions = model.predict(test_combined)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        data['test_labels'], predictions, average='binary'
    )
    
    logger.info(f"feature_selection_fusion: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, k={best_k}")
    return f1, precision, recall, best_k

def main():
    logger.info("="*60)
    logger.info("TESTING REMAINING FUSION STRATEGIES")
    logger.info("="*60)
    
    # Get data and precomputed embeddings/features
    data, train_embeddings, test_embeddings, train_features, test_features = get_cached_embeddings_and_features()
    
    # Test strategies
    baseline_f1 = 0.8378
    
    logger.info(f"Phase 2 baseline: F1 = {baseline_f1:.4f}")
    logger.info(f"Early fusion tokens: F1 = 0.8571 (Δ=+0.0193) [PREVIOUS RESULT]")
    logger.info("")
    
    # Strategy 3
    try:
        f1_3, p_3, r_3, alpha = test_weighted_fusion(data, train_embeddings, test_embeddings, train_features, test_features)
        improvement_3 = f1_3 - baseline_f1
        status_3 = "✅" if improvement_3 > 0 else "❌"
        logger.info(f"{status_3} weighted_fusion: Δ={improvement_3:+.4f}")
    except Exception as e:
        logger.error(f"Strategy 3 failed: {e}")
        f1_3 = 0
    
    # Strategy 4
    try:
        f1_4, p_4, r_4, k = test_feature_selection_fusion(data, train_embeddings, test_embeddings, train_features, test_features)
        improvement_4 = f1_4 - baseline_f1
        status_4 = "✅" if improvement_4 > 0 else "❌"
        logger.info(f"{status_4} feature_selection_fusion: Δ={improvement_4:+.4f}")
    except Exception as e:
        logger.error(f"Strategy 4 failed: {e}")
        f1_4 = 0
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("COMPLETE FUSION STRATEGIES RESULTS")
    logger.info("="*60)
    
    results = [
        ("late_fusion_concat", 0.8378, 0.0000),
        ("early_fusion_tokens", 0.8571, 0.0193),
        ("weighted_fusion", f1_3, f1_3 - baseline_f1 if f1_3 > 0 else 0),
        ("feature_selection_fusion", f1_4, f1_4 - baseline_f1 if f1_4 > 0 else 0)
    ]
    
    # Sort by F1
    results.sort(key=lambda x: x[1], reverse=True)
    
    for name, f1, improvement in results:
        if f1 > 0:
            status = "🏆" if improvement > 0.01 else "✅" if improvement > 0 else "❌"
            logger.info(f"{status} {name}: F1 = {f1:.4f} (Δ={improvement:+.4f})")
    
    best_strategy = results[0]
    logger.info(f"\n🎉 BEST FUSION STRATEGY: {best_strategy[0]}")
    logger.info(f"F1 Score: {best_strategy[1]:.4f}")

if __name__ == "__main__":
    main()