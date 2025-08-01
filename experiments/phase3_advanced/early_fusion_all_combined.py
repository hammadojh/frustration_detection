#!/usr/bin/env python3
"""
Test Early Fusion with ALL Feature Bundles Combined
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

def extract_all_features(texts):
    """Extract ALL features from ALL bundles"""
    extractor = FastFrustrationFeatureExtractor()
    
    all_bundles = [
        'linguistic_bundle',      # 8 features
        'dialogue_bundle',        # 4 features  
        'behavioral_bundle',      # 2 features
        'contextual_bundle',      # 3 features
        'emotion_dynamics_bundle', # 2 features
        'system_bundle',          # 2 features
        'user_model_bundle'       # 1 feature
    ]
    
    all_features = []
    
    for text in texts:
        text_features = []
        for bundle in all_bundles:
            bundle_features = extractor.extract_bundle_features([text], bundle)
            text_features.extend(list(bundle_features.values()))
        all_features.append(text_features)
    
    return np.array(all_features), all_bundles

def features_to_tokens(features_scaled):
    """Convert features to special tokens"""
    feature_tokens = []
    
    for feat_vec in features_scaled:
        tokens = []
        for i, feat_val in enumerate(feat_vec):
            # Bin feature value into discrete levels (0-9)
            level = int(np.clip((feat_val + 3) / 6 * 10, 0, 9))
            tokens.append(f"[FEAT{i}_{level}]")
        feature_tokens.append(" ".join(tokens))
    
    return feature_tokens

def get_roberta_embeddings(texts, tokenizer, roberta, device):
    """Extract RoBERTa [CLS] embeddings"""
    embeddings = []
    batch_size = 16
    
    logger.info(f"Processing {len(texts)} texts with RoBERTa...")
    
    roberta.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            outputs = roberta(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)

def main():
    logger.info("="*70)
    logger.info("EARLY FUSION: ALL BUNDLES COMBINED TEST")
    logger.info("="*70)
    
    # Load data
    data = load_data()
    
    # Initialize models
    tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
    roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roberta.to(device)
    
    # Extract ALL features from ALL bundles
    logger.info("Extracting ALL features from ALL 7 bundles...")
    train_features, bundles = extract_all_features(data['train_texts'])
    test_features, _ = extract_all_features(data['test_texts'])
    
    logger.info(f"Total features extracted: {train_features.shape[1]}")
    logger.info(f"Bundles included: {bundles}")
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Convert features to tokens
    logger.info("Converting features to special tokens...")
    train_feat_tokens = features_to_tokens(train_features_scaled)
    test_feat_tokens = features_to_tokens(test_features_scaled)
    
    # Show example of feature tokens
    logger.info(f"Example feature tokens: {train_feat_tokens[0][:100]}...")
    
    # Create enhanced texts with ALL feature tokens prepended
    train_texts_enhanced = [
        f"{feat_tokens} {text}" 
        for feat_tokens, text in zip(train_feat_tokens, data['train_texts'])
    ]
    test_texts_enhanced = [
        f"{feat_tokens} {text}" 
        for feat_tokens, text in zip(test_feat_tokens, data['test_texts'])
    ]
    
    logger.info(f"Example enhanced text: {train_texts_enhanced[0][:200]}...")
    
    # Get embeddings from enhanced text
    train_embeddings = get_roberta_embeddings(train_texts_enhanced, tokenizer, roberta, device)
    test_embeddings = get_roberta_embeddings(test_texts_enhanced, tokenizer, roberta, device)
    
    # Train classifier
    logger.info("Training classifier...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train_embeddings, data['train_labels'])
    
    # Predict
    predictions = model.predict(test_embeddings)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        data['test_labels'], predictions, average='binary'
    )
    accuracy = accuracy_score(data['test_labels'], predictions)
    
    # Results
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    
    baseline_linguistic = 0.8571  # Early fusion with linguistic bundle only
    baseline_phase2 = 0.8378     # Late fusion with linguistic bundle
    
    logger.info(f"Early Fusion ALL Bundles: F1 = {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Total features: {train_features.shape[1]}")
    logger.info("")
    
    improvement_vs_linguistic = f1 - baseline_linguistic
    improvement_vs_phase2 = f1 - baseline_phase2
    
    logger.info("COMPARISONS:")
    logger.info(f"vs. Early Fusion Linguistic Only: Δ={improvement_vs_linguistic:+.4f}")
    logger.info(f"vs. Phase 2 Late Fusion: Δ={improvement_vs_phase2:+.4f}")
    
    status_vs_linguistic = "🏆" if improvement_vs_linguistic > 0.01 else "✅" if improvement_vs_linguistic > 0 else "❌"
    logger.info(f"{status_vs_linguistic} Early Fusion ALL vs Linguistic Only")
    
    # Save result
    result = {
        'experiment': 'early_fusion_all_bundles',
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'num_features': train_features.shape[1],
        'improvement_vs_linguistic': improvement_vs_linguistic,
        'improvement_vs_phase2': improvement_vs_phase2
    }
    
    import pandas as pd
    df = pd.DataFrame([result])
    results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/early_fusion_all_combined_result.csv"
    df.to_csv(results_path, index=False)
    logger.info(f"\nResult saved to {results_path}")
    
    if improvement_vs_linguistic > 0:
        logger.info(f"\n🎉 SUCCESS! All bundles improved over linguistic-only by {improvement_vs_linguistic:+.4f}")
    else:
        logger.info(f"\n📊 Result: All bundles did not improve over linguistic-only ({improvement_vs_linguistic:+.4f})")
    
    return result

if __name__ == "__main__":
    main()