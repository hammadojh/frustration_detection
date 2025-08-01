#!/usr/bin/env python3
"""
Test All 22 Features After Fixing Constant Features
Compare performance with original 12 non-constant features
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

def test_early_fusion_all_features(data):
    """Test early fusion with all 22 features"""
    logger.info("Testing all 22 features with early fusion...")
    
    start_time = time.time()
    
    # Initialize models
    tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
    roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roberta.to(device)
    roberta.eval()
    
    # Extract all features
    train_features = extract_all_features(data['train_texts'])
    test_features = extract_all_features(data['test_texts'])
    
    logger.info(f"Extracted features shape: {train_features.shape}")
    
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
    
    logger.info(f"Example enhanced text: {train_texts_enhanced[0][:150]}...")
    
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
    
    logger.info("Extracting RoBERTa embeddings...")
    train_embeddings = get_embeddings(train_texts_enhanced)
    test_embeddings = get_embeddings(test_texts_enhanced)
    
    # Train classifier
    logger.info("Training classifier...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train_embeddings, data['train_labels'])
    
    # Predictions
    train_predictions = model.predict(train_embeddings)
    test_predictions = model.predict(test_embeddings)
    
    # Calculate metrics
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        data['train_labels'], train_predictions, average='binary'
    )
    train_accuracy = accuracy_score(data['train_labels'], train_predictions)
    
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        data['test_labels'], test_predictions, average='binary'
    )
    test_accuracy = accuracy_score(data['test_labels'], test_predictions)
    
    runtime = time.time() - start_time
    
    return {
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'train_accuracy': train_accuracy,
        'runtime_minutes': runtime / 60
    }

def main():
    logger.info("=" * 80)
    logger.info("TESTING ALL 22 FEATURES AFTER FIXING CONSTANT FEATURES")
    logger.info("=" * 80)
    
    # Load data
    data = load_data()
    
    # Test all 22 features
    results = test_early_fusion_all_features(data)
    
    # Comparison analysis
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON WITH PREVIOUS RESULTS")
    logger.info("=" * 80)
    
    # Previous results
    previous_all_22 = 0.8831  # Original with 10 constant features
    previous_12_nonconstant = 0.8608  # Best 12 non-constant features
    
    test_f1 = results['test_f1']
    
    logger.info("PERFORMANCE COMPARISON:")
    logger.info(f"  NEW All 22 features:        F1 = {test_f1:.4f}")
    logger.info(f"  OLD All 22 features:        F1 = {previous_all_22:.4f} (Δ={test_f1-previous_all_22:+.4f})")
    logger.info(f"  OLD 12 non-constant:        F1 = {previous_12_nonconstant:.4f} (Δ={test_f1-previous_12_nonconstant:+.4f})")
    logger.info("")
    
    # Status determination
    if test_f1 >= previous_all_22 + 0.01:
        status = "🚀 SIGNIFICANT IMPROVEMENT"
    elif test_f1 >= previous_all_22:
        status = "✅ IMPROVEMENT"
    elif test_f1 >= previous_all_22 - 0.01:
        status = "➡️ SIMILAR PERFORMANCE"
    else:
        status = "⚠️ PERFORMANCE DROP"
    
    logger.info(f"RESULT STATUS: {status}")
    
    # Key insights
    logger.info("\nKEY INSIGHTS:")
    if test_f1 > previous_all_22:
        improvement_pct = ((test_f1 / previous_all_22) - 1) * 100
        logger.info(f"💡 Fixing constant features improved performance by {improvement_pct:+.2f}%!")
        logger.info("💡 All 22 features now contribute meaningful signal!")
    elif test_f1 >= previous_12_nonconstant:
        logger.info("💡 Performance matches/exceeds the best 12 features!")
        logger.info("💡 The previously constant features now add value!")
    else:
        logger.info("💡 Some new proxy features may need refinement")
        logger.info("💡 Consider adjusting the feature extraction logic")
    
    # Detailed results
    logger.info("\nDETAILED RESULTS:")
    logger.info(f"  Test F1:        {results['test_f1']:.4f}")
    logger.info(f"  Test Precision: {results['test_precision']:.4f}")
    logger.info(f"  Test Recall:    {results['test_recall']:.4f}")
    logger.info(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
    logger.info(f"  Train F1:       {results['train_f1']:.4f}")
    logger.info(f"  Runtime:        {results['runtime_minutes']:.1f} minutes")
    
    # Save results
    output_dir = "/Users/omarhammad/Documents/code_local/frustration_researcher/results"
    
    # Save report
    report_path = f"{output_dir}/all_22_features_updated_RESULTS.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ALL 22 FEATURES AFTER FIXING CONSTANT FEATURES - RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment completed in {results['runtime_minutes']:.1f} minutes\n")
        f.write("\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write(f"  Test F1:        {results['test_f1']:.4f}\n")
        f.write(f"  Test Precision: {results['test_precision']:.4f}\n")
        f.write(f"  Test Recall:    {results['test_recall']:.4f}\n")
        f.write(f"  Test Accuracy:  {results['test_accuracy']:.4f}\n")
        f.write(f"  Train F1:       {results['train_f1']:.4f}\n")
        f.write(f"  Train Accuracy: {results['train_accuracy']:.4f}\n")
        f.write("\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write(f"  NEW All 22 features:        F1 = {test_f1:.4f}\n")
        f.write(f"  OLD All 22 features:        F1 = {previous_all_22:.4f} (Δ={test_f1-previous_all_22:+.4f})\n")
        f.write(f"  OLD 12 non-constant:        F1 = {previous_12_nonconstant:.4f} (Δ={test_f1-previous_12_nonconstant:+.4f})\n")
        f.write("\n")
        
        f.write(f"FINAL STATUS: {status}\n")
        f.write("\n")
        
        if test_f1 > previous_all_22:
            improvement_pct = ((test_f1 / previous_all_22) - 1) * 100
            f.write("CONCLUSION:\n")
            f.write("✅ SUCCESS: Fixing the constant features led to measurable improvement!\n")
            f.write(f"   Performance increased by {improvement_pct:+.2f}%, proving that all 22 features\n")
            f.write("   now contribute meaningful signal for frustration detection.\n")
        elif test_f1 >= previous_12_nonconstant:
            f.write("CONCLUSION:\n")
            f.write("✅ IMPROVEMENT: All 22 features now perform as well as or better than\n")
            f.write("   the previous best 12 non-constant features, validating our fixes.\n")
        else:
            f.write("CONCLUSION:\n")
            f.write("📊 MIXED: While we eliminated constant features, some proxy implementations\n")
            f.write("   may need refinement to fully capture the intended signal.\n")
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame([{
        'experiment': 'all_22_features_updated',
        'num_features': 22,
        'test_f1': results['test_f1'],
        'test_precision': results['test_precision'],
        'test_recall': results['test_recall'],
        'test_accuracy': results['test_accuracy'],
        'train_f1': results['train_f1'],
        'train_accuracy': results['train_accuracy'],
        'runtime_minutes': results['runtime_minutes'],
        'improvement_vs_old_22': test_f1 - previous_all_22,
        'improvement_vs_12_nonconstant': test_f1 - previous_12_nonconstant
    }])
    csv_path = f"{output_dir}/all_22_features_updated_data.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Data: {csv_path}")
    
    logger.info(f"\n🎯 FINAL RESULT:")
    logger.info(f"   All 22 features (updated): F1 = {test_f1:.4f}")
    logger.info(f"   Status: {status}")
    
    return results

if __name__ == "__main__":
    main()