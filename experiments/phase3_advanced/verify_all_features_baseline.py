#!/usr/bin/env python3
"""
Verify All 22 Features Early Fusion Baseline
Confirm the F1=0.8831 result was not a measurement error
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
    """Extract ALL 22 features"""
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

def test_all_features_early_fusion_multiple_runs(data, num_runs=3):
    """Test all 22 features with early fusion multiple times to ensure consistency"""
    
    logger.info("=" * 80)
    logger.info("VERIFYING ALL 22 FEATURES EARLY FUSION BASELINE")
    logger.info("=" * 80)
    logger.info(f"Running {num_runs} independent trials to verify consistency")
    logger.info("")
    
    # Initialize models
    tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
    roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roberta.to(device)
    roberta.eval()
    
    results = []
    
    for run in range(num_runs):
        logger.info(f"=== RUN {run+1}/{num_runs} ===")
        start_time = time.time()
        
        # Extract ALL features
        logger.info("Extracting all 22 features...")
        train_features = extract_all_features(data['train_texts'])
        test_features = extract_all_features(data['test_texts'])
        
        logger.info(f"Extracted features shape: {train_features.shape}")
        logger.info(f"Feature statistics - Mean: {train_features.mean():.4f}, Std: {train_features.std():.4f}")
        
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
        
        logger.info("Converting features to tokens...")
        train_feat_tokens = features_to_tokens(train_features_scaled)
        test_feat_tokens = features_to_tokens(test_features_scaled)
        
        # Show example
        logger.info(f"Example feature tokens (first 100 chars): {train_feat_tokens[0][:100]}...")
        
        # Create enhanced texts
        train_texts_enhanced = [f"{feat_tokens} {text}" for feat_tokens, text in zip(train_feat_tokens, data['train_texts'])]
        test_texts_enhanced = [f"{feat_tokens} {text}" for feat_tokens, text in zip(test_feat_tokens, data['test_texts'])]
        
        logger.info(f"Example enhanced text (first 200 chars): {train_texts_enhanced[0][:200]}...")
        
        # Get embeddings
        logger.info("Extracting RoBERTa embeddings...")
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
        
        logger.info(f"Embeddings shape: {train_embeddings.shape}")
        
        # Train classifier
        logger.info("Training classifier...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_embeddings, data['train_labels'])
        
        # Predictions
        logger.info("Making predictions...")
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
        
        run_time = time.time() - start_time
        
        result = {
            'run': run + 1,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_accuracy': train_accuracy,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_accuracy': test_accuracy,
            'num_features': train_features.shape[1],
            'run_time_minutes': run_time / 60
        }
        
        results.append(result)
        
        logger.info(f"Run {run+1} Results:")
        logger.info(f"  Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
        logger.info(f"  Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
        logger.info(f"  Runtime: {run_time/60:.1f} minutes")
        logger.info("")
    
    # Aggregate analysis
    logger.info("=" * 80)
    logger.info("VERIFICATION RESULTS SUMMARY")
    logger.info("=" * 80)
    
    test_f1_scores = [r['test_f1'] for r in results]
    test_f1_mean = np.mean(test_f1_scores)
    test_f1_std = np.std(test_f1_scores)
    test_f1_min = np.min(test_f1_scores)
    test_f1_max = np.max(test_f1_scores)
    
    logger.info(f"ALL 22 FEATURES EARLY FUSION VERIFICATION:")
    logger.info(f"  Mean Test F1: {test_f1_mean:.4f} ± {test_f1_std:.4f}")
    logger.info(f"  Range: {test_f1_min:.4f} - {test_f1_max:.4f}")
    logger.info(f"  Individual runs: {[f'{f1:.4f}' for f1 in test_f1_scores]}")
    logger.info("")
    
    # Compare to previous result
    previous_result = 0.8831
    logger.info(f"COMPARISON TO PREVIOUS RESULT:")
    logger.info(f"  Previous: F1 = {previous_result:.4f}")
    logger.info(f"  Current:  F1 = {test_f1_mean:.4f} ± {test_f1_std:.4f}")
    logger.info(f"  Difference: {test_f1_mean - previous_result:+.4f}")
    
    if abs(test_f1_mean - previous_result) < 0.01:
        status = "✅ CONFIRMED"
        logger.info(f"  Status: {status} - Results are consistent")
    elif test_f1_mean > previous_result:
        status = "📈 HIGHER"
        logger.info(f"  Status: {status} - New results are higher")
    else:
        status = "📉 LOWER"
        logger.info(f"  Status: {status} - New results are lower")
    
    # Save detailed results
    output_dir = "/Users/omarhammad/Documents/code_local/frustration_researcher/results"
    
    # Save verification report
    report_path = f"{output_dir}/all_22_features_verification_RESULTS.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ALL 22 FEATURES EARLY FUSION - VERIFICATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Verification completed with {num_runs} independent runs\n")
        f.write(f"Total experiment time: {sum(r['run_time_minutes'] for r in results):.1f} minutes\n")
        f.write("\n")
        
        f.write("DETAILED RESULTS BY RUN:\n")
        f.write("-" * 80 + "\n")
        for result in results:
            f.write(f"\nRUN {result['run']}:\n")
            f.write(f"  Test F1: {result['test_f1']:.4f}\n")
            f.write(f"  Test Precision: {result['test_precision']:.4f}\n")
            f.write(f"  Test Recall: {result['test_recall']:.4f}\n")
            f.write(f"  Test Accuracy: {result['test_accuracy']:.4f}\n")
            f.write(f"  Train F1: {result['train_f1']:.4f}\n")
            f.write(f"  Features Used: {result['num_features']}\n")
            f.write(f"  Runtime: {result['run_time_minutes']:.1f} minutes\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICAL SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test F1 Statistics ({num_runs} runs):\n")
        f.write(f"  Mean: {test_f1_mean:.4f}\n")
        f.write(f"  Std Dev: {test_f1_std:.4f}\n")
        f.write(f"  Min: {test_f1_min:.4f}\n")
        f.write(f"  Max: {test_f1_max:.4f}\n")
        f.write(f"  Range: {test_f1_max - test_f1_min:.4f}\n")
        f.write("\n")
        
        f.write("COMPARISON TO PREVIOUS RESULT:\n")
        f.write(f"  Previous Reported: F1 = {previous_result:.4f}\n")
        f.write(f"  Current Verified:  F1 = {test_f1_mean:.4f} ± {test_f1_std:.4f}\n")
        f.write(f"  Absolute Difference: {abs(test_f1_mean - previous_result):.4f}\n")
        f.write(f"  Relative Difference: {((test_f1_mean - previous_result) / previous_result * 100):+.2f}%\n")
        f.write(f"  Verification Status: {status}\n")
        f.write("\n")
        
        f.write("CONCLUSION:\n")
        if abs(test_f1_mean - previous_result) < 0.01:
            f.write("✅ The previous result of F1=0.8831 is CONFIRMED.\n")
            f.write("   The systematic feature selection comparison is valid.\n")
        elif test_f1_mean > previous_result + 0.01:
            f.write("📈 The new results are HIGHER than previously reported.\n")
            f.write("   This suggests potential measurement variability.\n")
        else:
            f.write("📉 The new results are LOWER than previously reported.\n")
            f.write("   The previous F1=0.8831 may have been optimistic.\n")
            f.write("   Our systematic feature selection conclusions remain valid.\n")
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = f"{output_dir}/all_22_features_verification_data.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Data: {csv_path}")
    
    logger.info(f"\n🎯 FINAL VERIFICATION CONCLUSION:")
    logger.info(f"   All 22 features early fusion: F1 = {test_f1_mean:.4f} ± {test_f1_std:.4f}")
    logger.info(f"   Verification status: {status}")
    
    return results, test_f1_mean

def main():
    """Run verification experiment"""
    
    # Load data
    data = load_data()
    
    # Run verification
    results, verified_f1 = test_all_features_early_fusion_multiple_runs(data, num_runs=3)
    
    return results, verified_f1

if __name__ == "__main__":
    main()