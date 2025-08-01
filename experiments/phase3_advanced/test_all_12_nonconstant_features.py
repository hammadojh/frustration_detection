#!/usr/bin/env python3
"""
Test All 12 Non-Constant Features Combined in Early Fusion
Fill the gap between 6 features and 22 features
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

def test_feature_combination(data, feature_indices, feature_names, combo_name):
    """Test early fusion with specific feature combination"""
    logger.info(f"Testing {combo_name} with {len(feature_indices)} features...")
    logger.info(f"Features: {feature_names}")
    
    start_time = time.time()
    
    # Initialize models
    tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
    roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roberta.to(device)
    roberta.eval()
    
    # Extract selected features
    train_features = extract_selected_features(data['train_texts'], feature_indices)
    test_features = extract_selected_features(data['test_texts'], feature_indices)
    
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
    
    result = {
        'combination': combo_name,
        'feature_indices': list(feature_indices),
        'feature_names': feature_names,
        'num_features': len(feature_indices),
        'train_f1': train_f1,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_accuracy': train_accuracy,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_accuracy': test_accuracy,
        'runtime_minutes': runtime / 60
    }
    
    logger.info(f"Results: Test F1={test_f1:.4f}, Train F1={train_f1:.4f}")
    logger.info(f"Runtime: {runtime/60:.1f} minutes")
    
    return result

def main():
    logger.info("=" * 80)
    logger.info("TESTING ALL 12 NON-CONSTANT FEATURES IN EARLY FUSION")
    logger.info("=" * 80)
    
    # Load data
    data = load_data()
    
    # Define the 12 non-constant features from our previous analysis
    # These are the features that showed non-zero variance
    nonconstant_feature_indices = [2, 4, 5, 7, 9, 11, 12, 14, 16, 19, 20, 21]
    nonconstant_feature_names = [
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
    
    logger.info(f"Testing combination of {len(nonconstant_feature_indices)} non-constant features")
    logger.info("")
    
    # Test the combination
    result = test_feature_combination(
        data, 
        nonconstant_feature_indices, 
        nonconstant_feature_names,
        "all_12_nonconstant_features"
    )
    
    # Comparison analysis
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON ANALYSIS")
    logger.info("=" * 80)
    
    # Reference results
    baseline_all_22 = 0.8831
    baseline_linguistic_8 = 0.8571
    best_2_features = 0.8533
    best_6_features = 0.8462
    
    test_f1 = result['test_f1']
    
    logger.info("PERFORMANCE COMPARISON:")
    logger.info(f"  All 22 features:           F1 = {baseline_all_22:.4f}")
    logger.info(f"  All 12 non-constant:       F1 = {test_f1:.4f} (Δ={test_f1-baseline_all_22:+.4f})")
    logger.info(f"  Linguistic 8 features:     F1 = {baseline_linguistic_8:.4f} (Δ={test_f1-baseline_linguistic_8:+.4f})")
    logger.info(f"  Best 6 features:           F1 = {best_6_features:.4f} (Δ={test_f1-best_6_features:+.4f})")
    logger.info(f"  Best 2 features:           F1 = {best_2_features:.4f} (Δ={test_f1-best_2_features:+.4f})")
    logger.info("")
    
    # Feature reduction analysis
    reduction_vs_22 = (22 - 12) / 22 * 100
    reduction_vs_8 = (8 - 12) / 8 * 100  # Negative because we're using more
    
    logger.info("FEATURE EFFICIENCY:")
    logger.info(f"  Features used: 12 out of 22 ({reduction_vs_22:.1f}% reduction)")
    logger.info(f"  F1/feature ratio: {test_f1/12:.4f}")
    logger.info(f"  Performance retention: {test_f1/baseline_all_22*100:.1f}% of full performance")
    logger.info("")
    
    # Status determination
    if test_f1 >= baseline_all_22:
        status = "🏆 EQUALS/BEATS ALL FEATURES"
    elif test_f1 >= baseline_all_22 - 0.01:
        status = "🔥 VERY CLOSE TO ALL FEATURES"
    elif test_f1 >= baseline_linguistic_8:
        status = "✅ BEATS LINGUISTIC BASELINE"
    elif test_f1 >= best_2_features:
        status = "✅ BEATS BEST 2 FEATURES"
    else:
        status = "❌ BELOW BEST 2 FEATURES"
    
    logger.info(f"RESULT STATUS: {status}")
    
    # Key insights
    logger.info("\nKEY INSIGHTS:")
    if test_f1 >= baseline_all_22 - 0.005:
        logger.info("💡 The 10 constant features add very little value!")
        logger.info("💡 We can achieve full performance with just the informative features!")
    elif test_f1 > best_6_features:
        logger.info("💡 There's value in using more than 6 features!")
        logger.info("💡 The 6-12 feature range contains additional signal!")
    elif test_f1 <= best_2_features:
        logger.info("💡 Confirms that more features don't always help!")
        logger.info("💡 The 2-feature combination remains optimal!")
    
    # Save detailed results
    output_dir = "/Users/omarhammad/Documents/code_local/frustration_researcher/results"
    
    # Save report
    report_path = f"{output_dir}/all_12_nonconstant_features_RESULTS.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ALL 12 NON-CONSTANT FEATURES EARLY FUSION TEST\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment completed in {result['runtime_minutes']:.1f} minutes\n")
        f.write("\n")
        
        f.write("FEATURE COMBINATION TESTED:\n")
        f.write(f"  Number of features: {result['num_features']}\n")
        f.write(f"  Feature indices: {result['feature_indices']}\n")
        f.write("  Feature names:\n")
        for i, name in enumerate(result['feature_names']):
            f.write(f"    {i+1:2d}. {name}\n")
        f.write("\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write(f"  Test F1:        {result['test_f1']:.4f}\n")
        f.write(f"  Test Precision: {result['test_precision']:.4f}\n")
        f.write(f"  Test Recall:    {result['test_recall']:.4f}\n")
        f.write(f"  Test Accuracy:  {result['test_accuracy']:.4f}\n")
        f.write(f"  Train F1:       {result['train_f1']:.4f}\n")
        f.write(f"  Train Accuracy: {result['train_accuracy']:.4f}\n")
        f.write("\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write(f"  vs All 22 features:     {test_f1-baseline_all_22:+.4f} ({((test_f1/baseline_all_22-1)*100):+.2f}%)\n")
        f.write(f"  vs Linguistic 8:        {test_f1-baseline_linguistic_8:+.4f} ({((test_f1/baseline_linguistic_8-1)*100):+.2f}%)\n")
        f.write(f"  vs Best 6 features:     {test_f1-best_6_features:+.4f} ({((test_f1/best_6_features-1)*100):+.2f}%)\n")
        f.write(f"  vs Best 2 features:     {test_f1-best_2_features:+.4f} ({((test_f1/best_2_features-1)*100):+.2f}%)\n")
        f.write("\n")
        
        f.write("EFFICIENCY METRICS:\n")
        f.write(f"  Feature reduction vs 22: {reduction_vs_22:.1f}%\n")
        f.write(f"  F1 per feature:          {test_f1/12:.4f}\n")
        f.write(f"  Performance retention:   {test_f1/baseline_all_22*100:.1f}%\n")
        f.write("\n")
        
        f.write(f"FINAL STATUS: {status}\n")
        f.write("\n")
        
        f.write("CONCLUSION:\n")
        if test_f1 >= baseline_all_22 - 0.005:
            f.write("✅ SUCCESS: The 12 non-constant features achieve virtually the same\n")
            f.write("   performance as all 22 features. The 10 constant features add no value.\n")
        elif test_f1 > best_6_features + 0.01:
            f.write("✅ IMPROVEMENT: Using 12 features provides meaningful improvement\n")
            f.write("   over the previous best 6-feature combination.\n")
        elif test_f1 <= best_2_features:
            f.write("📊 INSIGHT: Even with 12 informative features, the performance\n")
            f.write("   doesn't exceed the optimal 2-feature combination, confirming\n")
            f.write("   that feature selection quality matters more than quantity.\n")
        else:
            f.write("📊 MODERATE: The 12 non-constant features perform between the\n")
            f.write("   2-feature optimum and the full 22-feature baseline.\n")
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame([result])
    csv_path = f"{output_dir}/all_12_nonconstant_features_data.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Data: {csv_path}")
    
    logger.info(f"\n🎯 FINAL RESULT:")
    logger.info(f"   12 non-constant features: F1 = {test_f1:.4f}")
    logger.info(f"   Status: {status}")
    
    return result

if __name__ == "__main__":
    main()