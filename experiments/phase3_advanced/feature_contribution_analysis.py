#!/usr/bin/env python3
"""
Feature Contribution Analysis
Analyze which specific features contributed to the early fusion enhancement
"""

import json
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging
import sys

sys.path.append('/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase2_features')
from fast_feature_extractor import FastFrustrationFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureContributionAnalyzer:
    """Analyze which features contributed to the early fusion improvement"""
    
    def __init__(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.roberta.to(self.device)
        self.feature_extractor = FastFrustrationFeatureExtractor()
        
        # Feature names mapping
        self.feature_names = {
            'linguistic_bundle': [
                'sentiment_slope', 'sentiment_volatility', 'avg_politeness', 
                'politeness_decline', 'avg_confusion', 'avg_negation', 
                'avg_caps', 'avg_exclamation'
            ],
            'dialogue_bundle': [
                'total_turns', 'avg_turn_length', 'repeated_turns', 'corrections'
            ],
            'behavioral_bundle': [
                'escalation_requests', 'negative_feedback'
            ],
            'contextual_bundle': [
                'avg_urgency', 'urgency_increase', 'task_complexity'
            ],
            'emotion_dynamics_bundle': [
                'emotion_drift', 'emotion_volatility'
            ],
            'system_bundle': [
                'response_clarity', 'response_relevance'
            ],
            'user_model_bundle': [
                'trust_decline'
            ]
        }
        
        self.results = []
        
    def load_data(self):
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
    
    def extract_all_features_with_names(self, texts):
        """Extract all features and return with names"""
        all_bundles = [
            'linguistic_bundle', 'dialogue_bundle', 'behavioral_bundle',
            'contextual_bundle', 'emotion_dynamics_bundle', 'system_bundle', 'user_model_bundle'
        ]
        
        all_features = []
        feature_names_flat = []
        
        for text in texts:
            text_features = []
            for bundle in all_bundles:
                bundle_features = self.feature_extractor.extract_bundle_features([text], bundle)
                text_features.extend(list(bundle_features.values()))
                
                # Add feature names for first text only
                if len(feature_names_flat) < 22:
                    for feat_name in self.feature_names[bundle]:
                        feature_names_flat.append(f"{bundle}_{feat_name}")
            
            all_features.append(text_features)
        
        return np.array(all_features), feature_names_flat
    
    def test_early_fusion_subset(self, data, feature_indices, feature_names, test_name):
        """Test early fusion with a subset of features"""
        # Extract all features first
        train_features_full, _ = self.extract_all_features_with_names(data['train_texts'])
        test_features_full, _ = self.extract_all_features_with_names(data['test_texts'])
        
        # Select subset
        train_features = train_features_full[:, feature_indices]
        test_features = test_features_full[:, feature_indices]
        
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
        train_texts_enhanced = [
            f"{feat_tokens} {text}" 
            for feat_tokens, text in zip(train_feat_tokens, data['train_texts'])
        ]
        test_texts_enhanced = [
            f"{feat_tokens} {text}" 
            for feat_tokens, text in zip(test_feat_tokens, data['test_texts'])
        ]
        
        # Get embeddings
        train_embeddings = self.get_roberta_embeddings(train_texts_enhanced)
        test_embeddings = self.get_roberta_embeddings(test_texts_enhanced)
        
        # Train and predict
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_embeddings, data['train_labels'])
        predictions = model.predict(test_embeddings)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            data['test_labels'], predictions, average='binary'
        )
        
        return {
            'test_name': test_name,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'num_features': len(feature_indices),
            'features_used': [feature_names[i] for i in feature_indices]
        }
    
    def get_roberta_embeddings(self, texts):
        """Extract RoBERTa embeddings efficiently"""
        embeddings = []
        batch_size = 16
        
        self.roberta.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts, truncation=True, padding="max_length", 
                    max_length=512, return_tensors="pt"
                ).to(self.device)
                
                outputs = self.roberta(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def analyze_feature_importance(self, data):
        """Analyze feature importance using statistical methods"""
        logger.info("=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Extract all features
        train_features, feature_names = self.extract_all_features_with_names(data['train_texts'])
        
        # Scale features for analysis
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        
        # Method 1: F-statistic (ANOVA)
        logger.info("Computing F-statistics...")
        f_scores, f_pvalues = f_classif(train_features_scaled, data['train_labels'])
        
        # Method 2: Mutual Information
        logger.info("Computing mutual information...")
        mi_scores = mutual_info_classif(train_features_scaled, data['train_labels'], random_state=42)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature_name': feature_names,
            'f_score': f_scores,
            'f_pvalue': f_pvalues,
            'mutual_info': mi_scores,
        })
        
        # Add bundle information
        importance_df['bundle'] = importance_df['feature_name'].apply(
            lambda x: x.split('_')[0] + '_' + x.split('_')[1]
        )
        
        # Sort by mutual information (more robust)
        importance_df = importance_df.sort_values('mutual_info', ascending=False)
        
        logger.info("\nTOP 10 MOST IMPORTANT FEATURES (by Mutual Information):")
        logger.info("=" * 80)
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"{row['feature_name']:<35} | MI: {row['mutual_info']:.4f} | F: {row['f_score']:.2f}")
        
        return importance_df
    
    def test_incremental_features(self, data, importance_df):
        """Test performance with incrementally adding features by importance"""
        logger.info("\n=== INCREMENTAL FEATURE TESTING ===")
        
        baseline_f1 = 0.8571  # Linguistic-only early fusion
        all_bundles_f1 = 0.8831  # All bundles early fusion
        
        # Test with top k features
        feature_names = importance_df['feature_name'].tolist()
        incremental_results = []
        
        for k in [5, 10, 15, 20, 22]:  # Test different numbers of top features
            if k > len(feature_names):
                k = len(feature_names)
            
            top_features = feature_names[:k]
            feature_indices = [i for i, name in enumerate(importance_df['feature_name'].tolist()) if name in top_features]
            
            logger.info(f"Testing top {k} features...")
            result = self.test_early_fusion_subset(
                data, feature_indices, importance_df['feature_name'].tolist(), 
                f"top_{k}_features"
            )
            
            improvement_vs_linguistic = result['f1_score'] - baseline_f1
            improvement_vs_baseline = result['f1_score'] - baseline_f1
            
            logger.info(f"Top {k} features: F1 = {result['f1_score']:.4f} (Δ={improvement_vs_linguistic:+.4f})")
            
            incremental_results.append({
                'num_features': k,
                'f1_score': result['f1_score'],
                'improvement_vs_linguistic': improvement_vs_linguistic,
                'top_features': top_features[:5]  # Show top 5 for reference
            })
        
        return incremental_results
    
    def test_bundle_combinations(self, data, importance_df):
        """Test different bundle combinations"""
        logger.info("\n=== BUNDLE COMBINATION TESTING ===")
        
        # Get feature indices by bundle
        bundle_indices = {}
        feature_names = importance_df['feature_name'].tolist()
        
        for bundle in self.feature_names.keys():
            bundle_indices[bundle] = []
            for i, feat_name in enumerate(feature_names):
                if feat_name.startswith(bundle):
                    bundle_indices[bundle].append(i)
        
        baseline_f1 = 0.8571
        
        # Test effective bundles from Phase 2
        effective_bundles = ['linguistic_bundle', 'system_bundle', 'user_model_bundle']
        effective_indices = []
        for bundle in effective_bundles:
            effective_indices.extend(bundle_indices[bundle])
        
        logger.info("Testing effective bundles (linguistic + system + user_model)...")
        result = self.test_early_fusion_subset(
            data, effective_indices, feature_names, "effective_bundles"
        )
        improvement = result['f1_score'] - baseline_f1
        logger.info(f"Effective bundles: F1 = {result['f1_score']:.4f} (Δ={improvement:+.4f})")
        
        # Test adding dialogue bundle (was harmful in late fusion)
        dialogue_enhanced = effective_indices + bundle_indices['dialogue_bundle']
        logger.info("Testing effective + dialogue bundle...")
        result = self.test_early_fusion_subset(
            data, dialogue_enhanced, feature_names, "effective_plus_dialogue"
        )
        improvement = result['f1_score'] - baseline_f1
        logger.info(f"Effective + dialogue: F1 = {result['f1_score']:.4f} (Δ={improvement:+.4f})")
        
        return {
            'effective_bundles': effective_indices,
            'dialogue_enhanced': dialogue_enhanced
        }
    
    def run_full_analysis(self):
        """Run complete feature contribution analysis"""
        logger.info("=" * 80)
        logger.info("FEATURE CONTRIBUTION ANALYSIS")
        logger.info("=" * 80)
        
        # Load data
        data = self.load_data()
        
        # 1. Statistical importance analysis
        importance_df = self.analyze_feature_importance(data)
        
        # 2. Incremental feature testing
        incremental_results = self.test_incremental_features(data, importance_df)
        
        # 3. Bundle combination testing
        bundle_results = self.test_bundle_combinations(data, importance_df)
        
        # 4. Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        
        logger.info("BASELINES:")
        logger.info(f"  Linguistic-only early fusion: F1 = 0.8571")
        logger.info(f"  All bundles early fusion: F1 = 0.8831 (+0.0260)")
        logger.info("")
        
        logger.info("KEY FINDINGS:")
        top_5_features = importance_df.head(5)['feature_name'].tolist()
        logger.info(f"  Top 5 most important features: {top_5_features}")
        
        best_incremental = max(incremental_results, key=lambda x: x['f1_score'])
        logger.info(f"  Best incremental result: {best_incremental['num_features']} features, F1 = {best_incremental['f1_score']:.4f}")
        
        # Save results
        importance_df.to_csv("/Users/omarhammad/Documents/code_local/frustration_researcher/results/feature_importance_analysis.csv", index=False)
        
        incremental_df = pd.DataFrame(incremental_results)
        incremental_df.to_csv("/Users/omarhammad/Documents/code_local/frustration_researcher/results/incremental_feature_results.csv", index=False)
        
        logger.info("\nResults saved to CSV files")
        
        return importance_df, incremental_results

def main():
    analyzer = FeatureContributionAnalyzer()
    importance_df, incremental_results = analyzer.run_full_analysis()
    return importance_df, incremental_results

if __name__ == "__main__":
    main()