#!/usr/bin/env python3
"""
Phase 3b: Feature Selection Algorithms
Test different feature selection methods to optimize the 22-feature set
"""

import json
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
import logging
import sys

sys.path.append('/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase2_features')
from fast_feature_extractor import FastFrustrationFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelectionExperiment:
    """Test different feature selection algorithms with early fusion"""
    
    def __init__(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.roberta.to(self.device)
        self.feature_extractor = FastFrustrationFeatureExtractor()
        
        self.results = []
        
        # Feature names for interpretation
        self.feature_names = [
            'linguistic_bundle_sentiment_slope', 'linguistic_bundle_sentiment_volatility',
            'linguistic_bundle_avg_politeness', 'linguistic_bundle_politeness_decline',
            'linguistic_bundle_avg_confusion', 'linguistic_bundle_avg_negation',
            'linguistic_bundle_avg_caps', 'linguistic_bundle_avg_exclamation',
            'dialogue_bundle_total_turns', 'dialogue_bundle_avg_turn_length',
            'dialogue_bundle_repeated_turns', 'dialogue_bundle_corrections',
            'behavioral_bundle_escalation_requests', 'behavioral_bundle_negative_feedback',
            'contextual_bundle_avg_urgency', 'contextual_bundle_urgency_increase',
            'contextual_bundle_task_complexity', 'emotion_dynamics_bundle_emotion_drift',
            'emotion_dynamics_bundle_emotion_volatility', 'system_bundle_response_clarity',
            'system_bundle_response_relevance', 'user_model_bundle_trust_decline'
        ]
        
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
    
    def extract_all_features(self, texts):
        """Extract all 22 features"""
        all_bundles = [
            'linguistic_bundle', 'dialogue_bundle', 'behavioral_bundle',
            'contextual_bundle', 'emotion_dynamics_bundle', 'system_bundle', 'user_model_bundle'
        ]
        
        all_features = []
        for text in texts:
            text_features = []
            for bundle in all_bundles:
                bundle_features = self.feature_extractor.extract_bundle_features([text], bundle)
                text_features.extend(list(bundle_features.values()))
            all_features.append(text_features)
        
        return np.array(all_features)
    
    def test_early_fusion_with_features(self, data, feature_indices, method_name):
        """Test early fusion with selected features"""
        logger.info(f"Testing {method_name} with {len(feature_indices)} features...")
        
        # Extract all features
        train_features_full = self.extract_all_features(data['train_texts'])
        test_features_full = self.extract_all_features(data['test_texts'])
        
        # Select subset
        train_features = train_features_full[:, feature_indices]
        test_features = test_features_full[:, feature_indices]
        
        # Handle empty selection
        if train_features.shape[1] == 0:
            logger.warning(f"{method_name}: No features selected!")
            return None
        
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
        
        # Train classifier
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_embeddings, data['train_labels'])
        predictions = model.predict(test_embeddings)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            data['test_labels'], predictions, average='binary'
        )
        accuracy = accuracy_score(data['test_labels'], predictions)
        
        selected_features = [self.feature_names[i] for i in feature_indices]
        
        result = {
            'method': method_name,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'num_features': len(feature_indices),
            'selected_features': selected_features[:5]  # Top 5 for display
        }
        
        logger.info(f"{method_name}: F1={f1:.4f}, Features={len(feature_indices)}")
        return result
    
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
    
    def method_1_mutual_information(self, data):
        """Method 1: Mutual Information based selection"""
        logger.info("=== METHOD 1: MUTUAL INFORMATION SELECTION ===")
        
        train_features = self.extract_all_features(data['train_texts'])
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        
        # Remove constant features
        non_constant_mask = np.var(train_features_scaled, axis=0) > 1e-6
        train_features_clean = train_features_scaled[:, non_constant_mask]
        clean_indices = np.where(non_constant_mask)[0]
        
        if train_features_clean.shape[1] == 0:
            logger.warning("No non-constant features found!")
            return []
        
        # Compute mutual information
        mi_scores = mutual_info_classif(train_features_clean, data['train_labels'], random_state=42)
        
        # Test different k values
        mi_results = []
        for k in [3, 5, 8, 10, 12]:
            if k > len(mi_scores):
                k = len(mi_scores)
            
            # Get top k features by MI
            top_k_clean_indices = np.argsort(mi_scores)[-k:]
            top_k_original_indices = clean_indices[top_k_clean_indices]
            
            result = self.test_early_fusion_with_features(
                data, top_k_original_indices, f"MI_top_{k}"
            )
            if result:
                mi_results.append(result)
        
        return mi_results
    
    def method_2_lasso_selection(self, data):
        """Method 2: LASSO regularization based selection"""
        logger.info("=== METHOD 2: LASSO REGULARIZATION SELECTION ===")
        
        train_features = self.extract_all_features(data['train_texts'])
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        
        # Remove constant features
        non_constant_mask = np.var(train_features_scaled, axis=0) > 1e-6
        train_features_clean = train_features_scaled[:, non_constant_mask]
        clean_indices = np.where(non_constant_mask)[0]
        
        if train_features_clean.shape[1] == 0:
            logger.warning("No non-constant features found!")
            return []
        
        # LASSO with cross-validation for alpha selection
        lasso_cv = LassoCV(cv=5, random_state=42, max_iter=2000)
        lasso_cv.fit(train_features_clean, data['train_labels'])
        
        logger.info(f"LASSO optimal alpha: {lasso_cv.alpha_:.6f}")
        
        # Test different alpha multipliers for different sparsity levels
        lasso_results = []
        base_alpha = lasso_cv.alpha_
        
        for multiplier, name in [(0.5, "loose"), (1.0, "optimal"), (2.0, "strict"), (5.0, "very_strict")]:
            alpha = base_alpha * multiplier
            
            # Fit LASSO with this alpha
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
            lasso.fit(train_features_clean, data['train_labels'])
            
            # Get selected features (non-zero coefficients)
            selected_mask = np.abs(lasso.coef_) > 1e-6
            selected_clean_indices = np.where(selected_mask)[0]
            
            if len(selected_clean_indices) > 0:
                selected_original_indices = clean_indices[selected_clean_indices]
                
                result = self.test_early_fusion_with_features(
                    data, selected_original_indices, f"LASSO_{name}"
                )
                if result:
                    result['alpha'] = alpha
                    result['lasso_coefs'] = lasso.coef_[selected_mask].tolist()
                    lasso_results.append(result)
            else:
                logger.info(f"LASSO {name} (α={alpha:.6f}): No features selected")
        
        return lasso_results
    
    def method_3_recursive_elimination(self, data):
        """Method 3: Recursive Feature Elimination"""
        logger.info("=== METHOD 3: RECURSIVE FEATURE ELIMINATION ===")
        
        train_features = self.extract_all_features(data['train_texts'])
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        
        # Remove constant features
        non_constant_mask = np.var(train_features_scaled, axis=0) > 1e-6
        train_features_clean = train_features_scaled[:, non_constant_mask]
        clean_indices = np.where(non_constant_mask)[0]
        
        if train_features_clean.shape[1] == 0:
            logger.warning("No non-constant features found!")
            return []
        
        rfe_results = []
        
        # Test different numbers of features to select
        for n_features in [3, 5, 8, 10]:
            if n_features > train_features_clean.shape[1]:
                n_features = train_features_clean.shape[1]
            
            # Use logistic regression as base estimator for RFE
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            
            try:
                rfe.fit(train_features_clean, data['train_labels'])
                
                # Get selected features
                selected_clean_indices = np.where(rfe.support_)[0]
                selected_original_indices = clean_indices[selected_clean_indices]
                
                result = self.test_early_fusion_with_features(
                    data, selected_original_indices, f"RFE_{n_features}_features"
                )
                if result:
                    result['rfe_ranking'] = rfe.ranking_.tolist()
                    rfe_results.append(result)
                    
            except Exception as e:
                logger.error(f"RFE with {n_features} features failed: {e}")
        
        return rfe_results
    
    def method_4_hybrid_selection(self, data):
        """Method 4: Hybrid approach combining MI + LASSO"""
        logger.info("=== METHOD 4: HYBRID MI + LASSO SELECTION ===")
        
        train_features = self.extract_all_features(data['train_texts'])
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        
        # Remove constant features
        non_constant_mask = np.var(train_features_scaled, axis=0) > 1e-6
        train_features_clean = train_features_scaled[:, non_constant_mask]
        clean_indices = np.where(non_constant_mask)[0]
        
        if train_features_clean.shape[1] == 0:
            return []
        
        # Step 1: Use MI to get top 10 features
        mi_scores = mutual_info_classif(train_features_clean, data['train_labels'], random_state=42)
        top_10_clean_indices = np.argsort(mi_scores)[-10:]
        top_10_features = train_features_clean[:, top_10_clean_indices]
        
        # Step 2: Use LASSO on the top 10 MI features
        lasso_cv = LassoCV(cv=3, random_state=42, max_iter=2000)
        lasso_cv.fit(top_10_features, data['train_labels'])
        
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=lasso_cv.alpha_, random_state=42, max_iter=2000)
        lasso.fit(top_10_features, data['train_labels'])
        
        # Get final selected features
        lasso_selected_mask = np.abs(lasso.coef_) > 1e-6
        final_clean_indices = top_10_clean_indices[lasso_selected_mask]
        final_original_indices = clean_indices[final_clean_indices]
        
        if len(final_original_indices) > 0:
            result = self.test_early_fusion_with_features(
                data, final_original_indices, "Hybrid_MI_LASSO"
            )
            if result:
                result['method_details'] = "Top 10 MI + LASSO refinement"
                return [result]
        
        return []
    
    def run_all_methods(self):
        """Run all feature selection methods"""
        logger.info("=" * 80)
        logger.info("PHASE 3B: FEATURE SELECTION ALGORITHMS")
        logger.info("=" * 80)
        
        # Load data
        data = self.load_data()
        
        # Baseline: All features (from Phase 3a)
        baseline_f1 = 0.8831
        logger.info(f"Baseline (all 22 features): F1 = {baseline_f1:.4f}")
        logger.info("")
        
        # Run all methods
        all_results = []
        
        # Method 1: Mutual Information
        mi_results = self.method_1_mutual_information(data)
        all_results.extend(mi_results)
        
        # Method 2: LASSO
        lasso_results = self.method_2_lasso_selection(data)
        all_results.extend(lasso_results)
        
        # Method 3: RFE
        rfe_results = self.method_3_recursive_elimination(data)
        all_results.extend(rfe_results)
        
        # Method 4: Hybrid
        hybrid_results = self.method_4_hybrid_selection(data)
        all_results.extend(hybrid_results)
        
        self.results = all_results
        
        # Analysis
        self.print_summary(baseline_f1)
        return all_results
    
    def print_summary(self, baseline_f1):
        """Print comprehensive summary"""
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE SELECTION SUMMARY")
        logger.info("=" * 80)
        
        if not self.results:
            logger.info("No results to summarize")
            return
        
        # Sort by F1 score
        sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
        
        logger.info(f"Baseline (22 features): F1 = {baseline_f1:.4f}")
        logger.info("")
        
        best_result = sorted_results[0]
        logger.info("TOP RESULTS:")
        for i, result in enumerate(sorted_results[:10]):
            improvement = result['f1_score'] - baseline_f1
            status = "🏆" if improvement > 0.005 else "✅" if improvement > 0 else "❌"
            
            logger.info(f"{status} {result['method']:<20} | F1: {result['f1_score']:.4f} "
                       f"(Δ={improvement:+.4f}) | Features: {result['num_features']}")
        
        # Feature efficiency analysis
        logger.info(f"\nFEATURE EFFICIENCY:")
        for result in sorted_results[:5]:
            efficiency = result['f1_score'] / result['num_features']  # F1 per feature
            logger.info(f"{result['method']:<20} | Efficiency: {efficiency:.4f} F1/feature")
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/feature_selection_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to {results_path}")
        
        # Final recommendation
        logger.info(f"\n🎯 BEST METHOD: {best_result['method']}")
        logger.info(f"F1 Score: {best_result['f1_score']:.4f}")
        logger.info(f"Features: {best_result['num_features']}")
        if best_result['f1_score'] > baseline_f1:
            logger.info(f"🎉 IMPROVEMENT: +{best_result['f1_score'] - baseline_f1:.4f} over baseline!")
        else:
            logger.info(f"📊 No improvement over baseline, but {best_result['num_features']} features achieve similar performance")

def main():
    """Run feature selection experiment"""
    experiment = FeatureSelectionExperiment()
    
    try:
        results = experiment.run_all_methods()
        return results
        
    except Exception as e:
        logger.error(f"Feature selection experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()