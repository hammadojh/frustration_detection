#!/usr/bin/env python3
"""
Test Early Fusion with All Feature Bundles
Extend the successful early fusion approach to include all feature bundles
"""

import json
import torch
import numpy as np
import pandas as pd
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

class EarlyFusionAllBundlesExperiment:
    """Test early fusion with different combinations of feature bundles"""
    
    def __init__(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.roberta.to(self.device)
        self.feature_extractor = FastFrustrationFeatureExtractor()
        
        self.all_bundles = [
            'linguistic_bundle',      # 8 features - KNOWN TO WORK
            'dialogue_bundle',        # 4 features - Previously harmful
            'behavioral_bundle',      # 2 features - Previously neutral
            'contextual_bundle',      # 3 features - Previously harmful  
            'emotion_dynamics_bundle', # 2 features - Previously neutral
            'system_bundle',          # 2 features - Previously effective
            'user_model_bundle'       # 1 feature - Previously effective
        ]
        
        self.results = []
        
    def load_data(self):
        """Load processed examples"""
        processed_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/data/subset_processed.json"
        with open(processed_path, 'r') as f:
            examples = json.load(f)
        
        texts = [ex['text'] for ex in examples]
        labels = [ex['label'] for ex in examples]
        
        # Create splits
        n = len(texts)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        return {
            'train_texts': texts[:train_end],
            'train_labels': labels[:train_end],
            'test_texts': texts[val_end:],
            'test_labels': labels[val_end:]
        }
    
    def extract_bundle_features(self, texts, bundles):
        """Extract features for specified bundles"""
        all_features = []
        
        for text in texts:
            text_features = []
            for bundle in bundles:
                bundle_features = self.feature_extractor.extract_bundle_features([text], bundle)
                text_features.extend(list(bundle_features.values()))
            all_features.append(text_features)
        
        return np.array(all_features)
    
    def features_to_tokens(self, features_scaled, feature_names=None):
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
    
    def get_roberta_embeddings(self, texts):
        """Extract RoBERTa [CLS] embeddings"""
        embeddings = []
        batch_size = 16
        
        logger.info(f"Processing {len(texts)} texts...")
        
        self.roberta.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.roberta(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def test_early_fusion_bundles(self, data, bundles, experiment_name):
        """Test early fusion with specified bundles"""
        logger.info(f"Testing {experiment_name} with bundles: {bundles}")
        
        # Extract features for specified bundles
        train_features = self.extract_bundle_features(data['train_texts'], bundles)
        test_features = self.extract_bundle_features(data['test_texts'], bundles)
        
        logger.info(f"Extracted {train_features.shape[1]} features")
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Convert features to tokens
        train_feat_tokens = self.features_to_tokens(train_features_scaled)
        test_feat_tokens = self.features_to_tokens(test_features_scaled)
        
        # Create enhanced texts with feature tokens prepended
        train_texts_enhanced = [
            f"{feat_tokens} {text}" 
            for feat_tokens, text in zip(train_feat_tokens, data['train_texts'])
        ]
        test_texts_enhanced = [
            f"{feat_tokens} {text}" 
            for feat_tokens, text in zip(test_feat_tokens, data['test_texts'])
        ]
        
        # Get embeddings from enhanced text
        train_embeddings = self.get_roberta_embeddings(train_texts_enhanced)
        test_embeddings = self.get_roberta_embeddings(test_texts_enhanced)
        
        # Train classifier
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_embeddings, data['train_labels'])
        
        # Predict
        predictions = model.predict(test_embeddings)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            data['test_labels'], predictions, average='binary'
        )
        accuracy = accuracy_score(data['test_labels'], predictions)
        
        result = {
            'experiment_name': experiment_name,
            'bundles': ','.join(bundles),
            'num_bundles': len(bundles),
            'num_features': train_features.shape[1],
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }
        
        logger.info(f"Results: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
        return result
    
    def run_systematic_tests(self):
        """Run systematic tests with different bundle combinations"""
        logger.info("="*60)
        logger.info("EARLY FUSION: ALL BUNDLES EXPERIMENT")
        logger.info("="*60)
        
        data = self.load_data()
        baseline_f1 = 0.8571  # Early fusion with linguistic bundle
        
        logger.info(f"Current best (linguistic only): F1 = {baseline_f1:.4f}")
        logger.info("")
        
        # Test 1: Individual bundles with early fusion
        logger.info("--- TESTING INDIVIDUAL BUNDLES ---")
        for bundle in self.all_bundles:
            result = self.test_early_fusion_bundles(data, [bundle], f"early_fusion_{bundle}")
            self.results.append(result)
            
            improvement = result['f1_score'] - baseline_f1
            status = "🏆" if improvement > 0.01 else "✅" if improvement > 0 else "❌"
            logger.info(f"{status} {bundle}: F1 = {result['f1_score']:.4f} (Δ={improvement:+.4f})")
        
        # Test 2: Effective bundles combinations (linguistic + system + user_model)
        logger.info("\n--- TESTING EFFECTIVE BUNDLES COMBINATION ---")
        effective_bundles = ['linguistic_bundle', 'system_bundle', 'user_model_bundle']
        result = self.test_early_fusion_bundles(data, effective_bundles, "early_fusion_effective_3")
        self.results.append(result)
        
        improvement = result['f1_score'] - baseline_f1
        status = "🏆" if improvement > 0.01 else "✅" if improvement > 0 else "❌"
        logger.info(f"{status} effective_3_bundles: F1 = {result['f1_score']:.4f} (Δ={improvement:+.4f})")
        
        # Test 3: All bundles
        logger.info("\n--- TESTING ALL BUNDLES ---")
        result = self.test_early_fusion_bundles(data, self.all_bundles, "early_fusion_all_bundles")
        self.results.append(result)
        
        improvement = result['f1_score'] - baseline_f1
        status = "🏆" if improvement > 0.01 else "✅" if improvement > 0 else "❌"
        logger.info(f"{status} all_bundles: F1 = {result['f1_score']:.4f} (Δ={improvement:+.4f})")
        
        # Test 4: Best individual + linguistic (additive testing)
        logger.info("\n--- TESTING ADDITIVE COMBINATIONS ---")
        
        # Find best individual bundle (excluding linguistic)
        individual_results = [r for r in self.results if r['num_bundles'] == 1 and 'linguistic' not in r['bundles']]
        if individual_results:
            best_individual = max(individual_results, key=lambda x: x['f1_score'])
            best_bundle = best_individual['bundles']
            
            logger.info(f"Best individual non-linguistic bundle: {best_bundle} (F1={best_individual['f1_score']:.4f})")
            
            # Test linguistic + best individual
            combo_bundles = ['linguistic_bundle', best_bundle]
            result = self.test_early_fusion_bundles(data, combo_bundles, f"early_fusion_linguistic_plus_{best_bundle}")
            self.results.append(result)
            
            improvement = result['f1_score'] - baseline_f1
            status = "🏆" if improvement > 0.01 else "✅" if improvement > 0 else "❌"
            logger.info(f"{status} linguistic+{best_bundle}: F1 = {result['f1_score']:.4f} (Δ={improvement:+.4f})")
        
        # Summary
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print experiment summary"""
        logger.info("\n" + "="*70)
        logger.info("EARLY FUSION ALL BUNDLES SUMMARY")
        logger.info("="*70)
        
        # Sort results by F1 score
        sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
        
        baseline_f1 = 0.8571
        logger.info(f"Baseline (linguistic only): F1 = {baseline_f1:.4f}")
        logger.info("")
        
        for result in sorted_results:
            improvement = result['f1_score'] - baseline_f1
            status = "🏆" if improvement > 0.01 else "✅" if improvement > 0 else "❌"
            
            logger.info(f"{status} {result['experiment_name']}: F1 = {result['f1_score']:.4f} "
                       f"(Δ={improvement:+.4f}) [{result['num_features']} features]")
        
        # Save results
        results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/early_fusion_all_bundles_results.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to {results_path}")
        
        # Find best result
        best_result = sorted_results[0]
        logger.info(f"\n🎉 BEST EARLY FUSION COMBINATION: {best_result['experiment_name']}")
        logger.info(f"F1 Score: {best_result['f1_score']:.4f}")
        logger.info(f"Bundles: {best_result['bundles']}")
        logger.info(f"Features: {best_result['num_features']}")

def main():
    """Run early fusion all bundles experiment"""
    experiment = EarlyFusionAllBundlesExperiment()
    
    try:
        results = experiment.run_systematic_tests()
        return results
        
    except Exception as e:
        logger.error(f"Early fusion all bundles experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()