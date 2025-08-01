#!/usr/bin/env python3
"""
Phase 2: Combined Text + Features Experiment
Test features APPENDED to RoBERTa embeddings, not replacing them
"""

import json
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from fast_feature_extractor import FastFrustrationFeatureExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CombinedTextFeatureExperiment:
    """Test RoBERTa embeddings + engineered features together"""
    
    def __init__(self):
        self.feature_extractor = FastFrustrationFeatureExtractor()
        self.tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        
        self.feature_bundles = [
            'linguistic_bundle',
            'dialogue_bundle', 
            'behavioral_bundle',
            'contextual_bundle',
            'emotion_dynamics_bundle',
            'system_bundle',
            'user_model_bundle'
        ]
        self.baseline_f1 = 0.8108
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
            'val_texts': texts[train_end:val_end],
            'val_labels': labels[train_end:val_end],
            'test_texts': texts[val_end:],
            'test_labels': labels[val_end:]
        }
    
    def get_roberta_embeddings(self, texts):
        """Extract RoBERTa [CLS] embeddings"""
        logger.info(f"Extracting RoBERTa embeddings for {len(texts)} texts...")
        
        embeddings = []
        batch_size = 16
        
        self.roberta.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Get embeddings
                outputs = self.roberta(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def extract_features(self, texts, feature_bundles):
        """Extract engineered features for given bundles"""
        all_features = []
        
        for text in texts:
            feature_dict = {}
            for bundle in feature_bundles:
                bundle_features = self.feature_extractor.extract_bundle_features([text], bundle)
                feature_dict.update(bundle_features)
            
            all_features.append(list(feature_dict.values()))
        
        return np.array(all_features)
    
    def run_experiment(self, feature_bundles, experiment_name):
        """Run experiment with RoBERTa + features combined"""
        logger.info(f"Running {experiment_name} with bundles: {feature_bundles}")
        
        # Load data
        data = self.load_data()
        
        # Get RoBERTa embeddings (768-dim)
        train_embeddings = self.get_roberta_embeddings(data['train_texts'])
        test_embeddings = self.get_roberta_embeddings(data['test_texts'])
        
        if len(feature_bundles) > 0:
            # Extract engineered features
            train_features = self.extract_features(data['train_texts'], feature_bundles)
            test_features = self.extract_features(data['test_texts'], feature_bundles)
            
            # Handle edge case of no features
            if train_features.shape[1] == 0:
                logger.warning(f"No features extracted for {experiment_name}")
                return None
            
            # Scale features (but not RoBERTa embeddings)
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)
            
            # Combine RoBERTa embeddings with features
            train_combined = np.hstack([train_embeddings, train_features_scaled])
            test_combined = np.hstack([test_embeddings, test_features_scaled])
            
            logger.info(f"Combined features: RoBERTa ({train_embeddings.shape[1]}) + Features ({train_features.shape[1]}) = {train_combined.shape[1]} total")
        else:
            # Baseline: only RoBERTa embeddings
            train_combined = train_embeddings
            test_combined = test_embeddings
            logger.info(f"Using only RoBERTa embeddings: {train_combined.shape[1]} dimensions")
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_combined, data['train_labels'])
        
        # Predict
        predictions = model.predict(test_combined)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            data['test_labels'], predictions, average='binary'
        )
        accuracy = accuracy_score(data['test_labels'], predictions)
        
        result = {
            'experiment_name': experiment_name,
            'features_used': ','.join(feature_bundles) if feature_bundles else 'roberta_only',
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'num_features': train_combined.shape[1],
            'notes': f'RoBERTa + {len(feature_bundles)} feature bundles' if feature_bundles else 'RoBERTa embeddings only'
        }
        
        logger.info(f"Results: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        return result
    
    def run_additive_experiments(self):
        """Run additive experiments with RoBERTa + features"""
        logger.info("="*60)
        logger.info("RUNNING COMBINED ROBERTA + FEATURES EXPERIMENTS")
        logger.info("="*60)
        
        # Test RoBERTa-only baseline first
        roberta_baseline = self.run_experiment([], 'roberta_baseline')
        self.results.append(roberta_baseline)
        
        current_best_f1 = roberta_baseline['f1_score']
        current_best_features = []
        
        logger.info(f"RoBERTa baseline F1: {current_best_f1:.4f}")
        
        # Test each bundle additively
        for i, bundle in enumerate(self.feature_bundles):
            features_to_test = current_best_features + [bundle]
            experiment_name = f"roberta+{bundle}"
            
            result = self.run_experiment(features_to_test, experiment_name) 
            
            if result is None:
                logger.warning(f"Skipping {bundle}")
                continue
            
            self.results.append(result)
            
            # Check improvement
            new_f1 = result['f1_score']
            improvement = new_f1 - current_best_f1
            
            logger.info(f"F1: {new_f1:.4f} vs Best: {current_best_f1:.4f} (Δ={improvement:+.4f})")
            
            if new_f1 > current_best_f1:
                current_best_f1 = new_f1
                current_best_features = features_to_test.copy()
                logger.info(f"✅ IMPROVEMENT! New best: {current_best_f1:.4f}")
            else:
                logger.info(f"❌ No improvement with {bundle}")
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("COMBINED EXPERIMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"RoBERTa baseline F1: {roberta_baseline['f1_score']:.4f}")
        logger.info(f"Best combined F1: {current_best_f1:.4f}")
        logger.info(f"Best features: {current_best_features}")
        logger.info(f"Improvement: {(current_best_f1 - roberta_baseline['f1_score']):+.4f}")
        
        # Save results
        self.save_results()
        
        return current_best_features, current_best_f1
    
    def save_results(self):
        """Save experiment results"""
        # Save to CSV
        results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/combined_results.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(results_path, index=False)
        
        # Print summary table
        logger.info("\n" + "COMBINED RESULTS TABLE")
        logger.info("="*80)
        print(df[['experiment_name', 'features_used', 'f1_score', 'num_features']].to_string(index=False))
        
        logger.info(f"\nResults saved to {results_path}")

def main():
    """Run combined experiments"""
    experiment = CombinedTextFeatureExperiment()
    
    try:
        best_features, best_f1 = experiment.run_additive_experiments()
        
        logger.info(f"\n🎉 COMBINED EXPERIMENTS COMPLETED!")
        logger.info(f"Best combination: {best_features}")
        logger.info(f"Best F1 score: {best_f1:.4f}")
        
        return best_features, best_f1
        
    except Exception as e:
        logger.error(f"Combined experiments failed: {e}")
        raise

if __name__ == "__main__":
    main()