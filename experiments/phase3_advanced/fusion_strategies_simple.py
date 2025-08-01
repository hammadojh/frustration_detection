#!/usr/bin/env python3
"""
Phase 3a: Fusion Strategies Experiment (Simplified)
Test key fusion strategies without complex neural networks
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

# Add path for feature extractor
sys.path.append('/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase2_features')
from fast_feature_extractor import FastFrustrationFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFusionExperiment:
    """Test key fusion strategies efficiently"""
    
    def __init__(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.roberta.to(self.device)
        self.feature_extractor = FastFrustrationFeatureExtractor()
        
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
    
    def extract_linguistic_features(self, texts):
        """Extract linguistic features"""
        all_features = []
        for text in texts:
            features = self.feature_extractor.extract_linguistic_bundle([text])
            all_features.append(list(features.values()))
        return np.array(all_features)
    
    def get_roberta_embeddings(self, texts):
        """Extract RoBERTa [CLS] embeddings efficiently"""
        embeddings = []
        batch_size = 16
        
        logger.info(f"Extracting RoBERTa embeddings for {len(texts)} texts...")
        
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
    
    def strategy_1_late_fusion_concat(self, data):
        """Strategy 1: Late fusion - concatenate embeddings + features (baseline)"""
        logger.info("Testing Strategy 1: Late Fusion (Concatenation)")
        
        # Get embeddings and features
        train_embeddings = self.get_roberta_embeddings(data['train_texts'])
        test_embeddings = self.get_roberta_embeddings(data['test_texts'])
        
        train_features = self.extract_linguistic_features(data['train_texts'])
        test_features = self.extract_linguistic_features(data['test_texts'])
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Concatenate
        train_combined = np.hstack([train_embeddings, train_features_scaled])
        test_combined = np.hstack([test_embeddings, test_features_scaled])
        
        # Train and predict
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_combined, data['train_labels'])
        predictions = model.predict(test_combined)
        
        return self.calculate_metrics(data['test_labels'], predictions, "late_fusion_concat")
    
    def strategy_2_early_fusion_tokens(self, data):
        """Strategy 2: Early fusion - add features as special tokens"""
        logger.info("Testing Strategy 2: Early Fusion (Feature Tokens)")
        
        train_features = self.extract_linguistic_features(data['train_texts'])
        test_features = self.extract_linguistic_features(data['test_texts'])
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Convert features to tokens
        def features_to_tokens(features_scaled):
            feature_tokens = []
            for feat_vec in features_scaled:
                tokens = []
                for i, feat_val in enumerate(feat_vec):
                    # Bin feature value into discrete levels
                    level = int(np.clip((feat_val + 3) / 6 * 10, 0, 9))
                    tokens.append(f"[FEAT{i}_{level}]")
                feature_tokens.append(" ".join(tokens))
            return feature_tokens
        
        train_feat_tokens = features_to_tokens(train_features_scaled)
        test_feat_tokens = features_to_tokens(test_features_scaled)
        
        # Prepend feature tokens to text
        train_texts_enhanced = [f"{feat_tokens} {text}" for feat_tokens, text in zip(train_feat_tokens, data['train_texts'])]
        test_texts_enhanced = [f"{feat_tokens} {text}" for feat_tokens, text in zip(test_feat_tokens, data['test_texts'])]
        
        # Get embeddings from enhanced text
        train_embeddings = self.get_roberta_embeddings(train_texts_enhanced)
        test_embeddings = self.get_roberta_embeddings(test_texts_enhanced)
        
        # Train and predict
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_embeddings, data['train_labels'])
        predictions = model.predict(test_embeddings)
        
        return self.calculate_metrics(data['test_labels'], predictions, "early_fusion_tokens")
    
    def strategy_3_weighted_fusion(self, data):
        """Strategy 3: Learnable weighted fusion"""
        logger.info("Testing Strategy 3: Weighted Fusion")
        
        # Get embeddings and features
        train_embeddings = self.get_roberta_embeddings(data['train_texts'])
        test_embeddings = self.get_roberta_embeddings(data['test_texts'])
        
        train_features = self.extract_linguistic_features(data['train_texts'])
        test_features = self.extract_linguistic_features(data['test_texts'])
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Find optimal weights
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        best_alpha = 0.5
        best_score = 0
        
        logger.info("Finding optimal text/feature weights...")
        for alpha in alphas:
            # Weight text vs features  
            weighted_train = np.hstack([
                alpha * train_embeddings,
                (1 - alpha) * train_features_scaled
            ])
            
            # Quick validation
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(weighted_train, data['train_labels'])
            train_pred = model.predict(weighted_train)
            train_acc = accuracy_score(data['train_labels'], train_pred)
            
            if train_acc > best_score:
                best_score = train_acc
                best_alpha = alpha
        
        logger.info(f"Best alpha (text weight): {best_alpha}")
        
        # Final model with best alpha
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
        
        result = self.calculate_metrics(data['test_labels'], predictions, "weighted_fusion")
        result['best_alpha'] = best_alpha
        return result
    
    def strategy_4_feature_selection_fusion(self, data):
        """Strategy 4: Feature selection + fusion"""
        logger.info("Testing Strategy 4: Feature Selection Fusion")
        
        # Get embeddings and features
        train_embeddings = self.get_roberta_embeddings(data['train_texts'])
        test_embeddings = self.get_roberta_embeddings(data['test_texts'])
        
        train_features = self.extract_linguistic_features(data['train_texts'])
        test_features = self.extract_linguistic_features(data['test_texts'])
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Feature selection using correlation with labels
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Select top k features
        k_values = [3, 5, 8]  # 8 is all features
        best_k = 8
        best_score = 0
        
        for k in k_values:
            if k >= train_features.shape[1]:
                k = train_features.shape[1]
            
            selector = SelectKBest(f_classif, k=k)
            train_feat_selected = selector.fit_transform(train_features_scaled, data['train_labels'])
            test_feat_selected = selector.transform(test_features_scaled)
            
            # Combine with embeddings
            train_combined = np.hstack([train_embeddings, train_feat_selected])
            test_combined = np.hstack([test_embeddings, test_feat_selected])
            
            # Quick validation
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(train_combined, data['train_labels'])
            train_pred = model.predict(train_combined)
            train_acc = accuracy_score(data['train_labels'], train_pred)
            
            if train_acc > best_score:
                best_score = train_acc
                best_k = k
        
        logger.info(f"Best k (selected features): {best_k}")
        
        # Final model with best k
        selector = SelectKBest(f_classif, k=best_k)
        train_feat_selected = selector.fit_transform(train_features_scaled, data['train_labels'])
        test_feat_selected = selector.transform(test_features_scaled)
        
        train_combined = np.hstack([train_embeddings, train_feat_selected])
        test_combined = np.hstack([test_embeddings, test_feat_selected])
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_combined, data['train_labels'])
        predictions = model.predict(test_combined)
        
        result = self.calculate_metrics(data['test_labels'], predictions, "feature_selection_fusion")
        result['best_k'] = best_k
        return result
    
    def calculate_metrics(self, true_labels, predictions, strategy_name):
        """Calculate evaluation metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        accuracy = accuracy_score(true_labels, predictions)
        
        result = {
            'strategy': strategy_name,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }
        
        logger.info(f"{strategy_name}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
        return result
    
    def run_all_strategies(self):
        """Run all fusion strategies"""
        logger.info("="*60)
        logger.info("PHASE 3A: FUSION STRATEGIES EXPERIMENT")
        logger.info("="*60)
        
        # Load data once
        data = self.load_data()
        
        # Run each strategy
        strategies = [
            self.strategy_1_late_fusion_concat,
            self.strategy_2_early_fusion_tokens,
            self.strategy_3_weighted_fusion,
            self.strategy_4_feature_selection_fusion
        ]
        
        baseline_f1 = 0.8378  # From Phase 2
        
        for strategy_func in strategies:
            try:
                result = strategy_func(data)
                self.results.append(result)
                
                improvement = result['f1_score'] - baseline_f1
                status = "✅" if improvement > 0 else "❌"
                logger.info(f"{status} {result['strategy']}: Δ={improvement:+.4f}")
                
            except Exception as e:
                logger.error(f"Strategy {strategy_func.__name__} failed: {e}")
        
        # Summary
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print experiment summary"""
        logger.info("\n" + "="*60)
        logger.info("FUSION STRATEGIES SUMMARY")
        logger.info("="*60)
        
        # Sort by F1 score
        sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
        
        baseline_f1 = 0.8378
        logger.info(f"Phase 2 baseline: F1 = {baseline_f1:.4f}")
        logger.info("")
        
        for result in sorted_results:
            improvement = result['f1_score'] - baseline_f1
            status = "🏆" if improvement > 0.01 else "✅" if improvement > 0 else "❌"
            logger.info(f"{status} {result['strategy']}: F1 = {result['f1_score']:.4f} (Δ={improvement:+.4f})")
        
        # Save results
        results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/fusion_strategies_results.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to {results_path}")

def main():
    """Run fusion strategies experiment"""
    experiment = SimpleFusionExperiment()
    
    try:
        results = experiment.run_all_strategies()
        
        best_result = max(results, key=lambda x: x['f1_score'])
        logger.info(f"\n🎉 BEST FUSION STRATEGY: {best_result['strategy']}")
        logger.info(f"F1 Score: {best_result['f1_score']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Fusion strategies experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()