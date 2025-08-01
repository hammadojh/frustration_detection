#!/usr/bin/env python3
"""
Phase 3a: Fusion Strategies Experiment
Test different ways to combine text and features beyond simple concatenation
"""

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaModel, AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FusionStrategiesExperiment:
    """Test different fusion strategies for combining RoBERTa + features"""
    
    def __init__(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.roberta = RobertaModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.roberta.to(self.device)
        
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
    
    def extract_linguistic_features(self, texts):
        """Extract linguistic features (our best performing bundle)"""
        import sys
        sys.path.append('/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase2_features')
        from fast_feature_extractor import FastFrustrationFeatureExtractor
        extractor = FastFrustrationFeatureExtractor()
        
        all_features = []
        for text in texts:
            features = extractor.extract_linguistic_bundle([text])
            all_features.append(list(features.values()))
        
        return np.array(all_features)
    
    def strategy_1_late_fusion_concat(self, texts, labels, test_texts, test_labels):
        """Strategy 1: Late fusion - concatenate embeddings + features (baseline)"""
        logger.info("Testing Strategy 1: Late Fusion (Concatenation)")
        
        # Get RoBERTa embeddings
        train_embeddings = self.get_roberta_embeddings(texts)
        test_embeddings = self.get_roberta_embeddings(test_texts)
        
        # Get features
        train_features = self.extract_linguistic_features(texts)
        test_features = self.extract_linguistic_features(test_texts)
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Concatenate
        train_combined = np.hstack([train_embeddings, train_features_scaled])
        test_combined = np.hstack([test_embeddings, test_features_scaled])
        
        # Train classifier
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_combined, labels)
        
        # Predict
        predictions = model.predict(test_combined)
        
        return self.calculate_metrics(test_labels, predictions, "late_fusion_concat")
    
    def strategy_2_early_fusion_tokens(self, texts, labels, test_texts, test_labels):
        """Strategy 2: Early fusion - add features as special tokens"""
        logger.info("Testing Strategy 2: Early Fusion (Feature Tokens)")
        
        # Convert features to token representations
        train_features = self.extract_linguistic_features(texts)
        test_features = self.extract_linguistic_features(test_texts)
        
        # Scale features to reasonable token-like values
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Convert to discrete tokens (bin into vocabulary)
        def features_to_tokens(features_scaled):
            # Bin each feature into 10 discrete levels, map to token IDs
            feature_tokens = []
            for feat_vec in features_scaled:
                tokens = []
                for i, feat_val in enumerate(feat_vec):
                    # Bin feature value into discrete levels
                    level = int(np.clip((feat_val + 3) / 6 * 10, 0, 9))  # -3 to +3 → 0-9
                    # Create special token like [FEAT0_L5] 
                    token_text = f"[FEAT{i}_L{level}]"
                    tokens.append(token_text)
                feature_tokens.append(" ".join(tokens))
            return feature_tokens
        
        train_feat_tokens = features_to_tokens(train_features_scaled)
        test_feat_tokens = features_to_tokens(test_features_scaled)
        
        # Prepend feature tokens to text
        train_texts_enhanced = [f"{feat_tokens} {text}" for feat_tokens, text in zip(train_feat_tokens, texts)]
        test_texts_enhanced = [f"{feat_tokens} {text}" for feat_tokens, text in zip(test_feat_tokens, test_texts)]
        
        # Get embeddings from enhanced text
        train_embeddings = self.get_roberta_embeddings(train_texts_enhanced)
        test_embeddings = self.get_roberta_embeddings(test_texts_enhanced)
        
        # Train classifier
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(train_embeddings, labels)
        
        predictions = model.predict(test_embeddings)
        
        return self.calculate_metrics(test_labels, predictions, "early_fusion_tokens")
    
    def strategy_3_attention_fusion(self, texts, labels, test_texts, test_labels):
        """Strategy 3: Attention-based fusion"""
        logger.info("Testing Strategy 3: Attention Fusion")
        
        class AttentionFusionModel(nn.Module):
            def __init__(self, text_dim=768, feat_dim=8, hidden_dim=256):
                super().__init__()
                self.text_dim = text_dim
                self.feat_dim = feat_dim
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim, 
                    num_heads=8, 
                    dropout=0.1,
                    batch_first=True
                )
                
                # Projection layers
                self.text_proj = nn.Linear(text_dim, hidden_dim)
                self.feat_proj = nn.Linear(feat_dim, hidden_dim)
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 1)
                )
                
            def forward(self, text_emb, feat_emb):
                batch_size = text_emb.size(0)
                
                # Project to same dimension
                text_proj = self.text_proj(text_emb).unsqueeze(1)  # [B, 1, H]
                feat_proj = self.feat_proj(feat_emb).unsqueeze(1)  # [B, 1, H]
                
                # Concatenate for attention
                combined = torch.cat([text_proj, feat_proj], dim=1)  # [B, 2, H]
                
                # Self-attention
                attn_out, _ = self.attention(combined, combined, combined)
                
                # Global average pooling
                fused = attn_out.mean(dim=1)  # [B, H]
                
                # Classification
                logits = self.classifier(fused)
                return logits.squeeze()
        
        # Get embeddings and features
        train_embeddings = self.get_roberta_embeddings(texts)
        test_embeddings = self.get_roberta_embeddings(test_texts)
        
        train_features = self.extract_linguistic_features(texts)
        test_features = self.extract_linguistic_features(test_texts)
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Convert to tensors
        train_emb_tensor = torch.FloatTensor(train_embeddings).to(self.device)
        train_feat_tensor = torch.FloatTensor(train_features_scaled).to(self.device)
        train_labels_tensor = torch.FloatTensor(labels).to(self.device)
        
        test_emb_tensor = torch.FloatTensor(test_embeddings).to(self.device)
        test_feat_tensor = torch.FloatTensor(test_features_scaled).to(self.device)
        
        # Create model
        model = AttentionFusionModel(
            text_dim=train_embeddings.shape[1],
            feat_dim=train_features.shape[1]
        ).to(self.device)
        
        # Training setup
        optimizer = AdamW(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        # Create DataLoader
        train_dataset = TensorDataset(train_emb_tensor, train_feat_tensor, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Training loop
        model.train()
        for epoch in range(10):
            total_loss = 0
            for batch_emb, batch_feat, batch_labels in train_loader:
                optimizer.zero_grad()
                
                logits = model(batch_emb, batch_feat)
                loss = criterion(logits, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 3 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Prediction
        model.eval()
        with torch.no_grad():
            test_logits = model(test_emb_tensor, test_feat_tensor)
            test_probs = torch.sigmoid(test_logits)
            predictions = (test_probs > 0.5).cpu().numpy().astype(int)
        
        return self.calculate_metrics(test_labels, predictions, "attention_fusion")
    
    def strategy_4_weighted_fusion(self, texts, labels, test_texts, test_labels):
        """Strategy 4: Learnable weighted fusion"""
        logger.info("Testing Strategy 4: Weighted Fusion")
        
        # Get embeddings and features
        train_embeddings = self.get_roberta_embeddings(texts)
        test_embeddings = self.get_roberta_embeddings(test_texts)
        
        train_features = self.extract_linguistic_features(texts)
        test_features = self.extract_linguistic_features(test_texts)
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Learn optimal weights through cross-validation on training set
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import Ridge
        
        # Create weighted combinations with different alpha values
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        best_alpha = 0.5
        best_score = 0
        
        for alpha in alphas:
            # Weight text vs features
            weighted_train = np.hstack([
                alpha * train_embeddings,
                (1 - alpha) * train_features_scaled
            ])
            
            # Quick validation
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(weighted_train, labels)
            
            # Simple train accuracy as proxy
            train_pred = model.predict(weighted_train)
            train_acc = accuracy_score(labels, train_pred)
            
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
        model.fit(weighted_train, labels)
        predictions = model.predict(weighted_test)
        
        result = self.calculate_metrics(test_labels, predictions, "weighted_fusion")
        result['best_alpha'] = best_alpha
        return result
    
    def get_roberta_embeddings(self, texts):
        """Extract RoBERTa [CLS] embeddings"""
        embeddings = []
        batch_size = 16
        
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
        
        # Load data
        data = self.load_data()
        
        # Run each strategy
        strategies = [
            self.strategy_1_late_fusion_concat,
            self.strategy_2_early_fusion_tokens,
            self.strategy_3_attention_fusion,
            self.strategy_4_weighted_fusion
        ]
        
        baseline_f1 = 0.8378  # From Phase 2
        
        for strategy_func in strategies:
            try:
                result = strategy_func(
                    data['train_texts'], data['train_labels'],
                    data['test_texts'], data['test_labels']
                )
                
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
            logger.info(f"{result['strategy']}: F1 = {result['f1_score']:.4f} (Δ={improvement:+.4f})")
        
        # Save results
        results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/fusion_strategies_results.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to {results_path}")

def main():
    """Run fusion strategies experiment"""
    experiment = FusionStrategiesExperiment()
    
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