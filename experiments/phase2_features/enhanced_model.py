#!/usr/bin/env python3
"""
Phase 2: Enhanced RoBERTa Model with Feature Concatenation
Combine RoBERTa embeddings with engineered features
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from fast_feature_extractor import FastFrustrationFeatureExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFrustrationModel(nn.Module):
    """Enhanced model combining RoBERTa with engineered features"""
    
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions", num_feature_dims=32):
        super().__init__()
        
        # RoBERTa backbone
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.roberta_dim = self.roberta.config.hidden_size  # 768
        
        # Feature processing layers
        self.num_feature_dims = num_feature_dims
        self.feature_projection = nn.Linear(num_feature_dims, 128)
        
        # Combined classification layers
        combined_dim = self.roberta_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask, features=None):
        # Get RoBERTa embeddings
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = roberta_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        if features is not None:
            # Project features to higher dimension
            feature_proj = self.feature_projection(features)
            feature_proj = torch.relu(feature_proj)
            
            # Concatenate RoBERTa [CLS] with projected features
            combined = torch.cat([cls_output, feature_proj], dim=1)
        else:
            combined = cls_output
        
        # Classification
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        
        return {"logits": logits}

class CustomTrainer(Trainer):
    """Custom trainer for enhanced model with features"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Don't pop labels - keep them for the model
        labels = inputs.get("labels")
        
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if labels is not None:
            labels = labels.float().view(-1, 1)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        else:
            # Return a zero loss tensor instead of None
            loss = torch.tensor(0.0, requires_grad=True, device=logits.device if logits is not None else 'cpu')
            
        return (loss, outputs) if return_outputs else loss

class EnhancedFrustrationDetector:
    """Enhanced frustration detection with RoBERTa + features"""
    
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.feature_extractor = FastFrustrationFeatureExtractor()
        self.feature_scaler = None
        
    def load_data_with_features(self, data_path, processed_examples_path, feature_bundles=None):
        """Load tokenized data and extract features"""
        logger.info(f"Loading data from {data_path}...")
        
        # Load tokenized data
        tokenized_data = torch.load(data_path, weights_only=False)
        
        # Load processed examples to get text
        with open(processed_examples_path, 'r') as f:
            processed_examples = json.load(f)
        
        # Extract features for each example
        logger.info("Extracting features for all examples...")
        all_features = []
        
        for example in processed_examples:
            text = example['text']
            # Treat single utterance as a list for feature extraction
            texts = [text] if isinstance(text, str) else text
            
            if feature_bundles:
                # Extract only specified bundles
                feature_dict = {}
                for bundle in feature_bundles:
                    bundle_features = self.feature_extractor.extract_bundle_features(texts, bundle)
                    feature_dict.update(bundle_features)
            else:
                # Extract all features
                all_bundle_features = self.feature_extractor.extract_all_features(texts)
                feature_dict = {}
                for bundle_features in all_bundle_features.values():
                    feature_dict.update(bundle_features)
            
            # Convert to feature vector
            feature_vector = list(feature_dict.values())
            all_features.append(feature_vector)
        
        # Convert to numpy array and handle variable dimensions
        max_features = max(len(f) for f in all_features) if all_features else 0
        feature_matrix = np.zeros((len(all_features), max_features))
        
        for i, features in enumerate(all_features):
            feature_matrix[i, :len(features)] = features
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.feature_scaler = StandardScaler()
        feature_matrix = self.feature_scaler.fit_transform(feature_matrix)
        
        # Split data
        total_size = len(tokenized_data['input_ids'])
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        
        # Create data splits
        train_data = {
            'input_ids': tokenized_data['input_ids'][:train_size],
            'attention_mask': tokenized_data['attention_mask'][:train_size],
            'labels': tokenized_data['labels'][:train_size],
            'features': torch.FloatTensor(feature_matrix[:train_size])
        }
        
        val_data = {
            'input_ids': tokenized_data['input_ids'][train_size:train_size+val_size],
            'attention_mask': tokenized_data['attention_mask'][train_size:train_size+val_size],
            'labels': tokenized_data['labels'][train_size:train_size+val_size],
            'features': torch.FloatTensor(feature_matrix[train_size:train_size+val_size])
        }
        
        test_data = {
            'input_ids': tokenized_data['input_ids'][train_size+val_size:],
            'attention_mask': tokenized_data['attention_mask'][train_size+val_size:],
            'labels': tokenized_data['labels'][train_size+val_size:],
            'features': torch.FloatTensor(feature_matrix[train_size+val_size:])
        }
        
        logger.info(f"Data loaded - Train: {len(train_data['labels'])}, Val: {len(val_data['labels'])}, Test: {len(test_data['labels'])}")
        logger.info(f"Feature dimensions: {feature_matrix.shape[1]}")
        
        return train_data, val_data, test_data, feature_matrix.shape[1]
    
    def setup_model(self, num_feature_dims):
        """Initialize enhanced model and tokenizer"""
        logger.info("Setting up enhanced model...")
        
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
        self.model = EnhancedFrustrationModel(self.model_name, num_feature_dims)
        
        logger.info(f"Model setup complete with {num_feature_dims} feature dimensions")
    
    def compute_metrics(self, pred):
        """Compute evaluation metrics"""
        labels = pred.label_ids.flatten()
        preds = (pred.predictions > 0.0).astype(int).flatten()
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_data, val_data, output_dir, experiment_name="enhanced"):
        """Train the enhanced model"""
        logger.info(f"Starting enhanced model training for {experiment_name}...")
        
        # Convert to dataset format
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, input_ids, attention_mask, labels, features):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels
                self.features = features
                
            def __len__(self):
                return len(self.labels)
                
            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx],
                    'features': self.features[idx]
                }
        
        train_dataset = CustomDataset(
            train_data['input_ids'],
            train_data['attention_mask'],
            train_data['labels'],
            train_data['features']
        )
        
        val_dataset = CustomDataset(
            val_data['input_ids'],
            val_data['attention_mask'],
            val_data['labels'],
            val_data['features']
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=False,
            logging_steps=10,
            report_to=None,
        )
        
        # Create trainer
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        self.trainer.train()
        logger.info("Training completed!")
    
    def evaluate(self, test_data, output_dir, experiment_name="enhanced", features_used=""):
        """Evaluate enhanced model on test set"""
        logger.info("Evaluating enhanced model...")
        
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, input_ids, attention_mask, labels, features):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels
                self.features = features
                
            def __len__(self):
                return len(self.labels)
                
            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx],
                    'features': self.features[idx]
                }
        
        test_dataset = CustomDataset(
            test_data['input_ids'],
            test_data['attention_mask'],
            test_data['labels'],
            test_data['features']
        )
        
        results = self.trainer.evaluate(test_dataset)
        
        # Save results
        enhanced_results = {
            'experiment_name': experiment_name,
            'features_used': features_used,
            'f1_score': results['eval_f1'],
            'precision': results['eval_precision'],
            'recall': results['eval_recall'],
            'accuracy': results['eval_accuracy'],
            'notes': f'Enhanced RoBERTa with {features_used} features'
        }
        
        # Save to JSON
        with open(os.path.join(output_dir, f'{experiment_name}_results.json'), 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        logger.info("Enhanced Model Results:")
        for key, value in enhanced_results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return enhanced_results
    
    def save_model(self, output_dir, experiment_name="enhanced"):
        """Save trained model"""
        model_path = os.path.join(output_dir, f'{experiment_name}_model')
        self.trainer.save_model(model_path)
        
        # Save feature scaler
        if self.feature_scaler:
            import joblib
            scaler_path = os.path.join(output_dir, f'{experiment_name}_scaler.pkl')
            joblib.dump(self.feature_scaler, scaler_path)
        
        logger.info(f"Enhanced model saved to {model_path}")

def main():
    """Test enhanced model with features"""
    # Paths
    data_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/data/tokenized_subset.pt"
    processed_examples_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/data/subset_processed.json"
    output_dir = "/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase2_features/output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = EnhancedFrustrationDetector()
    
    # Test with linguistic bundle only
    feature_bundles = ['linguistic_bundle']
    
    # Load data with features
    train_data, val_data, test_data, num_features = detector.load_data_with_features(
        data_path, processed_examples_path, feature_bundles
    )
    
    # Setup model
    detector.setup_model(num_features)
    
    # Train model
    detector.train(train_data, val_data, output_dir, "linguistic_test")
    
    # Evaluate model
    results = detector.evaluate(test_data, output_dir, "linguistic_test", "linguistic_bundle")
    
    # Save model
    detector.save_model(output_dir, "linguistic_test")
    
    logger.info("Enhanced model test completed successfully!")
    return results

if __name__ == "__main__":
    main()