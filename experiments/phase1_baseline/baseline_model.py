#!/usr/bin/env python3
"""
Phase 1: Baseline RoBERTa Model for Frustration Detection
Train baseline model without engineered features
"""

import os
import json
import torch
import numpy as np
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    """Custom trainer for binary classification with proper loss handling"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        labels = inputs.get("labels")
        
        outputs = model(**model_inputs)
        logits = outputs.get('logits')
        
        if labels is not None:
            labels = labels.float().view(-1, 1)
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            
        return (loss, outputs) if return_outputs else loss

class BaselineFrustrationDetector:
    """Baseline frustration detection model using RoBERTa"""
    
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def load_data(self, data_path):
        """Load tokenized data"""
        logger.info(f"Loading tokenized data from {data_path}...")
        tokenized_data = torch.load(data_path)
        
        # Split into train/val/test
        total_size = len(tokenized_data['input_ids'])
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        
        train_data = {k: v[:train_size] for k, v in tokenized_data.items()}
        val_data = {k: v[train_size:train_size+val_size] for k, v in tokenized_data.items()}
        test_data = {k: v[train_size+val_size:] for k, v in tokenized_data.items()}
        
        logger.info(f"Data split - Train: {len(train_data['labels'])}, Val: {len(val_data['labels'])}, Test: {len(test_data['labels'])}")
        
        return train_data, val_data, test_data
    
    def setup_model(self):
        """Initialize model and tokenizer"""
        logger.info("Setting up baseline model...")
        
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
        logger.info("Model setup complete")
    
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
    
    def train(self, train_data, val_data, output_dir):
        """Train the baseline model"""
        logger.info("Starting baseline model training...")
        
        # Convert to proper dataset format for Trainer
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, input_ids, attention_mask, labels):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels
                
            def __len__(self):
                return len(self.labels)
                
            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx]
                }
        
        train_dataset = CustomDataset(
            train_data['input_ids'],
            train_data['attention_mask'],
            train_data['labels']
        )
        
        val_dataset = CustomDataset(
            val_data['input_ids'],
            val_data['attention_mask'],
            val_data['labels']
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=10,
            report_to=None,
        )
        
        # Create trainer
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        self.trainer.train()
        logger.info("Training completed!")
    
    def evaluate(self, test_data, output_dir):
        """Evaluate model on test set"""
        logger.info("Evaluating baseline model...")
        
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, input_ids, attention_mask, labels):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels
                
            def __len__(self):
                return len(self.labels)
                
            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx]
                }
        
        test_dataset = CustomDataset(
            test_data['input_ids'],
            test_data['attention_mask'],
            test_data['labels']
        )
        
        results = self.trainer.evaluate(test_dataset)
        
        # Save results
        baseline_results = {
            'experiment_name': 'baseline',
            'features_used': 'none',
            'f1_score': results['eval_f1'],
            'precision': results['eval_precision'],
            'recall': results['eval_recall'],
            'accuracy': results['eval_accuracy'],
            'notes': 'Baseline RoBERTa without engineered features'
        }
        
        # Save to JSON
        with open(os.path.join(output_dir, 'baseline_results.json'), 'w') as f:
            json.dump(baseline_results, f, indent=2)
        
        # Save to CSV format for results tracking
        results_csv_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/results.csv"
        os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
        
        import pandas as pd
        df = pd.DataFrame([baseline_results])
        df.to_csv(results_csv_path, index=False)
        
        logger.info("Baseline Results:")
        for key, value in baseline_results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return baseline_results
    
    def save_model(self, output_dir):
        """Save trained model"""
        model_path = os.path.join(output_dir, 'baseline_model')
        self.trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

def main():
    """Main function for baseline training"""
    # Paths
    data_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/data/tokenized_subset.pt"
    output_dir = "/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase1_baseline/output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = BaselineFrustrationDetector()
    
    # Load data
    train_data, val_data, test_data = detector.load_data(data_path)
    
    # Setup model
    detector.setup_model()
    
    # Train model
    detector.train(train_data, val_data, output_dir)
    
    # Evaluate model
    results = detector.evaluate(test_data, output_dir)
    
    # Save model
    detector.save_model(output_dir)
    
    logger.info("Phase 1 baseline training completed successfully!")
    return results

if __name__ == "__main__":
    main()