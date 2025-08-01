#!/usr/bin/env python3
"""
Frustration Detection in Dialogues with Transformers

This script implements a binary classifier to detect frustration in dialogues
using the EmpatheticDialogues dataset and RoBERTa transformer model.

Based on the implementation guide for frustration detection in task-oriented dialogue systems.
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast, 
    RobertaForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    """Custom trainer to handle binary classification loss properly."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation to handle shape mismatch in binary classification.
        """
        # Make a copy of inputs to avoid modifying the original
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        labels = inputs.get("labels")
        
        outputs = model(**model_inputs)  # Forward pass without labels
        logits = outputs.get('logits')
        
        if labels is not None:
            # Reshape labels to match logits for BCEWithLogitsLoss
            labels = labels.float().view(-1, 1)  # [batch_size] -> [batch_size, 1]
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step to handle evaluation.
        """
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs[1]
            else:
                loss = None
                model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                outputs = model(**model_inputs)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs[0]

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
            
        return (loss, logits.detach(), labels)

class FrustrationDetector:
    """
    A frustration detection classifier using RoBERTa transformer model.
    """
    
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions", max_length=512):
        """
        Initialize the frustration detector.
        
        Args:
            model_name (str): Pre-trained model name from Hugging Face
            max_length (int): Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Define which emotions are considered "frustrated"
        self.FRUSTRATED_LABELS = {
            "annoyed", "angry", "disgusted", "jealous", "embarrassed", 
            "ashamed", "afraid", "terrified", "furious", "devastated", 
            "disappointed"
        }
        
    def load_dataset(self):
        """Load and preprocess the EmpatheticDialogues dataset."""
        logger.info("Loading EmpatheticDialogues dataset...")
        dataset = load_dataset("facebook/empathetic_dialogues")
        logger.info(f"Dataset loaded. Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
        return dataset
    
    def map_to_binary(self, example):
        """
        Map emotion labels to binary frustrated/not frustrated labels.
        
        Args:
            example: Dataset example with 'prompt', 'utterance', and 'context' fields
            
        Returns:
            dict: Processed example with 'text' and binary 'label'
        """
        # Combine prompt and utterances into a single text
        if isinstance(example["utterance"], list):
            utterances = " ".join(example["utterance"])
        else:
            utterances = example["utterance"]
            
        text = example["prompt"] + " " + utterances
        
        # Map emotion to binary label (1 for frustrated, 0 for not frustrated)
        label = int(example["context"] in self.FRUSTRATED_LABELS)
        
        return {
            "text": text,
            "label": label
        }
    
    def preprocess_dataset(self, dataset):
        """
        Preprocess the dataset by mapping to binary labels and tokenizing.
        
        Args:
            dataset: Raw dataset from Hugging Face
            
        Returns:
            Tokenized dataset ready for training
        """
        logger.info("Mapping emotions to binary frustrated/not frustrated labels...")
        dataset = dataset.map(self.map_to_binary)
        
        # Print label distribution
        train_labels = [example["label"] for example in dataset["train"]]
        frustrated_count = sum(train_labels)
        total_count = len(train_labels)
        logger.info(f"Label distribution - Frustrated: {frustrated_count} ({frustrated_count/total_count:.2%}), "
                   f"Not Frustrated: {total_count - frustrated_count} ({(total_count - frustrated_count)/total_count:.2%})")
        
        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
        
        # Tokenize the dataset
        logger.info("Tokenizing dataset...")
        def tokenize(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=self.max_length
            )
        
        tokenized_dataset = dataset.map(tokenize, batched=True)
        
        # Convert labels to the right format for training
        def convert_labels(examples):
            examples["labels"] = [float(label) for label in examples["label"]]
            return examples
        
        tokenized_dataset = tokenized_dataset.map(convert_labels, batched=True)
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        return tokenized_dataset
    
    def setup_model(self, tokenized_dataset):
        """
        Setup the RoBERTa model with class weights for balanced training.
        
        Args:
            tokenized_dataset: Preprocessed and tokenized dataset
        """
        logger.info("Setting up RoBERTa model...")
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
        # Compute class weights for balanced training
        logger.info("Computing class weights...")
        train_labels = [int(example["labels"]) for example in tokenized_dataset["train"]]
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
        loss_weights = torch.tensor(class_weights, dtype=torch.float)
        
        logger.info(f"Class weights - Not Frustrated: {class_weights[0]:.3f}, Frustrated: {class_weights[1]:.3f}")
        
        # Store class weights for use in training
        self.class_weights = loss_weights
        
    def compute_metrics(self, pred):
        """
        Compute evaluation metrics for the model.
        
        Args:
            pred: Predictions object from Trainer
            
        Returns:
            dict: Dictionary containing accuracy, F1, precision, and recall
        """
        labels = pred.label_ids.flatten()
        # For binary classification with single output, apply sigmoid and threshold
        preds = (pred.predictions > 0.0).astype(int).flatten()
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, tokenized_dataset, output_dir="./frustration_detection_results"):
        """
        Train the frustration detection model.
        
        Args:
            tokenized_dataset: Preprocessed dataset
            output_dir (str): Directory to save training results
        """
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=4,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            report_to=None,  # Disable wandb/tensorboard logging
        )
        
        logger.info("Initializing trainer...")
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        logger.info("Starting training...")
        self.trainer.train()
        
        logger.info("Training completed!")
        
    def evaluate(self, tokenized_dataset):
        """
        Evaluate the trained model on the test set.
        
        Args:
            tokenized_dataset: Preprocessed dataset
            
        Returns:
            dict: Evaluation results
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation. Call train() first.")
            
        logger.info("Evaluating on test set...")
        results = self.trainer.evaluate(tokenized_dataset["test"])
        
        logger.info("Test Results:")
        for key, value in results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '')
                logger.info(f"  {metric_name}: {value:.4f}")
                
        return results
    
    def predict(self, text):
        """
        Predict frustration for a single text input.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Prediction results with label and confidence
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be trained before prediction. Call train() first.")
            
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply sigmoid to get probability for binary classification
            probability = torch.sigmoid(outputs.logits).item()
            predicted_class = int(probability > 0.5)
        
        return {
            "label": "frustrated" if predicted_class == 1 else "not_frustrated",
            "confidence": probability if predicted_class == 1 else (1 - probability),
            "probabilities": {
                "not_frustrated": 1 - probability,
                "frustrated": probability
            }
        }

def main():
    """Main function to run the complete frustration detection pipeline."""
    
    # Initialize detector
    detector = FrustrationDetector()
    
    try:
        # Load and preprocess dataset
        dataset = detector.load_dataset()
        tokenized_dataset = detector.preprocess_dataset(dataset)
        
        # Setup model
        detector.setup_model(tokenized_dataset)
        
        # Train model
        detector.train(tokenized_dataset)
        
        # Evaluate model
        results = detector.evaluate(tokenized_dataset)
        
        logger.info("Training and evaluation completed successfully!")
        logger.info("Use test_frustration_detection.py to test predictions with the trained model.")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
