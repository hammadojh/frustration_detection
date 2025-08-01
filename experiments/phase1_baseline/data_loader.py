#!/usr/bin/env python3
"""
Phase 1: Data Loading for EmoWOZ Dataset
Load EmoWOZ dataset and create 500 sample subset for fast experimentation
"""

import os
import json
import torch
import pandas as pd
from datasets import load_dataset
from transformers import RobertaTokenizerFast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmoWOZDataLoader:
    """Load and preprocess EmoWOZ dataset for frustration detection"""
    
    def __init__(self, subset_size=500, max_length=512):
        self.subset_size = subset_size
        self.max_length = max_length
        self.tokenizer = None
        
        # Map EmoWOZ emotion IDs to binary frustration labels
        # Based on EmoWOZ: 0=neutral, 1=excited, 2=dissatisfied, 3=satisfied, 4=apologetic, 5=abusive, 6=fearful
        self.FRUSTRATED_EMOTION_IDS = {2, 5, 6}  # dissatisfied, abusive, fearful
    
    def load_emowoz_dataset(self):
        """Load EmoWOZ dataset from HuggingFace"""
        logger.info("Loading EmoWOZ dataset...")
        try:
            dataset = load_dataset("hhu-dsml/emowoz")
            logger.info(f"Dataset loaded. Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load EmoWOZ dataset: {e}")
            raise
    
    def create_subset(self, dataset):
        """Create balanced subset of specified size"""
        logger.info(f"Creating balanced subset of {self.subset_size} examples...")
        
        # Get train split and extract turn-level examples
        train_data = dataset['train']
        
        # Extract turn-level examples with emotions
        frustrated_examples = []
        not_frustrated_examples = []
        
        for dialogue in train_data:
            texts = dialogue['log']['text']
            emotions = dialogue['log']['emotion']
            
            for text, emotion_id in zip(texts, emotions):
                if emotion_id == -1:  # Skip system turns
                    continue
                    
                example = {
                    'text': text,
                    'emotion_id': emotion_id,
                    'dialogue_id': dialogue['dialogue_id']
                }
                
                if emotion_id in self.FRUSTRATED_EMOTION_IDS:
                    frustrated_examples.append(example)
                else:
                    not_frustrated_examples.append(example)
        
        # Balance the subset
        target_frustrated = min(self.subset_size // 2, len(frustrated_examples))
        target_not_frustrated = min(self.subset_size - target_frustrated, len(not_frustrated_examples))
        
        import random
        
        # Shuffle examples to ensure good mixing
        random.shuffle(frustrated_examples)
        random.shuffle(not_frustrated_examples)
        
        subset_examples = (
            frustrated_examples[:target_frustrated] + 
            not_frustrated_examples[:target_not_frustrated]
        )
        
        # Shuffle the final subset to mix classes
        random.shuffle(subset_examples)
        
        logger.info(f"Subset created: {target_frustrated} frustrated, {target_not_frustrated} not frustrated")
        logger.info(f"Total frustrated available: {len(frustrated_examples)}, not frustrated: {len(not_frustrated_examples)}")
        return subset_examples
    
    def preprocess_example(self, example):
        """Convert example to model input format"""
        # Get text and emotion from our processed example
        text = example.get('text', '')
        emotion_id = example.get('emotion_id', 0)
        
        # Map emotion ID to binary label
        label = int(emotion_id in self.FRUSTRATED_EMOTION_IDS)
        
        return {
            'text': text,
            'label': label,
            'emotion_id': emotion_id
        }
    
    def tokenize_dataset(self, examples):
        """Tokenize dataset for model training"""
        if self.tokenizer is None:
            logger.info("Loading tokenizer...")
            self.tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
        
        processed_examples = [self.preprocess_example(ex) for ex in examples]
        
        # Tokenize texts
        texts = [ex['text'] for ex in processed_examples]
        labels = [ex['label'] for ex in processed_examples]
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Add labels
        tokenized['labels'] = torch.tensor(labels, dtype=torch.float)
        
        return tokenized, processed_examples
    
    def save_subset(self, examples, output_dir):
        """Save subset for reproducibility"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw examples
        with open(os.path.join(output_dir, 'subset_raw.json'), 'w') as f:
            json.dump(examples, f, indent=2)
        
        # Save processed examples
        processed = [self.preprocess_example(ex) for ex in examples]
        with open(os.path.join(output_dir, 'subset_processed.json'), 'w') as f:
            json.dump(processed, f, indent=2)
        
        # Save statistics
        emotion_ids = [ex.get('emotion_id', 0) for ex in examples]
        stats = {
            'total_examples': len(examples),
            'emotion_id_distribution': pd.Series(emotion_ids).value_counts().to_dict(),
            'frustrated_count': sum(1 for ex in processed if ex['label'] == 1),
            'not_frustrated_count': sum(1 for ex in processed if ex['label'] == 0)
        }
        
        with open(os.path.join(output_dir, 'subset_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Subset saved to {output_dir}")
        logger.info(f"Statistics: {stats}")

def main():
    """Main function to create dataset subset"""
    loader = EmoWOZDataLoader(subset_size=500)
    
    # Load dataset
    dataset = loader.load_emowoz_dataset()
    
    # Create subset
    subset = loader.create_subset(dataset)
    
    # Save subset
    output_dir = "/Users/omarhammad/Documents/code_local/frustration_researcher/data"
    loader.save_subset(subset, output_dir)
    
    # Tokenize for training
    tokenized, processed = loader.tokenize_dataset(subset)
    
    # Save tokenized data
    torch.save(tokenized, os.path.join(output_dir, 'tokenized_subset.pt'))
    
    logger.info("Phase 1 data loading completed successfully!")

if __name__ == "__main__":
    main()