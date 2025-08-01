#!/usr/bin/env python3
"""
Phase 1: Run Complete Baseline Experiment
Execute data loading and baseline model training
"""

import sys
import os
import logging
from data_loader import main as load_data
from baseline_model import main as train_baseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_phase1():
    """Run complete Phase 1: baseline experiment"""
    logger.info("="*50)
    logger.info("PHASE 1: BASELINE EXPERIMENT")
    logger.info("="*50)
    
    try:
        # Step 1: Load and prepare data
        logger.info("Step 1: Loading EmoWOZ data...")
        load_data()
        
        # Step 2: Train baseline model
        logger.info("Step 2: Training baseline model...")
        results = train_baseline()
        
        logger.info("Phase 1 completed successfully!")
        logger.info(f"Baseline F1 Score: {results['f1_score']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        raise

if __name__ == "__main__":
    run_phase1()