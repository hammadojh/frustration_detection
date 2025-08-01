#!/usr/bin/env python3
"""
Test each feature bundle individually with RoBERTa
"""

from combined_experiment import CombinedTextFeatureExperiment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_individual_bundles():
    """Test each bundle individually with RoBERTa"""
    experiment = CombinedTextFeatureExperiment()
    
    logger.info("="*60)
    logger.info("TESTING INDIVIDUAL BUNDLES WITH ROBERTA")
    logger.info("="*60)
    
    # Test RoBERTa baseline
    baseline = experiment.run_experiment([], 'roberta_baseline')
    baseline_f1 = baseline['f1_score']
    logger.info(f"RoBERTa baseline: F1 = {baseline_f1:.4f}")
    
    results = [baseline]
    
    # Test each bundle individually
    for bundle in experiment.feature_bundles:
        result = experiment.run_experiment([bundle], f'roberta+{bundle}_only')
        results.append(result)
        
        improvement = result['f1_score'] - baseline_f1
        status = "✅" if improvement > 0 else "❌"
        logger.info(f"{status} {bundle}: F1 = {result['f1_score']:.4f} (Δ={improvement:+.4f})")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("INDIVIDUAL BUNDLE SUMMARY")
    logger.info("="*60)
    
    # Sort by F1 score
    results_sorted = sorted(results[1:], key=lambda x: x['f1_score'], reverse=True)
    
    for result in results_sorted:
        improvement = result['f1_score'] - baseline_f1
        bundle_name = result['experiment_name'].replace('roberta+', '').replace('_only', '')
        logger.info(f"{bundle_name}: F1 = {result['f1_score']:.4f} (Δ={improvement:+.4f})")
    
    return results

if __name__ == "__main__":
    test_individual_bundles()