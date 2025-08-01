#!/usr/bin/env python3
"""
Phase 2: Run Feature Bundle Experiments
Execute additive feature testing following the ML experiment plan
"""

import os
import json
import pandas as pd
from enhanced_model import EnhancedFrustrationDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExperimentRunner:
    """Run systematic feature bundle experiments"""
    
    def __init__(self):
        self.feature_bundles = [
            'linguistic_bundle',
            'dialogue_bundle', 
            'behavioral_bundle',
            'contextual_bundle',
            'emotion_dynamics_bundle',
            'system_bundle',
            'user_model_bundle'
        ]
        
        self.data_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/data/tokenized_subset.pt"
        self.processed_examples_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/data/subset_processed.json"
        self.output_dir = "/Users/omarhammad/Documents/code_local/frustration_researcher/experiments/phase2_features/output"
        self.results_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/results/results.csv"
        
        # Initialize results tracking
        self.results = []
        self.current_best_f1 = 0.8108  # Baseline F1 score
        self.current_best_features = []
        
        # Load baseline results
        self._load_baseline_results()
    
    def _load_baseline_results(self):
        """Load baseline results to start comparison"""
        baseline_result = {
            'experiment_name': 'baseline',
            'features_used': 'none',
            'f1_score': 0.8108,
            'precision': 0.8108,
            'recall': 0.8108,
            'accuracy': 0.8133,
            'notes': 'Baseline RoBERTa without engineered features'
        }
        self.results.append(baseline_result)
        logger.info(f"Baseline F1 Score: {self.current_best_f1:.4f}")
    
    def run_experiment(self, features_to_test, experiment_name):
        """Run single experiment with specified features"""
        logger.info(f"Running experiment: {experiment_name}")
        logger.info(f"Features to test: {features_to_test}")
        
        try:
            # Initialize detector
            detector = EnhancedFrustrationDetector()
            
            # Load data with features
            train_data, val_data, test_data, num_features = detector.load_data_with_features(
                self.data_path, self.processed_examples_path, features_to_test
            )
            
            # Setup model
            detector.setup_model(num_features)
            
            # Train model
            experiment_output_dir = os.path.join(self.output_dir, experiment_name)
            os.makedirs(experiment_output_dir, exist_ok=True)
            
            detector.train(train_data, val_data, experiment_output_dir, experiment_name)
            
            # Evaluate model
            features_used_str = ','.join(features_to_test)
            results = detector.evaluate(test_data, experiment_output_dir, experiment_name, features_used_str)
            
            # Save model
            detector.save_model(experiment_output_dir, experiment_name)
            
            return results
            
        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed: {e}")
            return None
    
    def run_additive_experiments(self):
        """Run additive feature bundle experiments following the ML plan"""
        logger.info("="*60)
        logger.info("STARTING ADDITIVE FEATURE BUNDLE EXPERIMENTS")
        logger.info("="*60)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tracking
        current_best_features = []
        current_best_f1 = self.current_best_f1
        
        # Test each bundle additively
        for i, bundle in enumerate(self.feature_bundles):
            logger.info(f"\n--- Testing Bundle {i+1}/{len(self.feature_bundles)}: {bundle} ---")
            
            # Prepare features to test: current best + new bundle
            features_to_test = current_best_features + [bundle]
            experiment_name = f"run_{i+1}_{bundle}"
            
            # Run experiment
            results = self.run_experiment(features_to_test, experiment_name)
            
            if results is None:
                logger.warning(f"Skipping {bundle} due to experiment failure")
                continue
            
            # Add to results
            self.results.append(results)
            
            # Compare with current best
            new_f1 = results['f1_score']
            logger.info(f"New F1: {new_f1:.4f}, Current Best: {current_best_f1:.4f}")
            
            if new_f1 > current_best_f1:
                # Update best performance
                current_best_f1 = new_f1
                current_best_features = features_to_test.copy()
                logger.info(f"✅ IMPROVEMENT! New best F1: {current_best_f1:.4f}")
                logger.info(f"Best features: {current_best_features}")
                
                # Update checkpoints
                self._update_checkpoints(experiment_name, current_best_features, current_best_f1)
            else:
                logger.info(f"❌ No improvement. {bundle} did not help.")
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("ADDITIVE EXPERIMENTS COMPLETED")
        logger.info("="*60)
        logger.info(f"Final best F1 score: {current_best_f1:.4f}")
        logger.info(f"Final best features: {current_best_features}")
        logger.info(f"Improvement over baseline: {(current_best_f1 - self.current_best_f1):.4f}")
        
        # Save all results
        self._save_results()
        
        return current_best_features, current_best_f1
    
    def _update_checkpoints(self, experiment_name, best_features, best_f1):
        """Update checkpoints with current progress"""
        checkpoint_path = "/Users/omarhammad/Documents/code_local/frustration_researcher/checkpoints.md"
        
        # Read current checkpoints
        try:
            with open(checkpoint_path, 'r') as f:
                content = f.read()
            
            # Update the feature bundle results table
            new_row = f"| {experiment_name} | {','.join(best_features)} | {best_f1:.4f} | TBD | TBD | Feature bundle experiment |"
            
            # Find the table and add the new row
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '| baseline | none |' in line:
                    # Insert new row after baseline
                    lines.insert(i + 1, new_row)
                    break
            
            # Write back
            with open(checkpoint_path, 'w') as f:
                f.write('\n'.join(lines))
                
            logger.info("✅ Checkpoints updated")
            
        except Exception as e:
            logger.warning(f"Could not update checkpoints: {e}")
    
    def _save_results(self):
        """Save all experiment results to CSV"""
        # Create results directory
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_path, index=False)
        
        # Also save as JSON for detailed analysis
        results_json_path = self.results_path.replace('.csv', '.json')
        with open(results_json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"✅ Results saved to {self.results_path}")
        logger.info(f"✅ Detailed results saved to {results_json_path}")
        
        # Print summary table
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT RESULTS SUMMARY")
        logger.info("="*80)
        print(df[['experiment_name', 'features_used', 'f1_score', 'precision', 'recall']].to_string(index=False))

def main():
    """Run all feature experiments"""
    runner = FeatureExperimentRunner()
    
    try:
        best_features, best_f1 = runner.run_additive_experiments()
        
        logger.info("\n🎉 PHASE 2 COMPLETED SUCCESSFULLY!")
        logger.info(f"Best feature combination: {best_features}")
        logger.info(f"Best F1 score: {best_f1:.4f}")
        
        return best_features, best_f1
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        raise

if __name__ == "__main__":
    main()