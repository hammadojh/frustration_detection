# Frustration Detection Experiments

## Folder Structure

```
experiments/
├── phase1_baseline/       # Baseline RoBERTa model (no features)
├── phase2_features/       # Feature extraction implementations  
├── phase3_evaluation/     # Model evaluation and comparison
data/                     # Dataset files and preprocessed data
models/                   # Trained model checkpoints
results/                  # Experiment results and logs
```

## Experiment Phases

### Phase 1: Baseline (phase1_baseline/)
- Load EmoWOZ dataset (500 sample subset)
- Train baseline RoBERTa model without engineered features
- Evaluate and record baseline performance
- **Deliverable**: baseline_model.pt, baseline_results.json

### Phase 2: Feature Engineering (phase2_features/)
- Implement feature extraction for 7 bundles
- Create enhanced model architecture with feature concatenation
- Test each bundle individually and cumulatively
- **Deliverable**: feature_extractor.py, enhanced_model.py

### Phase 3: Final Evaluation (phase3_evaluation/)
- Execute full additive feature testing loop
- Perform ablation study on best combination
- Generate final results and report
- **Deliverable**: final_model.pt, experiment_report.md

## Git Commits Strategy
- Commit after each phase completion
- Commit after major experiment milestones
- Include performance metrics in commit messages

## Recovery Instructions
- Each phase is self-contained and can be run independently
- Check `checkpoints.md` for current progress
- Model files and results are saved at each phase