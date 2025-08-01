# Documentation Index

Welcome to the Frustration Detection Research documentation. This folder contains comprehensive guides organized by audience and purpose.

## 📂 Documentation Structure

### 📊 For Researchers & Data Scientists
- **[Feature Modeling Report](methodology/feature_modeling_report.md)** - Detailed implementation of all 22 features
- **[ML Experiment Plan](methodology/ml_experiment_plan.md)** - Systematic 3-phase experimental approach  
- **[ML Training Blueprint](methodology/ml_training.md)** - Efficient experimentation framework
- **[Paper Analysis Method](methodology/paper_analysis_method.md)** - Literature review methodology

### 🧪 For ML Engineers & Developers
- **[Ablation Study Guide](experiments/ablation_study_guide.md)** - Comprehensive feature importance analysis
- **[Model Architecture](methodology/feature_modeling_report.md#feature-integration-architecture)** - Technical implementation details
- **[Experiment Results](../results/)** - Raw data and performance metrics

### 📚 For Academic Research
- **[Papers Summary](papers/papers_summary.md)** - Synthesis of 50+ academic papers
- **[Research Datasets](../data/research_datasets/)** - Feature-literature mappings and analysis
- **[Downloaded Papers](../downloaded_papers/)** - Full academic paper collection (50+ PDFs)

### 📋 For Project Management
- **[Checkpoints](project/checkpoints.md)** - Detailed experiment progress log
- **[Project Overview](project/project_overview.md)** - High-level status and milestones

## 🎯 Quick Navigation by Task

### "I want to understand how features work"
→ Start with **[Feature Modeling Report](methodology/feature_modeling_report.md)**

### "I want to reproduce the experiments"  
→ See **[Experimental Methodology](methodology/experimental_methodology.md)** and **[Ablation Study Guide](experiments/ablation_study_guide.md)**

### "I want to understand the academic background"
→ Read **[Papers Summary](papers/papers_summary.md)** and explore **[Research Datasets](../data/research_datasets/)**

### "I want to extend this research"
→ Study **[Feature Implementation](methodology/feature_modeling_report.md)** and **[Experimental Methodology](methodology/experimental_methodology.md)**

### "I want to see the current status"
→ Check **[Checkpoints](project/checkpoints.md)** and **[Results](../results/)**

## 📊 Key Results Quick Reference

- **Best F1 Score**: 0.8831 (vs 0.8108 baseline)
- **Most Effective Features**: `avg_politeness`, `avg_exclamation`, `response_clarity`  
- **Best Bundles**: Linguistic (+1.12%), System (+1.12%), User Model (+1.12%)
- **Optimal Feature Count**: 8-12 features for best efficiency/performance tradeoff
- **Total Experiments**: 68+ systematic tests

## 🔄 Documentation Updates

This documentation is actively maintained. Last updated with the comprehensive ablation study results and feature importance rankings.

**Status**: ✅ Research Complete | **Coverage**: 50+ papers | **Experiments**: 68+