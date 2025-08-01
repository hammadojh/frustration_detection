# Frustration Detection Research Project

A comprehensive machine learning research project for detecting user frustration in human-computer interactions using transformer models and engineered features.

## 🎯 Project Overview

This repository contains a systematic study of frustration detection combining **RoBERTa transformer models** with **engineered linguistic features** derived from academic literature. We conducted 68+ experiments to identify optimal feature combinations for predicting user frustration in conversational AI systems.

### Key Achievements
- **📊 Best F1 Score**: 0.8831 (vs 0.8108 baseline)
- **📚 Literature Review**: 50+ academic papers analyzed
- **🔧 Feature Engineering**: 22 features across 7 conceptual bundles
- **🧪 Comprehensive Study**: Individual + bundle + ablation experiments
- **📋 Research Datasets**: Complete feature-literature mapping with 151 citations

## 🏗️ Repository Structure

```
frustration_researcher/
├── 📁 experiments/           # ML experiments organized by phases
│   ├── phase1_baseline/      # RoBERTa baseline (F1: 0.8108)
│   ├── phase2_features/      # Feature engineering implementation
│   └── phase3_advanced/      # Ablation studies and optimization
├── 📁 data/                  # Research datasets and experiment data
│   ├── research_datasets/    # 📊 Core research data (features, papers, mappings)
│   ├── subset_processed.json # EmoWOZ experiment subset (500 examples)
│   └── subset_*.json        # Additional dataset variants
├── 📁 results/               # Experiment results and analysis
├── 📁 docs/                  # 📖 Organized documentation
│   ├── methodology/          # Research methods and implementation
│   ├── experiments/          # Experiment guides and analysis
│   ├── papers/              # Literature review and synthesis
│   ├── project/             # Project management and overview
│   └── README.md            # Documentation navigation guide
└── 📁 downloaded_papers/     # Academic papers (50+ PDFs)
```

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/hammadojh/frustration_detection.git
cd frustration_researcher

# Install dependencies (requirements.txt recommended)
pip install torch transformers datasets sklearn numpy pandas
```

### 2. Run Baseline Experiment
```bash
cd experiments/phase1_baseline/
python run_baseline.py
```

### 3. Run Feature-Enhanced Model
```bash
cd experiments/phase2_features/
python simple_experiment.py
```

### 4. View Results
```bash
# Check latest results
cat results/comprehensive_ablation_study_REPORT.txt

# View feature analysis
cat experiments/phase2_features/feature_modeling_report.md
```

## 🧪 Experimental Phases

### Phase 1: Baseline Establishment
- **Model**: RoBERTa-base-go_emotions
- **Data**: EmoWOZ subset (500 balanced examples)
- **Result**: F1 = 0.8108 (text-only baseline)
- **Status**: ✅ Complete

### Phase 2: Feature Engineering
- **Features**: 22 engineered features from academic literature
- **Architecture**: Post-embedding concatenation with RoBERTa [CLS]
- **Best Single Bundle**: Linguistic (+1.12% F1 improvement)
- **Status**: ✅ Complete

### Phase 3: Optimization & Ablation
- **Experiments**: 68 total (individual, bundle, top-K, leave-one-out)
- **Best Configuration**: Early fusion with top features
- **Result**: F1 = 0.8831 (optimal feature combination)
- **Status**: ✅ Complete

## 📊 Key Results Summary

### Most Effective Features
1. **`avg_politeness`** - Politeness level decline over conversation
2. **`avg_exclamation`** - Exclamation mark density  
3. **`response_clarity`** - System response clarity proxy
4. **`sentiment_slope`** - Sentiment trajectory over turns
5. **`avg_negation`** - Negation word frequency

### Feature Bundle Performance
| Bundle | Features | F1 Impact | Status |
|--------|----------|-----------|---------|
| **Linguistic** | 8 | +1.12% | ✅ Effective |
| **System** | 2 | +1.12% | ✅ Effective |
| **User Model** | 1 | +1.12% | ✅ Effective |
| Dialogue | 4 | -1.09% | ❌ Harmful |
| Contextual | 3 | -0.47% | ❌ Harmful |
| Behavioral | 2 | 0.00% | ⚪ Neutral |
| Emotion Dynamics | 2 | 0.00% | ⚪ Neutral |

### Performance vs Feature Count
- **3 features**: F1 = 0.85+ (efficient)
- **8 features**: F1 = 0.88+ (balanced)
- **22 features**: F1 = 0.8831 (maximum)

## 📊 Core Research Datasets

The **[data/research_datasets/](data/research_datasets/)** folder contains comprehensive research data linking academic literature to implementable features:

- **[features.csv](data/research_datasets/features.csv)** - 45 features with implementation guidance and dataset applicability
- **[paper_features.csv](data/research_datasets/paper_features.csv)** - 57 academic papers with detailed analysis and results
- **[feature_papers.csv](data/research_datasets/feature_papers.csv)** - 151 feature-to-paper mappings with citations
- **[BERT_Frustration_Feature_Justification.csv](data/research_datasets/BERT_Frustration_Feature_Justification.csv)** - Expert analysis of which features help BERT

*See [data/research_datasets/README.md](data/research_datasets/README.md) for detailed documentation.*

## 📚 Documentation Guide

**📖 [Complete Documentation Index](docs/README.md)** - Start here for full navigation

Our documentation is organized by audience and purpose:

### For Researchers & Data Scientists
- **[Experimental Methodology](docs/methodology/experimental_methodology.md)** - Complete research approach
- **[Feature Implementation](docs/methodology/feature_modeling_report.md)** - How each feature is computed
- **[Literature Review](docs/papers/papers_summary.md)** - Academic research synthesis
- **[Paper Analysis Method](docs/methodology/paper_analysis_method.md)** - Literature review methodology

### For ML Engineers & Developers
- **[Ablation Study Guide](docs/experiments/ablation_study_guide.md)** - Comprehensive feature analysis
- **[Model Architecture](docs/methodology/feature_modeling_report.md#feature-integration-architecture)** - Technical implementation
- **[Results Analysis](results/)** - Raw experiment data and reports

### For Project Management
- **[Project Overview](docs/project/project_overview.md)** - High-level status and achievements
- **[Progress Tracking](docs/project/checkpoints.md)** - Detailed experiment log

## 🔍 Key Insights

### What Works for Frustration Detection
1. **Linguistic markers** are most predictive (politeness, sentiment, emphasis)
2. **System interaction quality** matters (clarity, relevance)
3. **Trust indicators** provide valuable signals
4. **Post-embedding feature fusion** outperforms pre-processing approaches

### What Doesn't Work
1. **Dialogue structure features** can hurt performance
2. **Complex contextual features** add noise without benefit
3. **Emotion dynamics** show limited predictive power in our setup

### Technical Learnings
- **Feature scaling** is critical for post-embedding concatenation
- **Early fusion** works better than late fusion for this task
- **3-8 features** provide the best efficiency/performance tradeoff
- **RoBERTa-base** captures sufficient semantic information for the task

## 📈 Reproduction & Extension

### Reproduce Main Results
```bash
# Run the complete pipeline
cd experiments/phase3_advanced/
python comprehensive_ablation_study.py

# Expected runtime: ~60-90 minutes
# Expected output: F1 = 0.8831 ± 0.01
```

### Extend the Research
- **New features**: Add your features to `features.csv`
- **New datasets**: Modify data loaders in `experiments/phase1_baseline/`
- **New architectures**: Extend `experiments/phase2_features/enhanced_model.py`

## 🤝 Contributing

1. **Feature requests**: Add new linguistic/behavioral features
2. **Dataset expansion**: Test on larger or different datasets  
3. **Architecture improvements**: Experiment with different fusion strategies
4. **Documentation**: Improve guides and add examples

## 📜 Citation

If you use this work, please cite:
```bibtex
@misc{frustration_detection_2025,
  title={Comprehensive Feature Engineering for Frustration Detection in Conversational AI},
  author={Hammad, Omar},
  year={2025},
  url={https://github.com/hammadojh/frustration_detection}
}
```

## 📞 Contact

- **Repository**: [github.com/hammadojh/frustration_detection](https://github.com/hammadojh/frustration_detection)
- **Issues**: Please use GitHub Issues for questions and bug reports
- **Documentation**: See `docs/` folder for detailed guides

---

**📋 Status**: Research Complete | **🎯 Best F1**: 0.8831 | **📚 Papers Analyzed**: 50+ | **🧪 Experiments**: 68+