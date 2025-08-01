# Research Datasets

This folder contains the core research data files that define features, map them to academic literature, and provide implementation guidance.

## 📊 Dataset Files

### 1. **features.csv** (45 features)
**Purpose**: Core feature definitions with implementation guidance  
**Columns**: `feature`, `category`, `description`, `example`, `how_to_model`, `valid_dataset`, `modeled_literature`

**Content**:
- 22 features actually implemented in experiments
- 7 conceptual bundles: linguistic, dialogue, behavioral, contextual, emotion_dynamics, system, user_model
- Implementation approaches for each feature
- Dataset applicability across 16+ dialogue datasets
- Literature citations for each feature

### 2. **paper_features.csv** (57 papers)
**Purpose**: Maps academic papers to their discussed features  
**Columns**: `paper`, `predict_features`, `detect_features`, `download_status`, `paper_year`, `training_dataset(s)`, `testing_dataset(s)`, `architecture_summary`, `results_summary`, `limitations_summary`

**Content**:
- 50+ academic papers analyzed
- Features categorized as prediction vs detection
- Complete bibliographic information
- Dataset usage across papers
- Research methodology summaries
- Performance results and limitations

### 3. **feature_papers.csv** (151 mappings)
**Purpose**: Reverse mapping from features to source literature  
**Columns**: `feature`, `citation`, `emotion`, `predict_or_detect`, `link_valid`

**Content**:
- Direct feature-to-paper mappings
- Emotion categories (frustration, satisfaction, etc.)
- Classification of prediction vs detection tasks
- Link validation status for reproducibility

### 4. **BERT_Frustration_Feature_Justification.csv** (46 analyzed features)
**Purpose**: Analysis of which features help BERT predict frustration  
**Columns**: `feature`, `category`, `description`, `example`, `how_to_model`, `helps_bert_predict_frustration`, `justification`

**Content**:
- Expert analysis of BERT's capabilities vs feature necessity
- Justifications for why certain features help or don't help
- Implementation complexity assessments
- Practical deployment considerations

## 🎯 Usage

### For Feature Implementation
```python
import pandas as pd

# Load feature definitions
features_df = pd.read_csv('data/research_datasets/features.csv')

# Get implementation guidance for a specific feature
politeness_info = features_df[features_df['feature'] == 'politeness_level']
print(politeness_info['how_to_model'].iloc[0])
```

### For Literature Review
```python
# Load paper mappings
papers_df = pd.read_csv('data/research_datasets/paper_features.csv')

# Find papers that predict frustration
prediction_papers = papers_df[papers_df['predict_features'].str.contains('frustration', na=False)]
```

### For BERT Enhancement Analysis
```python
# Load BERT justification analysis
bert_analysis = pd.read_csv('data/research_datasets/BERT_Frustration_Feature_Justification.csv')

# Find features that help BERT
helpful_features = bert_analysis[bert_analysis['helps_bert_predict_frustration'] == 'yes']
```

## 📈 Key Statistics

- **Total Features Analyzed**: 45
- **Features Implemented**: 22  
- **Academic Papers Reviewed**: 50+
- **Feature-Paper Mappings**: 151
- **Dataset Coverage**: 16+ dialogue datasets
- **Research Span**: 2001-2025

## 🔗 Related Documentation

- **[Feature Implementation](../../docs/methodology/feature_modeling_report.md)** - How features were coded
- **[Literature Review](../../docs/papers/papers_summary.md)** - Academic paper synthesis  
- **[Experimental Results](../../results/)** - Performance impact of features
- **[Project Overview](../../docs/project/project_overview.md)** - High-level research summary

## 📋 File Relationships

```
features.csv ←→ feature_papers.csv ←→ paper_features.csv
     ↑                                      ↑
     └── BERT_Frustration_Feature_Justification.csv
```

These files form a comprehensive research knowledge base linking academic literature to implementable features with practical guidance for frustration detection systems.