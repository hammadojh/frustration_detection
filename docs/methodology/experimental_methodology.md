# Experimental Methodology: Frustration Detection

*Consolidated methodology document combining systematic experimentation with efficiency-focused optimization*

## 🎯 Objective
Systematically identify the optimal combination of engineered features that, when combined with RoBERTa, yields the highest performance in detecting user frustration while maintaining practical deployment efficiency.

## 📋 Methodology Overview

Our approach balanced **empirical rigor** with **practical constraints** (compute time, feature overhead, deployment latency). We conducted 68+ systematic experiments across three phases to answer the core question: **"What actually helps RoBERTa predict frustration?"**

---

## 🏗️ Phase 1: Baseline Establishment

### Architecture Foundation
- **Model**: `RobertaForSequenceClassification` using `SamLowe/roberta-base-go_emotions`
- **Configuration**: Single output node (`num_labels=1`) for binary classification
- **Loss Function**: BCEWithLogitsLoss with class-balanced weights
- **Training**: 4 epochs, learning rate 2e-5, batch size 16

### Dataset Preparation  
- **Source**: EmoWOZ dataset (hhu-dsml/emowoz)
- **Subset**: 500 balanced examples (250 frustrated, 250 not frustrated)
- **Labels**: "dissatisfied", "abusive", "fearful" → frustration (binary)
- **Split**: 350 train, 75 validation, 75 test

### Baseline Results
- **F1 Score**: 0.8108
- **Precision**: 0.8108  
- **Recall**: 0.8108
- **Accuracy**: 0.8133

---

## 🔬 Phase 2: Feature Engineering & Integration

### Feature Bundle Definition
We organized 22 engineered features into 7 conceptual bundles for systematic testing:

| Bundle | Features | Count | Rationale |
|--------|----------|-------|-----------|
| **Linguistic** | sentiment_trajectory, politeness_level, intent_repetition, directness_abruptness, confusion_lexical_markers, negation_frequency, emphasis_capitalization, exclamation_density | 8 | Direct linguistic markers of frustration |
| **Dialogue** | system_failures, repeated_turns, conversation_length, user_corrections | 4 | Structural conversation patterns |
| **Behavioral** | escalation_request, negative_feedback | 2 | User actions indicating frustration |
| **Contextual** | task_complexity, expressed_urgency, goal_completion_status | 3 | Situational factors |
| **Emotion Dynamics** | emotion_drift, emotion_volatility | 2 | Emotional state changes over time |
| **System** | response_clarity, response_relevance | 2 | System interaction quality |
| **User Model** | trust_in_system | 1 | User confidence indicators |

### Feature Integration Architecture

**Critical Innovation**: Post-embedding concatenation rather than pre-processing integration.

```
Text → RoBERTa → [CLS] embedding (768d)
    ↓
Features → Extraction → Scaling → Feature vector (N-d)  
    ↓
Concatenation: [768d RoBERTa] + [N-d features] = [768+N total]
    ↓
LogisticRegression classifier
```

**Key Implementation Details**:
1. **No feature scaling for RoBERTa embeddings** - Use as-is from transformer
2. **StandardScaler for engineered features** - Essential for proper fusion
3. **Horizontal concatenation** - `np.hstack()` for combined representation
4. **Simple classification head** - LogisticRegression on combined vectors

### Bundle Testing Results
- **Linguistic Bundle**: +1.12% F1 improvement ✅
- **System Bundle**: +1.12% F1 improvement ✅  
- **User Model Bundle**: +1.12% F1 improvement ✅
- **Dialogue Bundle**: -1.09% F1 (harmful) ❌
- **Contextual Bundle**: -0.47% F1 (harmful) ❌
- **Behavioral Bundle**: 0.00% F1 (neutral) ⚪
- **Emotion Dynamics Bundle**: 0.00% F1 (neutral) ⚪

---

## 🎯 Phase 3: Comprehensive Optimization

### Experimental Design (68 total experiments)

#### 1. Individual Feature Importance (22 experiments)
- Test each feature alone with early fusion
- **Purpose**: Identify most powerful individual contributors
- **Output**: Complete feature ranking

#### 2. Bundle-Level Analysis (7 experiments)  
- Test each bundle separately
- **Purpose**: Understand conceptual group value
- **Insight**: Linguistic features dominate

#### 3. Top-K Feature Combinations (6 experiments)
- Test combinations of top 3, 5, 8, 10, 12, 15 features
- **Purpose**: Find efficiency/performance sweet spot
- **Finding**: 8-12 features optimal

#### 4. Leave-One-Out Ablation (22 experiments)
- Remove each feature from full 22-feature set
- **Purpose**: Identify most critical features
- **Method**: Largest performance drops = most important

#### 5. Cumulative Addition (10 experiments)
- Add features incrementally by importance ranking
- **Purpose**: Understand diminishing returns
- **Finding**: Sharp gains initially, plateau after 12 features

#### 6. Baseline Comparison (1 experiment)
- Text-only performance for reference
- **Result**: Confirms feature engineering value

### Optimization Results
- **Best Configuration**: F1 = 0.8831 (+8.9% vs baseline)
- **Most Important Features**: `avg_politeness`, `avg_exclamation`, `response_clarity`
- **Optimal Feature Count**: 8-12 for balanced performance/efficiency
- **Diminishing Returns**: Minimal gains beyond 12 features

---

## 🔍 Methodological Innovations

### 1. **Post-Embedding Feature Fusion**
Unlike typical approaches that modify input text, we concatenate features with already-computed RoBERTa embeddings. This preserves transformer representations while adding structured signals.

### 2. **Systematic Bundle Testing**
Rather than ad-hoc feature combinations, we tested conceptually grouped bundles to understand which domains (linguistic, dialogue, etc.) contribute most.

### 3. **Comprehensive Ablation Framework**
68 experiments across multiple ablation strategies (individual, bundle, top-K, leave-one-out) provided complete feature importance mapping.

### 4. **Efficiency-Focused Evaluation**
Considered not just performance but deployment practicality, identifying minimal feature sets that retain most benefits.

### 5. **Negative Results Documentation**
Explicitly documented harmful features (dialogue bundle) to guide future research away from unproductive paths.

---

## 📊 Key Methodological Findings

### What Works
1. **Post-embedding concatenation** > pre-processing integration
2. **Simple features** often > complex engineered features  
3. **Linguistic markers** > structural dialogue patterns
4. **System interaction quality** highly predictive
5. **Feature scaling critical** for successful fusion

### What Doesn't Work
1. **Complex dialogue structure features** can hurt performance
2. **Emotion dynamics** show limited predictive power
3. **Behavioral features** provide minimal benefit in text-only setting
4. **More features ≠ better performance** beyond optimal count

### Implementation Lessons
1. **Careful feature scaling** essential for multi-modal fusion
2. **Bundle-wise testing** more informative than individual feature testing
3. **Ablation studies crucial** for understanding true contributions
4. **Efficiency considerations** should guide feature selection

---

## 🔄 Reproducibility & Extension

### Reproduction Steps
1. Load EmoWOZ subset with specified train/val/test split
2. Implement 22 features using provided extraction code
3. Train RoBERTa baseline with specified hyperparameters
4. Run systematic ablation experiments with early fusion architecture
5. Analyze results using provided evaluation framework

### Extension Opportunities
1. **Scale to full EmoWOZ dataset** (11K+ dialogues)
2. **Test on other conversation datasets** (MultiWOZ, SGD)
3. **Experiment with other fusion strategies** (late fusion, attention)
4. **Add new feature categories** (temporal, multi-modal)
5. **Optimize for production deployment** (model distillation, quantization)

---

## 📈 Success Metrics

**Primary**: F1 score improvement over text-only baseline  
**Secondary**: Precision, recall, efficiency (features needed)  
**Tertiary**: Interpretability, deployment feasibility

**Final Achievement**: 8.9% F1 improvement with clear feature importance ranking and practical deployment guidance.

This methodology successfully balanced research rigor with practical applicability, providing both academic insights and deployment-ready recommendations.