# Feature Modeling Report: Frustration Detection

## Overview
This report documents how each feature was modeled in our frustration detection experiment. We used lightweight proxy implementations optimized for rapid experimentation rather than production accuracy. Each feature uses simple heuristics and lexicon-based approaches to provide fast, interpretable signals.

## Feature Integration Architecture

### How Features Are Combined with RoBERTa

The engineered features are combined with RoBERTa embeddings **AFTER** the text has been processed through RoBERTa, not before. Here's the exact process:

#### Step 1: Extract RoBERTa Embeddings (768 dimensions)
```python
def get_roberta_embeddings(self, texts):
    # Tokenize text
    inputs = self.tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length", 
        max_length=512,
        return_tensors="pt"
    )
    
    # Get RoBERTa embeddings
    outputs = self.roberta(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token (768-dim)
    return cls_embeddings
```

#### Step 2: Extract Engineered Features (varies by bundle)
```python
def extract_features(self, texts, feature_bundles):
    for text in texts:
        feature_dict = {}
        for bundle in feature_bundles:
            bundle_features = self.feature_extractor.extract_bundle_features([text], bundle)
            feature_dict.update(bundle_features)
        all_features.append(list(feature_dict.values()))
    return np.array(all_features)
```

#### Step 3: Scale Features (but NOT RoBERTa embeddings)
```python
# Scale features (but not RoBERTa embeddings)
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)
```

#### Step 4: Concatenate After Embedding
```python
# Combine RoBERTa embeddings with features
train_combined = np.hstack([train_embeddings, train_features_scaled])
test_combined = np.hstack([test_embeddings, test_features_scaled])

# Result: [768 RoBERTa dims] + [N feature dims] = [768+N total dims]
```

### Key Implementation Details:

1. **Post-Embedding Concatenation**: Features are appended to the **already-computed** RoBERTa [CLS] token embeddings (768-dim), not to the raw text input
2. **Differential Scaling**: RoBERTa embeddings are used as-is, while engineered features are standardized using StandardScaler
3. **Horizontal Concatenation**: Uses `np.hstack()` to create vectors like `[rob_1, rob_2, ..., rob_768, feat_1, feat_2, ..., feat_N]`
4. **Final Classification**: A simple LogisticRegression model operates on the combined vector space

### Example Dimensions:
- **Baseline**: 768 dimensions (RoBERTa only)
- **RoBERTa + Linguistic**: 768 + 8 = 776 dimensions  
- **RoBERTa + System**: 768 + 2 = 770 dimensions
- **RoBERTa + User Model**: 768 + 1 = 769 dimensions

This approach allows the engineered features to **augment** rather than **replace** the rich semantic representations learned by RoBERTa, which explains why we see performance improvements rather than degradation.

---

## 1. Linguistic Bundle Features ✅ (Effective: +1.12% F1)

### 1.1 Sentiment Trajectory

**Description**: Tracks how sentiment changes over conversation turns  
**Bundle**: linguistic_bundle

**Code Block**
```python
# Sentiment trajectory proxy: negative word trend
neg_ratios = [self._word_ratio(text, self.negative_words) for text in texts]
features['sentiment_slope'] = np.polyfit(range(len(neg_ratios)), neg_ratios, 1)[0] if len(neg_ratios) > 1 else 0.0
features['sentiment_volatility'] = float(np.var(neg_ratios))
```

**Modeling Approach**: We proxy sentiment using a negative word lexicon (hate, awful, terrible, horrible, stupid, useless, broken, wrong, bad, frustrated, annoyed, angry, mad, upset). For each turn, we calculate the ratio of negative words to total words, then fit a linear slope to detect trending sentiment and compute variance for volatility.

### 1.2 Politeness Level

**Description**: Measures courteous language usage and its decline over time  
**Bundle**: linguistic_bundle

**Code Block**:
```python
# Politeness level proxy: politeness word ratio
pol_ratios = [self._word_ratio(text, self.politeness_words) for text in texts]
features['avg_politeness'] = float(np.mean(pol_ratios))
features['politeness_decline'] = -np.polyfit(range(len(pol_ratios)), pol_ratios, 1)[0] if len(pol_ratios) > 1 else 0.0
```

**Modeling Approach**: Uses a politeness lexicon (please, thank, sorry, excuse, pardon) to compute the ratio of polite words per turn. We track both average politeness and the rate of decline (negative slope), as frustrated users typically become less polite over time.

### 1.3 Confusion Lexical Markers

**Description**: Detects words indicating user confusion  
**Bundle**: linguistic_bundle

**Code Block**:
```python
# Confusion markers proxy
conf_ratios = [self._word_ratio(text, self.confusion_words) for text in texts]
features['avg_confusion'] = float(np.mean(conf_ratios))
```

**Modeling Approach**: Employs a confusion word lexicon (what, why, how, confused, unclear, understand, mean, huh) to identify turns where users express confusion. Calculates the average ratio of confusion words across all turns.

### 1.4 Negation Frequency

**Description**: Frequency of negation words indicating negative sentiment  
**Bundle**: linguistic_bundle

**Code Block**:
```python
# Negation frequency proxy
neg_ratios = [self._word_ratio(text, self.negation_words) for text in texts]
features['avg_negation'] = float(np.mean(neg_ratios))
```

**Modeling Approach**: Uses a negation lexicon (not, no, never, nothing, cant, wont, dont, isnt, wasnt) to compute average negation word density, as frustrated users tend to use more negative constructions.

### 1.5 Emphasis Capitalization

**Description**: Use of ALL CAPS for emphasis indicating frustration  
**Bundle**: linguistic_bundle

**Code Block**:
```python
# Emphasis capitalization proxy: ALL CAPS ratio
caps_ratios = []
for text in texts:
    words = text.split()
    caps_count = sum(1 for word in words if word.isupper() and len(word) > 1)
    caps_ratios.append(caps_count / max(len(words), 1))
features['avg_caps'] = float(np.mean(caps_ratios))
```

**Modeling Approach**: Counts words in ALL CAPS (excluding single characters) as a proxy for emphasis and emotional intensity. Frustrated users often resort to capitalization to express urgency or anger.

### 1.6 Exclamation Density

**Description**: Density of exclamation and question marks  
**Bundle**: linguistic_bundle

**Code Block**:
```python
# Exclamation density proxy
excl_densities = [text.count('!') + text.count('?') for text in texts]
features['avg_exclamation'] = float(np.mean(excl_densities))
```

**Modeling Approach**: Simple count of exclamation marks (!) and question marks (?) per turn, averaged across the conversation. Higher density indicates emotional intensity and questioning behavior common in frustration.

---

## 2. Dialogue Bundle Features ❌ (Harmful: -1.09% F1)

### 2.1 Conversation Length

**Description**: Total turns and average turn length in dialogue  
**Bundle**: dialogue_bundle

**Code Block**:
```python
# Conversation length
features['total_turns'] = float(len(texts))
features['avg_turn_length'] = float(np.mean([len(text.split()) for text in texts])) if texts else 0.0
```

**Modeling Approach**: Counts total conversation turns and computes average words per turn. Longer conversations might indicate more complex problems or system failures leading to frustration.

### 2.2 Repeated Turns

**Description**: Detection of repetitive user attempts  
**Bundle**: dialogue_bundle

**Code Block**:
```python
# Repeated turns proxy: consecutive similar length turns
repeated_count = 0
for i in range(1, len(texts)):
    len_diff = abs(len(texts[i].split()) - len(texts[i-1].split()))
    if len_diff <= 2:  # Similar length heuristic
        repeated_count += 1
features['repeated_turns'] = float(repeated_count / max(len(texts), 1))
```

**Modeling Approach**: Uses a simple heuristic where consecutive turns with similar word counts (within 2 words) are considered "repeated attempts." This proxies for users repeating requests when the system doesn't understand.

### 2.3 User Corrections

**Description**: Patterns where users correct themselves or the system  
**Bundle**: dialogue_bundle

**Code Block**:
```python
# User corrections proxy: correction patterns
correction_patterns = [r'no,?\s*i', r'actually', r'wait', r'i\s*said']
correction_count = sum(self._count_patterns(text, correction_patterns) for text in texts)
features['corrections'] = float(correction_count)
```

**Modeling Approach**: Uses regex patterns to detect correction phrases like "no, I", "actually", "wait", "I said". These indicate the user is correcting previous statements or system misunderstandings.

---

## 3. Behavioral Bundle Features ❌ (Neutral: 0.00% F1)

### 3.1 Escalation Request

**Description**: Explicit requests for human help or escalation  
**Bundle**: behavioral_bundle

**Code Block**:
```python
# Escalation request proxy
escalation_count = sum(self._word_ratio(text, self.escalation_words) > 0 for text in texts)
features['escalation_requests'] = float(escalation_count)
```

**Modeling Approach**: Counts turns containing escalation words (human, person, agent, manager, supervisor, help, support). Any turn with escalation words is flagged, indicating user desire to speak with a human agent.

### 3.2 Negative Feedback

**Description**: Explicit negative feedback about the system  
**Bundle**: behavioral_bundle

**Code Block**:
```python
# Negative feedback proxy
negative_feedback_count = sum(self._word_ratio(text, self.negative_words) > 0.1 for text in texts)
features['negative_feedback'] = float(negative_feedback_count)
```

**Modeling Approach**: Counts turns where negative words comprise more than 10% of the text, indicating strong negative sentiment about the interaction or system performance.

---

## 4. Contextual Bundle Features ❌ (Harmful: -0.47% F1)

### 4.1 Expressed Urgency

**Description**: User expressions of urgency or time pressure  
**Bundle**: contextual_bundle

**Code Block**:
```python
# Expressed urgency proxy
urgency_ratios = [self._word_ratio(text, self.urgency_words) for text in texts]
features['avg_urgency'] = float(np.mean(urgency_ratios))
features['urgency_increase'] = np.polyfit(range(len(urgency_ratios)), urgency_ratios, 1)[0] if len(urgency_ratios) > 1 else 0.0
```

**Modeling Approach**: Uses urgency lexicon (urgent, asap, now, immediately, quickly, fast, need, must) to compute urgency ratios per turn. Tracks both average urgency and whether urgency increases over time.

### 4.2 Task Complexity

**Description**: Estimated complexity based on question density  
**Bundle**: contextual_bundle

**Code Block**:
```python
# Task complexity proxy: question density
question_densities = [text.count('?') / max(len(text.split()), 1) for text in texts]
features['task_complexity'] = float(np.mean(question_densities))
```

**Modeling Approach**: Proxies task complexity using question mark density, assuming more questions indicate more complex or unclear tasks that might lead to frustration.

---

## 5. Emotion Dynamics Bundle Features ❌ (Neutral: 0.00% F1)

### 5.1 Emotion Drift

**Description**: Shift in emotional state over conversation turns  
**Bundle**: emotion_dynamics_bundle

**Code Block**:
```python
# Emotion drift proxy: negative emotion word progression
neg_scores = [self._word_ratio(text, self.negative_words) for text in texts]
features['emotion_drift'] = np.polyfit(range(len(neg_scores)), neg_scores, 1)[0] if len(neg_scores) > 1 else 0.0
```

**Modeling Approach**: Tracks the slope of negative emotion words over time. Positive slope indicates increasing negativity (emotional drift toward frustration).

### 5.2 Emotion Volatility

**Description**: Sudden changes in emotional state  
**Bundle**: emotion_dynamics_bundle

**Code Block**:
```python
features['emotion_volatility'] = float(np.var(neg_scores))
```

**Modeling Approach**: Computes variance in negative emotion scores across turns. Higher variance indicates more volatile emotional states, which may correlate with frustration.

---

## 6. System Bundle Features ✅ (Effective: +1.12% F1)

### 6.1 Response Clarity

**Description**: Clarity of system responses (inverse of user questions)  
**Bundle**: system_bundle

**Code Block**:
```python
# Response clarity proxy: question ratio (more questions = less clarity)
question_ratios = [text.count('?') / max(len(text.split()), 1) for text in texts]
features['response_clarity'] = float(1.0 - np.mean(question_ratios))  # Inverse of questions
```

**Modeling Approach**: Assumes that more user questions indicate less clear system responses. Computes inverse of question density as a proxy for response clarity.

### 6.2 Response Relevance

**Description**: Relevance of system responses (inverse of user confusion)  
**Bundle**: system_bundle

**Code Block**:
```python
# Response relevance proxy: confusion word ratio (more confusion = less relevance)
conf_ratios = [self._word_ratio(text, self.confusion_words) for text in texts]
features['response_relevance'] = float(1.0 - np.mean(conf_ratios))  # Inverse of confusion
```

**Modeling Approach**: Uses inverse of confusion word density as a proxy for response relevance. More user confusion suggests less relevant system responses.

---

## 7. User Model Bundle Features ✅ (Effective: +1.12% F1)

### 7.1 Trust in System

**Description**: User's declining trust in the system  
**Bundle**: user_model_bundle

**Code Block**:
```python
# Trust in system proxy: doubt expressions
doubt_words = {'sure?', 'really?', 'certain?', 'doubt', 'wrong'}
doubt_ratios = [self._word_ratio(text, doubt_words) for text in texts]
features['trust_decline'] = float(np.mean(doubt_ratios))
```

**Modeling Approach**: Uses doubt expression lexicon (sure?, really?, certain?, doubt, wrong) to measure user skepticism and declining trust in system capabilities.

---

## Summary

**Effective Features** (improve F1 by +1.12%):
- **Linguistic Bundle**: Sentiment, politeness, confusion, negation, emphasis, exclamation patterns
- **System Bundle**: Response clarity and relevance proxies  
- **User Model Bundle**: Trust decline indicators

**Ineffective Features**:
- **Dialogue Bundle**: Conversation patterns (actually hurt performance)
- **Behavioral Bundle**: Escalation and feedback (no improvement)
- **Contextual Bundle**: Urgency and complexity (slight harm)
- **Emotion Dynamics Bundle**: Emotional drift patterns (no improvement)

The success of linguistic and trust-related features suggests that frustration detection benefits most from explicit linguistic markers and system-user relationship indicators rather than structural dialogue patterns.