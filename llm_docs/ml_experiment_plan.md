# Autonomous ML Experimentation Plan for Frustration Detection

**Objective:** To systematically and autonomously identify the optimal combination of engineered features that, when combined with a BERT-based model, yields the highest performance in detecting user frustration. This plan is designed for iterative execution, evaluation, and refinement until the best model is found.

---

## Phase 1: Setup & Baseline Establishment

### Step 1.1: Environment & Data Preparation
1.  **Load Data:** Load the primary dataset (e.g., EmoWOZ) for training and testing.
2.  **Load Features:** Load the `features.csv` file to be used for feature engineering.
3.  **Pre-process Data:** Clean and pre-process the dialogue data, aligning labels with dialogue turns.

### Step 1.2: Feature Bundle Definition
Based on `features.csv`, categorize all available features into the following bundles. These bundles will be the units for our experiments.

*   **`linguistic_bundle`**: `sentiment_trajectory`, `politeness_level`, `intent_repetition`, `directness_abruptness`, `confusion_lexical_markers`, `hedging_expressions`, `negation_frequency`, `emotion_words`, `discourse_markers`, `emphasis_capitalization`, `exclamation_density`, `sarcasm_indicators`
*   **`dialogue_bundle`**: `system_failures`, `repeated_turns`, `conversation_length`, `user_corrections`, `intent_switch_frequency`, `self_corrections`, `confirmation_count`, `system_misunderstanding_rate`, `alignment_failures`
*   **`behavioral_bundle`**: `escalation_request`, `negative_feedback` (Note: Text-derivable only)
*   **`contextual_bundle`**: `task_complexity`, `goal_completion_status`, `subgoal_block_count`, `expressed_urgency`
*   **`emotion_dynamics_bundle`**: `emotion_drift`, `emotion_volatility`, `frustration_delay`
*   **`system_bundle`**: `response_clarity`, `response_relevance`
*   **`user_model_bundle`**: `trust_in_system` (Note: Text-derivable proxy only)

### Step 1.3: Train Baseline Model
1.  **Action:** Train a standard RoBERTa-based classifier on the raw dialogue text *without* any engineered features using the existing `FrustrationDetector` architecture.
2.  **Architecture Details:**
    - **Model:** `RobertaForSequenceClassification` using `SamLowe/roberta-base-go_emotions` pre-trained model
    - **Configuration:** Single output node (`num_labels=1`) for binary classification
    - **Loss Function:** BCEWithLogitsLoss for binary classification
    - **Training Parameters:**
      - Learning rate: 2e-5
      - Batch size: 16 (train/eval)
      - Epochs: 4
      - Weight decay: 0.01
      - Class-balanced loss weights computed using `sklearn.utils.class_weight.compute_class_weight`
    - **Custom Trainer:** Uses `CustomTrainer` class to handle binary classification loss and prediction steps
    - **Metrics:** F1, Precision, Recall, and Accuracy computed via `compute_metrics` function
3.  **Input:** Raw dialogue history processed through `map_to_binary` function
4.  **Output:** A trained baseline model saved in the output directory
5.  **Evaluation:** Calculate F1, Precision, and Recall on the test set using the existing evaluation framework
6.  **Store Results:** Save the baseline performance metrics to a `results.csv` file.
    ```csv
    experiment_name,features_used,f1_score,precision,recall,notes
    baseline,none,0.75,0.73,0.77,"Raw text performance with RoBERTa"
    ```

---

## Phase 2: Iterative Feature Bundle Evaluation (The Loop)

This phase will proceed in an automated loop. The agent will add one feature bundle at a time, train a new model, and evaluate its performance against the current best model.

### Loop Logic:
1.  Initialize `current_best_features` = `[]`
2.  Initialize `current_best_f1` = `baseline_f1_score`
3.  For each `bundle` in `[linguistic_bundle, dialogue_bundle, ...]`:
    a. **Prepare Features**: `features_to_test` = `current_best_features` + `bundle`
    b. **Feature Integration Architecture**: 
       - Extract engineered features from dialogue text and convert to numerical vectors
       - Concatenate feature vectors with RoBERTa's `[CLS]` token embedding (768-dim)
       - Pass combined representation through additional dense layers before final classification
       - Maintain the same `CustomTrainer`, loss function, and training parameters as baseline
    c. **Train Model**: Train the enhanced RoBERTa model with the augmented input representation
    d. **Evaluate**: Calculate the F1 score using the same evaluation framework
    e. **Compare**:
        - If `new_f1 > current_best_f1`:
            - `current_best_f1` = `new_f1`
            - `current_best_features` = `features_to_test`
            - Log the improvement in `results.csv`.
        - Else:
            - Log that the bundle did not improve performance.
4.  Repeat until all bundles have been tested.

### Step 2.1: Execute the Additive Loop
1.  **Action:** Begin the iterative loop described above.
2.  **Artifacts:** For each iteration, a new model is trained and a new row is added to `results.csv`.
    ```csv
    experiment_name,features_used,f1_score,precision,recall,notes
    baseline,none,0.75,0.73,0.77,"Raw text performance"
    run_1,"linguistic_bundle",0.78,0.76,0.80,"Linguistic features added"
    run_2,"linguistic_bundle,dialogue_bundle",0.81,0.80,0.82,"Dialogue features added"
    ...
    ```

---

## Phase 3: Ablation Study & Finalization

After the additive loop identifies the best-performing combination of feature bundles, conduct an ablation study to ensure every included bundle is contributing positively.

### Step 3.1: Perform Ablation
1.  Let `best_feature_set` be the list of bundles from the end of Phase 2.
2.  For each `bundle_to_remove` in `best_feature_set`:
    a. **Prepare Features**: `features_for_ablation` = `best_feature_set` - `bundle_to_remove`
    b. **Train Model**: Train a new model using this reduced feature set.
    c. **Evaluate & Log**: Calculate the F1 score and log it to `results.csv` with a note indicating which bundle was removed.
    ```csv
    experiment_name,features_used,f1_score,precision,recall,notes
    ...
    ablation_1,"dialogue_bundle",0.79,0.78,0.80,"Removed linguistic_bundle"
    ...
    ```

### Step 3.2: Final Model Selection
1.  **Analyze Results:** Review the completed `results.csv`.
2.  **Identify Final Features:** The final feature set is the one from the additive phase, potentially with any bundles removed that showed no negative impact during the ablation study (i.e., their removal did not decrease the F1 score).
3.  **Train Final Model:** Train one last model using the finalized optimal feature set.
4.  **Action:** Save the final model as `frustration_detector_final.pt` and generate a `final_report.md` summarizing the results table and declaring the best feature combination found.

This structured plan will ensure that we methodically and efficiently converge on the most powerful and cost-effective feature set for the task.
