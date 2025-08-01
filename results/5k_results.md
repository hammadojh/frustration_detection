ubuntu@focused-mendel:~/frustration_detect$ python3 validation_study_5k.py 
INFO:__main__:================================================================================
INFO:__main__:VALIDATION STUDY: 5K BALANCED DATAPOINTS
INFO:__main__:================================================================================
INFO:__main__:This validation study will test key configurations from previous studies:
INFO:__main__:  1. Baseline (text-only): 1 test
INFO:__main__:  2. Top individual features: 5 tests
INFO:__main__:  3. Best bundles: 3 tests
INFO:__main__:  4. Key combinations: 5 tests
INFO:__main__:  5. Leave-one-out (top features): 5 tests
INFO:__main__:  Total: ~19 focused experiments on 5k data
INFO:__main__:
INFO:__main__:5k dataset not found, creating it...
INFO:__main__:Loading EmoWOZ dataset for 5k validation...
INFO:__main__:Processing train split with 9233 dialogues...
INFO:__main__:Processing validation split with 1100 dialogues...
INFO:__main__:Processing test split with 1100 dialogues...
INFO:__main__:Total available: 23620 frustrated, 59997 not frustrated
INFO:__main__:5k Dataset created: 2500 frustrated, 2500 not frustrated
INFO:__main__:5k Dataset saved to data
INFO:__main__:Statistics: {'total_examples': 5000, 'frustrated_count': 2500, 'not_frustrated_count': 2500, 'emotion_id_distribution': {0: 2439, 6: 1872, 2: 523, 5: 105, 3: 40, 1: 16, 4: 5}, 'split_distribution': {'train': 4005, 'test': 509, 'validation': 486}}
INFO:__main__:Data splits: Train=3500, Val=750, Test=750
INFO:__main__:
1. Testing BASELINE: Text-only (no features)
INFO:__main__:Initializing RoBERTa models...
Some weights of the model checkpoint at SamLowe/roberta-base-go_emotions were not used when initializing RobertaModel: ['classifier.out_proj.bias', 'classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.weight']
- This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaModel were not initialized from the model checkpoint at SamLowe/roberta-base-go_emotions and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
INFO:__main__:Models initialized successfully!
INFO:__main__:Testing text_only_baseline_5k with 0 features...
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8388, Precision=0.8473, Recall=0.8305
INFO:__main__:   5k Text-only F1: 0.8388
INFO:__main__:
2. Testing TOP INDIVIDUAL FEATURES from previous studies
INFO:__main__:   Testing individual feature 1/5: sentiment_slope
INFO:__main__:Testing individual_sentiment_slope_5k with 1 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8429, Precision=0.8526, Recall=0.8333
INFO:__main__:   Testing individual feature 2/5: avg_politeness
INFO:__main__:Testing individual_avg_politeness_5k with 1 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8441, Precision=0.8551, Recall=0.8333
INFO:__main__:   Testing individual feature 3/5: avg_confusion
INFO:__main__:Testing individual_avg_confusion_5k with 1 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8383, Precision=0.8493, Recall=0.8277
INFO:__main__:   Testing individual feature 4/5: avg_negation
INFO:__main__:Testing individual_avg_negation_5k with 1 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8371, Precision=0.8468, Recall=0.8277
INFO:__main__:   Testing individual feature 5/5: total_turns
INFO:__main__:Testing individual_total_turns_5k with 1 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8412, Precision=0.8522, Recall=0.8305
INFO:__main__:
3. Testing BEST BUNDLES from previous studies
INFO:__main__:   Testing linguistic_bundle: 8 features
INFO:__main__:Testing bundle_linguistic_bundle_5k with 8 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8450, Precision=0.8510, Recall=0.8390
INFO:__main__:   Testing behavioral_bundle: 2 features
INFO:__main__:Testing bundle_behavioral_bundle_5k with 2 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8452, Precision=0.8665, Recall=0.8249
INFO:__main__:   Testing system_bundle: 2 features
INFO:__main__:Testing bundle_system_bundle_5k with 2 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8432, Precision=0.8592, Recall=0.8277
INFO:__main__:
4. Testing KEY COMBINATIONS from previous studies
INFO:__main__:   Testing top_3_features: 3 features
INFO:__main__:Testing top_3_features_5k with 3 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8397, Precision=0.8433, Recall=0.8362
INFO:__main__:     Progress saved: 10 tests completed
INFO:__main__:   Testing top_8_features: 8 features
INFO:__main__:Testing top_8_features_5k with 8 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8442, Precision=0.8466, Recall=0.8418
INFO:__main__:     Progress saved: 11 tests completed
INFO:__main__:   Testing linguistic_8: 8 features
INFO:__main__:Testing linguistic_8_5k with 8 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8450, Precision=0.8510, Recall=0.8390
INFO:__main__:     Progress saved: 12 tests completed
INFO:__main__:   Testing all_22_features: 22 features
INFO:__main__:Testing all_22_features_5k with 22 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8438, Precision=0.8486, Recall=0.8390
INFO:__main__:     Progress saved: 13 tests completed
INFO:__main__:   Testing optimal_21_features: 21 features
INFO:__main__:Testing optimal_21_features_5k with 21 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8498, Precision=0.8609, Recall=0.8390
INFO:__main__:     Progress saved: 14 tests completed
INFO:__main__:
5. Testing LEAVE-ONE-OUT for most critical features
INFO:__main__:   Removing critical feature 1/5: sentiment_slope
INFO:__main__:Testing loo_remove_sentiment_slope_5k with 21 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8526, Precision=0.8638, Recall=0.8418
INFO:__main__:   Removing critical feature 2/5: avg_politeness
INFO:__main__:Testing loo_remove_avg_politeness_5k with 21 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8481, Precision=0.8605, Recall=0.8362
INFO:__main__:   Removing critical feature 3/5: avg_confusion
INFO:__main__:Testing loo_remove_avg_confusion_5k with 21 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8469, Precision=0.8580, Recall=0.8362
INFO:__main__:   Removing critical feature 4/5: avg_negation
INFO:__main__:Testing loo_remove_avg_negation_5k with 21 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8481, Precision=0.8605, Recall=0.8362
INFO:__main__:   Removing critical feature 5/5: avg_exclamation
INFO:__main__:Testing loo_remove_avg_exclamation_5k with 21 features...
INFO:__main__:Extracting features for 3500 texts in batches of 100...
INFO:__main__:  Processing batch 1/35
INFO:__main__:  Processing batch 11/35
INFO:__main__:  Processing batch 21/35
INFO:__main__:  Processing batch 31/35
INFO:__main__:Extracting features for 750 texts in batches of 100...
INFO:__main__:  Processing batch 1/8
INFO:__main__:  Extracting embeddings for 3500 texts...
INFO:__main__:    Embedding batch 1/110
INFO:__main__:    Embedding batch 21/110
INFO:__main__:    Embedding batch 41/110
INFO:__main__:    Embedding batch 61/110
INFO:__main__:    Embedding batch 81/110
INFO:__main__:    Embedding batch 101/110
INFO:__main__:  Extracting embeddings for 750 texts...
INFO:__main__:    Embedding batch 1/24
INFO:__main__:    Embedding batch 21/24
INFO:__main__:  Training classifier...
INFO:__main__:  Result: F1=0.8498, Precision=0.8609, Recall=0.8390
INFO:__main__:
🎯 VALIDATION STUDY COMPLETE!
INFO:__main__:Results saved to:
INFO:__main__:  Data: results/validation_study_5k.csv
INFO:__main__:  Report: results/validation_study_5k_REPORT.txt
INFO:__main__:
🔍 KEY VALIDATION FINDINGS:
INFO:__main__:  5k Text-only baseline: 0.8388
INFO:__main__:  Best 5k configuration: loo_remove_sentiment_slope_5k (F1: 0.8526)
INFO:__main__:  Improvement: +0.0138
