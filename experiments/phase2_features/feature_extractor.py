#!/usr/bin/env python3
"""
Phase 2: Feature Extraction for Frustration Detection
Extract engineered features from dialogue text for each bundle
"""

import os
import re
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrustrationFeatureExtractor:
    """Extract engineered features for frustration detection"""
    
    def __init__(self):
        # Initialize models and resources
        self.sentiment_analyzer = None
        self.emotion_analyzer = None
        
        # Feature bundles from features.csv
        self.feature_bundles = {
            'linguistic_bundle': [
                'sentiment_trajectory', 'politeness_level', 'intent_repetition', 
                'directness_abruptness', 'confusion_lexical_markers', 'hedging_expressions',
                'negation_frequency', 'emotion_words', 'discourse_markers', 
                'emphasis_capitalization', 'exclamation_density', 'sarcasm_indicators'
            ],
            'dialogue_bundle': [
                'system_failures', 'repeated_turns', 'conversation_length', 
                'user_corrections', 'intent_switch_frequency', 'self_corrections',
                'confirmation_count', 'system_misunderstanding_rate', 'alignment_failures'
            ],
            'behavioral_bundle': [
                'escalation_request', 'negative_feedback'
            ],
            'contextual_bundle': [
                'task_complexity', 'goal_completion_status', 'subgoal_block_count', 'expressed_urgency'
            ],
            'emotion_dynamics_bundle': [
                'emotion_drift', 'emotion_volatility', 'frustration_delay'
            ],
            'system_bundle': [
                'response_clarity', 'response_relevance'
            ],
            'user_model_bundle': [
                'trust_in_system'
            ]
        }
        
        # Initialize lexicons and patterns
        self._init_lexicons()
    
    def _init_lexicons(self):
        """Initialize lexicons and patterns for feature extraction"""
        # Confusion markers
        self.confusion_markers = {
            'what', 'why', 'how', 'confused', 'unclear', 'understand', 'mean', 'huh'
        }
        
        # Hedging expressions
        self.hedging_words = {
            'maybe', 'perhaps', 'possibly', 'probably', 'might', 'could', 'would',
            'i think', 'i guess', 'i suppose', 'kind of', 'sort of'
        }
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither',
            'cant', "can't", 'wont', "won't", 'dont', "don't", 'isnt', "isn't",
            'wasnt', "wasn't", 'arent', "aren't", 'werent', "weren't"
        }
        
        # Emotion words (negative)
        self.negative_emotion_words = {
            'angry', 'frustrated', 'annoyed', 'irritated', 'mad', 'furious',
            'disappointed', 'upset', 'sad', 'terrible', 'awful', 'horrible',
            'stupid', 'useless', 'broken', 'wrong', 'bad', 'hate'
        }
        
        # Discourse markers
        self.adversative_markers = {'but', 'however', 'though', 'although', 'yet', 'still'}
        self.additive_markers = {'and', 'also', 'furthermore', 'moreover', 'besides'}
        
        # Escalation phrases
        self.escalation_phrases = {
            'talk to', 'speak to', 'human', 'person', 'agent', 'manager', 'supervisor',
            'escalate', 'help', 'support', 'service', 'representative'
        }
        
        # System failure indicators
        self.system_failure_patterns = {
            'sorry', "i don't understand", "didn't get that", 'error', 'failed',
            'try again', 'something went wrong', 'not working'
        }
        
        # User correction patterns
        self.correction_patterns = [
            r'no,?\s*i\s*meant?',
            r'actually,?\s*',
            r'wait,?\s*',
            r'i\s*said',
            r'correction',
            r'let\s*me\s*clarify'
        ]
        
        # Urgency expressions
        self.urgency_words = {
            'urgent', 'asap', 'now', 'immediately', 'quickly', 'fast', 'hurry',
            'need', 'must', 'important', 'critical', 'emergency'
        }
    
    def _load_models(self):
        """Load heavy models on demand"""
        if self.sentiment_analyzer is None:
            logger.info("Loading sentiment analyzer...")
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                             return_all_scores=True)
        
        if self.emotion_analyzer is None:
            logger.info("Loading emotion analyzer...")
            self.emotion_analyzer = pipeline("text-classification",
                                           model="j-hartmann/emotion-english-distilroberta-base",
                                           return_all_scores=True)
    
    # LINGUISTIC BUNDLE FEATURES
    def extract_sentiment_trajectory(self, texts: List[str]) -> Dict[str, float]:
        """Extract sentiment trajectory features"""
        self._load_models()
        
        sentiments = []
        for text in texts:
            if not text.strip():
                continue
            result = self.sentiment_analyzer(text)[0]
            # Convert to numeric scale: negative=-1, neutral=0, positive=1
            for item in result:
                if item['label'] == 'LABEL_2':  # Positive
                    sentiments.append(item['score'])
                elif item['label'] == 'LABEL_0':  # Negative
                    sentiments.append(-item['score'])
                else:  # Neutral
                    sentiments.append(0)
                break
        
        if len(sentiments) < 2:
            return {'sentiment_slope': 0.0, 'sentiment_volatility': 0.0, 'final_sentiment': 0.0}
        
        # Calculate slope (trend)
        x = np.arange(len(sentiments))
        slope = np.polyfit(x, sentiments, 1)[0] if len(sentiments) > 1 else 0.0
        
        # Calculate volatility (variance)
        volatility = np.var(sentiments)
        
        return {
            'sentiment_slope': float(slope),
            'sentiment_volatility': float(volatility),
            'final_sentiment': float(sentiments[-1])
        }
    
    def extract_politeness_level(self, texts: List[str]) -> Dict[str, float]:
        """Extract politeness level features"""
        politeness_indicators = {'please', 'thank', 'sorry', 'excuse', 'pardon', 'could you', 'would you'}
        
        politeness_scores = []
        for text in texts:
            text_lower = text.lower()
            score = sum(1 for word in politeness_indicators if word in text_lower)
            # Normalize by text length
            normalized_score = score / max(len(text.split()), 1)
            politeness_scores.append(normalized_score)
        
        if not politeness_scores:
            return {'avg_politeness': 0.0, 'politeness_decline': 0.0}
        
        avg_politeness = np.mean(politeness_scores)
        
        # Calculate decline (negative slope)
        if len(politeness_scores) > 1:
            x = np.arange(len(politeness_scores))
            decline = -np.polyfit(x, politeness_scores, 1)[0]
        else:
            decline = 0.0
        
        return {
            'avg_politeness': float(avg_politeness),
            'politeness_decline': float(decline)
        }
    
    def extract_confusion_markers(self, texts: List[str]) -> Dict[str, float]:
        """Extract confusion lexical markers"""
        confusion_counts = []
        for text in texts:
            text_lower = text.lower()
            count = sum(1 for marker in self.confusion_markers if marker in text_lower)
            confusion_counts.append(count / max(len(text.split()), 1))
        
        return {
            'avg_confusion_markers': float(np.mean(confusion_counts)) if confusion_counts else 0.0,
            'confusion_increase': float(np.polyfit(range(len(confusion_counts)), confusion_counts, 1)[0]) if len(confusion_counts) > 1 else 0.0
        }
    
    def extract_negation_frequency(self, texts: List[str]) -> Dict[str, float]:
        """Extract negation frequency"""
        negation_counts = []
        for text in texts:
            text_lower = text.lower()
            count = sum(1 for neg in self.negation_words if neg in text_lower)
            negation_counts.append(count / max(len(text.split()), 1))
        
        return {
            'avg_negation_frequency': float(np.mean(negation_counts)) if negation_counts else 0.0,
            'negation_increase': float(np.polyfit(range(len(negation_counts)), negation_counts, 1)[0]) if len(negation_counts) > 1 else 0.0
        }
    
    def extract_emotion_words(self, texts: List[str]) -> Dict[str, float]:
        """Extract emotional word frequency"""
        emotion_counts = []
        for text in texts:
            text_lower = text.lower()
            count = sum(1 for word in self.negative_emotion_words if word in text_lower)
            emotion_counts.append(count / max(len(text.split()), 1))
        
        return {
            'avg_emotion_words': float(np.mean(emotion_counts)) if emotion_counts else 0.0,
            'emotion_words_increase': float(np.polyfit(range(len(emotion_counts)), emotion_counts, 1)[0]) if len(emotion_counts) > 1 else 0.0
        }
    
    def extract_emphasis_capitalization(self, texts: List[str]) -> Dict[str, float]:
        """Extract emphasis through capitalization"""
        caps_ratios = []
        for text in texts:
            if not text:
                caps_ratios.append(0.0)
                continue
            
            # Count capitalized words (excluding first word and proper nouns heuristically)
            words = text.split()
            caps_count = 0
            for i, word in enumerate(words):
                if i > 0 and word.isupper() and len(word) > 1:  # Skip single letters
                    caps_count += 1
            
            caps_ratios.append(caps_count / max(len(words), 1))
        
        return {
            'avg_capitalization': float(np.mean(caps_ratios)) if caps_ratios else 0.0,
            'capitalization_increase': float(np.polyfit(range(len(caps_ratios)), caps_ratios, 1)[0]) if len(caps_ratios) > 1 else 0.0
        }
    
    def extract_exclamation_density(self, texts: List[str]) -> Dict[str, float]:
        """Extract exclamation mark density"""
        exclamation_densities = []
        for text in texts:
            exclamation_count = text.count('!') + text.count('?')
            density = exclamation_count / max(len(text.split()), 1)
            exclamation_densities.append(density)
        
        return {
            'avg_exclamation_density': float(np.mean(exclamation_densities)) if exclamation_densities else 0.0,
            'exclamation_increase': float(np.polyfit(range(len(exclamation_densities)), exclamation_densities, 1)[0]) if len(exclamation_densities) > 1 else 0.0
        }
    
    # DIALOGUE BUNDLE FEATURES
    def extract_conversation_length(self, texts: List[str]) -> Dict[str, float]:
        """Extract conversation length feature"""
        return {
            'total_turns': float(len(texts)),
            'avg_turn_length': float(np.mean([len(text.split()) for text in texts])) if texts else 0.0
        }
    
    def extract_repeated_turns(self, texts: List[str]) -> Dict[str, float]:
        """Extract repeated turn patterns"""
        if len(texts) < 2:
            return {'repeated_turn_ratio': 0.0}
        
        # Simple similarity check using word overlap
        repeated_count = 0
        for i in range(1, len(texts)):
            current_words = set(texts[i].lower().split())
            prev_words = set(texts[i-1].lower().split())
            
            if len(current_words) > 0 and len(prev_words) > 0:
                overlap = len(current_words & prev_words) / len(current_words | prev_words)
                if overlap > 0.5:  # 50% similarity threshold
                    repeated_count += 1
        
        return {
            'repeated_turn_ratio': float(repeated_count / len(texts))
        }
    
    def extract_user_corrections(self, texts: List[str]) -> Dict[str, float]:
        """Extract user correction patterns"""
        correction_count = 0
        for text in texts:
            text_lower = text.lower()
            for pattern in self.correction_patterns:
                if re.search(pattern, text_lower):
                    correction_count += 1
                    break
        
        return {
            'correction_frequency': float(correction_count / max(len(texts), 1))
        }
    
    # BEHAVIORAL BUNDLE FEATURES
    def extract_escalation_request(self, texts: List[str]) -> Dict[str, float]:
        """Extract escalation request indicators"""
        escalation_count = 0
        for text in texts:
            text_lower = text.lower()
            if any(phrase in text_lower for phrase in self.escalation_phrases):
                escalation_count += 1
        
        return {
            'escalation_requests': float(escalation_count),
            'escalation_ratio': float(escalation_count / max(len(texts), 1))
        }
    
    def extract_negative_feedback(self, texts: List[str]) -> Dict[str, float]:
        """Extract explicit negative feedback"""
        negative_patterns = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'sucks', 'useless']
        negative_count = 0
        
        for text in texts:
            text_lower = text.lower()
            if any(pattern in text_lower for pattern in negative_patterns):
                negative_count += 1
        
        return {
            'negative_feedback_count': float(negative_count),
            'negative_feedback_ratio': float(negative_count / max(len(texts), 1))
        }
    
    # CONTEXTUAL BUNDLE FEATURES
    def extract_expressed_urgency(self, texts: List[str]) -> Dict[str, float]:
        """Extract expressed urgency indicators"""
        urgency_scores = []
        for text in texts:
            text_lower = text.lower()
            score = sum(1 for word in self.urgency_words if word in text_lower)
            urgency_scores.append(score / max(len(text.split()), 1))
        
        return {
            'avg_urgency': float(np.mean(urgency_scores)) if urgency_scores else 0.0,
            'urgency_increase': float(np.polyfit(range(len(urgency_scores)), urgency_scores, 1)[0]) if len(urgency_scores) > 1 else 0.0
        }
    
    # EMOTION DYNAMICS BUNDLE FEATURES
    def extract_emotion_drift(self, texts: List[str]) -> Dict[str, float]:
        """Extract emotion drift over conversation"""
        self._load_models()
        
        emotion_scores = []
        for text in texts:
            if not text.strip():
                continue
            result = self.emotion_analyzer(text)[0]
            # Focus on negative emotions (anger, sadness, fear)
            negative_score = 0
            for item in result:
                if item['label'].lower() in ['anger', 'sadness', 'fear']:
                    negative_score += item['score']
            emotion_scores.append(negative_score)
        
        if len(emotion_scores) < 2:
            return {'emotion_drift': 0.0, 'emotion_volatility': 0.0}
        
        # Calculate drift (slope towards negative emotions)
        x = np.arange(len(emotion_scores))
        drift = np.polyfit(x, emotion_scores, 1)[0]
        volatility = np.var(emotion_scores)
        
        return {
            'emotion_drift': float(drift),
            'emotion_volatility': float(volatility)
        }
    
    # USER MODEL BUNDLE FEATURES
    def extract_trust_in_system(self, texts: List[str]) -> Dict[str, float]:
        """Extract trust indicators"""
        trust_indicators = ['sure', 'confident', 'trust', 'believe', 'right', 'correct']
        distrust_indicators = ['doubt', 'wrong', 'sure?', 'really?', 'certain?', 'positive?']
        
        trust_scores = []
        for text in texts:
            text_lower = text.lower()
            trust_count = sum(1 for word in trust_indicators if word in text_lower)
            distrust_count = sum(1 for word in distrust_indicators if word in text_lower)
            
            # Net trust score
            net_trust = (trust_count - distrust_count) / max(len(text.split()), 1)
            trust_scores.append(net_trust)
        
        return {
            'avg_trust': float(np.mean(trust_scores)) if trust_scores else 0.0,
            'trust_decline': float(-np.polyfit(range(len(trust_scores)), trust_scores, 1)[0]) if len(trust_scores) > 1 else 0.0
        }
    
    def extract_bundle_features(self, texts: List[str], bundle_name: str) -> Dict[str, float]:
        """Extract features for a specific bundle"""
        features = {}
        
        if bundle_name == 'linguistic_bundle':
            features.update(self.extract_sentiment_trajectory(texts))
            features.update(self.extract_politeness_level(texts))
            features.update(self.extract_confusion_markers(texts))
            features.update(self.extract_negation_frequency(texts))
            features.update(self.extract_emotion_words(texts))
            features.update(self.extract_emphasis_capitalization(texts))
            features.update(self.extract_exclamation_density(texts))
            
        elif bundle_name == 'dialogue_bundle':
            features.update(self.extract_conversation_length(texts))
            features.update(self.extract_repeated_turns(texts))
            features.update(self.extract_user_corrections(texts))
            
        elif bundle_name == 'behavioral_bundle':
            features.update(self.extract_escalation_request(texts))
            features.update(self.extract_negative_feedback(texts))
            
        elif bundle_name == 'contextual_bundle':
            features.update(self.extract_expressed_urgency(texts))
            
        elif bundle_name == 'emotion_dynamics_bundle':
            features.update(self.extract_emotion_drift(texts))
            
        elif bundle_name == 'user_model_bundle':
            features.update(self.extract_trust_in_system(texts))
        
        return features
    
    def extract_all_features(self, texts: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract features for all bundles"""
        all_features = {}
        
        for bundle_name in self.feature_bundles.keys():
            logger.info(f"Extracting {bundle_name} features...")
            all_features[bundle_name] = self.extract_bundle_features(texts, bundle_name)
        
        return all_features

def test_feature_extraction():
    """Test feature extraction with sample dialogue"""
    logger.info("Testing feature extraction...")
    
    extractor = FrustrationFeatureExtractor()
    
    # Sample frustrated dialogue
    sample_texts = [
        "Hi, I need help with booking a flight",
        "I said I want to book a flight to Paris",
        "No, I meant PARIS, not London!",
        "This is getting really frustrating. Can I talk to a human?",
        "This system is useless. I HATE this!"
    ]
    
    # Test individual bundles
    for bundle_name in extractor.feature_bundles.keys():
        features = extractor.extract_bundle_features(sample_texts, bundle_name)
        logger.info(f"{bundle_name}: {features}")
    
    logger.info("Feature extraction test completed!")

if __name__ == "__main__":
    test_feature_extraction()