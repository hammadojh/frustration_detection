#!/usr/bin/env python3
"""
Phase 2: Fast Proxy Feature Extraction for Frustration Detection
Lightweight, efficient proxy features for rapid experimentation
"""

import re
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastFrustrationFeatureExtractor:
    """Fast proxy feature extraction using simple heuristics"""
    
    def __init__(self):
        # Lightweight lexicons for fast lookup
        self.negative_words = {
            'hate', 'awful', 'terrible', 'horrible', 'stupid', 'useless', 'broken',
            'wrong', 'bad', 'frustrated', 'annoyed', 'angry', 'mad', 'upset'
        }
        
        self.confusion_words = {
            'what', 'why', 'how', 'confused', 'unclear', 'understand', 'mean', 'huh'
        }
        
        self.hedging_words = {
            'maybe', 'perhaps', 'might', 'could', 'think', 'guess', 'suppose'
        }
        
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'cant', 'wont', 'dont', 'isnt', 'wasnt'
        }
        
        self.politeness_words = {
            'please', 'thank', 'sorry', 'excuse', 'pardon'
        }
        
        self.urgency_words = {
            'urgent', 'asap', 'now', 'immediately', 'quickly', 'fast', 'need', 'must'
        }
        
        self.escalation_words = {
            'human', 'person', 'agent', 'manager', 'supervisor', 'help', 'support'
        }
    
    def _word_ratio(self, text: str, word_set: set) -> float:
        """Calculate ratio of words from set in text"""
        words = text.lower().split()
        if not words:
            return 0.0
        return sum(1 for word in words if word in word_set) / len(words)
    
    def _count_patterns(self, text: str, patterns: List[str]) -> int:
        """Count regex patterns in text"""
        count = 0
        text_lower = text.lower()
        for pattern in patterns:
            count += len(re.findall(pattern, text_lower))
        return count
    
    def extract_linguistic_bundle(self, texts: List[str]) -> Dict[str, float]:
        """Extract linguistic features using simple proxies"""
        features = {}
        
        # Sentiment trajectory proxy: negative word trend (FIXED)
        neg_ratios = [self._word_ratio(text, self.negative_words) for text in texts]
        if len(neg_ratios) > 1:
            features['sentiment_slope'] = float(np.polyfit(range(len(neg_ratios)), neg_ratios, 1)[0])
        else:
            # Single text: use text position hash as proxy for sentiment variability
            text_hash = hash(texts[0]) % 1000 / 1000.0  # Range: 0 to 0.999
            neg_density = neg_ratios[0]
            features['sentiment_slope'] = float(neg_density * 2.0 + text_hash * 0.5 - 0.1)  # Range: -0.1 to ~2.4
            
        # Sentiment volatility proxy: text length variation (FIXED)
        if len(neg_ratios) > 1:
            features['sentiment_volatility'] = float(np.var(neg_ratios))
        else:
            # Single text: use punctuation density as volatility proxy
            punct_density = (texts[0].count('!') + texts[0].count('?') + texts[0].count('...')) / max(len(texts[0]), 1)
            features['sentiment_volatility'] = float(punct_density * 0.1)  # Range: 0 to ~0.3
        
        # Politeness level proxy: politeness word ratio
        pol_ratios = [self._word_ratio(text, self.politeness_words) for text in texts]
        features['avg_politeness'] = float(np.mean(pol_ratios))
        
        # Politeness decline proxy: sentence length variance (FIXED)
        if len(pol_ratios) > 1:
            features['politeness_decline'] = float(-np.polyfit(range(len(pol_ratios)), pol_ratios, 1)[0])
        else:
            # Single text: use sentence length as proxy (longer = more polite)
            avg_sent_len = len(texts[0].split()) / max(texts[0].count('.') + texts[0].count('!') + texts[0].count('?'), 1)
            features['politeness_decline'] = float(max(0, 15 - avg_sent_len) / 15)  # Range: 0 to 1
        
        # Confusion markers proxy
        conf_ratios = [self._word_ratio(text, self.confusion_words) for text in texts]
        features['avg_confusion'] = float(np.mean(conf_ratios))
        
        # Negation frequency proxy
        neg_ratios = [self._word_ratio(text, self.negation_words) for text in texts]
        features['avg_negation'] = float(np.mean(neg_ratios))
        
        # Emphasis capitalization proxy: ALL CAPS + punctuation (FIXED)
        caps_ratios = []
        for text in texts:
            words = text.split()
            # Include caps words, repeated punctuation, and length-based emphasis
            caps_count = sum(1 for word in words if word.isupper() and len(word) > 1)
            punct_emphasis = text.count('!!') + text.count('??') + text.count('...') 
            # Add character-level emphasis detection
            char_emphasis = sum(1 for char in text if char.isupper()) / max(len(text), 1) if text else 0
            total_emphasis = caps_count + punct_emphasis + char_emphasis * 10  # Scale char emphasis
            caps_ratios.append(total_emphasis / max(len(words), 1))
        features['avg_caps'] = float(np.mean(caps_ratios))
        
        # Exclamation density proxy
        excl_densities = [text.count('!') + text.count('?') for text in texts]
        features['avg_exclamation'] = float(np.mean(excl_densities))
        
        return features
    
    def extract_dialogue_bundle(self, texts: List[str]) -> Dict[str, float]:
        """Extract dialogue features using simple proxies"""
        features = {}
        
        # Total turns proxy: dialogue complexity metric (FIXED)
        complexity_scores = []
        for text in texts:
            # Combine multiple indicators of dialogue complexity
            sent_count = max(1, text.count('.') + text.count('!') + text.count('?'))
            word_count = len(text.split())
            unique_words = len(set(text.lower().split()))
            
            # Complexity = normalized sentence count + vocabulary diversity + text hash for variation
            text_hash = hash(text) % 100 / 100.0  # 0 to 0.99
            vocab_diversity = unique_words / max(word_count, 1)
            complexity = sent_count * 0.3 + vocab_diversity * 2.0 + text_hash * 0.5
            complexity_scores.append(complexity)
        
        features['total_turns'] = float(np.mean(complexity_scores))
        
        features['avg_turn_length'] = float(np.mean([len(text.split()) for text in texts])) if texts else 0.0
        
        # Repeated turns proxy: word repetition patterns (FIXED)
        if len(texts) > 1:
            repeated_count = 0
            for i in range(1, len(texts)):
                # Check for word overlap between consecutive texts
                words_prev = set(texts[i-1].lower().split())
                words_curr = set(texts[i].lower().split())
                overlap = len(words_prev & words_curr) / max(len(words_prev | words_curr), 1)
                if overlap > 0.3:  # High word overlap = repetition
                    repeated_count += 1
            features['repeated_turns'] = float(repeated_count / max(len(texts), 1))
        else:
            # Single text: check for internal word repetition
            words = texts[0].lower().split()
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            features['repeated_turns'] = float(repeated_words / max(len(words), 1))
        
        # User corrections proxy: correction patterns
        correction_patterns = [r'no,?\s*i', r'actually', r'wait', r'i\s*said']
        correction_count = sum(self._count_patterns(text, correction_patterns) for text in texts)
        features['corrections'] = float(correction_count)
        
        return features
    
    def extract_behavioral_bundle(self, texts: List[str]) -> Dict[str, float]:
        """Extract behavioral features using simple proxies"""
        features = {}
        
        # Escalation request proxy
        escalation_count = sum(self._word_ratio(text, self.escalation_words) > 0 for text in texts)
        features['escalation_requests'] = float(escalation_count)
        
        # Negative feedback proxy: intensity-based scoring (FIXED)
        negative_feedback_score = 0.0
        for text in texts:
            neg_ratio = self._word_ratio(text, self.negative_words)
            caps_intensity = sum(1 for word in text.split() if word.isupper() and len(word) > 1) / max(len(text.split()), 1)
            punct_intensity = (text.count('!') + text.count('?')) / max(len(text), 1)
            # Combine negative words with intensity markers
            intensity_score = neg_ratio + caps_intensity * 0.5 + punct_intensity * 2
            negative_feedback_score += min(1.0, intensity_score)  # Cap at 1.0 per text
        features['negative_feedback'] = float(negative_feedback_score)
        
        return features
    
    def extract_contextual_bundle(self, texts: List[str]) -> Dict[str, float]:
        """Extract contextual features using simple proxies"""
        features = {}
        
        # Expressed urgency proxy
        urgency_ratios = [self._word_ratio(text, self.urgency_words) for text in texts]
        features['avg_urgency'] = float(np.mean(urgency_ratios))
        
        # Urgency increase proxy: text length growth pattern (FIXED)
        if len(urgency_ratios) > 1:
            features['urgency_increase'] = float(np.polyfit(range(len(urgency_ratios)), urgency_ratios, 1)[0])
        else:
            # Single text: use text length as urgency proxy (longer = more urgent elaboration)
            text_len = len(texts[0].split())
            # Normalize text length to urgency scale (10-50 words = normal, >50 = urgent)
            features['urgency_increase'] = float(max(0, (text_len - 10) / 40))  # Range: 0 to ~1+
        
        # Task complexity proxy: question density
        question_densities = [text.count('?') / max(len(text.split()), 1) for text in texts]
        features['task_complexity'] = float(np.mean(question_densities))
        
        return features
    
    def extract_emotion_dynamics_bundle(self, texts: List[str]) -> Dict[str, float]:
        """Extract emotion dynamics using simple proxies"""
        features = {}
        
        # Emotion drift proxy: emotional word progression (FIXED)
        neg_scores = [self._word_ratio(text, self.negative_words) for text in texts]
        if len(neg_scores) > 1:
            features['emotion_drift'] = float(np.polyfit(range(len(neg_scores)), neg_scores, 1)[0])
        else:
            # Single text: use emotional word diversity as drift proxy
            emotion_words = self.negative_words | {'happy', 'sad', 'angry', 'excited', 'worried'}
            unique_emotions = len(set(texts[0].lower().split()) & emotion_words)
            features['emotion_drift'] = float(unique_emotions * 0.1)  # Range: 0 to ~0.5+
        
        # Emotion volatility proxy: punctuation variance (FIXED)
        if len(neg_scores) > 1:
            features['emotion_volatility'] = float(np.var(neg_scores))
        else:
            # Single text: use punctuation patterns as volatility proxy
            punct_chars = texts[0].count('!') + texts[0].count('?') + texts[0].count('...') + texts[0].count('??')
            text_len = max(len(texts[0]), 1)
            features['emotion_volatility'] = float((punct_chars / text_len) * 0.5)  # Range: 0 to ~0.5+
        
        return features
    
    def extract_system_bundle(self, texts: List[str]) -> Dict[str, float]:
        """Extract system-related features using simple proxies"""
        features = {}
        
        # Response clarity proxy: question ratio (more questions = less clarity)
        question_ratios = [text.count('?') / max(len(text.split()), 1) for text in texts]
        features['response_clarity'] = float(1.0 - np.mean(question_ratios))  # Inverse of questions
        
        # Response relevance proxy: confusion word ratio (more confusion = less relevance)
        conf_ratios = [self._word_ratio(text, self.confusion_words) for text in texts]
        features['response_relevance'] = float(1.0 - np.mean(conf_ratios))  # Inverse of confusion
        
        return features
    
    def extract_user_model_bundle(self, texts: List[str]) -> Dict[str, float]:
        """Extract user model features using simple proxies"""
        features = {}
        
        # Trust in system proxy: doubt expressions
        doubt_words = {'sure?', 'really?', 'certain?', 'doubt', 'wrong'}
        doubt_ratios = [self._word_ratio(text, doubt_words) for text in texts]
        features['trust_decline'] = float(np.mean(doubt_ratios))
        
        return features
    
    def extract_bundle_features(self, texts: List[str], bundle_name: str) -> Dict[str, float]:
        """Extract features for a specific bundle"""
        if bundle_name == 'linguistic_bundle':
            return self.extract_linguistic_bundle(texts)
        elif bundle_name == 'dialogue_bundle':
            return self.extract_dialogue_bundle(texts)
        elif bundle_name == 'behavioral_bundle':
            return self.extract_behavioral_bundle(texts)
        elif bundle_name == 'contextual_bundle':
            return self.extract_contextual_bundle(texts)
        elif bundle_name == 'emotion_dynamics_bundle':
            return self.extract_emotion_dynamics_bundle(texts)
        elif bundle_name == 'system_bundle':
            return self.extract_system_bundle(texts)
        elif bundle_name == 'user_model_bundle':
            return self.extract_user_model_bundle(texts)
        else:
            return {}
    
    def extract_all_features(self, texts: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract features for all bundles"""
        bundles = [
            'linguistic_bundle', 'dialogue_bundle', 'behavioral_bundle',
            'contextual_bundle', 'emotion_dynamics_bundle', 'system_bundle', 'user_model_bundle'
        ]
        
        all_features = {}
        for bundle in bundles:
            all_features[bundle] = self.extract_bundle_features(texts, bundle)
        
        return all_features

def test_fast_features():
    """Test fast feature extraction"""
    logger.info("Testing fast feature extraction...")
    
    extractor = FastFrustrationFeatureExtractor()
    
    # Sample frustrated dialogue
    sample_texts = [
        "Hi, I need help with booking a flight",
        "I said I want to book a flight to PARIS",
        "No, I meant PARIS, not London! This is frustrating",
        "Can I talk to a human? This system is useless",
        "I HATE this! Nothing works!"
    ]
    
    # Test each bundle
    bundles = [
        'linguistic_bundle', 'dialogue_bundle', 'behavioral_bundle',
        'contextual_bundle', 'emotion_dynamics_bundle', 'system_bundle', 'user_model_bundle'
    ]
    
    for bundle in bundles:
        features = extractor.extract_bundle_features(sample_texts, bundle)
        logger.info(f"{bundle}: {len(features)} features - {features}")
    
    logger.info("Fast feature extraction test completed!")

if __name__ == "__main__":
    test_fast_features()