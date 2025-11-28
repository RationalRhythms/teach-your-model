# uncertainty.py
import numpy as np
from typing import List

class UncertaintyCalculator:
    def __init__(self, forsale_boost: float = 0.3):
        self.forsale_boost = forsale_boost
        self.class_names = ['technology', 'science', 'recreation', 'politics', 'forsale']
    
    
    def should_request_human_review(self, probabilities: np.ndarray ):
        uncertainty_threshold = 0.6
        
        confidence = np.max(probabilities)
        base_uncertainty = 1 - confidence
        reasons = ""

        # Boost uncertainty for potential forsale samples
        forsale_prob = probabilities[4]  # forsale is class 4
        boosted_uncertainty = base_uncertainty + (forsale_prob * self.forsale_boost)

        uncertainty = np.max(boosted_uncertainty)
        if uncertainty > uncertainty_threshold:
            reasons = reasons+(f"High Uncertainity ({uncertainty:.2f})")
            return True, reasons
        
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) > 1 and (sorted_probs[0] - sorted_probs[1]) <= 0.1:
            second_class = np.argsort(probabilities)[-2]
            reasons = reasons+(f"close call with {self.class_names[second_class]}")
            return True, reasons
        return False, reasons
        
