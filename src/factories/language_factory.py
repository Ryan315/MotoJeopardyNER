# src/factories/language_factory.py (LEMMA-BASED SOLUTION)

import re
import spacy
import pandas as pd
from typing import List, Dict, Set
from .base_factory import AbstractDatasetFactory

class NonEnglishDatasetFactory(AbstractDatasetFactory):
    """
    External package:
    - spaCy for lemmatization and tokenization
    - NLTK for English dictionary lookup

    Ideas:
    Dictionary-based method like NLTK can't handle lemmatization,
    so I need to use spaCy's lemmatization to get the base form of words
    and then check against the NLTK dictionary.    
    
    Example:
    'kisses' → lemma: 'kiss' → In dictionary ✅
    'became' → lemma: 'become' → In dictionary ✅  
    'ciao' → lemma: 'ciao' → Not in dictionary ❌ (correctly detected!)
    """
    
    def __init__(self, dataframe: pd.DataFrame, n_processes: int = None):
        super().__init__(dataframe, n_processes)
        self.target_columns = ['question', 'answer']
        self._setup_detection_resources()
    
    def _setup_detection_resources(self):
        """Setup detection with lemma-based approach."""
        
        # Non-ASCII pattern for accented characters.
        self.non_ascii_pattern = re.compile(r'[à-ÿÀ-ßĀ-žА-я]')
    
    def get_subset_name(self) -> str:
        return "non_english"
    
    def detect_features_batch(self, text_batch: List[str]) -> List[bool]:
        """
        Lemma-based detection using spaCy's lemmatization.
        """
        nlp, english_words = self._load_detection_resources()
        
        if nlp is None or english_words is None:
            return self._detect_with_patterns_only(text_batch)
        
        results = []
        
        for text in text_batch:
            if not text or pd.isna(text) or text.strip() == "":
                results.append(False)
                continue
            
            text_str = str(text).strip()
            has_non_english = self._lemma_based_detection(text_str, nlp, english_words)
            results.append(has_non_english)
        
        return results
    
    def _load_detection_resources(self):
        """Load spaCy model and NLTK dictionary."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not available")
            nlp = None
        
        try:
            import nltk
            from nltk.corpus import words
            
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                self.logger.info("Downloading NLTK words corpus...")
                nltk.download('words', quiet=True)
            
            # Use NLTK dictionary as-is (no need for variants with lemma approach)
            english_words = set(word.lower() for word in words.words())
            
        except Exception as e:
            self.logger.warning(f"NLTK dictionary not available: {e}")
            english_words = None
        
        return nlp, english_words
    
    def _lemma_based_detection(self, text: str, nlp, english_words: set) -> bool:
        """
        Clean lemma-based detection approach.
        """
        if self.non_ascii_pattern.search(text):
            return True
        
        try:
            doc = nlp(text)
            
            for token in doc:
                if (token.is_alpha and                    # Alphabetic only
                    len(token.text) > 2 and               # Skip very short words
                    token.pos_ != 'PROPN'):               # Skip proper nouns
                    
                    # Check LEMMA against dictionary
                    lemma_lower = token.lemma_.lower()
                    
                    if lemma_lower not in english_words:
                        # Additional validation for edge cases
                        if not self._is_likely_english_artifact(token.text.lower()):
                            return True
            
        except Exception:
            # Fallback to pattern matching if spaCy fails
            pass
        
        return False
    
    def _is_likely_english_artifact(self, word: str) -> bool:
        """
        Handle edge cases where lemmatization might not work perfectly.
        """
        # Common contractions that might confuse lemmatizer
        english_artifacts = {
            "n't", "'s", "'re", "'ll", "'ve", "'d", "'m",  # Contractions
            "gon", "na", "wan", "ta",                      # Informal speech
            "ok", "okay", "yeah", "yep", "nope", "hmm",    # Informal words
        }
        
        if word in english_artifacts:
            return True
        
        # Very short words are often English function words
        if len(word) <= 2:
            return True
        
        return False
    
    def _detect_with_patterns_only(self, text_batch: List[str]) -> List[bool]:
        """Fallback detection using only non-ASCII pattern."""
        results = []
        
        for text in text_batch:
            if not text or pd.isna(text) or text.strip() == "":
                results.append(False)
                continue
            
            text_str = str(text).strip()
            
            # Only detect non-ASCII characters as fallback
            has_non_ascii = bool(self.non_ascii_pattern.search(text_str))
            results.append(has_non_ascii)
        
        return results
    
    def get_detection_details(self, text: str) -> dict:
        """Get detailed detection results using lemma approach."""
        nlp, english_words = self._load_detection_resources()
        
        if nlp is None or english_words is None:
            return {"error": "Detection resources not available"}
        
        details = {
            'non_english_words': [],
            'non_ascii_chars': [],
            'analysis_method': 'spacy_lemma_based'
        }
        
        # Check non-ASCII characters
        non_ascii_matches = self.non_ascii_pattern.findall(text)
        if non_ascii_matches:
            details['non_ascii_chars'] = list(set(non_ascii_matches))
        
        # Lemma-based analysis
        try:
            doc = nlp(str(text).strip())
            
            for token in doc:
                if (token.is_alpha and 
                    len(token.text) > 2 and 
                    token.pos_ != 'PROPN'):
                    
                    lemma_lower = token.lemma_.lower()
                    
                    if (lemma_lower not in english_words and 
                        not self._is_likely_english_artifact(token.text.lower())):
                        
                        details['non_english_words'].append({
                            'word': token.text,
                            'lemma': token.lemma_,
                            'pos': token.pos_,
                            'start': token.idx,
                            'end': token.idx + len(token.text)
                        })
        
        except Exception as e:
            details['error'] = str(e)
        
        return details
    
    def find_non_english_lemma_check(self, phrase: str) -> List[str]:
        nlp, english_words = self._load_detection_resources()
        
        if nlp is None or english_words is None:
            return []
        
        doc = nlp(phrase)
        non_english_words = []
        
        for token in doc:
            # Check the LEMMA against the wordlist
            if token.is_alpha and token.lemma_.lower() not in english_words:
                # We still ignore proper nouns for better accuracy
                if token.pos_ != 'PROPN':
                    non_english_words.append(token.text)
        
        return non_english_words
    

def main():
    """Simple test function for NonEnglishDatasetFactory."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data with mixed English and non-English content
    test_data = pd.DataFrame({
        'question': [
            'What is the capital of France?',
            'This French word is "bonjour" meaning hello',
            'What color is café con leche?',
            'The German word Schadenfreude means what?',
            'How do you say gracias in English?'
        ],
        'answer': [
            'What is Paris?',
            'What is hello?',
            'What is brown?',
            'What is pleasure from others misfortune?',
            'What is thank you?'
        ],
        'category': ['GEOGRAPHY', 'LANGUAGE', 'FOOD', 'PSYCHOLOGY', 'LANGUAGE']
    })
    
    print("Testing NonEnglishDatasetFactory...")
    print(f"Test data: {len(test_data)} records")
    
    # Test factory
    factory = NonEnglishDatasetFactory(test_data, n_processes=1)
    
    # Test detection
    results = factory.apply_detection_parallel(show_progress=False)
    print(f"Detection results: {results}")
    
    # Test lemma check directly
    test_phrases = [
        "bonjour my friend",
        "café con leche",
        "Schadenfreude is interesting",
        "completely normal English"
    ]
    
    for phrase in test_phrases:
        non_english = factory.find_non_english_lemma_check(phrase)
        print(f"'{phrase}' → non-English words: {non_english}")
    
    # Test detailed detection
    details = factory.get_detection_details("This café serves excellent croissants")
    print(f"Detection details: {details}")
    
    # Generate subset
    subset = factory.generate_subset(size=3)
    print(f"Generated subset: {len(subset)} records")
    
    print("✓ Test completed")


if __name__ == "__main__":
    main()