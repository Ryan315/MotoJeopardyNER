# src/factories/numbers_factory.py

import re
import spacy
import pandas as pd
from typing import List
from .base_factory import AbstractDatasetFactory

class NumbersDatasetFactory(AbstractDatasetFactory):
    """
    Factory for generating datasets containing numbers for NER validation.
    
    External packages:
    - spaCy: For natural language processing and number detection
    - pandas: For data manipulation and handling

    Ideas:
    Uses spaCy's optimized token.like_num feature which automatically detects:
    - Written numbers: "five", "twenty-one", "first", "second"
    - Digit numbers: "123", "7pm", "3.14", "$50"
    - Mixed formats: "twenty-first", "1990s"
    - Complex expressions: "twenty-five", "3.5 million"
    
    Falls back to Roman numeral regex for edge cases spaCy might miss.
    """
    
    def __init__(self, dataframe: pd.DataFrame, n_processes: int = None):
        """
        Initialization of the NumbersDatasetFactory.
        
        Args:
            dataframe (pd.DataFrame): Jeopardy dataset
            n_processes (int, optional): Number of processes for parallel execution
        """
        super().__init__(dataframe, n_processes)
        
        # Focus on question and answer primarily, category secondarily
        self.target_columns = ['question', 'answer', 'category']
        
        # Roman numeral regex as fallback (spaCy sometimes misses these)
        self.roman_numeral_regex = re.compile(r'\b[MDCLXVI]+\b', re.IGNORECASE)
    
    def get_subset_name(self) -> str:
        return "numbers"
    
    def detect_features_batch(self, text_batch: List[str]) -> List[bool]:
        """
        Detect numbers using spaCy's token.like_num feature with Roman numeral fallback.
        
        1. Use spaCy's token.like_num for comprehensive number detection
        2. Roman numeral regex as fallback for edge cases
        
        Args:
            text_batch (List[str]): Batch of combined text strings
            
        Returns:
            List[bool]: Detection results for each text
        """
        # Load spaCy model in each process (required for multiprocessing)
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not available, using regex-only fallback")
            return self._detect_with_regex_only(text_batch)
        
        results = []
        
        for text in text_batch:
            if not text or pd.isna(text) or text.strip() == "":
                results.append(False)
                continue
            
            text_str = str(text).strip()
            has_numbers = False
            
            try:
                # Primary method: spaCy's like_num feature
                doc = nlp(text_str)
                
                # Check if any token is number-like
                for token in doc:
                    if token.like_num:
                        has_numbers = True
                        break  # Found one, that's enough
                
            except Exception:
                # Continue with fallback if spaCy processing fails
                pass
            
            # Fallback: Roman numeral detection (spaCy sometimes misses these)
            if not has_numbers:
                if self.roman_numeral_regex.search(text_str):
                    has_numbers = True
            
            results.append(has_numbers)
        
        return results
    
    def _detect_with_regex_only(self, text_batch: List[str]) -> List[bool]:
        """Fallback detection using only regex patterns if spaCy unavailable."""
        results = []
        
        # Basic number patterns for fallback
        number_patterns = [
            r'\b\d+\b',                           # Basic digits
            r'\$[\d,]+(?:\.\d{2})?',              # Currency
            r'\b\d+(?:\.\d+)?%',                  # Percentages
            r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',
            self.roman_numeral_regex.pattern      # Roman numerals
        ]
        
        combined_pattern = re.compile('|'.join(f'({pattern})' for pattern in number_patterns), re.IGNORECASE)
        
        for text in text_batch:
            if not text or pd.isna(text) or text.strip() == "":
                results.append(False)
                continue
            
            text_str = str(text).strip()
            has_numbers = bool(combined_pattern.search(text_str))
            results.append(has_numbers)
        
        return results
    
    def get_detection_details(self, text: str) -> dict:
        """
        Get detailed detection results for a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Detailed detection results showing what numbers were found
        """
        if not text or pd.isna(text):
            return {}
        
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(str(text).strip())
            
            number_tokens = []
            spacy_entities = []
            
            # Collect number-like tokens
            for token in doc:
                if token.like_num:
                    number_tokens.append({
                        'text': token.text,
                        'pos': token.pos_,
                        'lemma': token.lemma_,
                        'start': token.idx,
                        'end': token.idx + len(token.text),
                        'is_alpha': token.is_alpha,
                        'is_digit': token.is_digit
                    })
            
            # Collect number-related entities
            for ent in doc.ents:
                if ent.label_ in ["CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT", "DATE"]:
                    spacy_entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            
            # Check for Roman numerals
            roman_matches = []
            for match in self.roman_numeral_regex.finditer(str(text)):
                roman_matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
            
            return {
                'number_tokens': number_tokens,
                'spacy_entities': spacy_entities,
                'roman_numerals': roman_matches,
                'total_numbers_found': len(number_tokens) + len(roman_matches),
                'detection_method': 'spacy_like_num'
            }
            
        except Exception as e:
            return {'error': str(e)}
        

def main():
    """Simple test function for NumbersDatasetFactory."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data with various number formats
    test_data = pd.DataFrame({
        'question': [
            'This element has atomic number 6',
            'This Roman emperor ruled from VI to IX',
            'How many sides does a triangle have?',
            'What color is the sky?',
            'This happened in the 1990s'
        ],
        'answer': [
            'What is Carbon?',
            'Who is Augustus?',
            'What is three?',
            'What is blue?',
            'What is the Internet?'
        ],
        'category': ['SCIENCE', 'HISTORY', 'MATH', 'NATURE', 'TECHNOLOGY']
    })
    
    print("Testing NumbersDatasetFactory...")
    print(f"Test data: {len(test_data)} records")
    
    # Test factory
    factory = NumbersDatasetFactory(test_data, n_processes=1)
    
    # Test detection
    results = factory.apply_detection_parallel(show_progress=False)
    print(f"Detection results: {results}")
    
    # Test detailed detection
    sample_text = "This element has atomic number 6"
    details = factory.get_detection_details(sample_text)
    print(f"Details for '{sample_text}': {details}")
    
    # Generate subset
    subset = factory.generate_subset(size=3)
    print(f"Generated subset: {len(subset)} records")
    
    print("✓ Test completed")


if __name__ == "__main__":
    main()