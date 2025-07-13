# src/factories/entities_factory.py

import spacy
import pandas as pd
import re
from typing import List, Dict, Set
from collections import Counter
from .base_factory import AbstractDatasetFactory

class ProperNounsDatasetFactory(AbstractDatasetFactory):
    """
    Factory for generating datasets containing unusual proper nouns for NER validation in a data-driven manner.
    
    External packages:
    - spaCy for Named Entity Recognition (NER)
    - pandas for data manipulation
    - re for string matching

    Idea:
    Unusual proper nouns are defined based on their frequency in the Jeopardy dataset.
    1. Extract all proper nouns from entire dataset using spaCy NER
    2. Count frequency of each unique proper noun across all records (might be slow, need optimization)
    3. Define "unusual" as appearing ≤3 times in the dataset
    4. Use simple string matching for final detection
    """
    
    def __init__(self, dataframe: pd.DataFrame, n_processes: int = None, frequency_threshold: int = 3):
        """
        Initialize the proper nouns detection factory.
        
        Args:
            dataframe (pd.DataFrame): Jeopardy dataset
            n_processes (int, optional): Number of processes for parallel execution
            frequency_threshold (int): Maximum frequency for "unusual" (default: 3)
        """
        super().__init__(dataframe, n_processes)
        
        # Create combined text column for efficient processing
        self.df['text'] = self.df['question'].fillna('') + ' ' + self.df['answer'].fillna('')
        
        # Focus on question and answer fields
        self.target_columns = ['question', 'answer']
        
        # Frequency threshold for "unusual" proper nouns
        self.frequency_threshold = frequency_threshold
        
        # Storage for analysis results
        self.noun_counts: Dict[str, int] = {}
        self.unusual_nouns: Set[str] = set()
        self.frequency_analysis_complete = False
        
        # Proper noun entity labels (from reference implementation)
        self.proper_noun_labels = {'PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART', 'LOC'}
    
    def get_subset_name(self) -> str:
        """Return the subset identifier."""
        return "proper_nouns"
    
    def analyze_entity_frequencies(self, show_progress: bool = True) -> Dict[str, int]:
        """
        Phase 1: Analyze entire dataset to count proper noun frequencies.
        
        Uses spaCy's efficient nlp.pipe() for optimal performance with multiprocessing.
        
        Args:
            show_progress (bool): Whether to show progress
            
        Returns:
            Dict[str, int]: Entity frequencies across entire dataset
        """
        if self.frequency_analysis_complete:
            self.logger.info("Frequency analysis already completed")
            return self.noun_counts
        
        self.logger.info(f"Analyzing proper noun frequencies across {len(self.df):,} records...")
        
        # Use multiprocessing to split the work across processes
        if self.n_processes > 1:
            return self._analyze_frequencies_parallel(show_progress)
        else:
            return self._analyze_frequencies_single(show_progress)
    
    def _analyze_frequencies_single(self, show_progress: bool) -> Dict[str, int]:
        """Single-threaded frequency analysis using spaCy's efficient pipe."""
        import spacy
        from tqdm import tqdm
        
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error("spaCy model not available")
            return {}
        
        self.logger.info("Extracting proper nouns using spaCy NER (single-threaded)...")
        noun_counts = Counter()
        
        # Use spaCy's efficient pipe processing
        texts = self.df['text'].astype(str)
        
        if show_progress:
            # Process with progress bar
            for doc in tqdm(nlp.pipe(texts, batch_size=100), 
                           total=len(texts), desc="Processing texts"):
                for ent in doc.ents:
                    if ent.label_ in self.proper_noun_labels:
                        # Normalize to lowercase for counting
                        noun_counts[ent.text.lower()] += 1
        else:
            # Process without progress bar
            for doc in nlp.pipe(texts, batch_size=100):
                for ent in doc.ents:
                    if ent.label_ in self.proper_noun_labels:
                        noun_counts[ent.text.lower()] += 1
        
        self.noun_counts = dict(noun_counts)
        self._finalize_analysis()
        return self.noun_counts
    
    def _analyze_frequencies_parallel(self, show_progress: bool) -> Dict[str, int]:
        """Parallel frequency analysis by splitting data across processes."""
        from multiprocessing import Pool
        from tqdm import tqdm
        
        # Split dataframe into chunks for parallel processing
        chunk_size = len(self.df) // self.n_processes
        chunks = [
            self.df.iloc[i:i + chunk_size] 
            for i in range(0, len(self.df), chunk_size)
        ]
        
        self.logger.info(f"Processing {len(self.df):,} records in {len(chunks)} chunks "
                        f"using {self.n_processes} processes")
        
        try:
            with Pool(self.n_processes) as pool:
                if show_progress:
                    results = []
                    with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                        for result in pool.imap(self._process_chunk_for_entities, chunks):
                            results.append(result)
                            pbar.update(1)
                else:
                    results = pool.map(self._process_chunk_for_entities, chunks)
            
            # Combine results from all processes
            combined_counts = Counter()
            for chunk_counts in results:
                combined_counts.update(chunk_counts)
            
            self.noun_counts = dict(combined_counts)
            self._finalize_analysis()
            return self.noun_counts
            
        except Exception as e:
            self.logger.warning(f"Parallel processing failed: {e}, falling back to single-threaded")
            return self._analyze_frequencies_single(show_progress)
    
    def _process_chunk_for_entities(self, chunk_df: pd.DataFrame) -> Dict[str, int]:
        """Process a chunk of data to extract entity frequencies."""
        import spacy
        
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return {}
        
        noun_counts = Counter()
        
        # Create text column if not exists
        if 'text' not in chunk_df.columns:
            chunk_df = chunk_df.copy()
            chunk_df['text'] = chunk_df['question'].fillna('') + ' ' + chunk_df['answer'].fillna('')
        
        # Use spaCy's efficient pipe processing
        texts = chunk_df['text'].astype(str)
        
        for doc in nlp.pipe(texts, batch_size=50):  # Smaller batch for memory efficiency
            for ent in doc.ents:
                if ent.label_ in self.proper_noun_labels:
                    noun_counts[ent.text.lower()] += 1
        
        return dict(noun_counts)
    
    def _finalize_analysis(self):
        """Finalize the frequency analysis by identifying unusual nouns."""
        # Identify unusual nouns based on frequency threshold
        self.unusual_nouns = {
            noun for noun, count in self.noun_counts.items() 
            if count <= self.frequency_threshold
        }
        
        self.frequency_analysis_complete = True
        
        total_entities = len(self.noun_counts)
        unusual_count = len(self.unusual_nouns)
        
        self.logger.info(f"Frequency analysis completed:")
        self.logger.info(f"  - Total unique proper nouns: {total_entities:,}")
        self.logger.info(f"  - Unusual nouns (≤{self.frequency_threshold}): {unusual_count:,}")
        self.logger.info(f"  - Percentage unusual: {(unusual_count/total_entities*100):.1f}%")
        
        # Log most common entities for insight
        most_common = sorted(self.noun_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        self.logger.info(f"  - Most common: {dict(most_common)}")
    
    def detect_features_batch(self, text_batch: List[str]) -> List[bool]:
        """
        Detect texts containing unusual proper nouns using simple string matching.

        Args:
            text_batch (List[str]): Batch of combined text strings
            
        Returns:
            List[bool]: Detection results for each text
        """
        if not self.frequency_analysis_complete:
            raise ValueError("Must run analyze_entity_frequencies() before detection")
        
        results = []
        
        for text in text_batch:
            if not text or pd.isna(text) or text.strip() == "":
                results.append(False)
                continue
            
            text_lower = str(text).lower()
            
            # Simple string matching
            # Check if any unusual noun appears in the text
            has_unusual = any(noun in text_lower for noun in self.unusual_nouns)
            
            results.append(has_unusual)
        
        return results
    
    def apply_detection_parallel(self, show_progress: bool = True) -> List[bool]:
        """
        Override to ensure frequency analysis is completed first.
        
        Args:
            show_progress (bool): Whether to show progress bars
            
        Returns:
            List[bool]: Detection results for all records
        """
        # Step 1: Analyze frequencies if not done
        if not self.frequency_analysis_complete:
            self.analyze_entity_frequencies(show_progress)
        
        # Step 2: Run detection using the optimized approach
        return super().apply_detection_parallel(show_progress)
    
    def get_frequency_stats(self) -> Dict[str, any]:
        """Get comprehensive frequency statistics."""
        if not self.frequency_analysis_complete:
            return {"error": "Frequency analysis not completed"}
        
        # Categorize by frequency
        categories = {
            'very_rare': [],      # Appears 1 time
            'rare': [],           # Appears 2-3 times
            'uncommon': [],       # Appears 4-10 times
            'common': [],         # Appears >10 times
        }
        
        for noun, freq in self.noun_counts.items():
            if freq == 1:
                categories['very_rare'].append((noun, freq))
            elif freq <= 3:
                categories['rare'].append((noun, freq))
            elif freq <= 10:
                categories['uncommon'].append((noun, freq))
            else:
                categories['common'].append((noun, freq))
        
        return {
            'total_unique_entities': len(self.noun_counts),
            'unusual_entities_count': len(self.unusual_nouns),
            'frequency_threshold': self.frequency_threshold,
            'categories': {
                cat: {
                    'count': len(entities),
                    'percentage': len(entities) / len(self.noun_counts) * 100,
                    'examples': entities[:10]  # First 10 examples
                }
                for cat, entities in categories.items()
            },
            'most_common_20': sorted(self.noun_counts.items(), 
                                   key=lambda x: x[1], reverse=True)[:20],
            'sample_unusual_20': list(self.unusual_nouns)[:20]
        }
    
    def get_detection_details(self, text: str) -> dict:
        """Get detailed results showing which unusual entities were found."""
        if not self.frequency_analysis_complete:
            return {"error": "Frequency analysis not completed"}
        
        text_lower = str(text).lower()
        found_unusual = []
        
        # Find which unusual nouns appear in this text
        for noun in self.unusual_nouns:
            if noun in text_lower:
                found_unusual.append({
                    'text': noun,
                    'frequency': self.noun_counts[noun],
                    'positions': [m.start() for m in re.finditer(re.escape(noun), text_lower)]
                })
        
        return {
            'has_unusual': len(found_unusual) > 0,
            'unusual_count': len(found_unusual),
            'unusual_entities_found': found_unusual,
            'text_length': len(text)
        }
    

def main():
    """Simple test function for ProperNounsDatasetFactory."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data with various proper nouns (some common, some rare)
    test_data = pd.DataFrame({
        'question': [
            'Who was Shakespeare?',
            'What is the capital of Zarathia?',
            'Who founded Microsoft?',
            'Where is Mount Everest?',
            'What did Nostradamus predict?',
            'Who painted the Mona Lisa?',
            'What is Blorbnoxia famous for?'
        ],
        'answer': [
            'Who is William Shakespeare?',
            'What is Mythopolis?',
            'Who is Bill Gates?',
            'What is Nepal?',
            'What are future events?',
            'Who is Leonardo da Vinci?',
            'What is nothing?'
        ],
        'category': ['LITERATURE', 'GEOGRAPHY', 'BUSINESS', 'GEOGRAPHY', 'HISTORY', 'ART', 'FICTIONAL']
    })
    
    print("Testing ProperNounsDatasetFactory...")
    print(f"Test data: {len(test_data)} records")
    
    # Test factory
    factory = ProperNounsDatasetFactory(test_data, n_processes=1, frequency_threshold=2)
    
    # Test frequency analysis
    print("\nAnalyzing entity frequencies...")
    frequencies = factory.analyze_entity_frequencies(show_progress=False)
    print(f"Found {len(frequencies)} unique entities")
    print(f"Sample frequencies: {dict(list(frequencies.items())[:5])}")
    
    # Test detection
    print("\nRunning detection...")
    results = factory.apply_detection_parallel(show_progress=False)
    print(f"Detection results: {results}")
    
    # Test frequency stats
    stats = factory.get_frequency_stats()
    print(f"\nFrequency stats:")
    print(f"  Total entities: {stats['total_unique_entities']}")
    print(f"  Unusual entities: {stats['unusual_entities_count']}")
    
    # Test detection details
    sample_text = "What is the capital of Zarathia called Mythopolis?"
    details = factory.get_detection_details(sample_text)
    print(f"\nDetection details for '{sample_text}':")
    print(f"  Has unusual: {details['has_unusual']}")
    print(f"  Found entities: {details['unusual_entities_found']}")
    
    # Generate subset
    subset = factory.generate_subset(size=3)
    print(f"\nGenerated subset: {len(subset)} records")
    
    print("✓ Test completed")


if __name__ == "__main__":
    main()