from abc import ABC, abstractmethod
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Dict, Any, Optional
import time
import os
import logging
from tqdm import tqdm

class AbstractDatasetFactory(ABC):
    """
    Abstract factory for generating specialized NER validation subsets.
    
    Implementation targets:
    - Multiprocessing support for fast processing of 200K+ records
    - Batch processing with progress tracking
    - Subset generation and export to JSONL format
    - Performance monitoring and statistics
    - Abstract methods for independent detection logic
    """
    
    def __init__(self, dataframe: pd.DataFrame, n_processes: Optional[int] = None):
        """
        Initialize the factory with a DataFrame and processing configuration.
        
        Args:
            dataframe (pd.DataFrame): The Jeopardy dataset to process
            n_processes (int, optional): Number of processes to use. Defaults to CPU count.
        """
        self.df = dataframe.copy()  # Work with a copy to avoid modifying original
        self.n_processes = n_processes or mp.cpu_count()
        self.detection_results: Optional[List[bool]] = None
        self.processing_stats: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Columns to analyze (can be overridden by subclasses)
        self.target_columns = ['question', 'answer', 'category']
    
    @abstractmethod
    def detect_features_batch(self, text_batch: List[str]) -> List[bool]:
        """
        Detect features in a batch of texts using specific detection logic.
        
        Args:
            text_batch (List[str]): Batch of combined text strings to analyze
            
        Returns:
            List[bool]: Detection results for each text in the batch
        """
        pass
    
    @abstractmethod
    def get_subset_name(self) -> str:
        """
        Get the name identifier for this subset type.
        
        Returns:
            str: Subset name (e.g., 'numbers', 'non_english', 'proper_nouns')
        """
        pass
    
    def prepare_texts(self) -> List[str]:
        """
        Prepare combined text strings from target columns for processing.
        
        Returns:
            List[str]: Combined text strings from specified columns
        """
        combined_texts = []
        
        for _, row in self.df.iterrows():
            # Combine text from target columns, handling NaN values
            text_parts = []
            for col in self.target_columns:
                if col in self.df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            
            combined_text = " ".join(text_parts)
            combined_texts.append(combined_text)
        
        return combined_texts
    
    def apply_detection_parallel(self, show_progress: bool = True) -> List[bool]:
        """
        Apply detection using multiprocessing with progress tracking.
        
        Args:
            show_progress (bool): Whether to show progress bar
            
        Returns:
            List[bool]: Detection results for all records
        """
        subset_name = self.get_subset_name()
        self.logger.info(f"Starting {subset_name} detection using {self.n_processes} processes...")
        
        start_time = time.time()
        
        # Prepare combined texts
        combined_texts = self.prepare_texts()
        total_records = len(combined_texts)
        
        # Calculate optimal chunk size for load balancing
        chunk_size = max(500, total_records // (self.n_processes * 3))
        
        # Split into chunks
        chunks = [
            combined_texts[i:i + chunk_size] 
            for i in range(0, len(combined_texts), chunk_size)
        ]
        
        self.logger.info(f"Processing {total_records:,} records in {len(chunks)} chunks "
                        f"(~{chunk_size} records per chunk)")
        
        # Process chunks in parallel
        try:
            with Pool(self.n_processes) as pool:
                if show_progress:
                    # Use tqdm for progress tracking
                    results = []
                    with tqdm(total=len(chunks), desc=f"Processing {subset_name}") as pbar:
                        for result in pool.imap(self.detect_features_batch, chunks):
                            results.append(result)
                            pbar.update(1)
                else:
                    results = pool.map(self.detect_features_batch, chunks)
            
            # Flatten results
            self.detection_results = [item for sublist in results for item in sublist]
            
        except Exception as e:
            self.logger.error(f"Multiprocessing failed: {e}")
            # Fallback to single-threaded processing
            self.logger.info("Falling back to single-threaded processing...")
            self.detection_results = self.detect_features_batch(combined_texts)
        
        # Record processing statistics
        processing_time = time.time() - start_time
        detected_count = sum(self.detection_results)
        
        self.processing_stats = {
            'total_records': total_records,
            'detected_records': detected_count,
            'detection_rate': (detected_count / total_records * 100) if total_records > 0 else 0,
            'processing_time': processing_time,
            'records_per_second': total_records / processing_time if processing_time > 0 else 0,
            'chunks_processed': len(chunks),
            'processes_used': self.n_processes
        }
        
        self.logger.info(f"✓ {subset_name} detection completed: {detected_count:,}/{total_records:,} "
                        f"records ({self.processing_stats['detection_rate']:.1f}%) "
                        f"in {processing_time:.1f}s")
        
        return self.detection_results
    
    def generate_subset(self, size: int = 1000, random_state: Optional[int] = 42) -> pd.DataFrame:
        """
        Generate a subset based on detection results.
        
        Args:
            size (int): Target size of the subset
            random_state (int, optional): Random seed for reproducible sampling
            
        Returns:
            pd.DataFrame: Generated subset
            
        Raises:
            ValueError: If detection hasn't been run yet
        """
        if self.detection_results is None:
            raise ValueError("Detection must be run before generating subset. "
                           "Call apply_detection_parallel() first.")
        
        # Add detection results to DataFrame
        filter_column = f"has_{self.get_subset_name()}"
        self.df[filter_column] = self.detection_results
        
        # Filter records that match the detection criteria
        filtered_df = self.df[self.df[filter_column] == True]
        
        if len(filtered_df) == 0:
            self.logger.warning(f"No records found matching {self.get_subset_name()} criteria")
            return pd.DataFrame()
        
        # Sample subset
        subset_size = min(size, len(filtered_df))
        
        if len(filtered_df) >= size:
            subset = filtered_df.sample(n=size, random_state=random_state)
            self.logger.info(f"Sampled {size:,} records from {len(filtered_df):,} candidates")
        else:
            subset = filtered_df.copy()
            self.logger.warning(f"Only {len(filtered_df):,} records available, "
                              f"returning all (requested {size:,})")
        
        return subset
    
    def export_subset(self, subset: pd.DataFrame, output_file: str) -> int:
        """
        Export subset to JSONL format.
        
        Args:
            subset (pd.DataFrame): Subset to export
            output_file (str): Output file path
            
        Returns:
            int: Number of records exported
        """
        if len(subset) == 0:
            self.logger.warning(f"Empty subset - no records to export to {output_file}")
            return 0
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Export only original Jeopardy columns
        original_columns = [
            'category', 'value', 'question', 'answer', 
            'round', 'show_number', 'air_date'
        ]
        
        export_subset = subset[original_columns]
        
        # Export to JSONL format
        export_subset.to_json(output_file, orient='records', lines=True)
        
        self.logger.info(f"✓ Exported {len(export_subset):,} records to {output_file}")
        return len(export_subset)
    
    def process_and_generate(self, size: int = 1000, output_file: Optional[str] = None, 
                           show_progress: bool = True) -> Dict[str, Any]:
        """
        Complete processing pipeline: detect, generate subset, and export.
        
        Args:
            size (int): Target subset size
            output_file (str, optional): Output file path. Auto-generated if None.
            show_progress (bool): Whether to show progress bars
            
        Returns:
            Dict[str, Any]: Processing results including subset and statistics
        """
        subset_name = self.get_subset_name()
        
        # Step 1: Detection
        self.apply_detection_parallel(show_progress=show_progress)
        
        # Step 2: Generate subset
        subset = self.generate_subset(size=size)
        
        # Step 3: Export (if output file specified)
        exported_count = 0
        if output_file:
            exported_count = self.export_subset(subset, output_file)
        
        # Return comprehensive results
        return {
            'subset_name': subset_name,
            'subset_dataframe': subset,
            'exported_count': exported_count,
            'output_file': output_file,
            'processing_stats': self.processing_stats,
            'detection_summary': {
                'total_candidates': self.processing_stats.get('detected_records', 0),
                'subset_size': len(subset),
                'sampling_ratio': len(subset) / max(1, self.processing_stats.get('detected_records', 1))
            }
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get detailed processing statistics.
        
        Returns:
            Dict[str, Any]: Processing performance metrics
        """
        return self.processing_stats.copy()

# ---
# Test implementation of the abstract factory
class TestDatasetFactory(AbstractDatasetFactory):
    def detect_features_batch(self, text_batch: List[str]) -> List[bool]:
        """
        Detect if text contains any digits.
        
        Args:
            text_batch (List[str]): Batch of text strings to analyze
            
        Returns:
            List[bool]: True if text contains digits, False otherwise
        """
        return [any(char.isdigit() for char in text) for text in text_batch]
    
    def get_subset_name(self) -> str:
        """
        Get the name identifier for this subset type.
        
        Returns:
            str: Subset name
        """
        return "numbers_test"

def main():
    """
    Test function to verify the AbstractDatasetFactory implementation.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing AbstractDatasetFactory...")
    
    # Create sample test data (simulating Jeopardy dataset)
    test_data = {
        'category': ['SCIENCE', 'HISTORY', 'SPORTS', 'MOVIES', 'LITERATURE'],
        'value': [200, 400, 600, 800, 1000],
        'question': [
            'This element has atomic number 6',
            'This war ended in 1945',
            'This team won the World Series in 2020',
            'This movie won Best Picture in 2019',
            'This author wrote "1984"'
        ],
        'answer': [
            'What is Carbon?',
            'What is World War II?',
            'Who are the Los Angeles Dodgers?',
            'What is Parasite?',
            'Who is George Orwell?'
        ],
        'round': ['Jeopardy!'] * 5,
        'show_number': [1, 2, 3, 4, 5],
        'air_date': ['2023-01-01'] * 5
    }
    
    df = pd.DataFrame(test_data)
    print(f"Created test dataset with {len(df)} records")
    print(f"Sample record: {df.iloc[0].to_dict()}")
    
    try:
        # Test the factory
        factory = TestDatasetFactory(df, n_processes=2)
        print(f"Initialized factory with {factory.n_processes} processes")
        
        # Test text preparation
        texts = factory.prepare_texts()
        print(f"Prepared {len(texts)} text strings")
        print(f"Sample text: '{texts[0]}'")
        
        # Test detection
        print("\nRunning detection...")
        results = factory.apply_detection_parallel(show_progress=True)
        print(f"Detection results: {results}")
        
        # Test subset generation
        print("\nGenerating subset...")
        subset = factory.generate_subset(size=3)
        print(f"Generated subset with {len(subset)} records")
        
        # Test export (create temp directory)
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_subset.jsonl")
            exported_count = factory.export_subset(subset, output_file)
            print(f"Exported {exported_count} records to {output_file}")
            
            # Verify the file exists and has content
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read()
                print(f"Output file size: {len(content)} characters")
        
        # Test complete pipeline
        print("\nTesting complete pipeline...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "pipeline_test.jsonl")
            results = factory.process_and_generate(
                size=2, 
                output_file=output_file,
                show_progress=False
            )
            
            print("Pipeline results:")
            print(f"  Subset name: {results['subset_name']}")
            print(f"  Exported count: {results['exported_count']}")
            print(f"  Detection rate: {results['processing_stats']['detection_rate']:.1f}%")
            print(f"  Processing time: {results['processing_stats']['processing_time']:.3f}s")
        
        print("\n✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)