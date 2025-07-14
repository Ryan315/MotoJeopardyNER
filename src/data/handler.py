# src/data/handler.py

import pandas as pd
import os
import logging
from typing import Dict, Any, Optional

class JeopardyDataHandler:    
    def __init__(self, json_file_path: str, test_limit: Optional[int] = None):
        self.json_file = json_file_path
        self.df = None
        self.logger = logging.getLogger(__name__)
        self.test_limit = test_limit
        
    def load_and_validate(self) -> pd.DataFrame:
        """Load and validate the Jeopardy dataset."""
        
        # Check file exists
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"File not found: {self.json_file}")
        
        # Load data
        self.logger.info(f"Loading data from {self.json_file}...")
        print(f"Loading data from {self.json_file}...")
        
        try:
            # Add test limit if specified for testing data_cleaner
            if self.test_limit:
                self.df = pd.read_json(self.json_file)
                if len(self.df) > self.test_limit:
                    self.df = self.df.head(self.test_limit)
                self.logger.info(f"Successfully loaded {len(self.df):,} records (limited to {self.test_limit:,} for testing)")
                print(f"✓ Loaded {len(self.df):,} records (TEST MODE - limited to {self.test_limit:,})")
            else:
                self.df = pd.read_json(self.json_file)
                self.logger.info(f"Successfully loaded {len(self.df):,} records")
                print(f"✓ Loaded {len(self.df):,} records")
        except Exception as e:
            raise ValueError(f"Failed to load JSON data: {e}")
        
        # Validate schema
        self._validate_schema()
        
        # Clean data
        self._clean_data()
        
        return self.df
    
    def _validate_schema(self) -> None:
        """Validate expected Jeopardy data structure."""
        required_columns = [
            'category', 'value', 'question', 'answer', 
            'round', 'show_number', 'air_date'
        ]
        
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.logger.info("✓ Schema validation passed")
        print("✓ Schema validation passed")
    
    def _clean_data(self) -> None:
        """Clean and normalize the dataset."""
        initial_count = len(self.df)
        
        # Remove records with missing critical fields
        self.df = self.df.dropna(subset=['question', 'answer'])
        
        # Clean text fields
        text_columns = ['category', 'question', 'answer']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                self.df[col] = self.df[col].replace('', pd.NA)
        
        # Clean other fields
        if 'show_number' in self.df.columns:
            self.df['show_number'] = self.df['show_number'].astype(str)
        
        if 'value' in self.df.columns:
            self.df['value'] = self.df['value'].astype(str)
        
        # Handle air_date
        # INFO: Don't convert to datetime for simplicity, just ensure it's a string currently.
        if 'air_date' in self.df.columns:
            self.df['air_date'] = self.df['air_date'].astype(str)
        
        # Remove completely empty rows
        self.df = self.df.dropna(how='all')
        
        cleaned_count = len(self.df)
        removed_count = initial_count - cleaned_count
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count:,} records during cleaning")
            print(f"✓ Cleaned data: {cleaned_count:,} records (removed {removed_count:,})")
        else:
            self.logger.info("No records removed during cleaning")
            print(f"✓ Data already clean: {cleaned_count:,} records")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the loaded DataFrame."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_validate() first.")
        return self.df.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive dataset summary."""
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        try:
            # Basic info
            summary = {
                'total_records': len(self.df),
                'memory_mb': round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 1),
                'columns': list(self.df.columns)
            }
            
            # Date range
            if 'air_date' in self.df.columns:
                date_min = self.df['air_date'].min()
                date_max = self.df['air_date'].max()
                summary['date_range'] = {
                    'earliest': date_min.strftime('%Y-%m-%d') if pd.notna(date_min) else None,
                    'latest': date_max.strftime('%Y-%m-%d') if pd.notna(date_max) else None
                }
            
            # Round distribution
            if 'round' in self.df.columns:
                summary['rounds'] = self.df['round'].value_counts().to_dict()
            
            # Category info
            if 'category' in self.df.columns:
                summary['categories'] = {
                    'unique_count': self.df['category'].nunique(),
                    'top_5': self.df['category'].value_counts().head(5).to_dict()
                }
            
            # Text lengths
            if 'question' in self.df.columns and 'answer' in self.df.columns:
                summary['text_stats'] = {
                    'avg_question_length': round(self.df['question'].str.len().mean(), 1),
                    'avg_answer_length': round(self.df['answer'].str.len().mean(), 1)
                }
            
            # Sample record
            summary['sample_record'] = self.df.iloc[0].to_dict()
            
        except Exception as e:
            self.logger.warning(f"Could not generate complete summary: {e}")
            # Minimal summary
            summary = {
                'total_records': len(self.df),
                'memory_mb': round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 1),
                'sample_record': self.df.iloc[0].to_dict() if len(self.df) > 0 else {}
            }
        
        return summary
    
if __name__ == "__main__":
    """Test the JeopardyDataHandler with sample data or actual file."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 TESTING JEOPARDY DATA HANDLER")
    print("=" * 50)
    
    # Test file path - adjust as needed
    test_file = 'data/raw/JEOPARDY_QUESTIONS1.json'
    
    # Check if file exists
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("\nTo test the handler:")
        print(f"1. Download JEOPARDY_QUESTIONS1.json")
        print(f"2. Place it at: {test_file}")
        print(f"3. Run this test again")
    
    try:
        # Initialize handler
        print(f"📂 Initializing handler for: {test_file}")
        handler = JeopardyDataHandler(test_file)
        
        # Load and validate data
        print("\n🔄 Loading and validating data...")
        df = handler.load_and_validate()
        
        # Get summary
        print("\n📊 Generating summary...")
        summary = handler.get_summary()
        
        # Print summary
        print("\n" + "=" * 50)
        print("📈 DATASET SUMMARY")
        print("=" * 50)
        
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Memory Usage: {summary['memory_mb']} MB")
        print(f"Columns: {summary['columns']}")
        
        if 'date_range' in summary:
            print(f"Date Range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")
        
        if 'rounds' in summary:
            print(f"Round Distribution:")
            for round_name, count in summary['rounds'].items():
                print(f"  - {round_name}: {count:,}")
        
        if 'categories' in summary:
            print(f"Categories: {summary['categories']['unique_count']:,} unique")
            print(f"Top 5 Categories:")
            for cat, count in summary['categories']['top_5'].items():
                print(f"  - {cat}: {count:,}")
        
        if 'text_stats' in summary:
            print(f"Text Statistics:")
            print(f"  - Avg Question Length: {summary['text_stats']['avg_question_length']} chars")
            print(f"  - Avg Answer Length: {summary['text_stats']['avg_answer_length']} chars")
        
        # Show sample record
        print(f"\n📝 SAMPLE RECORD:")
        sample = summary['sample_record']
        for key, value in sample.items():
            if isinstance(value, str) and len(str(value)) > 100:
                print(f"  {key}: {str(value)[:100]}...")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ HANDLER TEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ HANDLER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
