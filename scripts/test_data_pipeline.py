# scripts/test_data_pipeline.py

import sys
import os
import logging
import pandas as pd
from datetime import datetime

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.handler import JeopardyDataHandler
from data.cleaner import JeopardyDataCleaner

def setup_logging():
    """Setup logging configuration."""
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'test_pipeline.log')),
            logging.StreamHandler()
        ]
    )

def test_data_pipeline(test_size: int = 1000):
    """
    Test the data loading and cleaning pipeline.
    
    Args:
        test_size: Number of records to load for testing
    """
    
    print("🧪 TESTING JEOPARDY DATA PIPELINE")
    print("=" * 60)
    print(f"Test size: {test_size:,} records")
    print()
    
    # Define paths
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'JEOPARDY_QUESTIONS1.json')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    
    # Create output directories
    for subdir in ['processed', 'reports', 'validation']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    try:
        # STEP 1: Test Data Handler
        print("📂 STEP 1: Testing Data Handler")
        print("-" * 40)
        
        # Initialize handler with test limit
        handler = JeopardyDataHandler(raw_data_path, test_limit=test_size)
        
        # Load data
        df_raw = handler.load_and_validate()
        print(f"✓ Loaded {len(df_raw):,} records")
        
        # Get and display summary
        summary = handler.get_summary()
        print(f"✓ Memory usage: {summary['memory_mb']} MB")
        
        if 'rounds' in summary:
            print(f"✓ Round distribution:")
            for round_name, count in summary['rounds'].items():
                print(f"  - {round_name}: {count:,}")
        
        print()
        
        # STEP 2: Test Data Cleaner
        print("🧹 STEP 2: Testing Data Cleaner")
        print("-" * 40)
        
        # Initialize cleaner
        cleaner = JeopardyDataCleaner()
        
        # Show debug sample before cleaning
        print("🔍 Debug sample (first 3 records):")
        debug_results = cleaner.debug_validation_sample(df_raw, n_samples=3)
        cleaner.print_debug_results(debug_results)
        
        # Clean and validate data
        df_cleaned, validation_report = cleaner.clean_and_validate_dataframe(df_raw)
        
        # Print validation report
        cleaner.print_validation_report(validation_report)
        
        print()
        
        # STEP 3: Save Results
        print("💾 STEP 3: Saving Results")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save cleaned data as JSONL
        cleaned_file = os.path.join(output_dir, 'processed', f'cleaned_test_data_{timestamp}.jsonl')
        df_cleaned.to_json(cleaned_file, orient='records', lines=True)
        print(f"✓ Cleaned data saved: {cleaned_file}")
        
        # Save validation report
        report_file = os.path.join(output_dir, 'validation', f'validation_report_{timestamp}.json')
        cleaner.save_validation_report(validation_report, report_file)
        
        # Save sample comparison
        comparison_file = os.path.join(output_dir, 'reports', f'before_after_comparison_{timestamp}.csv')
        
        # Create before/after comparison for first 10 records
        if len(df_raw) > 0 and len(df_cleaned) > 0:
            comparison_data = []
            
            for i in range(min(10, len(df_raw))):
                original_row = df_raw.iloc[i]
                
                # Find corresponding cleaned row (if it exists)
                cleaned_row = None
                if i < len(df_cleaned):
                    cleaned_row = df_cleaned.iloc[i]
                
                comparison_data.append({
                    'record_index': i,
                    'original_category': str(original_row.get('category', ''))[:100],
                    'cleaned_category': str(cleaned_row.get('category', '') if cleaned_row is not None else 'REMOVED')[:100],
                    'original_question': str(original_row.get('question', ''))[:200],
                    'cleaned_question': str(cleaned_row.get('question', '') if cleaned_row is not None else 'REMOVED')[:200],
                    'original_answer': str(original_row.get('answer', ''))[:100],
                    'cleaned_answer': str(cleaned_row.get('answer', '') if cleaned_row is not None else 'REMOVED')[:100],
                    'original_value': str(original_row.get('value', '')),
                    'cleaned_value': str(cleaned_row.get('value', '') if cleaned_row is not None else 'REMOVED'),
                    'original_round': str(original_row.get('round', '')),
                    'cleaned_round': str(cleaned_row.get('round', '') if cleaned_row is not None else 'REMOVED')
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(comparison_file, index=False)
            print(f"✓ Before/after comparison saved: {comparison_file}")
        
        # STEP 4: Final Summary
        print()
        print("📊 STEP 4: Final Summary")
        print("-" * 40)
        
        print(f"Original records: {len(df_raw):,}")
        print(f"Cleaned records: {len(df_cleaned):,}")
        print(f"Retention rate: {(len(df_cleaned)/len(df_raw)*100):.1f}%")
        print(f"Removed records: {len(df_raw) - len(df_cleaned):,}")
        
        # Show sample of cleaned data
        if len(df_cleaned) > 0:
            print(f"\n📝 Sample cleaned record:")
            sample = df_cleaned.iloc[0]
            for col in ['category', 'question', 'answer', 'value', 'round']:
                if col in sample:
                    value = str(sample[col])
                    if len(value) > 100:
                        value = value[:100] + "..."
                    print(f"  {col}: {value}")
        
        print()
        print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print(f"📁 Check the outputs/ directory for saved files")
        
        return df_cleaned, validation_report
        
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {e}")
        logging.error(f"Pipeline test failed: {e}", exc_info=True)
        raise

def show_cleaning_preview(test_size: int = 5):
    """Show a preview of what cleaning does to the data."""
    
    print("\n🔍 CLEANING PREVIEW")
    print("=" * 50)
    
    # Define paths
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'JEOPARDY_QUESTIONS1.json')
    
    try:
        # Load small sample
        handler = JeopardyDataHandler(raw_data_path, test_limit=test_size)
        df_sample = handler.load_and_validate()
        
        # Initialize cleaner
        cleaner = JeopardyDataCleaner()
        
        # Show cleaning preview for each text field
        for field in ['question', 'answer', 'category']:
            if field in df_sample.columns:
                print(f"\n📝 {field.upper()} CLEANING PREVIEW:")
                preview = cleaner.get_cleaning_preview(df_sample, field, n_samples=3)
                
                for i, row in preview.iterrows():
                    print(f"\nRecord {i+1}:")
                    original = str(row[field])[:150]
                    cleaned = str(row[f'{field}_cleaned'])[:150]
                    print(f"  Original: {original}")
                    print(f"  Cleaned:  {cleaned}")
                    if original != cleaned:
                        print(f"  Changed:  {'Yes' if original != cleaned else 'No'}")
                    print()
        
    except Exception as e:
        print(f"❌ Preview failed: {e}")

if __name__ == "__main__":
    """Run the test pipeline."""
    
    # Setup logging
    setup_logging()
    
    # Check command line arguments
    test_size = 1000
    if len(sys.argv) > 1:
        try:
            test_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid test size: {sys.argv[1]}. Using default: {test_size}")
    
    # Run cleaning preview first
    show_cleaning_preview(5)
    
    # Run main test
    test_data_pipeline(test_size)