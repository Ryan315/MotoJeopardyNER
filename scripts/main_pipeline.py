import os
import sys
import time
import json
import logging
from datetime import datetime
import pandas as pd
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.handler import JeopardyDataHandler
from data.cleaner import JeopardyDataCleaner
from factories.numbers_factory import NumbersDatasetFactory
from factories.language_factory import NonEnglishDatasetFactory
from factories.entities_factory import ProperNounsDatasetFactory

def setup_logging():
    """Setup comprehensive logging."""
    log_dir = 'outputs/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'pipeline_execution.log')),
            logging.StreamHandler()
        ]
    )

def load_data():
    """Load and validate the Jeopardy dataset."""
    print("🔄 STEP 1: LOADING RAW DATA")
    print("=" * 60)
    
    json_file = 'data/raw/JEOPARDY_QUESTIONS1.json'
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"❌ Data file not found: {json_file}\n"
                              f"Please download JEOPARDY_QUESTIONS1.json and place it in data/raw/")
    
    handler = JeopardyDataHandler(json_file)
    df = handler.load_and_validate()
    
    summary = handler.get_summary()
    print(f"✓ Raw dataset loaded successfully:")
    print(f"  - Total records: {summary['total_records']:,}")
    print(f"  - Memory usage: {summary['memory_mb']:.1f} MB")
    print(f"  - Columns: {list(df.columns)}")
    
    # Show data quality preview
    print(f"\n📋 RAW DATA PREVIEW:")
    for col in ['category', 'value', 'question', 'answer', 'round', 'show_number', 'air_date']:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"  {col}: {null_count:,} null values ({null_pct:.1f}%)")
    
    return df

def clean_and_validate_data(df):
    """Clean and validate the dataset with comprehensive reporting."""
    print(f"\n🧹 STEP 2: DATA CLEANING & VALIDATION")
    print("=" * 60)
    
    # Initialize cleaner
    cleaner = JeopardyDataCleaner()
    
    # First, let's debug a few samples to understand the data format
    print("🔍 DEBUGGING SAMPLE RECORDS...")
    debug_results = cleaner.debug_validation_sample(df, n_samples=3)
    cleaner.print_debug_results(debug_results)
    
    # Show cleaning preview for key fields
    # print("📝 CLEANING PREVIEW (before processing):")
    # for field in ['question', 'answer', 'category']:
    #     if field in df.columns:
    #         preview = cleaner.get_cleaning_preview(df, field, n_samples=2)
    #         print(f"\n{field.upper()} field samples:")
    #         for idx, row in preview.iterrows():
    #             print(f"  Original: {str(row[field])[:80]}...")
    #             print(f"  Cleaned:  {str(row[f'{field}_cleaned'])[:80]}...")
    #             print()
    
    # Perform cleaning and validation
    print("🔄 Processing all records...")
    start_time = time.time()
    
    cleaned_df, validation_report = cleaner.clean_and_validate_dataframe(df)
    
    processing_time = time.time() - start_time
    
    # Print comprehensive report
    cleaner.print_validation_report(validation_report)
    
    # If we have very few valid records, let's investigate further
    if len(cleaned_df) < len(df) * 0.5:  # If we lost more than 50% of records
        print(f"\n⚠️  WARNING: High removal rate detected! Investigating...")
        print(f"Let's examine some failed records in detail:")
        
        # Show more debug info for failed records
        failed_samples = df.head(10)  # Look at first 10 records
        debug_results = cleaner.debug_validation_sample(failed_samples, n_samples=5)
        cleaner.print_debug_results(debug_results)
    
    # Save validation report
    os.makedirs('outputs/validation', exist_ok=True)
    report_path = 'outputs/validation/cleaning_report.json'
    cleaner.save_validation_report(validation_report, report_path)
    
    # Save cleaned dataset
    cleaned_data_path = 'data/processed/jeopardy_cleaned.jsonl'
    os.makedirs('data/processed', exist_ok=True)
    
    with open(cleaned_data_path, 'w') as f:
        for _, row in cleaned_df.iterrows():
            # Convert row to dict and handle non-serializable types
            row_dict = row.to_dict()
            
            # Handle any non-JSON-serializable values
            for key, value in row_dict.items():
                if pd.isna(value):  # Handle NaN values
                    row_dict[key] = None
            
            # Use default=str to handle any remaining non-serializable types (like numpy int64)
            f.write(json.dumps(row_dict, default=str) + '\n')
    
    print(f"\n💾 CLEANED DATA SAVED:")
    print(f"  - Clean dataset: {cleaned_data_path}")
    print(f"  - Validation report: {report_path}")
    print(f"  - Processing time: {processing_time:.1f} seconds")
    print(f"  - Final dataset size: {len(cleaned_df):,} records")
    
    return cleaned_df, validation_report

def process_numbers_subset(df):
    """Process numbers subset using spaCy token.like_num."""
    print(f"\n🔢 STEP 3: PROCESSING NUMBERS SUBSET")
    print("=" * 60)
    
    factory = NumbersDatasetFactory(df, n_processes=None)  # Auto-detect CPU count
    
    # Detection phase
    start_time = time.time()
    factory.apply_detection_parallel(show_progress=True)
    detection_time = time.time() - start_time
    
    # Generate subset
    subset = factory.generate_subset(size=1000)
    
    # Export
    output_file = 'data/subsets/numbers_subset.jsonl'
    os.makedirs('data/subsets', exist_ok=True)
    exported_count = factory.export_subset(subset, output_file)
    
    # Results
    stats = factory.get_processing_stats()
    print(f"\n✓ Numbers subset completed:")
    print(f"  - Candidates found: {stats['detected_records']:,}/{stats['total_records']:,} ({stats['detection_rate']:.1f}%)")
    print(f"  - Subset generated: {len(subset):,} records")
    print(f"  - Processing time: {detection_time:.1f}s")
    print(f"  - Speed: {stats['records_per_second']:.0f} records/second")
    print(f"  - Output: {output_file}")
    
    return {
        'factory': factory,
        'subset': subset,
        'stats': stats,
        'output_file': output_file,
        'processing_time': detection_time
    }

def process_non_english_subset(df):
    """Process non-English words subset using NLTK dictionary."""
    print(f"\n🌍 STEP 4: PROCESSING NON-ENGLISH SUBSET")
    print("=" * 60)
    
    factory = NonEnglishDatasetFactory(df, n_processes=None)
    
    # Detection phase
    start_time = time.time()
    factory.apply_detection_parallel(show_progress=True)
    detection_time = time.time() - start_time
    
    # Generate subset
    subset = factory.generate_subset(size=1000)
    
    # Export
    output_file = 'data/subsets/non_english_subset.jsonl'
    os.makedirs('data/subsets', exist_ok=True)
    exported_count = factory.export_subset(subset, output_file)
    
    # Results
    stats = factory.get_processing_stats()
    print(f"\n✓ Non-English subset completed:")
    print(f"  - Candidates found: {stats['detected_records']:,}/{stats['total_records']:,} ({stats['detection_rate']:.1f}%)")
    print(f"  - Subset generated: {len(subset):,} records")
    print(f"  - Processing time: {detection_time:.1f}s")
    print(f"  - Speed: {stats['records_per_second']:.0f} records/second")
    print(f"  - Output: {output_file}")
    
    return {
        'factory': factory,
        'subset': subset,
        'stats': stats,
        'output_file': output_file,
        'processing_time': detection_time
    }

def process_proper_nouns_subset(df):
    """Process unusual proper nouns subset using frequency analysis."""
    print(f"\n👤 STEP 5: PROCESSING PROPER NOUNS SUBSET")
    print("=" * 60)
    
    factory = ProperNounsDatasetFactory(df, n_processes=None, frequency_threshold=3)
    
    # Phase 1: Frequency analysis
    print("Phase 1: Analyzing entity frequencies across entire dataset...")
    freq_start = time.time()
    factory.analyze_entity_frequencies(show_progress=True)
    freq_time = time.time() - freq_start
    
    # Phase 2: Detection
    print("Phase 2: Detecting records with unusual entities...")
    detection_start = time.time()
    factory.apply_detection_parallel(show_progress=True)
    detection_time = time.time() - detection_start
    
    # Generate subset
    subset = factory.generate_subset(size=1000)
    
    # Export
    output_file = 'data/subsets/proper_nouns_subset.jsonl'
    os.makedirs('data/subsets', exist_ok=True)
    exported_count = factory.export_subset(subset, output_file)
    
    # Results
    stats = factory.get_processing_stats()
    freq_stats = factory.get_frequency_stats()
    
    print(f"\n✓ Proper nouns subset completed:")
    print(f"  - Total entities analyzed: {freq_stats['total_unique_entities']:,}")
    print(f"  - Unusual entities (≤3): {freq_stats['unusual_entities_count']:,}")
    print(f"  - Candidates found: {stats['detected_records']:,}/{stats['total_records']:,} ({stats['detection_rate']:.1f}%)")
    print(f"  - Subset generated: {len(subset):,} records")
    print(f"  - Frequency analysis: {freq_time:.1f}s")
    print(f"  - Detection time: {detection_time:.1f}s")
    print(f"  - Total time: {freq_time + detection_time:.1f}s")
    print(f"  - Output: {output_file}")
    
    return {
        'factory': factory,
        'subset': subset,
        'stats': stats,
        'freq_stats': freq_stats,
        'output_file': output_file,
        'processing_time': freq_time + detection_time
    }

def generate_analysis_report(results, validation_report, total_time):
    """Generate comprehensive analysis report including cleaning results."""
    print(f"\n📊 STEP 6: GENERATING ANALYSIS REPORT")
    print("=" * 60)
    
    # Create reports directory
    reports_dir = 'outputs/reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    # Compile comprehensive report
    report = {
        'generation_info': {
            'timestamp': datetime.now().isoformat(),
            'total_processing_time': round(total_time, 2),
            'dataset_source': 'JEOPARDY_QUESTIONS1.json',
            'original_records': validation_report['total_records'],
            'cleaned_records': validation_report['valid_records'],
            'cleaning_removal_rate': validation_report['removal_rate']
        },
        'data_cleaning_summary': {
            'records_removed': validation_report['removed_count'],
            'removal_rate_percent': round(validation_report['removal_rate'], 2),
            'field_validation_results': validation_report['field_stats'],
            'text_cleaning_stats': validation_report['cleaning_summary']
        },
        'subset_results': {
            'numbers': {
                'candidates_found': results['numbers']['stats']['detected_records'],
                'detection_rate': round(results['numbers']['stats']['detection_rate'], 2),
                'subset_size': len(results['numbers']['subset']),
                'processing_time': round(results['numbers']['processing_time'], 2),
                'method': 'spaCy token.like_num + Roman numeral regex',
                'output_file': results['numbers']['output_file']
            },
            'non_english': {
                'candidates_found': results['non_english']['stats']['detected_records'],
                'detection_rate': round(results['non_english']['stats']['detection_rate'], 2),
                'subset_size': len(results['non_english']['subset']),
                'processing_time': round(results['non_english']['processing_time'], 2),
                'method': 'spaCy tokenization + NLTK English dictionary',
                'output_file': results['non_english']['output_file']
            },
            'proper_nouns': {
                'total_entities_analyzed': results['proper_nouns']['freq_stats']['total_unique_entities'],
                'unusual_entities_found': results['proper_nouns']['freq_stats']['unusual_entities_count'],
                'candidates_found': results['proper_nouns']['stats']['detected_records'],
                'detection_rate': round(results['proper_nouns']['stats']['detection_rate'], 2),
                'subset_size': len(results['proper_nouns']['subset']),
                'processing_time': round(results['proper_nouns']['processing_time'], 2),
                'method': 'Frequency analysis (≤3 occurrences) + spaCy NER',
                'output_file': results['proper_nouns']['output_file']
            }
        },
        'performance_metrics': {
            'average_speed_records_per_second': round(
                validation_report['valid_records'] * 3 / total_time, 0
            ),
            'multiprocessing_enabled': True,
            'processes_used': results['numbers']['stats']['processes_used']
        }
    }
    
    # Save JSON report
    report_file = os.path.join(reports_dir, 'analysis_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate markdown methodology report
    methodology_file = os.path.join(reports_dir, 'methodology_report.md')
    generate_methodology_report(methodology_file, report)
    
    print(f"✓ Analysis reports generated:")
    print(f"  - Detailed analysis: {report_file}")
    print(f"  - Methodology: {methodology_file}")
    
    return report

def generate_methodology_report(output_file, report_data):
    """Generate methodology documentation including cleaning process."""
    methodology = f"""# Jeopardy NER Dataset Curation - Methodology Report

Generated: {report_data['generation_info']['timestamp']}

## Overview
This report documents the methodology used to curate three specialized subsets from the Jeopardy questions dataset for Named Entity Recognition (NER) validation, including comprehensive data cleaning and validation procedures.

## Dataset Source & Processing
- **Source**: JEOPARDY_QUESTIONS1.json
- **Original Records**: {report_data['generation_info']['original_records']:,}
- **Cleaned Records**: {report_data['generation_info']['cleaned_records']:,}
- **Records Removed**: {report_data['data_cleaning_summary']['records_removed']:,} ({report_data['data_cleaning_summary']['removal_rate_percent']}%)
- **Total Processing Time**: {report_data['generation_info']['total_processing_time']} seconds

## Data Cleaning & Validation Process

### Cleaning Methodology
The cleaning process involved comprehensive text normalization and validation:

1. **HTML Content Processing**: Removed HTML tags while preserving semantic content
2. **Text Normalization**: 
   - Decoded HTML entities (&amp; → &, etc.)
   - Normalized quotation marks and apostrophes
   - Standardized whitespace and punctuation
3. **Jeopardy-Specific Cleaning**:
   - Removed audio/video markers that don't add semantic value
   - Cleaned Daily Double indicators
   - Handled hyperlinks and fill-in-the-blank markers
4. **Field-Specific Rules**:
   - Categories: Uppercase standardization, removed markers
   - Questions: Aggressive cleaning for semantic clarity
   - Answers: Conservative cleaning to preserve content

### Validation Rules
- **Required Fields**: All core fields must be present and non-empty
- **Format Validation**: 
  - Values: Must match $XXX pattern or be "None" for Final Jeopardy
  - Dates: Must follow YYYY-MM-DD format
  - Show Numbers: Must be numeric
- **Content Quality**: Minimum length requirements, no whitespace-only content

### Cleaning Results Summary
"""
    
    # Add field-specific cleaning stats
    if 'text_cleaning_stats' in report_data['data_cleaning_summary']:
        methodology += "\n#### Text Cleaning Statistics\n"
        for field, stats in report_data['data_cleaning_summary']['text_cleaning_stats'].items():
            methodology += f"- **{field.upper()}**: Avg reduction {stats['avg_reduction']:.1f} chars, {stats['empty_after_cleaning']} empty after cleaning\n"
    
    methodology += f"""

## Subset Generation Process

### Subset 1: Numbers Detection
**Method**: spaCy token.like_num + Roman numeral regex fallback
**Logic**: 
- Primary detection using spaCy's `token.like_num` feature
- Catches written numbers ("five", "twenty-one"), digits ("123", "7pm"), mixed formats
- Roman numeral regex fallback for edge cases spaCy might miss

**Results**:
- Candidates found: {report_data['subset_results']['numbers']['candidates_found']:,} ({report_data['subset_results']['numbers']['detection_rate']}%)
- Final subset: {report_data['subset_results']['numbers']['subset_size']} records
- Processing time: {report_data['subset_results']['numbers']['processing_time']} seconds

### Subset 2: Non-English Words Detection  
**Method**: spaCy tokenization + NLTK English dictionary
**Logic**:
- Use spaCy for accurate tokenization
- Check tokens against NLTK's comprehensive English dictionary (~230k words)
- Filter out proper nouns (naturally foreign)
- Fallback to non-ASCII character detection

**Results**:
- Candidates found: {report_data['subset_results']['non_english']['candidates_found']:,} ({report_data['subset_results']['non_english']['detection_rate']}%)
- Final subset: {report_data['subset_results']['non_english']['subset_size']} records
- Processing time: {report_data['subset_results']['non_english']['processing_time']} seconds

### Subset 3: Unusual Proper Nouns Detection
**Method**: Frequency analysis + spaCy NER
**Logic**:
- Phase 1: Extract all proper nouns using spaCy NER across entire dataset
- Phase 2: Count frequency of each unique entity
- Define "unusual" as appearing ≤3 times in dataset
- Use simple string matching for final detection

**Results**:
- Total entities analyzed: {report_data['subset_results']['proper_nouns']['total_entities_analyzed']:,}
- Unusual entities (≤3): {report_data['subset_results']['proper_nouns']['unusual_entities_found']:,}
- Candidates found: {report_data['subset_results']['proper_nouns']['candidates_found']:,} ({report_data['subset_results']['proper_nouns']['detection_rate']}%)
- Final subset: {report_data['subset_results']['proper_nouns']['subset_size']} records
- Processing time: {report_data['subset_results']['proper_nouns']['processing_time']} seconds

## Technical Implementation
- **Multiprocessing**: {report_data['performance_metrics']['processes_used']} processes
- **Average speed**: {report_data['performance_metrics']['average_speed_records_per_second']} records/second
- **Memory efficient**: In-memory processing with chunking
- **Output format**: JSONL files maintaining original Jeopardy structure
- **Data quality**: Comprehensive validation and cleaning with detailed reporting

## Quality Assurance
- Multi-stage data validation and cleaning
- Field-specific validation rules with detailed error reporting
- Comprehensive cleaning statistics and sample tracking
- Multiprocessing error handling with fallbacks
- Export verification and format validation
- Detailed logging and progress tracking throughout the pipeline

## Files Generated
- **Cleaned dataset**: `data/processed/jeopardy_cleaned.jsonl`
- **Validation report**: `outputs/validation/cleaning_report.json`
- **Subset files**: `data/subsets/[numbers|non_english|proper_nouns]_subset.jsonl`
- **Processing logs**: `outputs/logs/pipeline_execution.log`
"""
    
    with open(output_file, 'w') as f:
        f.write(methodology)

def show_sample_results(results):
    """Show sample results from each subset."""
    print(f"\n🔍 SAMPLE RESULTS")
    print("=" * 60)
    
    for subset_name, result in results.items():
        subset = result['subset']
        factory = result['factory']
        
        print(f"\n{subset_name.upper()} SUBSET SAMPLES:")
        if len(subset) > 0:
            for i, (_, row) in enumerate(subset.head(2).iterrows()):
                print(f"  {i+1}. Category: {row['category']}")
                print(f"     Question: {row['question'][:80]}...")
                print(f"     Answer: {row['answer']}")
                
                # Show detection details
                if hasattr(factory, 'get_detection_details'):
                    text = f"{row['question']} {row['answer']}"
                    details = factory.get_detection_details(text)
                    if 'number_tokens' in details and details['number_tokens']:
                        tokens = [t['text'] for t in details['number_tokens'][:3]]
                        print(f"     Detected: {tokens}")
                    elif 'non_english_words' in details and details['non_english_words']:
                        words = [w['word'] for w in details['non_english_words'][:3]]
                        print(f"     Detected: {words}")
                    elif 'unusual_entities_found' in details and details['unusual_entities_found']:
                        entities = [e['text'] for e in details['unusual_entities_found'][:3]]
                        print(f"     Detected: {entities}")
                print()

def main():
    """Main pipeline execution with integrated cleaning."""
    setup_logging()
    
    print("🚀 JEOPARDY NER DATASET CURATION - ENHANCED PIPELINE")
    print("Processing ~217K Jeopardy questions with comprehensive cleaning & validation...\n")
    
    pipeline_start = time.time()
    
    try:
        # Step 1: Load raw data
        df = load_data()
        
        # Step 2: Clean and validate data
        cleaned_df, validation_report = clean_and_validate_data(df)
        
        # Step 3-5: Process all three subsets using cleaned data
        results = {}
        results['numbers'] = process_numbers_subset(cleaned_df)
        results['non_english'] = process_non_english_subset(cleaned_df)
        results['proper_nouns'] = process_proper_nouns_subset(cleaned_df)
        
        # Step 6: Generate reports
        total_time = time.time() - pipeline_start
        report = generate_analysis_report(results, validation_report, total_time)
        
        # Show samples
        # show_sample_results(results)
        
        # Success summary
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✓ Total processing time: {total_time:.1f} seconds")
        print(f"✓ Original records: {validation_report['total_records']:,}")
        print(f"✓ Cleaned records: {validation_report['valid_records']:,}")
        print(f"✓ Data quality improvement: {validation_report['removal_rate']:.1f}% problematic records removed")
        print(f"✓ Subsets generated: 3 files with 1000 records each")
        print(f"✓ Output directories:")
        print(f"  - Clean dataset: data/processed/")
        print(f"  - Validation reports: outputs/validation/")
        print(f"  - Subsets: data/subsets/")
        print(f"  - Analysis reports: outputs/reports/")
        
        print(f"\n📁 Generated Files:")
        print(f"  - data/processed/jeopardy_cleaned.jsonl ({validation_report['valid_records']:,} records)")
        for subset_name, result in results.items():
            if os.path.exists(result['output_file']):
                file_size = os.path.getsize(result['output_file']) / 1024
                print(f"  - {result['output_file']} ({file_size:.1f} KB)")
                
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)