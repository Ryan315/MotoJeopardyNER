# src/data/cleaner.py

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import html
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

class JeopardyDataCleaner:
    """
    Enhanced cleaner for Jeopardy data with comprehensive validation and cleaning.
    
    External packages:
    - BeautifulSoup for HTML parsing

    Idea:
    The raw data contains a mix of structured metadata and unstructured text fields.
    This cleaner will:
    1. Validate metadata fields strictly (category, air_date, value, round, show_number)
    2. Clean text fields (question, answer) for records with valid metadata
    3. Apply minimal validation to cleaned text fields
    """
    
    def __init__(self):
        self.setup_patterns()
        self.setup_validation_rules()
        self.cleaning_stats = {}
        self.logger = logging.getLogger(__name__)
    
    def setup_patterns(self):
        """Setup comprehensive cleaning patterns."""
        
        self.patterns = {
            # HTML and markup patterns
            'html_tags': re.compile(r'<[^>]+>', re.IGNORECASE),
            'jeopardy_markup': re.compile(r'</?(?:i|b|u|em|strong|br|p)/?>', re.IGNORECASE),
            'html_entities': re.compile(r'&[#\w]+;'),
            'backslash_escapes': re.compile(r'\\([\'"])'),
            
            # Jeopardy-specific patterns
            'category_markers': re.compile(r'\[.*?\]|\(.*?\)'),
            'audio_video_markers': re.compile(r'\b(?:audio|video|seen here|heard here)\b', re.IGNORECASE),
            'daily_double': re.compile(r'\b(?:daily double|dd)\b', re.IGNORECASE),
            'fill_in_blank': re.compile(r'_+'),
            'hyperlinks': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            
            # Punctuation normalization
            'smart_quotes': re.compile(r'[""''"]'),
            'smart_apostrophes': re.compile(r'[\u2018\u2019\u201A\u201B\u2032\u2035]'),
            'em_en_dashes': re.compile(r'[–—―]'),
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_punctuation': re.compile(r'([.!?]){2,}'),
            
            # Validation patterns
            'dollar_values': re.compile(r'^\$\d{1,4}(?:,\d{3})*$'),
            'show_numbers': re.compile(r'^\d+$'),
            'air_dates': re.compile(r'^\d{4}-\d{2}-\d{2}$'),
            
            # Content validation
            'empty_content': re.compile(r'^\s*$'),
            'minimal_content': re.compile(r'^\s*\W{0,3}\s*$'),
        }
    
    def setup_validation_rules(self):
        """Setup data validation rules with different strictness for metadata vs text fields."""
        
        # Metadata fields - strict validation
        self.metadata_fields = ['category', 'air_date', 'value', 'round', 'show_number']
        # Text fields - will be cleaned and leniently validated
        self.text_fields = ['question', 'answer']
    
    def clean_text_field(self, text: str, field_type: str = 'general') -> str:
        """
        Clean individual text field with field-specific rules.
        
        Args:
            text: Raw text to clean
            field_type: Type of field ('question', 'answer', 'category', 'general')
        
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # Handle HTML content
        if '<' in text and '>' in text:
            # Try to preserve some structure for questions/answers
            if field_type in ['question', 'answer']:
                soup = BeautifulSoup(text, 'html.parser')
                # Replace <br> with spaces
                for br in soup.find_all('br'):
                    br.replace_with(' ')
                text = soup.get_text(separator=' ', strip=True)
            else:
                text = self.patterns['html_tags'].sub('', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Handle escaped quotes from raw data (double-encoded JSON)
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        
        # Handle other backslash escapes
        text = self.patterns['backslash_escapes'].sub(r'\1', text)
        
        # Remove outer single quotes that wrap the entire text
        # if len(text) >= 2 and text.startswith("'") and text.endswith("'"):
        #     text = text[1:-1]
        
        # Field-specific cleaning
        if field_type == 'question':
            # Remove audio/video markers that don't add semantic value
            text = self.patterns['audio_video_markers'].sub('', text)
            # Remove daily double indicators
            text = self.patterns['daily_double'].sub('', text)
            # Clean fill-in-the-blank markers
            text = self.patterns['fill_in_blank'].sub('[BLANK]', text)
            # Remove hyperlinks
            text = self.patterns['hyperlinks'].sub('[LINK]', text)
        
        elif field_type == 'answer':
            # For answers, be more conservative - keep most content
            text = self.patterns['hyperlinks'].sub('[LINK]', text)
        
        elif field_type == 'category':
            # Categories should be clean and standardized
            text = self.patterns['category_markers'].sub('', text)
            text = text.upper()  # Standardize case
        
        # General text normalization
        text = self.patterns['smart_quotes'].sub('"', text)
        text = self.patterns['smart_apostrophes'].sub("'", text)
        text = self.patterns['em_en_dashes'].sub('-', text)
        text = self.patterns['multiple_punctuation'].sub(r'\1', text)
        
        # Clean up remaining HTML entities
        text = self.patterns['html_entities'].sub('', text)
        
        # Normalize whitespace
        text = self.patterns['multiple_spaces'].sub(' ', text)
        
        return text.strip()

    
    def validate_metadata_field(self, value: str, field_name: str) -> Dict[str, any]:
        """
        Validate metadata fields with simplified rules.
        
        Rules:
        - category: check if empty
        - value: check if is $<number> or "None"
        - round: check if is one of "Jeopardy!", "Double Jeopardy!", "Final Jeopardy!", "Tiebreaker"
        - show_number: check if is a number
        - air_date: check if format is YYYY-MM-DD
        
        Args:
            value: Field value to validate
            field_name: Name of the field
        
        Returns:
            Dictionary with validation results: {'valid': bool, 'errors': list, 'warnings': list}
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if field is missing (None or NaN)
        if pd.isna(value) or value is None:
            result['valid'] = False
            result['errors'].append(f'{field_name} field is missing')
            return result
        
        # Convert to string and strip whitespace
        value_str = str(value).strip()
        
        # Validate each field according to specific rules
        if field_name == 'category':
            # Rule: check if empty
            if len(value_str) == 0:
                result['valid'] = False
                result['errors'].append('Category is empty')
        
        elif field_name == 'value':
            # Rule: check if is $<number> or "None"
            if value_str == 'None':
                # Valid for Final Jeopardy
                pass
            elif value_str.startswith('$') and len(value_str) > 1:
                # Check if everything after $ is a number (allowing commas)
                number_part = value_str[1:].replace(',', '')
                if not number_part.isdigit():
                    result['valid'] = False
                    result['errors'].append(f'Invalid value format: "{value_str}" (expected $<number> or "None")')
            else:
                result['valid'] = False
                result['errors'].append(f'Invalid value format: "{value_str}" (expected $<number> or "None")')
        
        elif field_name == 'round':
            # Rule: check if is one of the allowed round types
            allowed_rounds = ["Jeopardy!", "Double Jeopardy!", "Final Jeopardy!", "Tiebreaker"]
            if value_str not in allowed_rounds:
                result['valid'] = False
                result['errors'].append(f'Invalid round: "{value_str}" (must be one of: {allowed_rounds})')
        
        elif field_name == 'show_number':
            # Rule: check if is a number
            if not value_str.isdigit():
                result['valid'] = False
                result['errors'].append(f'Show number must be a number: "{value_str}"')
        
        elif field_name == 'air_date':
            # Rule: check if format is YYYY-MM-DD
            # if not self.patterns['air_dates'].match(value_str):
            #     result['valid'] = False
            #     result['errors'].append(f'Invalid date format: "{value_str}" (expected YYYY-MM-DD)')
            if len(value_str) == 0:
                result['valid'] = False
                result['errors'].append('air_date is empty')
        
        else:
            # Unknown field - add warning but don't fail
            result['warnings'].append(f'Unknown metadata field: {field_name}')
        
        return result
    
    def validate_text_field(self, value: str, field_name: str) -> Dict[str, any]:
        """
        Validate text fields leniently (after cleaning).
        
        Args:
            value: Field value to validate (already cleaned)
            field_name: Name of the field
        
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if pd.isna(value) or value is None:
            result['valid'] = False
            result['errors'].append('Required field is missing')
            return result
        
        value_str = str(value).strip()
        
        # Basic checks only
        if len(value_str) == 0:
            result['valid'] = False
            result['errors'].append('Empty after cleaning')
        elif len(value_str) < 2 and field_name == 'question':
            result['valid'] = False
            result['errors'].append('Too short after cleaning')
        
        return result
    
    def clean_and_validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean and validate entire DataFrame with two-stage process.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (cleaned_df, validation_report)
        """
        self.logger.info(f"Starting cleaning process for {len(df)} records")
        
        # Initialize tracking
        validation_report = {
            'total_records': len(df),
            'processed_records': 0,
            'valid_records': 0,
            'field_stats': {},
            'cleaning_summary': {},
            'removed_records': []
        }
        
        # Create copy for processing
        cleaned_df = df.copy()
        
        # STAGE 1: Validate metadata fields strictly
        self.logger.info("Stage 1: Validating metadata fields...")
        
        valid_indices = []
        metadata_error_counts = {field: 0 for field in self.metadata_fields}
        
        for idx, row in cleaned_df.iterrows():
            metadata_valid = True
            record_errors = []
            
            # Validate each metadata field
            for field_name in self.metadata_fields:
                if field_name in row:
                    # print(row[field_name])
                    validation_result = self.validate_metadata_field(row[field_name], field_name)
                    
                    if not validation_result['valid']:
                        metadata_valid = False
                        metadata_error_counts[field_name] += 1
                        record_errors.extend([f"{field_name}: {err}" for err in validation_result['errors']])
                else:
                    metadata_valid = False
                    metadata_error_counts[field_name] += 1
                    record_errors.append(f"{field_name}: Field missing")
            
            if metadata_valid:
                valid_indices.append(idx)
            else:
                validation_report['removed_records'].append({
                    'index': idx,
                    'errors': record_errors,
                    'reason': 'Invalid metadata fields',
                    'sample_data': {
                        'category': str(row.get('category', 'N/A'))[:50],
                        'value': str(row.get('value', 'N/A')),
                        'round': str(row.get('round', 'N/A')),
                        'show_number': str(row.get('show_number', 'N/A'))
                    }
                })
        
        # Filter to records with valid metadata
        cleaned_df = cleaned_df.loc[valid_indices].reset_index(drop=True)
        
        self.logger.info(f"Stage 1 complete: {len(cleaned_df)}/{len(df)} records have valid metadata")
        
        # STAGE 2: Clean text fields for valid records
        self.logger.info("Stage 2: Cleaning text fields...")
        
        # Track original lengths for text fields
        original_lengths = {}
        
        for field in self.text_fields:
            if field in cleaned_df.columns:
                self.logger.info(f"Cleaning {field} field...")
                
                # Store original lengths
                original_lengths[field] = cleaned_df[field].fillna('').astype(str).str.len()
                
                # Clean the field
                cleaned_df[field] = cleaned_df[field].apply(
                    lambda x: self.clean_text_field(x, field_type=field)
                )
                
                # Calculate cleaning stats
                cleaned_lengths = cleaned_df[field].str.len()
                validation_report['cleaning_summary'][field] = {
                    'original_avg_length': original_lengths[field].mean(),
                    'cleaned_avg_length': cleaned_lengths.mean(),
                    'avg_reduction': (original_lengths[field] - cleaned_lengths).mean(),
                    'empty_after_cleaning': (cleaned_lengths == 0).sum(),
                    'significantly_reduced': ((original_lengths[field] - cleaned_lengths) > 20).sum()
                }
        
        # STAGE 3: Final validation of cleaned text fields
        self.logger.info("Stage 3: Final validation of cleaned text fields...")
        
        final_valid_indices = []
        text_error_counts = {field: 0 for field in self.text_fields}
        
        for idx, row in cleaned_df.iterrows():
            text_valid = True
            record_errors = []
            
            # Validate cleaned text fields
            for field_name in self.text_fields:
                if field_name in row:
                    validation_result = self.validate_text_field(row[field_name], field_name)
                    
                    if not validation_result['valid']:
                        text_valid = False
                        text_error_counts[field_name] += 1
                        record_errors.extend([f"{field_name}: {err}" for err in validation_result['errors']])
            
            if text_valid:
                final_valid_indices.append(idx)
            else:
                validation_report['removed_records'].append({
                    'index': len(df) + idx,  # Different index to distinguish from metadata failures
                    'errors': record_errors,
                    'reason': 'Invalid text fields after cleaning',
                    'sample_data': {
                        'category': str(row.get('category', 'N/A'))[:50],
                        'question_cleaned': str(row.get('question', 'N/A'))[:50],
                        'answer_cleaned': str(row.get('answer', 'N/A'))[:50]
                    }
                })
        
        # Final filtered dataset
        cleaned_df = cleaned_df.loc[final_valid_indices].reset_index(drop=True)
        
        # Compile field statistics
        for field in self.metadata_fields + self.text_fields:
            if field in df.columns:
                if field in self.metadata_fields:
                    validation_report['field_stats'][field] = {
                        'total_records': len(df),
                        'non_empty_records': df[field].notna().sum(),
                        'empty_records': df[field].isna().sum(),
                        'error_records': metadata_error_counts[field],
                        'valid_records': len(df) - metadata_error_counts[field],
                        'error_rate': (metadata_error_counts[field] / len(df)) * 100,
                        'validation_type': 'strict_metadata'
                    }
                else:
                    validation_report['field_stats'][field] = {
                        'total_records': len(df),
                        'non_empty_records': df[field].notna().sum(),
                        'empty_records': df[field].isna().sum(),
                        'error_records': text_error_counts[field],
                        'valid_records': len(df) - text_error_counts[field],
                        'error_rate': (text_error_counts[field] / len(df)) * 100,
                        'validation_type': 'cleaned_text'
                    }
        
        # Final summary
        validation_report['processed_records'] = len(df)
        validation_report['valid_records'] = len(cleaned_df)
        validation_report['removed_count'] = len(df) - len(cleaned_df)
        validation_report['removal_rate'] = (validation_report['removed_count'] / len(df)) * 100
        
        self.logger.info(f"Cleaning completed: {len(cleaned_df)}/{len(df)} records retained")
        
        return cleaned_df, validation_report
    
    def print_validation_report(self, report: Dict):
        """Print comprehensive validation report."""
        
        print("\n" + "="*70)
        print("📊 DATA CLEANING & VALIDATION REPORT")
        print("="*70)
        
        # Overall summary
        print(f"📈 OVERALL SUMMARY:")
        print(f"  Total records processed: {report['total_records']:,}")
        print(f"  Valid records retained: {report['valid_records']:,}")
        print(f"  Records removed: {report['removed_count']:,}")
        print(f"  Retention rate: {100 - report['removal_rate']:.2f}%")
        
        # Field-by-field statistics
        print(f"\n📋 FIELD VALIDATION STATISTICS:")
        
        # Separate metadata and text fields
        metadata_fields = []
        text_fields = []
        
        for field, stats in report['field_stats'].items():
            if stats.get('validation_type') == 'strict_metadata':
                metadata_fields.append((field, stats))
            elif stats.get('validation_type') == 'cleaned_text':
                text_fields.append((field, stats))
        
        if metadata_fields:
            print(f"  METADATA FIELDS (Strict Validation):")
            for field, stats in metadata_fields:
                print(f"    {field.upper()}:")
                print(f"      Non-empty: {stats['non_empty_records']:,}/{stats['total_records']:,} ({(stats['non_empty_records']/stats['total_records']*100):.1f}%)")
                print(f"      Errors: {stats['error_records']:,} ({stats['error_rate']:.1f}%)")
                print(f"      Valid: {stats['valid_records']:,}")
        
        if text_fields:
            print(f"  TEXT FIELDS (Cleaned & Leniently Validated):")
            for field, stats in text_fields:
                print(f"    {field.upper()}:")
                print(f"      Non-empty: {stats['non_empty_records']:,}/{stats['total_records']:,} ({(stats['non_empty_records']/stats['total_records']*100):.1f}%)")
                print(f"      Errors after cleaning: {stats['error_records']:,} ({stats['error_rate']:.1f}%)")
                print(f"      Valid: {stats['valid_records']:,}")
        
        # Cleaning summary
        if report['cleaning_summary']:
            print(f"\n🧹 TEXT CLEANING SUMMARY:")
            for field, stats in report['cleaning_summary'].items():
                print(f"  {field.upper()}:")
                print(f"    Avg length: {stats['original_avg_length']:.0f} → {stats['cleaned_avg_length']:.0f} chars")
                print(f"    Avg reduction: {stats['avg_reduction']:.1f} chars")
                print(f"    Empty after cleaning: {stats['empty_after_cleaning']:,}")
                print(f"    Significantly reduced: {stats['significantly_reduced']:,}")
        
        # Sample errors by type
        if report['removed_records']:
            metadata_failures = [r for r in report['removed_records'] if r.get('reason') == 'Invalid metadata fields']
            text_failures = [r for r in report['removed_records'] if r.get('reason') == 'Invalid text fields after cleaning']
            
            if metadata_failures:
                print(f"\n❌ SAMPLE METADATA VALIDATION FAILURES (first 3):")
                for i, record in enumerate(metadata_failures[:3]):
                    print(f"  {i+1}. Index {record['index']}:")
                    print(f"     Category: {record['sample_data'].get('category', 'N/A')}")
                    print(f"     Value: {record['sample_data'].get('value', 'N/A')}")
                    print(f"     Round: {record['sample_data'].get('round', 'N/A')}")
                    print(f"     Errors: {'; '.join(record['errors'][:2])}")
                    print()
            
            if text_failures:
                print(f"\n❌ SAMPLE TEXT CLEANING FAILURES (first 3):")
                for i, record in enumerate(text_failures[:3]):
                    print(f"  {i+1}. Index {record['index']}:")
                    print(f"     Category: {record['sample_data'].get('category', 'N/A')}")
                    print(f"     Question: {record['sample_data'].get('question_cleaned', 'N/A')}")
                    print(f"     Answer: {record['sample_data'].get('answer_cleaned', 'N/A')}")
                    print(f"     Errors: {'; '.join(record['errors'][:2])}")
                    print()
    
    def save_validation_report(self, report: Dict, output_path: str):
        """Save validation report to JSON file."""
        
        # Add timestamp
        report['generation_timestamp'] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📁 Validation report saved to: {output_path}")
    
    def debug_validation_sample(self, df: pd.DataFrame, n_samples: int = 5) -> List[Dict]:
        """
        Debug validation by showing detailed results for sample records.
        
        Args:
            df: DataFrame to debug
            n_samples: Number of samples to analyze
        
        Returns:
            List of debug information dictionaries
        """
        debug_results = []
        
        for idx, row in df.head(n_samples).iterrows():
            record_debug = {
                'index': idx,
                'overall_valid': True,
                'metadata_valid': True,
                'text_valid': True,
                'field_results': {}
            }
            
            # Test metadata fields
            for field_name in self.metadata_fields:
                if field_name in row:
                    validation_result = self.validate_metadata_field(row[field_name], field_name)
                    record_debug['field_results'][field_name] = {
                        'value': str(row[field_name])[:50],
                        'valid': validation_result['valid'],
                        'errors': validation_result['errors'],
                        'warnings': validation_result['warnings'],
                        'field_type': 'metadata'
                    }
                    if not validation_result['valid']:
                        record_debug['metadata_valid'] = False
                        record_debug['overall_valid'] = False
                else:
                    record_debug['field_results'][field_name] = {
                        'value': 'MISSING',
                        'valid': False,
                        'errors': ['Field not found in data'],
                        'warnings': [],
                        'field_type': 'metadata'
                    }
                    record_debug['metadata_valid'] = False
                    record_debug['overall_valid'] = False
            
            # Test text fields (clean first, then validate)
            for field_name in self.text_fields:
                if field_name in row:
                    original_text = row[field_name]
                    cleaned_text = self.clean_text_field(original_text, field_type=field_name)
                    validation_result = self.validate_text_field(cleaned_text, field_name)
                    
                    record_debug['field_results'][field_name] = {
                        'original_value': str(original_text)[:50],
                        'cleaned_value': str(cleaned_text)[:50],
                        'valid': validation_result['valid'],
                        'errors': validation_result['errors'],
                        'warnings': validation_result['warnings'],
                        'field_type': 'text'
                    }
                    if not validation_result['valid']:
                        record_debug['text_valid'] = False
                        record_debug['overall_valid'] = False
                else:
                    record_debug['field_results'][field_name] = {
                        'original_value': 'MISSING',
                        'cleaned_value': 'MISSING',
                        'valid': False,
                        'errors': ['Field not found in data'],
                        'warnings': [],
                        'field_type': 'text'
                    }
                    record_debug['text_valid'] = False
                    record_debug['overall_valid'] = False
            
            debug_results.append(record_debug)
        
        return debug_results
    
    def print_debug_results(self, debug_results: List[Dict]):
        print("\n🔍 VALIDATION DEBUG RESULTS")
        print("=" * 50)
        
        for i, result in enumerate(debug_results):
            status = "✓ VALID" if result['overall_valid'] else "❌ INVALID"
            meta_status = "✓" if result['metadata_valid'] else "❌"
            text_status = "✓" if result['text_valid'] else "❌"
            
            print(f"\nRecord {i+1} (Index {result['index']}): {status}")
            print(f"  Metadata: {meta_status} | Text: {text_status}")
            
            for field_name, field_result in result['field_results'].items():
                status = "✓" if field_result['valid'] else "❌"
                
                if field_result['field_type'] == 'metadata':
                    print(f"  {status} {field_name}: {field_result['value']}")
                else:
                    print(f"  {status} {field_name}: {field_result['original_value']} → {field_result['cleaned_value']}")
                
                if field_result['errors']:
                    for error in field_result['errors']:
                        print(f"ERROR: {error}")
                
                if field_result['warnings']:
                    for warning in field_result['warnings']:
                        print(f"WARNING: {warning}")
            print()
    
    def get_cleaning_preview(self, df: pd.DataFrame, field: str, n_samples: int = 3) -> pd.DataFrame:
        """Get preview of cleaning results for manual inspection."""
        
        if field not in df.columns:
            return pd.DataFrame({'error': [f'Field {field} not found']})
        
        preview_df = df[[field]].head(n_samples).copy()
        preview_df[f'{field}_cleaned'] = preview_df[field].apply(
            lambda x: self.clean_text_field(x, field_type=field)
        )
        
        return preview_df