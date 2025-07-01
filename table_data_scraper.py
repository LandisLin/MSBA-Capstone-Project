"""
Universal Table Data Scraper
A flexible, reusable scraper for extracting tabular data from websites
Supports date-value pairs with automatic pattern detection and duplicate handling
"""

import requests
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json

class UniversalTableScraper:
    """
    Universal scraper for extracting date-value pairs from web tables
    
    Features:
    - Multiple extraction patterns for robustness
    - Automatic duplicate detection and removal
    - Flexible data validation
    - Excel export with analysis
    """
    
    def __init__(self, output_dir: str = "./extracted_data"):
        """
        Initialize the scraper
        
        Args:
            output_dir: Directory to save extracted data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with browser-like headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        })
        return session
    
    def extract_table_data(self, 
                          url: str,
                          data_name: str,
                          date_format: str = "%d/%m/%Y",
                          value_range: Tuple[float, float] = (0.0, 100.0),
                          country: str = "Unknown",
                          data_type: str = "Numeric_Data",
                          unit: str = "Units",
                          verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Extract date-value pairs from a web table
        
        Args:
            url: Website URL to scrape
            data_name: Descriptive name for the data (e.g., "Malaysia OPR")
            date_format: Expected date format (default: DD/MM/YYYY)
            value_range: Valid range for values as (min, max)
            country: Country name for metadata
            data_type: Type of data for metadata
            unit: Unit of measurement
            verbose: Enable detailed logging
            
        Returns:
            DataFrame with extracted data or None if failed
        """
        
        if verbose:
            print(f"🌐 EXTRACTING: {data_name}")
            print(f"🔗 URL: {url}")
            print("=" * 50)
        
        try:
            # Fetch webpage content
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                if verbose:
                    print(f"❌ HTTP Error: {response.status_code}")
                return None
            
            if verbose:
                print(f"✅ Content fetched: {len(response.text):,} characters")
            
            # Extract raw records
            records = self._extract_date_value_pairs(response.text, date_format, value_range, verbose)
            
            if not records:
                if verbose:
                    print("❌ No valid records found")
                return None
            
            # Process and standardize records
            df = self._process_records(records, data_name, country, data_type, unit, date_format, verbose)
            
            return df
            
        except Exception as e:
            if verbose:
                print(f"❌ Extraction error: {e}")
            return None
    
    def _extract_date_value_pairs(self, 
                                 html_content: str, 
                                 date_format: str,
                                 value_range: Tuple[float, float],
                                 verbose: bool) -> List[Dict]:
        """
        Extract date-value pairs using multiple patterns
        
        Args:
            html_content: HTML content to parse
            date_format: Expected date format
            value_range: Valid value range (min, max)
            verbose: Enable logging
            
        Returns:
            List of extracted records
        """
        
        # Convert date format to regex pattern
        if date_format == "%d/%m/%Y":
            date_pattern = r'\d{2}/\d{2}/\d{4}'
        elif date_format == "%Y-%m-%d":
            date_pattern = r'\d{4}-\d{2}-\d{2}'
        elif date_format == "%m/%d/%Y":
            date_pattern = r'\d{2}/\d{2}/\d{4}'
        else:
            date_pattern = r'\d{2}/\d{2}/\d{4}'  # Default fallback
        
        # Multiple extraction patterns for different table structures
        patterns = [
            # Pattern 1: Flexible date-value matching
            (rf'({date_pattern})[\s\S]*?(\d+\.\d{{2}})', "Flexible date-value pattern"),
            
            # Pattern 2: Table row structure
            (rf'<tr[^>]*>.*?({date_pattern}).*?(\d+\.\d{{2}}).*?</tr>', "Table row pattern"),
            
            # Pattern 3: Table cell structure  
            (rf'<td[^>]*>.*?({date_pattern}).*?</td>.*?<td[^>]*>.*?(\d+\.\d{{2}}).*?</td>', "Table cell pattern"),
            
            # Pattern 4: Direct sequence
            (rf'({date_pattern})\s+(\d+\.\d{{2}})', "Direct sequence pattern"),
            
            # Pattern 5: JSON-like structure
            (rf'["\']({date_pattern})["\'][\s:,]*(\d+\.\d{{2}})', "JSON-like pattern")
        ]
        
        all_records = []
        
        for i, (pattern, description) in enumerate(patterns):
            if verbose:
                print(f"🔎 Pattern {i+1}: {description}")
            
            matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
            
            if verbose:
                print(f"   Raw matches: {len(matches)}")
            
            valid_records = []
            
            for date_str, value_str in matches:
                try:
                    # Clean the extracted strings
                    date_clean = date_str.strip()
                    value_clean = value_str.strip()
                    
                    # Validate date format
                    date_obj = pd.to_datetime(date_clean, format=date_format)
                    
                    # Validate and convert value
                    value_num = float(value_clean)
                    
                    # Check if value is in valid range
                    if value_range[0] <= value_num <= value_range[1]:
                        valid_records.append({
                            'date': date_clean,
                            'value': value_num,
                            'pattern': f'pattern_{i+1}',
                            'raw_match': f"{date_str} -> {value_str}"
                        })
                    
                except (ValueError, pd.errors.ParserError):
                    continue
            
            if verbose:
                print(f"   Valid records: {len(valid_records)}")
            
            if valid_records:
                all_records.extend(valid_records)
                if verbose:
                    print(f"   ✅ Using pattern {i+1}")
                break
            elif verbose:
                print(f"   ❌ No valid records")
        
        if verbose:
            print(f"\n📊 Total records extracted: {len(all_records)}")
        
        return all_records
    
    def _process_records(self, 
                        records: List[Dict], 
                        data_name: str,
                        country: str,
                        data_type: str,
                        unit: str,
                        date_format: str,
                        verbose: bool) -> pd.DataFrame:
        """
        Process records with duplicate analysis and standardization
        
        Args:
            records: Raw extracted records
            data_name: Name of the dataset
            country: Country name
            data_type: Type of data
            unit: Unit of measurement
            date_format: Date format used
            verbose: Enable logging
            
        Returns:
            Processed DataFrame
        """
        
        if verbose:
            print("\n🔬 PROCESSING & DUPLICATE ANALYSIS")
            print("-" * 40)
        
        # Analyze duplicates before removal
        dates_seen = {}
        duplicates_found = []
        
        for record in records:
            date = record['date']
            if date in dates_seen:
                duplicates_found.append({
                    'date': date,
                    'first_value': dates_seen[date]['value'],
                    'duplicate_value': record['value']
                })
                if verbose:
                    print(f"🔄 DUPLICATE: {date}")
                    print(f"   First: {dates_seen[date]['value']} | Duplicate: {record['value']}")
            else:
                dates_seen[date] = record
        
        if verbose:
            print(f"\n📈 Summary:")
            print(f"   Raw records: {len(records)}")
            print(f"   Duplicates: {len(duplicates_found)}")
            print(f"   Unique records: {len(dates_seen)}")
        
        # Create standardized DataFrame
        standardized_data = []
        
        for date_str, record in dates_seen.items():
            try:
                # Convert date to standard format (YYYY-MM-DD)
                date_obj = pd.to_datetime(date_str, format=date_format)
                
                standardized_data.append({
                    'date': date_obj.strftime('%Y-%m-%d'),
                    'value': record['value'],
                    'source_name': data_name,
                    'data_type': data_type,
                    'country': country,
                    'unit': unit,
                    'extraction_time': datetime.now().isoformat(),
                    'original_date_format': date_str,
                    'extraction_pattern': record['pattern']
                })
                
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error processing {date_str}: {e}")
                continue
        
        # Create and sort DataFrame
        df = pd.DataFrame(standardized_data)
        df = df.sort_values('date').reset_index(drop=True)
        
        if verbose and len(df) > 0:
            print(f"\n✅ Final dataset: {len(df)} records")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
            print(f"   Latest: {df['value'].iloc[-1]:.2f} ({df['date'].iloc[-1]})")
        
        return df
    
    def save_to_excel(self, 
                     df: pd.DataFrame, 
                     filename_prefix: str,
                     data_name: str) -> Optional[str]:
        """
        Save DataFrame to Excel with analysis sheets
        
        Args:
            df: DataFrame to save
            filename_prefix: Prefix for filename
            data_name: Descriptive name for the data
            
        Returns:
            File path if successful, None if failed
        """
        
        if df is None or df.empty:
            print("❌ No data to save")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.xlsx"
        filepath = self.output_dir / filename
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Dataset Name', 'Total Records', 'Date Range Start', 'Date Range End',
                        'Latest Value', 'Value Range', 'Unique Values', 'Data Type',
                        'Country', 'Unit', 'Extraction Time'
                    ],
                    'Value': [
                        data_name, len(df), df['date'].min(), df['date'].max(),
                        f"{df['value'].iloc[-1]:.2f}", 
                        f"{df['value'].min():.2f} - {df['value'].max():.2f}",
                        len(df['value'].unique()), df['data_type'].iloc[0],
                        df['country'].iloc[0], df['unit'].iloc[0],
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Value changes analysis
                if len(df) > 1:
                    df_changes = df.copy()
                    df_changes['value_change'] = df_changes['value'].diff()
                    df_changes['change_type'] = df_changes['value_change'].apply(
                        lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'No Change')
                    )
                    
                    # Only actual changes
                    changes_only = df_changes[df_changes['value_change'].notna() & (df_changes['value_change'] != 0)]
                    
                    if not changes_only.empty:
                        changes_export = changes_only[['date', 'value', 'value_change', 'change_type']]
                        changes_export.to_excel(writer, sheet_name='Value_Changes', index=False)
            
            print(f"💾 Saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            return None

# Convenience functions for common use cases

def extract_interest_rates(url: str, 
                          country: str = "Unknown",
                          data_name: str = "Interest Rates",
                          rate_range: Tuple[float, float] = (0.0, 20.0)) -> Optional[pd.DataFrame]:
    """
    Extract interest rate data from a website
    
    Args:
        url: Website URL
        country: Country name
        data_name: Descriptive name
        rate_range: Valid rate range (min, max)
        
    Returns:
        DataFrame with interest rate data
    """
    scraper = UniversalTableScraper()
    return scraper.extract_table_data(
        url=url,
        data_name=data_name,
        date_format="%d/%m/%Y",
        value_range=rate_range,
        country=country,
        data_type="Interest_Rate",
        unit="Percent per annum"
    )

def extract_economic_indicator(url: str,
                              indicator_name: str,
                              country: str = "Unknown",
                              value_range: Tuple[float, float] = (0.0, 1000.0),
                              unit: str = "Index") -> Optional[pd.DataFrame]:
    """
    Extract general economic indicator data
    
    Args:
        url: Website URL
        indicator_name: Name of the indicator
        country: Country name
        value_range: Valid value range
        unit: Unit of measurement
        
    Returns:
        DataFrame with indicator data
    """
    scraper = UniversalTableScraper()
    return scraper.extract_table_data(
        url=url,
        data_name=indicator_name,
        date_format="%d/%m/%Y",
        value_range=value_range,
        country=country,
        data_type="Economic_Indicator",
        unit=unit
    )

# Example usage functions

def extract_malaysia_opr() -> Optional[pd.DataFrame]:
    """Extract Malaysia OPR data - example usage"""
    url = "https://financialmarkets.bnm.gov.my/data-download-opr?date_range=all_data&format=json"
    
    scraper = UniversalTableScraper()
    df = scraper.extract_table_data(
        url=url,
        data_name="Malaysia Overnight Policy Rate",
        date_format="%d/%m/%Y",
        value_range=(0.25, 15.0),
        country="Malaysia",
        data_type="Overnight_Policy_Rate",
        unit="Percent per annum"
    )
    
    if df is not None:
        # Save to Excel
        scraper.save_to_excel(df, "malaysia_opr", "Malaysia OPR")
    
    return df

def main():
    """Example usage of the universal scraper"""
    print("🚀 UNIVERSAL TABLE DATA SCRAPER")
    print("=" * 40)
    print("Example: Extracting Malaysia OPR data")
    
    # Extract Malaysia OPR as example
    df = extract_malaysia_opr()
    
    if df is not None:
        print(f"\n🎉 SUCCESS!")
        print(f"📊 Extracted {len(df)} records")
        print(f"📅 Range: {df['date'].min()} to {df['date'].max()}")
    else:
        print("\n❌ Extraction failed")

if __name__ == "__main__":
    main()
