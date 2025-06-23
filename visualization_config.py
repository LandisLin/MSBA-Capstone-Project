"""
Visualization Configuration and Utilities
Centralized configuration, data processing, and chart utilities for all visualization files
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import warnings
import webbrowser
import os
from datetime import datetime

# ============================================================================
# SECTION 1: GLOBAL CONFIGURATION AND CONSTANTS
# ============================================================================

# Suppress warnings globally
warnings.filterwarnings('ignore')

# Configure Plotly renderer
try:
    pio.renderers.default = "browser"
except:
    pio.renderers.default = "html"

# Default data directory
DEFAULT_DATA_DIR = "./extracted_data"

# Country configuration - single source of truth
COUNTRIES = {
    'singapore': {
        'pattern': 'standardized_cleaned_macro_data_singapore_*.xlsx',
        'display': 'Singapore',
        'code': 'SG'
    },
    'us': {
        'pattern': 'cleaned_macro_data_us_*.xlsx', 
        'display': 'United States',
        'code': 'US'
    },
    'euro_area': {
        'pattern': 'cleaned_macro_data_euro area_*.xlsx',
        'display': 'Euro Area',
        'code': 'EU'
    },
    'japan': {
        'pattern': 'cleaned_macro_data_japan_*.xlsx',
        'display': 'Japan',
        'code': 'JP'
    }
}

# Market indices configuration
MARKET_INDICES_PATTERN = "market_indices_data_*.xlsx"

# Name mapping for display formatting
NAME_CORRECTIONS = {
    # Countries
    'singapore': 'Singapore',
    'us': 'US', 
    'euro_area': 'Euro Area',
    'japan': 'Japan',
    
    # Market indices
    'sandp_500_index': 'S&P 500 Index',
    's&p_500_index': 'S&P 500 Index',
    'nasdaq_composite_index': 'NASDAQ Composite Index',
    'shanghai_composite_index': 'Shanghai Composite Index',
    'hang_seng_index': 'Hang Seng Index',
    'nikkei_225': 'Nikkei 225',
    'euro_stoxx_50': 'EURO STOXX 50',
    'straits_times_index': 'Straits Times Index',
    'cboe_volatility_index_(vix)': 'VIX',
    'vix': 'VIX'
}

# Color schemes for charts
CHART_COLORS = {
    'countries': {
        'singapore': '#1f77b4',
        'us': '#ff7f0e', 
        'euro_area': '#2ca02c',
        'japan': '#d62728'
    },
    'indicators': {
        'GDP': 'blue',
        'CPI': 'orange', 
        'Interest_Rate': 'green',
        'Population': 'purple',
        'Property_Price': 'brown'
    },
    'qualitative': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
}

# Chart styling defaults
CHART_DEFAULTS = {
    'line_width': 2,
    'height': 600,
    'title_x': 0.5,
    'hovermode': 'x unified'
}

# ============================================================================
# SECTION 2: DATA UTILITIES
# ============================================================================

class DataLoader:
    """Centralized data loading utilities"""
    
    @staticmethod
    def find_latest_file(pattern, data_dir=DEFAULT_DATA_DIR):
        """Find the latest file matching pattern"""
        data_path = Path(data_dir)
        if not data_path.exists():
            return None
        
        files = list(data_path.glob(pattern))
        if not files:
            return None
        
        return max(files, key=lambda x: x.stat().st_mtime)
    
    @staticmethod
    def find_latest_country_file(country_key, data_dir=DEFAULT_DATA_DIR):
        """Find latest file for specific country"""
        if country_key not in COUNTRIES:
            return None
        
        pattern = COUNTRIES[country_key]['pattern']
        latest_file = DataLoader.find_latest_file(pattern, data_dir)
        
        if latest_file:
            print(f"📄 Using: {latest_file.name}")
        
        return str(latest_file) if latest_file else None
    
    @staticmethod
    def find_latest_market_file(data_dir=DEFAULT_DATA_DIR):
        """Find latest market indices file"""
        return DataLoader.find_latest_file(MARKET_INDICES_PATTERN, data_dir)
    
    @staticmethod
    def load_excel_sheets(file_path):
        """Load all sheets from Excel file"""
        try:
            return pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return None
    
    @staticmethod
    def get_available_countries(data_dir=DEFAULT_DATA_DIR):
        """Get list of available countries from data files"""
        available = []
        for country_key in COUNTRIES.keys():
            if DataLoader.find_latest_country_file(country_key, data_dir):
                available.append(country_key)
        return available


class DataProcessor:
    """Centralized data processing utilities"""
    
    @staticmethod
    def process_singapore_data(df, sheet_name):
        """Process Singapore data format - standardized version"""
        try:
            # Handle Property_Price special case
            if sheet_name == 'Property_Price':
                if 'property_type' in df.columns:
                    # Filter for All Residential
                    df_filtered = df[df['property_type'].str.contains('All Residential', case=False, na=False)]
                    if df_filtered.empty:
                        df_filtered = df[df['property_type'].str.contains('All', case=False, na=False)]
                    if df_filtered.empty:
                        df_filtered = df.iloc[:1]  # Fallback to first row
                else:
                    df_filtered = df
                
                # Handle different column combinations
                if 'date' in df_filtered.columns and 'index' in df_filtered.columns:
                    result = pd.DataFrame({
                        'date': pd.to_datetime(df_filtered['date']),
                        'value': pd.to_numeric(df_filtered['index'], errors='coerce'),
                        'series': 'Property Price Index',
                        'country': 'Singapore',
                        'indicator': sheet_name
                    })
                elif 'date' in df_filtered.columns and 'value' in df_filtered.columns:
                    result = pd.DataFrame({
                        'date': pd.to_datetime(df_filtered['date']),
                        'value': pd.to_numeric(df_filtered['value'], errors='coerce'),
                        'series': 'Property Price Index',
                        'country': 'Singapore',
                        'indicator': sheet_name
                    })
                else:
                    return None
            else:
                # Standard processing for other indicators
                if 'date' not in df.columns or 'value' not in df.columns:
                    return None
                
                result = pd.DataFrame({
                    'date': pd.to_datetime(df['date']),
                    'value': pd.to_numeric(df['value'], errors='coerce'),
                    'series': df.get('series_name', sheet_name).iloc[0] if len(df) > 0 else sheet_name,
                    'country': 'Singapore',
                    'indicator': sheet_name
                })
            
            return result.dropna(subset=['date', 'value']).sort_values('date')
            
        except Exception as e:
            print(f"   ❌ Error processing Singapore {sheet_name}: {e}")
            return None
    
    @staticmethod
    def process_fred_data(df, sheet_name, country_key):
        """Process FRED data format - standardized version"""
        try:
            if 'date' not in df.columns or 'value' not in df.columns:
                return None
            
            # Fix US population (convert from thousands to actual numbers)
            values = pd.to_numeric(df['value'], errors='coerce')
            if sheet_name == 'Population' and country_key == 'us':
                values = values * 1000
            
            result = pd.DataFrame({
                'date': pd.to_datetime(df['date']),
                'value': values,
                'series': df.get('source_name', sheet_name).iloc[0] if len(df) > 0 else sheet_name,
                'country': country_key,
                'indicator': sheet_name
            })
            
            return result.dropna(subset=['date', 'value']).sort_values('date')
            
        except Exception as e:
            print(f"   ❌ Error processing FRED {sheet_name}: {e}")
            return None
    
    @staticmethod
    def process_market_data(df, index_name):
        """Process market indices data - standardized version"""
        try:
            # Expected columns: Index_Name, Symbol, Country, Date, Close
            if 'Date' not in df.columns or 'Close' not in df.columns:
                return None
            
            result = pd.DataFrame({
                'date': pd.to_datetime(df['Date']),
                'close': pd.to_numeric(df['Close'], errors='coerce'),
                'index_name': index_name,
                'symbol': df.get('Symbol', index_name),
                'country': df.get('Country', 'Unknown')
            })
            
            return result.dropna(subset=['close']).sort_values('date')
            
        except Exception as e:
            print(f"   ❌ Error processing market data {index_name}: {e}")
            return None


class AnalysisUtils:
    """Analysis and calculation utilities"""
    
    @staticmethod
    def calculate_growth_rates(df, periods=1):
        """Calculate percentage growth rates"""
        if len(df) < periods + 1:
            return df
        
        df_copy = df.copy().sort_values('date')
        df_copy['growth_rate'] = df_copy['value'].pct_change(periods=periods) * 100
        return df_copy
    
    @staticmethod
    def calculate_returns(df, periods=1):
        """Calculate percentage returns (alias for calculate_growth_rates)"""
        return AnalysisUtils.calculate_growth_rates(df, periods)
    
    @staticmethod
    def detect_inflation_periods(data, country_key):
        """Detect proper periods for YoY inflation calculation"""
        if len(data) < 2:
            return 4
        
        date_diff = data['date'].diff().median()
        
        if date_diff <= pd.Timedelta(days=32):
            return 12  # Monthly data
        elif date_diff <= pd.Timedelta(days=100):
            return 4   # Quarterly data
        else:
            return 1   # Annual data
    
    @staticmethod
    def calculate_ytd_performance(df):
        """Calculate year-to-date performance"""
        df_sorted = df.copy().sort_values('date')
        current_year = datetime.now().year
        
        # Get data for current year
        current_year_data = df_sorted[df_sorted['date'].dt.year == current_year]
        
        if len(current_year_data) < 2:
            return None
        
        start_price = current_year_data['close'].iloc[0] if 'close' in current_year_data.columns else current_year_data['value'].iloc[0]
        latest_price = current_year_data['close'].iloc[-1] if 'close' in current_year_data.columns else current_year_data['value'].iloc[-1]
        
        ytd_return = ((latest_price - start_price) / start_price) * 100
        
        return {
            'ytd_return': ytd_return,
            'start_price': start_price,
            'latest_price': latest_price,
            'start_date': current_year_data['date'].iloc[0],
            'latest_date': current_year_data['date'].iloc[-1]
        }


class SourceUrlLoader:
    """Source URL loading utilities"""
    
    @staticmethod
    def load_source_urls_from_excel(file_path, sheet_name='Contents'):
        """Load source URLs from Excel Contents sheet"""
        source_urls = {}
        
        try:
            # Skip merged title rows and handle both patterns
            df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=2)
            
            # Find the correct columns flexibly
            name_col = None
            url_col = None
            
            for col in df.columns:
                col_lower = str(col).lower().strip()
                if any(keyword in col_lower for keyword in ['data type', 'index name']):
                    name_col = col
                elif any(keyword in col_lower for keyword in ['source url', 'url']):
                    url_col = col
            
            if name_col and url_col:
                for _, row in df.iterrows():
                    name_value = row[name_col]
                    source_url = row[url_col]
                    
                    if (pd.notna(name_value) and pd.notna(source_url) and 
                        str(source_url) not in ['N/A', 'nan', '', 'None']):
                        
                        # Store with original name
                        source_urls[str(name_value)] = str(source_url)
                        
                        # For market indices, also store with sheet name format
                        if 'index name' in str(name_col).lower():
                            sheet_name_clean = str(name_value).replace(' ', '_').replace('&', 'and')
                            sheet_name_clean = sheet_name_clean.replace('(', '').replace(')', '')[:31]
                            source_urls[sheet_name_clean] = str(source_url)
        
        except Exception as e:
            print(f"Warning: Could not load source URLs: {e}")
        
        return source_urls
    
    @staticmethod
    def get_source_url_for_index(index_name, source_urls):
        """Get source URL for specific index with name matching"""
        # Try direct lookup
        if index_name in source_urls:
            return source_urls[index_name]
        
        # Try with display name conversion
        display_name = NameFormatter.format_display_name(index_name)
        if display_name in source_urls:
            return source_urls[display_name]
        
        # Try partial matching
        index_lower = index_name.lower().replace('_', ' ')
        for stored_name, url in source_urls.items():
            stored_lower = stored_name.lower()
            if index_lower in stored_lower or stored_lower in index_lower:
                return url
        
        return "N/A"


# ============================================================================
# SECTION 3: CHART UTILITIES
# ============================================================================

class NameFormatter:
    """Name formatting utilities"""
    
    @staticmethod
    def format_display_name(name):
        """Format internal names to display names"""
        if not name:
            return name
        
        # Check direct mapping first
        lower_name = name.lower()
        if lower_name in NAME_CORRECTIONS:
            return NAME_CORRECTIONS[lower_name]
        
        # Handle S&P patterns
        if 'sandp' in lower_name or 's&p' in lower_name:
            return 'S&P 500 Index'
        
        # Default formatting
        return name.replace('_', ' ').title()
    
    @staticmethod
    def format_country_name(country_key):
        """Format country key to display name"""
        if country_key in COUNTRIES:
            return COUNTRIES[country_key]['display']
        return NameFormatter.format_display_name(country_key)


class ChartDisplayer:
    """Chart display and export utilities"""
    
    @staticmethod
    def show_chart(fig, filename_prefix="chart", auto_open=True):
        """Display chart with fallback to HTML file"""
        if not fig:
            print("❌ Cannot display - no chart provided")
            return False
        
        try:
            if auto_open:
                fig.show()
                print("✅ Chart displayed in browser!")
                return True
        except Exception:
            pass
        
        # Fallback to HTML file
        try:
            timestamp = datetime.now().strftime('%H%M%S')
            filename = f"{filename_prefix}_{timestamp}.html"
            fig.write_html(filename)
            print(f"💾 Chart saved as '{filename}'")
            
            if auto_open:
                try:
                    webbrowser.open(f"file://{os.path.abspath(filename)}")
                    print("🌐 Opening in browser...")
                    return True
                except:
                    print("💡 Please open the HTML file manually")
            
            return True
            
        except Exception as e:
            print(f"❌ Could not save chart: {e}")
            return False
    
    @staticmethod
    def save_chart(fig, filename, format="html"):
        """Save chart to file"""
        try:
            if format.lower() == "html":
                fig.write_html(filename)
            elif format.lower() == "png":
                fig.write_image(filename)
            elif format.lower() == "pdf":
                fig.write_image(filename)
            else:
                fig.write_html(filename)
            
            print(f"💾 Chart saved as '{filename}'")
            return True
            
        except Exception as e:
            print(f"❌ Could not save chart: {e}")
            return False


class ChartStyler:
    """Chart styling utilities"""
    
    @staticmethod
    def apply_default_layout(fig, title, height=None):
        """Apply default layout styling"""
        layout_updates = {
            'title': {'text': title, 'x': 0.5, 'xanchor': 'center'},
            'hovermode': CHART_DEFAULTS['hovermode']
        }
        
        if height:
            layout_updates['height'] = height
        else:
            layout_updates['height'] = CHART_DEFAULTS['height']
        
        fig.update_layout(**layout_updates)
        return fig
    
    @staticmethod
    def get_country_color(country_key):
        """Get color for country"""
        return CHART_COLORS['countries'].get(country_key, '#1f77b4')
    
    @staticmethod
    def get_indicator_color(indicator):
        """Get color for indicator"""
        return CHART_COLORS['indicators'].get(indicator, '#1f77b4')
    
    @staticmethod
    def get_qualitative_colors():
        """Get qualitative color palette"""
        return CHART_COLORS['qualitative']
    
    @staticmethod
    def add_reference_lines(fig, y_values, row=None, col=None, **kwargs):
        """Add horizontal reference lines"""
        default_style = {'line_dash': "dash", 'opacity': 0.7}
        default_style.update(kwargs)
        
        for y_val in y_values:
            if row and col:
                fig.add_hline(y=y_val, row=row, col=col, **default_style)
            else:
                fig.add_hline(y=y_val, **default_style)
        
        return fig


# ============================================================================
# SECTION 4: CONVENIENCE FUNCTIONS
# ============================================================================

def load_country_data(country_key, data_dir=DEFAULT_DATA_DIR):
    """Convenience function to load complete country data"""
    file_path = DataLoader.find_latest_country_file(country_key, data_dir)
    if not file_path:
        return None
    
    excel_data = DataLoader.load_excel_sheets(file_path)
    if not excel_data:
        return None
    
    processed_data = {}
    print(f"🔄 Loading {NameFormatter.format_country_name(country_key)} data...")
    
    for sheet_name, df in excel_data.items():
        if sheet_name == 'Contents' or df.empty:
            continue
        
        # Process based on country
        if country_key == 'singapore':
            processed_df = DataProcessor.process_singapore_data(df, sheet_name)
        else:
            processed_df = DataProcessor.process_fred_data(df, sheet_name, country_key)
        
        if processed_df is not None and not processed_df.empty:
            processed_data[sheet_name] = processed_df
            print(f"   ✅ {sheet_name}: {len(processed_df)} records")
    
    return processed_data

def load_market_data(data_dir=DEFAULT_DATA_DIR):
    """Convenience function to load market indices data"""
    file_path = DataLoader.find_latest_market_file(data_dir)
    if not file_path:
        return None
    
    excel_data = DataLoader.load_excel_sheets(str(file_path))
    if not excel_data:
        return None
    
    processed_data = {}
    print(f"🔄 Loading market indices data...")
    
    for sheet_name, df in excel_data.items():
        if sheet_name == 'Contents' or df.empty:
            continue
        
        processed_df = DataProcessor.process_market_data(df, sheet_name)
        if processed_df is not None and not processed_df.empty:
            processed_data[sheet_name] = processed_df
            print(f"   ✅ {sheet_name}: {len(processed_df)} records")
    
    return processed_data

def get_all_available_data(data_dir=DEFAULT_DATA_DIR):
    """Get summary of all available data"""
    countries = DataLoader.get_available_countries(data_dir)
    market_file = DataLoader.find_latest_market_file(data_dir)
    
    return {
        'countries': countries,
        'market_available': market_file is not None,
        'country_display_names': [NameFormatter.format_country_name(c) for c in countries]
    }

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_configuration():
    """Test the configuration and utilities"""
    print("🧪 Testing Visualization Configuration")
    print("=" * 50)
    
    # Test data availability
    available_data = get_all_available_data()
    print(f"✅ Available countries: {len(available_data['countries'])}")
    print(f"✅ Market data available: {available_data['market_available']}")
    
    # Test name formatting
    test_names = ['singapore', 'us', 'sandp_500_index', 'vix']
    print(f"\n🏷️ Name formatting test:")
    for name in test_names:
        formatted = NameFormatter.format_display_name(name)
        print(f"   {name} → {formatted}")
    
    # Test color schemes
    print(f"\n🎨 Color schemes loaded:")
    print(f"   Countries: {len(CHART_COLORS['countries'])} colors")
    print(f"   Indicators: {len(CHART_COLORS['indicators'])} colors")
    print(f"   Qualitative: {len(CHART_COLORS['qualitative'])} colors")
    
    print("🎉 Configuration test completed!")

if __name__ == "__main__":
    test_configuration()