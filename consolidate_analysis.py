"""
Consolidate Analysis Module - FIXED VERSION
Combines macroeconomic, market, and news data into a unified Excel sheet
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import streamlit as st

class ConsolidateAnalysis:
    """
    Handles consolidation of all data categories into a unified format
    """
    
    def __init__(self):
        self.consolidated_data = None
        self.data_summary = {
            'total_records': 0,
            'date_range': None,
            'categories': {'macro': 0, 'market': 0, 'news': 0}
        }
    
    def get_most_recent_file(self, pattern: str, folder: str = ".") -> Optional[str]:
        """Get the most recent file matching the pattern"""
        search_path = os.path.join(folder, pattern) if folder != "." else pattern
        files = glob.glob(search_path)
        if not files:
            return None
        # Sort by modification time, return the most recent
        return max(files, key=os.path.getmtime)
    
    def load_macro_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Load and process macroeconomic data for all countries"""
        macro_records = []
        
        # Region mapping for file names
        regions = [
            'vietnam', 'china', 'euro area', 'india', 'indonesia', 
            'japan', 'malaysia', 'thailand', 'uk', 'us'
        ]
        
        # Standard macro indicators (sheet names)
        indicators = ['GDP', 'CPI', 'Interest_Rate', 'Population', 'Property_Price']
        
        st.write(f"🔍 Loading macro data from {start_date} to {end_date}")
        
        try:
            # Process each region's macro data
            for region in regions:
                # Get the most recent file for this region
                pattern = f"cleaned_macro_data_{region}_*.xlsx"
                file_path = self.get_most_recent_file(pattern, "extracted_data")
                
                if not file_path:
                    st.write(f"⚠️ No macro file found for {region}")
                    continue
                
                st.write(f"📂 Processing {region}: {os.path.basename(file_path)}")
                
                # Read each indicator sheet
                try:
                    xl_file = pd.ExcelFile(file_path)
                    available_sheets = xl_file.sheet_names
                    st.write(f"   Available sheets: {available_sheets}")
                    
                    for indicator in indicators:
                        if indicator not in available_sheets:
                            continue
                            
                        try:
                            df = pd.read_excel(file_path, sheet_name=indicator)
                            st.write(f"   ✅ Loaded {indicator}: {len(df)} rows")
                            
                            # Find date column
                            date_col = None
                            for col in df.columns:
                                if 'date' in col.lower():
                                    date_col = col
                                    break
                            
                            if date_col is None:
                                st.write(f"   ⚠️ No date column found in {indicator}")
                                continue
                                
                            # Convert date and filter by range
                            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                            mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
                            filtered_df = df[mask]
                            
                            # Find value column (usually the second column)
                            value_col = None
                            for col in df.columns:
                                if col != date_col and df[col].dtype in ['float64', 'int64']:
                                    value_col = col
                                    break
                            
                            if value_col is None and len(df.columns) > 1:
                                value_col = df.columns[1]
                            
                            # Create records (no filtering needed for other regions)
                            for _, row in filtered_df.iterrows():
                                if pd.notna(row[date_col]) and pd.notna(row[value_col]):
                                    # FIXED: Standardize country names to full versions
                                    country_name = region.title().replace('_', ' ')
                                    if country_name.lower() == 'us':
                                        country_name = 'United States'
                                    elif country_name.lower() == 'uk':
                                        country_name = 'United Kingdom'
                                    
                                    macro_records.append({
                                        'Date': row[date_col].date(),
                                        'Country': country_name,
                                        'Type': 'Macro',
                                        'Attribute': indicator,
                                        'Value': row[value_col]
                                    })
                                    
                        except Exception as e:
                            st.write(f"   ❌ Error reading {indicator}: {e}")
                            continue
                            
                except Exception as e:
                    st.write(f"❌ Error processing {region}: {e}")
                    continue
            
            # FIXED: Process Singapore separately with proper debugging
            singapore_pattern = "standardized_cleaned_macro_data_singapore_*.xlsx"
            singapore_file = self.get_most_recent_file(singapore_pattern, "extracted_data")
            
            if singapore_file:
                st.write(f"📂 Processing Singapore: {os.path.basename(singapore_file)}")
                try:
                    xl_file = pd.ExcelFile(singapore_file)
                    available_sheets = xl_file.sheet_names
                    st.write(f"   Available sheets: {available_sheets}")
                    
                    for indicator in indicators:
                        if indicator not in available_sheets:
                            continue
                            
                        try:
                            df = pd.read_excel(singapore_file, sheet_name=indicator)
                            st.write(f"   ✅ Loaded {indicator}: {len(df)} rows")
                            st.write(f"   Columns: {list(df.columns)}")
                            
                            # Find date column
                            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                            
                            if date_col is None:
                                st.write(f"   ⚠️ No date column found in {indicator}")
                                continue
                                
                            # Convert date and filter by range
                            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                            mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
                            filtered_df = df[mask]
                            
                            st.write(f"   Filtered to {len(filtered_df)} rows for date range")
                            
                            # FIXED: Find value column (specifically look for 'value' column for Singapore)
                            value_col = 'value' if 'value' in df.columns else None
                            if value_col is None:
                                # Fallback to first numeric column that's not date
                                for col in df.columns:
                                    if col != date_col and df[col].dtype in ['float64', 'int64']:
                                        value_col = col
                                        break
                            
                            if value_col is None:
                                st.write(f"   ⚠️ No value column found in {indicator}")
                                continue
                            
                            st.write(f"   Using value column: {value_col}")
                            
                            # Apply property price filtering for Singapore
                            for _, row in filtered_df.iterrows():
                                if pd.notna(row[date_col]) and pd.notna(row[value_col]):
                                    # For Property_Price, filter to only "All Residential"
                                    if indicator == 'Property_Price':
                                        if 'property_type' in df.columns:
                                            if row['property_type'] != 'All Residential':
                                                continue
                                    
                                    macro_records.append({
                                        'Date': row[date_col].date(),
                                        'Country': 'Singapore',
                                        'Type': 'Macro',
                                        'Attribute': indicator,
                                        'Value': row[value_col]
                                    })
                                    
                        except Exception as e:
                            st.write(f"   ❌ Error reading {indicator}: {e}")
                            continue
                            
                except Exception as e:
                    st.write(f"❌ Error processing Singapore: {e}")
            else:
                st.write("⚠️ No Singapore macro file found")
            
            # FIXED: Process World Bank GDP data with proper debugging and attribute names
            worldbank_pattern = "world_bank_gdp_data_*.xlsx"
            worldbank_file = self.get_most_recent_file(worldbank_pattern, "extracted_data")
            
            if worldbank_file:
                st.write(f"📂 Processing World Bank GDP: {os.path.basename(worldbank_file)}")
                try:
                    xl_file = pd.ExcelFile(worldbank_file)
                    available_sheets = xl_file.sheet_names
                    st.write(f"   Available sheets: {available_sheets}")
                    
                    for sheet_name in available_sheets:
                        if sheet_name.lower() == 'contents':
                            continue
                        
                        st.write(f"   Processing sheet: {sheet_name}")
                        
                        df = pd.read_excel(worldbank_file, sheet_name=sheet_name)
                        st.write(f"   ✅ Loaded {sheet_name}: {len(df)} rows")
                        st.write(f"   Columns: {list(df.columns)}")
                        
                        # Look for required columns: date, value, country, indicator
                        if all(col in df.columns for col in ['date', 'value', 'country', 'indicator']):
                            # Convert date and keep only date part
                            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
                            
                            # Filter by date range
                            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                            filtered_df = df[mask]
                            
                            st.write(f"   Filtered to {len(filtered_df)} rows for date range")
                            
                            for _, row in filtered_df.iterrows():
                                if pd.notna(row['date']) and pd.notna(row['value']):
                                    # FIXED: Standardize country names (World Bank uses lowercase)
                                    country_name = row['country'].title().replace('_', ' ')
                                    if country_name.lower() == 'us':
                                        country_name = 'United States'
                                    elif country_name.lower() == 'uk':
                                        country_name = 'United Kingdom'
                                    
                                    # FIXED: Create descriptive attribute names
                                    if 'gdp_per_capita' in sheet_name.lower():
                                        attribute_name = 'GDP per Capita in USD'
                                    elif 'gdp' in sheet_name.lower():
                                        attribute_name = 'GDP in Trillion USD'
                                    else:
                                        attribute_name = f"WorldBank_{row['indicator']}"
                                    
                                    macro_records.append({
                                        'Date': row['date'],
                                        'Country': country_name,
                                        'Type': 'Macro',
                                        'Attribute': attribute_name,
                                        'Value': row['value']
                                    })
                        else:
                            st.write(f"   ⚠️ Missing required columns in {sheet_name}")
                            
                except Exception as e:
                    st.write(f"❌ Error processing World Bank data: {e}")
            else:
                st.write("⚠️ No World Bank GDP file found")
                        
        except Exception as e:
            st.error(f"Error loading macro data: {e}")
            
        st.write(f"📈 Total macro records loaded: {len(macro_records)}")
        return pd.DataFrame(macro_records)
    
    def load_market_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Load and process market indices data"""
        market_records = []
        
        try:
            # Get the most recent market indices file
            pattern = "market_indices_data_*.xlsx"
            file_path = self.get_most_recent_file(pattern, "extracted_data")
            
            if not file_path:
                st.write("⚠️ No market indices file found")
                return pd.DataFrame(market_records)
            
            st.write(f"📈 Processing market data: {os.path.basename(file_path)}")
            
            xl_file = pd.ExcelFile(file_path)
            available_sheets = xl_file.sheet_names
            st.write(f"   Available sheets: {available_sheets}")
            
            # FIXED: Updated country mapping for market indices
            market_countries = {
                'STI': 'Singapore',
                'Straits_Times_Index': 'Singapore',
                'Nikkei_225': 'Japan', 
                'NIKKEI': 'Japan',
                'Shanghai_Composite_Index': 'China',
                'Hang_Seng_Index': 'Hong Kong',
                'HANG_SENG': 'Hong Kong',
                'KOSPI': 'South Korea',
                'SET_Index': 'Thailand',
                'SET': 'Thailand',
                'FTSE_Bursa_Malaysia_KLCI': 'Malaysia',
                'KLCI': 'Malaysia',
                'Jakarta_Composite_Index': 'Indonesia',
                'JKSE': 'Indonesia',
                'VN_INDEX': 'Vietnam',
                'BSE_Sensex': 'India',
                'NIFTY': 'India',
                'FTSE_100_Index': 'United Kingdom',  # Changed from UK to United Kingdom
                'FTSE': 'United Kingdom',
                'EURO_STOXX_50': 'Euro Area',  # FIXED: Changed from EU to Euro Area
                'DAX': 'Germany', 
                'CAC': 'France',
                'SandP_500_Index': 'United States',  # Changed from US to United States
                'SP500': 'United States',
                'NASDAQ_Composite_Index': 'United States',
                'NASDAQ': 'United States',
                'DOW_JONES': 'United States',
                'CBOE_Volatility_Index_(VIX)': 'United States'
            }
            
            # Process each market index sheet (skip Contents)
            for sheet_name in available_sheets:
                if sheet_name.lower() == 'contents':
                    continue
                    
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    st.write(f"   ✅ Processing {sheet_name}: {len(df)} rows")
                    
                    # Check for required columns: Date, Close
                    required_cols = ['Date', 'Close']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.write(f"   ⚠️ Missing columns in {sheet_name}: {missing_cols}")
                        continue
                    
                    # Convert date and filter by range
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
                    filtered_df = df[mask]
                    
                    st.write(f"   Filtered to {len(filtered_df)} rows for date range")
                    
                    # Extract data for each row
                    for _, row in filtered_df.iterrows():
                        if pd.notna(row['Date']) and pd.notna(row['Close']):
                            # Use sheet name as Index_Name if not available
                            index_name = row.get('Index_Name', sheet_name)
                            
                            # FIXED: Get country with Euro Area mapping
                            country = market_countries.get(sheet_name, 'Unknown')
                            volume = row.get('Volume', None)
                            
                            # Create record for Close price
                            market_records.append({
                                'Date': row['Date'].date(),
                                'Country': country,
                                'Type': 'Market',
                                'Attribute': f"{index_name}_Close",
                                'Value': row['Close']
                            })
                            
                            # Create record for Volume if available
                            if pd.notna(volume):
                                market_records.append({
                                    'Date': row['Date'].date(),
                                    'Country': country,
                                    'Type': 'Market',
                                    'Attribute': f"{index_name}_Volume",
                                    'Value': volume
                                })
                                
                except Exception as e:
                    st.write(f"   ❌ Error processing {sheet_name}: {e}")
                    continue
                    
        except Exception as e:
            st.error(f"Error loading market data: {e}")
            
        st.write(f"📈 Total market records loaded: {len(market_records)}")
        return pd.DataFrame(market_records)
    
    def load_news_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Load and process news analysis data with topic-level sentiment averaging"""
        news_records = []
        
        try:
            # News file path
            file_path = os.path.join("news_analysis_output", "master_news_analysis.xlsx")
            
            if not os.path.exists(file_path):
                st.write("⚠️ News analysis file not found")
                return pd.DataFrame(news_records)
            
            st.write(f"📰 Processing news data: {os.path.basename(file_path)}")
            
            xl_file = pd.ExcelFile(file_path)
            available_sheets = xl_file.sheet_names
            st.write(f"   Available sheets: {available_sheets}")
            
            # Collect all news data for topic analysis
            all_news_data = []
            
            # Process each news source sheet (skip Contents)
            for sheet_name in available_sheets:
                if sheet_name.lower() == 'contents':
                    continue
                    
                try:
                    # Read with header starting from row 2 to skip merged title rows
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
                    st.write(f"   ✅ Processing {sheet_name}: {len(df)} rows")
                    st.write(f"   Columns found: {list(df.columns)}")
                    
                    # Check for required columns
                    required_cols = ['Region', 'Published', 'Sentiment Score', 'Topic Analysis']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.write(f"   ⚠️ Missing columns in {sheet_name}: {missing_cols}")
                        continue
                    
                    # Convert date and filter by range
                    df['Published'] = pd.to_datetime(df['Published'], errors='coerce').dt.date
                    mask = (df['Published'] >= start_date) & (df['Published'] <= end_date)
                    filtered_df = df[mask]
                    
                    st.write(f"   Filtered to {len(filtered_df)} rows for date range")
                    
                    # Process each article
                    for _, row in filtered_df.iterrows():
                        if pd.notna(row['Published']) and pd.notna(row['Sentiment Score']):
                            regions = str(row['Region']).split(',')
                            
                            # Extract topics from Topic Analysis
                            topics = []
                            if pd.notna(row['Topic Analysis']):
                                topic_text = str(row['Topic Analysis'])
                                # Parse topics (assuming format like "Topics: Economic Growth, Consumer & Retail, Government & Fiscal Policy")
                                if 'Topics:' in topic_text:
                                    topic_part = topic_text.split('Topics:')[1].split('Indicators:')[0]
                                    topics = [topic.strip() for topic in topic_part.split(',') if topic.strip()]
                            
                            # Add record for each region mentioned
                            for region in regions:
                                region = region.strip()
                                if region:
                                    # Add overall sentiment record
                                    all_news_data.append({
                                        'Date': row['Published'],
                                        'Country': region,
                                        'Topic': 'Overall',
                                        'Sentiment_Score': row['Sentiment Score'],
                                        'Source': sheet_name
                                    })
                                    
                                    # Add topic-specific records
                                    for topic in topics:
                                        all_news_data.append({
                                            'Date': row['Published'],
                                            'Country': region,
                                            'Topic': topic,
                                            'Sentiment_Score': row['Sentiment Score'],
                                            'Source': sheet_name
                                        })
                                    
                except Exception as e:
                    st.write(f"   ❌ Error processing {sheet_name}: {e}")
                    continue
            
            # Convert to DataFrame and calculate averages
            if all_news_data:
                news_df = pd.DataFrame(all_news_data)
                
                # Group by Date, Country, and Topic to calculate average sentiment
                grouped = news_df.groupby(['Date', 'Country', 'Topic']).agg({
                    'Sentiment_Score': 'mean',
                    'Source': 'count'
                }).round(4)
                
                grouped.columns = ['Avg_Sentiment_Score', 'Article_Count']
                grouped = grouped.reset_index()
                
                st.write(f"   📊 Calculated topic averages for {len(grouped)} date-country-topic combinations")
                
                # Create consolidated records
                for _, row in grouped.iterrows():
                    # FIXED: Create attribute name based on topic (keep natural language)
                    if row['Topic'] == 'Overall':
                        attribute_name = 'Overall Average Sentiment'
                    else:
                        attribute_name = f"{row['Topic']} Sentiment"
                    
                    news_records.append({
                        'Date': row['Date'],
                        'Country': row['Country'],
                        'Type': 'News',
                        'Attribute': attribute_name,
                        'Value': row['Avg_Sentiment_Score']
                    })
                        
        except Exception as e:
            st.error(f"Error loading news data: {e}")
            
        st.write(f"📰 Total news records loaded: {len(news_records)}")
        return pd.DataFrame(news_records)
    
    def consolidate_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Main function to consolidate all data categories"""
        
        with st.spinner("Loading macroeconomic data..."):
            macro_df = self.load_macro_data(start_date, end_date)
            
        with st.spinner("Loading market indices data..."):
            market_df = self.load_market_data(start_date, end_date)
            
        with st.spinner("Loading news analysis data..."):
            news_df = self.load_news_data(start_date, end_date)
        
        # Combine all dataframes
        all_dfs = []
        if not macro_df.empty:
            all_dfs.append(macro_df)
        if not market_df.empty:
            all_dfs.append(market_df)
        if not news_df.empty:
            all_dfs.append(news_df)
        
        if not all_dfs:
            return pd.DataFrame()
        
        # Concatenate all data
        self.consolidated_data = pd.concat(all_dfs, ignore_index=True)
        
        # FIXED: Multi-level sorting for proper organization
        # 1. First by Date
        # 2. Then by Type (Macro, Market, News)
        # 3. Then by Country  
        # 4. Then by Attribute (to group Close/Volume for same index)
        type_order = {'Macro': 0, 'Market': 1, 'News': 2}
        self.consolidated_data['Type_Order'] = self.consolidated_data['Type'].map(type_order)
        
        self.consolidated_data = self.consolidated_data.sort_values([
            'Date', 'Type_Order', 'Country', 'Attribute'
        ]).drop('Type_Order', axis=1).reset_index(drop=True)
        
        # Update summary statistics
        self.data_summary = {
            'total_records': len(self.consolidated_data),
            'date_range': (self.consolidated_data['Date'].min(), self.consolidated_data['Date'].max()),
            'categories': {
                'macro': len(macro_df),
                'market': len(market_df),
                'news': len(news_df)
            }
        }
        
        return self.consolidated_data
    
    def save_to_excel(self, filename: str = None) -> str:
        """Save consolidated data to Excel file"""
        if self.consolidated_data is None or self.consolidated_data.empty:
            raise ValueError("No consolidated data available to save")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consolidated_data_{timestamp}.xlsx"
        
        # Save to Excel
        self.consolidated_data.to_excel(filename, index=False, engine='openpyxl')
        
        return filename
    
    def get_preview_data(self, n_rows: int = 200) -> pd.DataFrame:
        """Get preview of consolidated data (first n rows)"""
        if self.consolidated_data is None or self.consolidated_data.empty:
            return pd.DataFrame()
        
        return self.consolidated_data.head(n_rows)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of consolidated data"""
        return self.data_summary

# Streamlit interface functions
def create_consolidate_analysis_page():
    """Create the Consolidate Analysis page for Streamlit"""
    
    st.markdown('<h1 class="main-header">📋 Consolidate Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Consolidate multi-source economic data into unified Excel exports and perform correlation analysis across indicators.")

    # Instructions
    with st.expander("📝 How to Use This Page", expanded=False):
        st.markdown("""
        ## Step 1: Data Consolidation
        1. **Select Date Range**: Choose the time period for data consolidation
        2. **Fetch Data**: Click the button to combine all data sources  
        3. **Review Preview**: Examine the first 200 rows in the preview table
        4. **Download**: Get the complete dataset as an Excel file

        ## Step 2: Correlation Analysis
        5. **Pick Target**: Select a country and specific indicator to analyze
        6. **Choose Categories**: Select which data types to include (Macro/Market/News)
        7. **Run Analysis**: Generate correlation chart and detailed results
                    
        ---     
        ### Data Structure
        - **Date**: The date of the data point
        - **Country**: The country/region associated with the data  
        - **Type**: Category of data (Macro, Market, News)
        - **Attribute**: Specific indicator (e.g., GDP, S&P 500, Overall Average Sentiment)
        - **Value**: The actual data value
        
        ### Data Sources
        - **Macro**: Regional economic indicators + World Bank GDP data
        - **Market**: Stock indices with Close prices and Volume (EURO STOXX 50 mapped to Euro Area)
        - **News**: Overall and topic-specific sentiment scores by region and date
        
        ### Correlation Methodology
        - Uses only **actual overlapping measurement dates** (no data filling)
        - Maintains statistical integrity by respecting different update frequencies
        - Requires minimum 3 overlapping data points for valid correlation
        """)
    
    # Initialize consolidate analysis object
    if 'consolidate_analyzer' not in st.session_state:
        st.session_state.consolidate_analyzer = ConsolidateAnalysis()
    
    consolidator = st.session_state.consolidate_analyzer
    
    # Initialize session state for maintaining results
    if 'consolidated_results' not in st.session_state:
        st.session_state.consolidated_results = None
    if 'summary_stats' not in st.session_state:
        st.session_state.summary_stats = None
    
    # Date range selector
    st.subheader("🗓️ Select Time Duration")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2025, 1, 1),
            min_value=date(2000, 1, 1),
            max_value=date.today()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=date.today(),
            min_value=date(2000, 1, 1),
            max_value=date.today()
        )
    
    # Validate date range
    if start_date >= end_date:
        st.error("❌ Start date must be before end date")
        return
    
    # Fetch consolidate data button
    if st.button("🔄 Fetch Consolidated Data", type="primary"):
        try:
            with st.spinner("Consolidating data from all sources..."):
                consolidated_df = consolidator.consolidate_data(start_date, end_date)
            
            if consolidated_df.empty:
                st.warning("⚠️ No data found for the selected date range")
                st.session_state.consolidated_results = None
                st.session_state.summary_stats = None
                return
                
            # Store results in session state
            st.session_state.consolidated_results = consolidated_df.copy()
            st.session_state.summary_stats = consolidator.get_summary_stats()
            
            st.success("✅ Data consolidated successfully!")
                
        except Exception as e:
            st.error(f"❌ Error consolidating data: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    # Display results if available (from session state)
    if st.session_state.consolidated_results is not None:
        
        # Display summary statistics
        summary = st.session_state.summary_stats
        
        st.subheader("📊 Data Summary")
        # Add date range display
        if summary.get('date_range'):
            date_start, date_end = summary['date_range']
            st.caption(f"📅 Data Period: {date_start} to {date_end}")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{summary['total_records']:,}")
        
        with col2:
            st.metric("Macro Records", f"{summary['categories']['macro']:,}")
        
        with col3:
            st.metric("Market Records", f"{summary['categories']['market']:,}")
        
        with col4:
            st.metric("News Records", f"{summary['categories']['news']:,}")
        
        # Display preview of consolidated data
        st.subheader("👁️ Data Preview (First 200 rows)")
        preview_df = st.session_state.consolidated_results.head(200)
        st.dataframe(preview_df, use_container_width=True)
        
        # Download button
        st.subheader("💾 Download Consolidated Data")
        
        try:
            # Create in-memory Excel data
            from io import BytesIO
            
            buffer = BytesIO()
            st.session_state.consolidated_results.to_excel(buffer, index=False, engine='openpyxl')
            excel_data = buffer.getvalue()
            buffer.close()
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consolidated_data_{timestamp}.xlsx"
            
            st.download_button(
                label="📥 Download Excel File",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="secondary",
                key="download_btn"  # Add unique key to prevent rerun issues
            )
            
            st.info(f"📁 File ready for download: {filename}")
                    
        except Exception as e:
            st.error(f"❌ Error creating download file: {e}")

    # Add correlation analysis section
    create_correlation_analysis_section()

def create_correlation_analysis_section():
    """Create correlation analysis section for the consolidated data"""
    
    # Only show if consolidated data is available
    if st.session_state.consolidated_results is None:
        return
    
    st.markdown("---")
    st.subheader("🔍 Correlation Analysis")
    st.markdown("Analyze correlations between your selected target indicator and all other indicators using only actual overlapping data points.")
    
    df = st.session_state.consolidated_results
    
    # Get unique countries and attributes from the fetched data
    available_countries = sorted(df['Country'].unique())
    available_attributes = sorted(df['Attribute'].unique())
    available_categories = ['Macro', 'Market', 'News']
    
    # Selection interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_country = st.selectbox(
            "Select Target Country",
            options=available_countries,
            key="target_country"
        )
    
    with col2:
        # Filter attributes available for the selected country
        country_attributes = df[df['Country'] == target_country]['Attribute'].unique()
        target_attribute = st.selectbox(
            "Select Target Attribute",
            options=sorted(country_attributes),
            key="target_attribute"
        )

    with col3:
        # Category multi-selection
        selected_categories = st.multiselect(
            "Select Data Categories",
            options=available_categories,
            default=available_categories,
            key="analysis_categories",
            help="Choose which data types to include in correlation analysis"
        )
    
    if not selected_categories:
        st.warning("Please select at least one data category for analysis")
        return
    
    # Run correlation analysis button
    if st.button("📈 Run Correlation Analysis", type="primary"):
        try:
            with st.spinner("Calculating correlations using actual overlapping data points..."):
                
                # Filter data by selected categories
                filtered_df = df[df['Type'].isin(selected_categories)].copy()
                
                # Get target data
                target_data = filtered_df[
                    (filtered_df['Country'] == target_country) & 
                    (filtered_df['Attribute'] == target_attribute)
                ][['Date', 'Value']].rename(columns={'Value': 'target'})
                
                if target_data.empty:
                    st.error(f"No data found for target: {target_country} - {target_attribute}")
                    return
                
                st.write(f"📊 Target has {len(target_data)} data points")
                
                # Calculate correlations with actual overlapping data only
                correlations = []
                
                for country in available_countries:
                    country_data = filtered_df[filtered_df['Country'] == country]
                    for attribute in country_data['Attribute'].unique():
                        if country == target_country and attribute == target_attribute:
                            continue  # Skip self-correlation
                            
                        other_data = filtered_df[
                            (filtered_df['Country'] == country) & 
                            (filtered_df['Attribute'] == attribute)
                        ][['Date', 'Value']].rename(columns={'Value': 'other'})
                        
                        # Find actual overlapping dates (no filling)
                        merged = pd.merge(target_data, other_data, on='Date', how='inner')
                        
                        if len(merged) >= 3:  # Need at least 3 overlapping points
                            corr = merged['target'].corr(merged['other'])
                            if pd.notna(corr):
                                correlations.append({
                                    'Country': country,
                                    'Attribute': attribute,
                                    'Correlation': round(corr, 4),
                                    'Data_Points': len(merged)  # Real overlapping points only
                                })
                
                if not correlations:
                    st.warning("No sufficient overlapping data found for correlation analysis")
                    return
                
                # Create correlation DataFrame and sort (positive on top)
                corr_df = pd.DataFrame(correlations)
                corr_df = corr_df.sort_values('Correlation', ascending=False)  # Positive on top
                
                # Display results
                st.subheader(f"📊 Correlation Analysis Results")
                st.caption(f"Target: {target_country} - {target_attribute} | Categories: {', '.join(selected_categories)}")
                
                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Comparisons", len(corr_df))
                with col2:
                    max_corr = corr_df['Correlation'].max()
                    st.metric("Highest Correlation", f"{max_corr:.4f}")
                with col3:
                    min_corr = corr_df['Correlation'].min()
                    st.metric("Lowest Correlation", f"{min_corr:.4f}")
                with col4:
                    avg_points = corr_df['Data_Points'].mean()
                    st.metric("Avg Data Points", f"{avg_points:.0f}")
                
                # Create correlation chart
                import plotly.graph_objects as go
                
                # Create horizontal bar chart
                fig = go.Figure()
                
                # Color coding: positive = green, negative = red
                colors = ['#2E8B57' if x >= 0 else '#DC143C' for x in corr_df['Correlation']]
                
                # Create labels for y-axis (reverse order for positive on top)
                y_labels = [f"{row['Country']} - {row['Attribute']}" for _, row in corr_df.iterrows()]
                y_labels.reverse()  # Reverse to put positive correlations at top
                correlations_reversed = corr_df['Correlation'].tolist()
                correlations_reversed.reverse()
                colors.reverse()
                data_points_reversed = corr_df['Data_Points'].tolist()
                data_points_reversed.reverse()
                
                fig.add_trace(go.Bar(
                    x=correlations_reversed,
                    y=y_labels,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{x:.3f}" for x in correlations_reversed],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Correlation: %{x:.4f}<br>Overlapping Data Points: %{customdata}<extra></extra>',
                    customdata=data_points_reversed
                ))
                
                # Add vertical line at x=0
                fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
                
                # Update layout
                fig.update_layout(
                    title=f"Correlation with {target_country} - {target_attribute}",
                    xaxis_title="Correlation Coefficient",
                    yaxis_title="Country - Indicator",
                    height=max(600, len(corr_df) * 25),  # Dynamic height
                    showlegend=False,
                    xaxis=dict(range=[-1.1, 1.1], tickformat='.2f'),
                    margin=dict(l=300, r=50, t=80, b=50)  # More space for y-axis labels
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display correlation table
                st.subheader("📋 Detailed Correlation Results")
                
                # Format correlation table for better display
                display_df = corr_df.copy()
                display_df['Correlation'] = display_df['Correlation'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Country": st.column_config.TextColumn("Country", width="medium"),
                        "Attribute": st.column_config.TextColumn("Indicator", width="large"), 
                        "Correlation": st.column_config.TextColumn("Correlation", width="small"),
                        "Data_Points": st.column_config.NumberColumn("Actual Overlap", width="small")
                    }
                )
                
        except Exception as e:
            st.error(f"Error performing correlation analysis: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())