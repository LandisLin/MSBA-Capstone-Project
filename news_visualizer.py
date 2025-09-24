"""
News Data Visualizer - Complete Dynamic News Analysis for Economic Dashboard
Handles news data loading, processing, and visualization with automatic sheet detection
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import warnings
import glob
import os

warnings.filterwarnings('ignore')

class NewsDataVisualizer:
    """Handle news analysis data visualization with dynamic detection"""
    
    def __init__(self, news_folder="./news_analysis_output"):
        """
        Initialize News Data Visualizer with automatic file detection
        
        Args:
            news_folder (str): Folder containing news analysis files
        """
        self.news_folder = news_folder
        self.news_file_path = self._find_latest_news_file()
        self.news_data = {}
        
        if self.news_file_path:
            print(f"📁 Using news file: {self.news_file_path}")
            self.load_news_data()
        else:
            print(f"❌ No news analysis files found in {news_folder}")
    
    def _find_latest_news_file(self):
        """Find the master news analysis file"""
        try:
            # Look for master file only
            master_file = os.path.join(self.news_folder, "master_news_analysis.xlsx")
            
            if os.path.exists(master_file):
                print(f"✅ Found master file: {master_file}")
                return master_file
            else:
                print(f"❌ Master news file not found in folder: {self.news_folder}")
                return None
                
        except Exception as e:
            print(f"❌ Error accessing folder {self.news_folder}: {e}")
            return None

    def debug_excel_structure(self):
        """Debug method to inspect Excel file structure"""
        if not self.news_file_path:
            print("❌ No file path available")
            return
        
        try:
            print(f"\n🔍 DEBUGGING EXCEL STRUCTURE: {self.news_file_path}")
            excel_file = pd.ExcelFile(self.news_file_path)
            print(f"📋 All sheets: {excel_file.sheet_names}")
            
            for sheet_name in excel_file.sheet_names:
                if any(skip_word in sheet_name.lower() for skip_word in ['contents', 'summary', 'index', 'overview']):
                    continue
                    
                print(f"\n--- SHEET: {sheet_name} ---")
                
                # Read first few rows without skipping
                df_raw = pd.read_excel(self.news_file_path, sheet_name=sheet_name, nrows=5)
                print(f"Raw data (first 5 rows):")
                print(df_raw)
                
                # Read with skiprows=2
                df_skip = pd.read_excel(self.news_file_path, sheet_name=sheet_name, skiprows=2, nrows=5)
                print(f"\nWith skiprows=2 (first 5 rows):")
                print(df_skip)
                
                break  # Just check first non-content sheet
                
        except Exception as e:
            print(f"❌ Error debugging Excel structure: {e}")
            import traceback
            traceback.print_exc()

    def load_news_data(self):
        """Dynamically load news analysis data from Excel file"""
        if not self.news_file_path:
            print("❌ No news file path available")
            return False
        
        # DEBUGGING: Add structure inspection
        self.debug_excel_structure()
        
        try:
            # First, get all sheet names
            excel_file = pd.ExcelFile(self.news_file_path)
            all_sheets = excel_file.sheet_names
            
            print(f"📋 Detected sheets: {all_sheets}")
            
            self.news_data = {}
            
            for sheet_name in all_sheets:
                try:
                    # Skip summary/content sheets that don't contain article data
                    if any(skip_word in sheet_name.lower() for skip_word in ['contents', 'summary', 'index', 'overview']):
                        print(f"⏭️ Skipping summary sheet: {sheet_name}")
                        continue
                    
                    # FIXED: Read from row 3 (skiprows=2) to skip title and empty row
                    df = pd.read_excel(self.news_file_path, sheet_name=sheet_name, skiprows=2)
                    
                    # Auto-detect column structure
                    processed_df = self._auto_detect_and_clean_columns(df, sheet_name)
                    
                    if processed_df is not None and len(processed_df) > 0:
                        self.news_data[sheet_name] = processed_df
                        print(f"✅ Successfully loaded {len(processed_df)} articles from: {sheet_name}")
                    else:
                        print(f"⚠️ No valid data found in: {sheet_name}")
                        
                except Exception as e:
                    print(f"❌ Error loading sheet '{sheet_name}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n📊 FINAL SUMMARY:")
            print(f"Loaded {len(self.news_data)} sheets with data")
            for sheet_name, df in self.news_data.items():
                print(f"  - {sheet_name}: {len(df)} articles")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading news analysis file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _auto_detect_and_clean_columns(self, df, sheet_name):
        """Automatically detect and standardize column structure"""
        try:
            print(f"\n🔍 Analyzing sheet: {sheet_name}")
            print(f"Original columns: {list(df.columns)}")
            print(f"Data shape: {df.shape}")
            print(f"First few rows of raw data:")
            print(df.head(2))
            
            if df.empty:
                print(f"❌ DataFrame is empty for {sheet_name}")
                return None
            
            # Since we're reading from row 3 (skiprows=2), we should have the headers
            # Let's check if the first row contains the actual headers
            if len(df) > 0:
                first_row = df.iloc[0]
                print(f"🔍 First row values: {first_row.tolist()}")
                
                # Check if first row looks like headers
                if 'Index' in str(first_row.iloc[0]) or 'Title' in str(first_row.iloc[1]):
                    print("🔄 First row appears to be headers, using them")
                    df.columns = df.iloc[0].astype(str)
                    df = df.iloc[1:].reset_index(drop=True)
                    print(f"🔄 New columns after using first row: {list(df.columns)}")
                else:
                    # Use standard column names based on the expected structure
                    expected_columns = ['Index', 'Title', 'Description', 'Region', 'Published', 
                                        'URL', 'Summary', 'Impact Analysis', 'Topic Analysis', 
                                        'Sentiment Score', 'Sentiment Label']
                    
                    # Only assign as many columns as we actually have
                    new_columns = expected_columns[:min(len(expected_columns), len(df.columns))]
                    df.columns = new_columns
                    print(f"🔄 Applied standard column names: {list(df.columns)}")
            
            # # ONLY ADD THIS: Apply standardization to handle underscore vs space differences
            # df = self._standardize_column_names(df)
            # print(f"🔄 Standardized columns: {list(df.columns)}")

            # Clean the data
            # Remove rows where Title is NaN or empty
            original_len = len(df)
            if 'Title' in df.columns:
                df = df.dropna(subset=['Title']).reset_index(drop=True)
                df = df[df['Title'].astype(str).str.strip() != ''].reset_index(drop=True)
                df = df[df['Title'].astype(str) != 'nan'].reset_index(drop=True)
                df = df[~df['Title'].astype(str).str.isdigit()].reset_index(drop=True)  # Remove numeric-only titles
                print(f"🧹 Cleaned data: {original_len} → {len(df)} rows (removed empty titles)")
            
            # Final validation
            if len(df) == 0:
                print(f"⚠️ No valid articles found in {sheet_name} after cleaning")
                return None
            
            # Check if we have the essential columns
            if 'Title' not in df.columns:
                print(f"❌ No 'Title' column found in {sheet_name}")
                return None
            
            print(f"✅ Final data shape: {df.shape}")
            print(f"✅ Final columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing sheet {sheet_name}: {e}")
            return None

    def _standardize_column_names(self, df):
        """Standardize column names across different formats"""
        
        # Column mapping dictionary for different naming patterns
        column_mapping = {
            # Handle various title patterns
            'title': 'Title',
            'article_title': 'Title',
            'headline': 'Title',
            
            # Handle description patterns  
            'description': 'Description',
            'summary': 'Summary',
            'content': 'Description',
            
            # Handle region patterns
            'region': 'Region',
            'country': 'Region',
            'location': 'Region',
            
            # Handle date patterns
            'date': 'Published',
            'published_date': 'Published',
            'publish_date': 'Published',
            'published': 'Published',
            
            # Handle sentiment patterns
            'sentiment_score': 'Sentiment Score',
            'sentiment': 'Sentiment Score',
            'score': 'Sentiment Score',
            'sentiment_label': 'Sentiment Label',
            'label': 'Sentiment Label',
            
            # Handle other patterns
            'link': 'URL',
            'url': 'URL',
            'source': 'Source',
            'impact_analysis': 'Impact Analysis',
            'impact': 'Impact Analysis',
            'topic_analysis': 'Topic Analysis',
            'topics': 'Topic Analysis',
            'index': 'Index'
        }
        
        # Create new column names
        new_columns = []
        for col in df.columns:
            col_lower = str(col).lower().strip()
            
            # Direct mapping
            if col_lower in column_mapping:
                new_columns.append(column_mapping[col_lower])
            # Partial matching for complex names
            elif any(key in col_lower for key in column_mapping.keys()):
                matched_key = next(key for key in column_mapping.keys() if key in col_lower)
                new_columns.append(column_mapping[matched_key])
            else:
                # Keep original name but clean it
                new_columns.append(str(col).replace('__EMPTY', 'Column').replace('_', ' ').title())
        
        df.columns = new_columns
        return df
    
    def get_news_source_display_name(self, sheet_name):
        """Convert sheet name to user-friendly display name"""
        # Handle common patterns
        name_mapping = {
            'bbc': 'BBC News',
            'yahoo finance - economic': 'Yahoo Economic News',
            'yahoo finance - stock market': 'Yahoo Market News',
            'yahoo economic': 'Yahoo Economic News',
            'yahoo market': 'Yahoo Market News',
            'business times singapore': 'Business Times Singapore',
            'reuters': 'Reuters',
            'bloomberg': 'Bloomberg',
            'cnbc': 'CNBC'
        }
        
        sheet_lower = sheet_name.lower()
        
        # Check for exact matches first
        for key, display_name in name_mapping.items():
            if key in sheet_lower:
                return display_name
        
        # If no match, clean up the original name
        return sheet_name.replace('_', ' ').replace('-', ' ').title()
    
    def get_summary_metrics(self):
        """Get summary metrics for the dashboard"""
        if not self.news_data:
            return None
        
        total_articles = sum(len(df) for df in self.news_data.values())
        sources_count = len(self.news_data)
        
        # Get latest date from Contents sheet or file modification time - FIXED
        latest_update = self._get_latest_extraction_date()
        
        return {
            'total_articles': total_articles,
            'sources_count': sources_count,
            'latest_update': latest_update
        }
    
    def create_country_distribution_chart(self):
        """Create country/region distribution chart showing article counts by region"""
        try:
            regional_data = []
            
            for source, df in self.news_data.items():
                if 'Region' in df.columns:
                    # Expand regions to count each individual region
                    df_expanded = self.expand_regions_for_analysis(df)
                    
                    # Count articles by region
                    region_counts = df_expanded['Region'].value_counts()
                    
                    for region, count in region_counts.items():
                        if region and region.lower() not in ['other', 'unknown', 'nan']:
                            regional_data.append({
                                'Region': region,
                                'Count': count
                            })
            
            if regional_data:
                # Aggregate across all sources
                region_df = pd.DataFrame(regional_data)
                total_counts = region_df.groupby('Region')['Count'].sum().sort_values(ascending=False)
                
                fig = px.bar(
                    x=total_counts.index, 
                    y=total_counts.values,
                    title="🌍 Article Count by Region/Country",
                    labels={'x': 'Region', 'y': 'Number of Articles'},
                    color=total_counts.values,
                    color_continuous_scale='Teal'
                )
                
                fig.update_layout(
                    height=400,
                    xaxis_tickangle=-45,
                    showlegend=False
                )
                
                return fig
            
        except Exception as e:
            print(f"❌ Error creating country distribution chart: {e}")
            return None

    def create_source_distribution_chart(self):
        """Create news source distribution chart showing article counts by source"""
        try:
            source_data = []
            
            for source_name, df in self.news_data.items():
                display_name = self.get_news_source_display_name(source_name)
                article_count = len(df)
                
                source_data.append({
                    'Source': display_name,
                    'Count': article_count
                })
            
            if source_data:
                source_df = pd.DataFrame(source_data)
                source_df = source_df.sort_values('Count', ascending=False)
                
                fig = px.bar(
                    source_df, 
                    x='Source', 
                    y='Count',
                    title="📺 Article Count by News Source",
                    labels={'Source': 'News Source', 'Count': 'Number of Articles'},
                    color='Count',
                    color_continuous_scale='Sunset'
                )
                
                fig.update_layout(
                    height=400,
                    xaxis_tickangle=-45,
                    showlegend=False
                )
                
                return fig
            
        except Exception as e:
            print(f"❌ Error creating source distribution chart: {e}")
            return None

    def _get_latest_extraction_date(self):
        """Get the latest extraction date from Contents sheet or file modification"""
        try:
            # Try to read Contents sheet for Last Extraction Time
            if self.news_file_path:
                try:
                    contents_df = pd.read_excel(self.news_file_path, sheet_name='Contents', skiprows=2)
                    if 'Last Extraction Time' in contents_df.columns:
                        extraction_times = contents_df['Last Extraction Time'].dropna()
                        if len(extraction_times) > 0:
                            latest_time = pd.to_datetime(extraction_times.iloc[0])
                            return latest_time.strftime('%Y-%m-%d')  # Date only, no time
                except:
                    pass
                
                # Fallback to file modification time
                try:
                    import os
                    mod_time = os.path.getmtime(self.news_file_path)
                    return pd.to_datetime(mod_time, unit='s').strftime('%Y-%m-%d')
                except:
                    pass
        except:
            pass
        
        return "Unknown"
    
    def expand_regions_for_analysis(self, df):
        """Expand comma-separated regions into individual rows for proper grouping"""
        expanded_rows = []
        
        for _, row in df.iterrows():
            region_str = str(row.get('Region', '')).strip()
            if region_str and region_str != 'nan':
                # Split comma-separated regions
                individual_regions = [r.strip() for r in region_str.split(',')]
                
                # Create a copy of the row for each individual region
                for region in individual_regions:
                    if region:  # Skip empty regions
                        row_copy = row.copy()
                        row_copy['Region'] = region
                        expanded_rows.append(row_copy)
            else:
                # Keep rows without region data as-is
                expanded_rows.append(row)
        
        return pd.DataFrame(expanded_rows)

    def create_sentiment_chart(self):
        """Create sentiment analysis chart with separate distribution plots for each source"""
        all_data = []
        for source, df in self.news_data.items():
            display_name = self.get_news_source_display_name(source)
            
            # Check if Sentiment Label column exists and has data
            if 'Sentiment Label' in df.columns:
                try:
                    # Clean sentiment data first - FIXED: Better error handling
                    sentiment_clean = df['Sentiment Label'].dropna().astype(str)
                    if len(sentiment_clean) > 0:
                        sentiment_counts = sentiment_clean.value_counts()
                        for sentiment, count in sentiment_counts.items():
                            all_data.append({
                                'Source': display_name, 
                                'Sentiment': str(sentiment), 
                                'Count': int(count),  # FIXED: Ensure count is int
                                'Percentage': round((count / len(sentiment_clean)) * 100, 1)  # Add percentage
                            })
                        print(f"📊 {display_name}: {dict(sentiment_counts)}")
                    else:
                        print(f"⚠️ {display_name}: No sentiment data available")
                except Exception as e:
                    print(f"❌ Error processing sentiment for {display_name}: {e}")
                    continue
            else:
                print(f"⚠️ {display_name}: No 'Sentiment Label' column found")
        
        if all_data:
            try:
                sentiment_df = pd.DataFrame(all_data)
                print(f"📊 Total sentiment data points: {len(sentiment_df)}")
                
                # IMPROVED: Create subplots for each source to show distribution better
                sources = sentiment_df['Source'].unique()
                fig = make_subplots(
                    rows=1, cols=len(sources),
                    subplot_titles=sources,
                    specs=[[{"secondary_y": False} for _ in sources]]
                )
                
                colors = {'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#ffc107', 
                         'Very Positive': '#155724', 'Very Negative': '#721c24',
                         'Slightly Positive': '#6c757d', 'Slightly Negative': '#495057'}
                
                for i, source in enumerate(sources, 1):
                    source_data = sentiment_df[sentiment_df['Source'] == source]
                    
                    fig.add_trace(
                        go.Bar(
                            x=source_data['Sentiment'],
                            y=source_data['Percentage'],
                            name=source,
                            marker_color=[colors.get(s, '#6c757d') for s in source_data['Sentiment']],
                            text=[f"{row['Count']} ({row['Percentage']}%)" for _, row in source_data.iterrows()],
                            textposition='auto',
                            showlegend=False
                        ),
                        row=1, col=i
                    )
                
                fig.update_layout(
                    title="📊 News Sentiment Distribution by Source",
                    height=400,
                    showlegend=True
                )
                fig.update_yaxes(title_text="Percentage (%)")
                
                return fig
            except Exception as e:
                print(f"❌ Error creating sentiment chart: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("❌ No sentiment data available for chart")
            return None

    # def create_regional_sentiment_chart(self):
    #     """Create regional sentiment analysis chart with individual countries"""
    #     regional_data = []
        
    #     for source, df in self.news_data.items():
    #         if 'Region' in df.columns and 'Sentiment Score' in df.columns:
    #             try:
    #                 # Expand regions before processing
    #                 df_expanded = self.expand_regions_for_analysis(df)
                    
    #                 # Clean and process data
    #                 df_clean = df_expanded.copy()
    #                 df_clean['Sentiment Score'] = pd.to_numeric(df_clean['Sentiment Score'], errors='coerce')
    #                 df_clean = df_clean.dropna(subset=['Region', 'Sentiment Score'])
                    
    #                 if len(df_clean) > 0:
    #                     regional_sentiment = df_clean.groupby('Region')['Sentiment Score'].agg(['mean', 'count']).reset_index()
    #                     regional_sentiment['Source'] = self.get_news_source_display_name(source)
                        
    #                     for _, row in regional_sentiment.iterrows():
    #                         regional_data.append({
    #                             'Region': row['Region'],
    #                             'Avg_Sentiment': row['mean'],
    #                             'Article_Count': row['count'],
    #                             'Source': row['Source']
    #                         })
    #             except Exception as e:
    #                 print(f"❌ Error processing regional sentiment for {source}: {e}")
    #                 continue
        
    #     if regional_data:
    #         try:
    #             regional_df = pd.DataFrame(regional_data)
                
    #             # Aggregate across sources for overall regional sentiment
    #             overall_regional = regional_df.groupby('Region').agg({
    #                 'Avg_Sentiment': 'mean',
    #                 'Article_Count': 'sum'
    #             }).reset_index()
                
    #             fig = px.bar(overall_regional, x='Region', y='Avg_Sentiment',
    #                         title="🌍 Average Sentiment Score by Region",
    #                         color='Avg_Sentiment',
    #                         color_continuous_scale=['red', 'yellow', 'green'],
    #                         color_continuous_midpoint=0)
                
    #             # Add article count as hover info
    #             fig.update_traces(
    #                 hovertemplate="<b>%{x}</b><br>Avg Sentiment: %{y:.2f}<br>Articles: %{customdata}<extra></extra>",
    #                 customdata=overall_regional['Article_Count']
    #             )
                
    #             fig.update_layout(height=400)
    #             fig.update_yaxes(title="Average Sentiment Score", range=[-1, 1])
    #             fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                
    #             return fig
    #         except Exception as e:
    #             print(f"❌ Error creating regional sentiment chart: {e}")
    #             return None
    #     else:
    #         print("❌ No regional sentiment data available")
    #         return None

    def create_regional_sentiment_chart(self):
        """FIXED: Calculate regional sentiment correctly with all improvements"""
        regional_sentiment_data = {}  # {region: [sentiment_scores]}
        
        print(f"\n📊 FIXED REGIONAL SENTIMENT PROCESSING:")
        
        for source, df in self.news_data.items():
            display_name = self.get_news_source_display_name(source)
            print(f"\n🔍 Processing {display_name}:")
            print(f"   Columns: {list(df.columns)}")
            
            if 'Region' in df.columns and 'Sentiment Score' in df.columns:
                try:
                    # Clean data
                    df_clean = df.copy()
                    df_clean['Sentiment Score'] = pd.to_numeric(df_clean['Sentiment Score'], errors='coerce')
                    df_clean = df_clean.dropna(subset=['Region', 'Sentiment Score'])
                    
                    print(f"   Articles with valid sentiment: {len(df_clean)}")
                    
                    # FIXED: Process each article directly - NO expansion DataFrame
                    for idx, row in df_clean.iterrows():
                        region_str = str(row['Region']).strip()
                        sentiment_score = row['Sentiment Score']
                        
                        if region_str and region_str != 'nan':
                            # Split comma-separated regions
                            individual_regions = [r.strip() for r in region_str.split(',')]
                            
                            # Add this article's sentiment to each region it mentions
                            for region in individual_regions:
                                if region:
                                    if region not in regional_sentiment_data:
                                        regional_sentiment_data[region] = []
                                    regional_sentiment_data[region].append(sentiment_score)
                    
                except Exception as e:
                    print(f"❌ Error processing {source}: {e}")
                    continue
            else:
                print(f"   ⚠️ Missing required columns")
        
        if regional_sentiment_data:
            try:
                regional_results = []
                
                print(f"\n📊 FINAL REGIONAL CALCULATIONS:")
                for region, sentiment_scores in regional_sentiment_data.items():
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    article_count = len(sentiment_scores)
                    
                    regional_results.append({
                        'Region': region,
                        'Avg_Sentiment': avg_sentiment,
                        'Article_Count': article_count
                    })
                
                    print(f"🌍 {region}: {article_count} articles, avg: {avg_sentiment:.2f}")
                    
                    # Show detailed scores for Vietnam and Indonesia
                    if region.lower() in ['vietnam', 'indonesia']:
                        print(f"   📊 Individual scores: {sentiment_scores}")
                
                regional_df = pd.DataFrame(regional_results)

                # Fix: Order regions according to macro sources
                macro_region_order = [
                    'Singapore', 'United States', 'Euro Area', 'United Kingdom', 
                    'China', 'Japan', 'India', 'Indonesia', 'Malaysia', 'Thailand', 'Vietnam'
                ]
                
                def get_region_order(region_name):
                    try:
                        return macro_region_order.index(region_name)
                    except ValueError:
                        return 999  # Put unknown regions at the end
                
                regional_df['Order'] = regional_df['Region'].apply(get_region_order)
                regional_df = regional_df.sort_values('Order').drop('Order', axis=1).reset_index(drop=True)
                
                print(f"📊 Regions ordered: {regional_df['Region'].tolist()}")
                
                # Create the bar chart
                fig = px.bar(regional_df, x='Region', y='Avg_Sentiment',
                            title="🌍 Average Sentiment Score by Region",
                            color='Avg_Sentiment',
                            color_continuous_scale=['red', 'yellow', 'green'],
                            color_continuous_midpoint=0)
                
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>Avg Sentiment: %{y:.2f}<br>Articles: %{customdata}<extra></extra>",
                    customdata=regional_df['Article_Count']
                )
                
                # FIXED: Dynamic zoom for better visibility
                min_sentiment = min(regional_df['Avg_Sentiment'])
                max_sentiment = max(regional_df['Avg_Sentiment'])
                
                # Add padding for better visibility (20% on each side)
                range_padding = max(abs(min_sentiment), abs(max_sentiment)) * 0.2
                y_min = max(-1, min_sentiment - range_padding)  # Don't go below -1
                y_max = min(1, max_sentiment + range_padding)   # Don't go above 1
                
                # Ensure minimum visible range for small values
                if (y_max - y_min) < 0.4:  # If range is too small, expand it
                    center = (y_max + y_min) / 2
                    y_min = max(-1, center - 0.2)
                    y_max = min(1, center + 0.2)
                
                print(f"📊 Y-axis range for better visibility: {y_min:.2f} to {y_max:.2f}")
                
                # Update layout
                fig.update_layout(height=400)
                fig.update_yaxes(title="Average Sentiment Score", range=[y_min, y_max])
                fig.update_xaxes(tickangle=-45)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                
                return fig
                
            except Exception as e:
                print(f"❌ Error creating chart: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("❌ No regional sentiment data available")
            return None

    def create_sentiment_timeline(self, selected_region=None, selected_topic=None, selected_indicator=None, show_topic_breakdown=False, max_topics=5):
        """
        Create sentiment timeline with SIMPLE and CORRECT logic:
        1. Filter articles based on selected criteria
        2. Group by date (and topic if breakdown is enabled)
        3. Calculate average sentiment per date
        4. Plot timeline
        
        Topic breakdown is ONLY available when:
        - Region is selected (not "All Regions")
        - Topic is "All Topics" 
        - Indicator is "All Indicators"
        """
        
        print(f"\n🔍 TIMELINE DEBUG - Input parameters:")
        print(f"   selected_region: {selected_region}")
        print(f"   selected_topic: {selected_topic}")  
        print(f"   selected_indicator: {selected_indicator}")
        print(f"   show_topic_breakdown: {show_topic_breakdown}")
        
        # Check if topic breakdown is valid
        topic_breakdown_available = (
            selected_region and selected_region != "All Regions" and
            (not selected_topic or selected_topic == "All Topics") and
            (not selected_indicator or selected_indicator == "All Indicators")
        )
        
        # Override show_topic_breakdown if conditions aren't met
        if show_topic_breakdown and not topic_breakdown_available:
            print("⚠️ Topic breakdown not available with current filter combination")
            show_topic_breakdown = False
        
        print(f"🔍 Topic breakdown: Available={topic_breakdown_available}, Using={show_topic_breakdown}")
        
        all_filtered_articles = []
        
        # Step 1: Filter articles from all sources based on criteria
        for source, df in self.news_data.items():
            if 'Published' in df.columns and 'Sentiment Score' in df.columns:
                display_name = self.get_news_source_display_name(source)
                
                try:
                    # Clean the data first
                    df_clean = df.copy()
                    df_clean['Published'] = pd.to_datetime(df_clean['Published'], errors='coerce')
                    df_clean['Sentiment Score'] = pd.to_numeric(df_clean['Sentiment Score'], errors='coerce')
                    df_clean = df_clean.dropna(subset=['Published', 'Sentiment Score'])
                    
                    print(f"\n🔍 Processing {display_name}: {len(df_clean)} articles")
                    
                    # Apply filters - NO expansion, just check if article matches criteria
                    for _, row in df_clean.iterrows():
                        article_matches = True
                        
                        # Region filter: Check if selected region is mentioned in article's regions
                        if selected_region and selected_region != "All Regions":
                            article_regions = str(row.get('Region', '')).strip()
                            if article_regions and article_regions != 'nan':
                                individual_regions = [r.strip() for r in article_regions.split(',')]
                                region_match = any(selected_region.lower() == region.lower() for region in individual_regions if region)
                            else:
                                region_match = False
                            article_matches = article_matches and region_match
                            if not region_match:
                                continue  # Skip this article early if region doesn't match
                        
                        # Topic filter: Check if selected topic is mentioned in article's topic analysis
                        if selected_topic and selected_topic != "All Topics":
                            topic_text = str(row.get('Topic Analysis', '')).strip()
                            topic_match = selected_topic.lower() in topic_text.lower() if topic_text and topic_text != 'nan' else False
                            article_matches = article_matches and topic_match
                            if not topic_match:
                                continue  # Skip this article early if topic doesn't match
                        
                        # Indicator filter: Check if selected indicator is mentioned in article's indicators
                        if selected_indicator and selected_indicator != "All Indicators":
                            topic_text = str(row.get('Topic Analysis', '')).strip()
                            if 'Indicators:' in topic_text:
                                indicators_part = topic_text.split('Indicators:')[1].strip()
                                indicator_match = selected_indicator.lower() in indicators_part.lower()
                            else:
                                indicator_match = False
                            article_matches = article_matches and indicator_match
                            if not indicator_match:
                                continue  # Skip this article early if indicator doesn't match
                        
                        # If article matches all filters, add it to our collection
                        if article_matches:
                            article_data = {
                                'Published': row['Published'],
                                'Sentiment Score': row['Sentiment Score'],
                                'Source': display_name,
                                'Title': str(row.get('Title', '')),
                                'Region': str(row.get('Region', '')),
                            }
                            
                            # If topic breakdown is enabled, extract topics for this article
                            if show_topic_breakdown:
                                article_data['Topic Analysis'] = str(row.get('Topic Analysis', ''))
                            
                            all_filtered_articles.append(article_data)
                    
                    articles_from_this_source = len([a for a in all_filtered_articles if a['Source'] == display_name])
                    print(f"✅ {display_name}: Found {articles_from_this_source} matching articles")
                    
                    # DEBUG: Show a sample of the data structure
                    if len(df_clean) > 0:
                        sample_row = df_clean.iloc[0]
                        print(f"   📋 Sample data structure:")
                        print(f"      Columns: {list(df_clean.columns)}")
                        print(f"      Sample regions: {sample_row.get('Region', 'N/A')}")
                        print(f"      Sample date: {sample_row.get('Published', 'N/A')}")
                        print(f"      Sample sentiment: {sample_row.get('Sentiment Score', 'N/A')}")
                    
                except Exception as e:
                    print(f"❌ Error processing {source}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                print(f"⚠️ Skipping {source}: Missing required columns (Published: {'Published' in df.columns}, Sentiment Score: {'Sentiment Score' in df.columns})")
        
        print(f"\n📊 TOTAL filtered articles from all sources: {len(all_filtered_articles)}")
        
        if not all_filtered_articles:
            print("❌ No articles match the selected criteria")
            print("🔍 Debugging info:")
            print(f"   - selected_region: '{selected_region}' (is 'All Regions'? {selected_region == 'All Regions'})")
            print(f"   - selected_topic: '{selected_topic}' (is 'All Topics'? {selected_topic == 'All Topics'})")
            print(f"   - selected_indicator: '{selected_indicator}' (is 'All Indicators'? {selected_indicator == 'All Indicators'})")
            return None
        
        # Step 2: Process based on whether topic breakdown is enabled
        if show_topic_breakdown:
            print(f"📊 Processing topic breakdown for {len(all_filtered_articles)} articles")
            
            # Extract topics from each article and create separate entries
            topic_articles = []
            
            for article in all_filtered_articles:
                topics = self._extract_topics_from_analysis(article.get('Topic Analysis', ''))
                if not topics:
                    topics = ["General Economic"]
                
                # Create separate entry for each topic (NO weight division - your logic)
                for topic in topics:
                    topic_articles.append({
                        'Published': article['Published'],
                        'Sentiment Score': article['Sentiment Score'],  # Keep original score
                        'Topic': topic,
                        'Source': article['Source'],
                        'Title': article['Title']
                    })
            
            # Convert to DataFrame and get ALL topics (or limit if needed)
            df_topics = pd.DataFrame(topic_articles)
            topic_counts = df_topics['Topic'].value_counts()
            
            # Show ALL topics instead of limiting to top N
            all_topics = topic_counts.index.tolist()
            
            print(f"🏷️ All topics found ({len(all_topics)}): {all_topics}")
            print(f"📊 Topic distribution: {dict(topic_counts)}")
            
            # Use all topics instead of filtering to top N
            df_filtered_topics = df_topics  # Use all topic data
            daily_sentiment = df_filtered_topics.groupby([df_filtered_topics['Published'].dt.date, 'Topic'])['Sentiment Score'].agg(['mean', 'count']).reset_index()
            daily_sentiment.columns = ['Date', 'Topic', 'Avg_Sentiment', 'Article_Count']
            
            # Create the chart with multiple lines (one per topic)
            chart_title = f"📈 Sentiment Timeline for {selected_region} by Topic"
            
            fig = px.line(
                daily_sentiment, 
                x='Date', 
                y='Avg_Sentiment',
                color='Topic',
                markers=True, 
                title=chart_title
            )
            
            # Customize traces
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=6),
                hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Avg Sentiment: %{y:.3f}<extra></extra>"
            )
            
        else:
            # Standard single-line timeline (your original logic)
            df_all = pd.DataFrame(all_filtered_articles)
            print(f"\n📊 Total filtered articles: {len(df_all)}")
            
            # Group by date and calculate average sentiment per date
            daily_sentiment = df_all.groupby(df_all['Published'].dt.date)['Sentiment Score'].agg(['mean', 'count']).reset_index()
            daily_sentiment.columns = ['Date', 'Avg_Sentiment', 'Article_Count']
            
            print(f"📅 Date range: {daily_sentiment['Date'].min()} to {daily_sentiment['Date'].max()}")
            print(f"📊 Daily sentiment summary:")
            for _, row in daily_sentiment.iterrows():
                print(f"   {row['Date']}: {row['Avg_Sentiment']:.3f} (from {row['Article_Count']} articles)")
            
            # Create dynamic title based on selected filters
            title_parts = ["📈 Sentiment Timeline"]
            filter_parts = []
            
            if selected_region and selected_region != "All Regions":
                filter_parts.append(f"Region: {selected_region}")
            if selected_topic and selected_topic != "All Topics":
                filter_parts.append(f"Topic: {selected_topic}")
            if selected_indicator and selected_indicator != "All Indicators":
                filter_parts.append(f"Indicator: {selected_indicator}")
            
            if filter_parts:
                title_parts.append(f"({', '.join(filter_parts)})")
            else:
                title_parts.append("(All Articles)")
            
            chart_title = " ".join(title_parts)
            
            # Create single-line chart
            fig = px.line(
                daily_sentiment, 
                x='Date', 
                y='Avg_Sentiment',
                markers=True, 
                title=chart_title,
                hover_data={'Article_Count': True}
            )
            
            # Customize the chart
            fig.update_traces(
                line=dict(width=3, color='#1f77b4'),
                marker=dict(size=8),
                hovertemplate="<b>%{x}</b><br>Avg Sentiment: %{y:.3f}<br>Articles: %{customdata[0]}<extra></extra>"
            )
        
        # Step 3: Final chart customization (common for both cases)
        try:
            if len(daily_sentiment) == 0:
                print("❌ No valid daily sentiment data")
                return None
            
            # Add neutral line
            fig.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.7,
                annotation_text="Neutral", 
                annotation_position="right"
            )
            
            # Update layout
            fig.update_layout(
                height=500,
                xaxis_title="Date",
                yaxis_title="Average Sentiment Score",
                yaxis=dict(range=[-1, 1]),
                showlegend=show_topic_breakdown
            )
            
            # Position legend if showing topic breakdown
            if show_topic_breakdown:
                fig.update_layout(
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
                )
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating timeline chart: {e}")
            return None

    def _extract_topics_from_analysis(self, topic_analysis_text):
        """Helper method to extract topics from analysis text"""
        try:
            topic_text = str(topic_analysis_text)
            if 'Topics:' in topic_text:
                topics_part = topic_text.split('Topics:')[1].split('Indicators:')[0] if 'Indicators:' in topic_text else topic_text.split('Topics:')[1]
                topics_part = topics_part.replace('*', '').strip()
                topics = [topic.strip().replace('**', '').replace('*', '') for topic in topics_part.split(',')]
                return [t for t in topics if t and len(t) > 2]
            return []
        except:
            return []
    
    def create_topic_analysis_chart(self):
        """Create topic analysis visualization with cleaned formatting"""
        topic_data = []
        
        for source, df in self.news_data.items():
            if 'Topic Analysis' in df.columns:
                for _, row in df.iterrows():
                    topics_text = str(row['Topic Analysis'])
                    if 'Topics:' in topics_text:
                        try:
                            # Extract topics after "Topics:" and clean formatting
                            topics_part = topics_text.split('Topics:')[1].split('Indicators:')[0] if 'Indicators:' in topics_text else topics_text.split('Topics:')[1]
                            
                            # Remove asterisks and clean the text
                            topics_part = topics_part.replace('*', '').strip()
                            
                            topics = [topic.strip() for topic in topics_part.split(',')]
                            
                            for topic in topics:
                                if topic and len(topic) > 2:  # Filter out very short topics
                                    # Additional cleaning to remove any remaining formatting
                                    clean_topic = topic.strip().replace('**', '').replace('*', '')
                                    if clean_topic:
                                        topic_data.append({
                                            'Topic': clean_topic,
                                            'Source': self.get_news_source_display_name(source)
                                        })
                        except Exception as e:
                            print(f"❌ Error parsing topics from: {topics_text[:100]}... Error: {e}")
                            continue
        
        if topic_data:
            try:
                topic_df = pd.DataFrame(topic_data)
                topic_counts = topic_df.groupby('Topic').size().sort_values(ascending=False)
                
                # Take top 15 topics for readability
                top_topics = topic_counts.head(15)
                
                fig = px.bar(x=top_topics.values, y=top_topics.index, orientation='h',
                            title="📊 Most Frequent Economic Topics in News",
                            labels={'x': 'Number of Articles', 'y': 'Topics'})
                
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                return fig
            except Exception as e:
                print(f"❌ Error creating topic chart: {e}")
                return None
        else:
            print("❌ No topic data available for chart")
            return None
    
    def get_detailed_articles(self, source_name, limit=5):
        """Get detailed articles for a specific source - FIXED: Get latest articles (from end of sheet)"""
        if source_name not in self.news_data:
            return []
        
        df = self.news_data[source_name]
        articles = []
        
        # FIXED: Get the last N articles (most recent) instead of first N
        for _, row in df.tail(limit).iterrows():  # Changed from head() to tail()
            try:
                article = {
                    'title': str(row.get('Title', 'No title')) if pd.notna(row.get('Title')) else 'No title',
                    'summary': str(row.get('Summary', row.get('Description', 'No summary available'))) if pd.notna(row.get('Summary', row.get('Description'))) else 'No summary available',
                    'impact_analysis': str(row.get('Impact Analysis', 'No impact analysis available')) if pd.notna(row.get('Impact Analysis')) else 'No impact analysis available',
                    'region': str(row.get('Region', 'Unknown')) if pd.notna(row.get('Region')) else 'Unknown',
                    'published_date': str(row.get('Published', 'Unknown')) if pd.notna(row.get('Published')) else 'Unknown',
                    'sentiment_label': str(row.get('Sentiment Label', 'Unknown')) if pd.notna(row.get('Sentiment Label')) else 'Unknown',
                    'sentiment_score': float(row.get('Sentiment Score', 0)) if pd.notna(row.get('Sentiment Score')) and str(row.get('Sentiment Score')).replace('.', '').replace('-', '').isdigit() else 0.0,
                    'link': str(row.get('URL', '')) if pd.notna(row.get('URL')) else None,
                    'source': self.get_news_source_display_name(source_name),
                    'topic_analysis': str(row.get('Topic Analysis', 'No topic analysis available')) if pd.notna(row.get('Topic Analysis')) else 'No topic analysis available',
                }
                articles.append(article)
            except Exception as e:
                print(f"❌ Error processing article: {e}")
                continue
        
        # FIXED: Reverse order so most recent articles appear first
        return list(reversed(articles))  # Most recent first

    def get_all_regions(self):
        """Get list of all individual regions (split comma-separated)"""
        regions = set()
        for df in self.news_data.values():
            if 'Region' in df.columns:
                region_values = df['Region'].dropna().astype(str)
                for region_str in region_values:
                    # Split comma-separated regions
                    individual_regions = [r.strip() for r in region_str.split(',')]
                    for region in individual_regions:
                        if region and region != 'nan':
                            regions.add(region)
        return sorted(list(regions))
    
    def get_all_topics(self):
        """Get list of all available topics from Topic Analysis with cleaned formatting"""
        topics = set()
        for df in self.news_data.values():
            if 'Topic Analysis' in df.columns:
                for _, row in df.iterrows():
                    topic_text = str(row.get('Topic Analysis', ''))
                    if 'Topics:' in topic_text:
                        try:
                            # Extract topics after "Topics:" and clean formatting
                            topics_part = topic_text.split('Topics:')[1].split('Indicators:')[0] if 'Indicators:' in topic_text else topic_text.split('Topics:')[1]
                            
                            # Remove asterisks and clean the text
                            topics_part = topics_part.replace('*', '').strip()
                            
                            topic_list = [topic.strip() for topic in topics_part.split(',')]
                            
                            for topic in topic_list:
                                if topic and len(topic) > 2:  # Filter out very short topics
                                    # Additional cleaning to remove any remaining formatting
                                    clean_topic = topic.strip().replace('**', '').replace('*', '')
                                    if clean_topic:
                                        topics.add(clean_topic)
                        except Exception as e:
                            print(f"❌ Error parsing topics: {e}")
                            continue
        return sorted(list(topics))
    
    def get_all_indicators(self):
        """Get list of all available indicators from Topic Analysis with cleaned formatting"""
        indicators = set()
        for df in self.news_data.values():
            if 'Topic Analysis' in df.columns:
                for _, row in df.iterrows():
                    topic_text = str(row.get('Topic Analysis', ''))
                    if 'Indicators:' in topic_text:
                        try:
                            # Extract indicators after "Indicators:"
                            indicators_part = topic_text.split('Indicators:')[1].strip()
                            
                            # Remove asterisks and clean the text
                            indicators_part = indicators_part.replace('*', '').strip()
                            
                            indicator_list = [indicator.strip() for indicator in indicators_part.split(',')]
                            
                            for indicator in indicator_list:
                                if indicator and len(indicator) > 1:  # Filter out very short indicators
                                    # Additional cleaning to remove any remaining formatting
                                    clean_indicator = indicator.strip().replace('**', '').replace('*', '')
                                    if clean_indicator:
                                        indicators.add(clean_indicator)
                        except Exception as e:
                            print(f"❌ Error parsing indicators: {e}")
                            continue
        return sorted(list(indicators))

    # def get_filtered_sentiment_summary(self, selected_region=None, selected_topic=None, selected_indicator=None):
    #     """Get sentiment summary for filtered articles by region/topic/indicator"""
    #     try:
    #         filtered_articles = []
            
    #         for source_name, df in self.news_data.items():
    #             for idx, row in df.iterrows():
    #                 try:
    #                     # Check region filter
    #                     region_match = True
    #                     if selected_region and selected_region != "All Regions":
    #                         article_region = str(row.get('Region', '')).strip()
    #                         if article_region:
    #                             individual_regions = [r.strip() for r in article_region.split(',')]
    #                             region_match = any(selected_region.lower() == region.lower() for region in individual_regions)
    #                         else:
    #                             region_match = False
                        
    #                     # Check topic filter
    #                     topic_match = True
    #                     if selected_topic and selected_topic != "All Topics":
    #                         topic_text = str(row.get('Topic Analysis', '')).strip()
    #                         topic_match = selected_topic.lower() in topic_text.lower() if topic_text else False
                        
    #                     # Check indicator filter
    #                     indicator_match = True
    #                     if selected_indicator and selected_indicator != "All Indicators":
    #                         topic_text = str(row.get('Topic Analysis', '')).strip()
    #                         # Check if the selected indicator appears in the indicators section
    #                         if 'Indicators:' in topic_text:
    #                             indicators_part = topic_text.split('Indicators:')[1].strip()
    #                             indicator_match = selected_indicator.lower() in indicators_part.lower()
    #                         else:
    #                             indicator_match = False

    #                     # If filters match, collect sentiment data
    #                     if region_match and topic_match and indicator_match:
    #                         sentiment_score = row.get('Sentiment Score')
    #                         if pd.notna(sentiment_score):
    #                             try:
    #                                 score = float(sentiment_score)
    #                                 filtered_articles.append({
    #                                     'sentiment_score': score,
    #                                     'sentiment_label': str(row.get('Sentiment Label', 'Unknown')),
    #                                     'region': str(row.get('Region', 'Unknown')),
    #                                     'source': self.get_news_source_display_name(source_name)
    #                                 })
    #                             except:
    #                                 continue
                                    
    #                 except Exception as e:
    #                     continue
            
    #         if filtered_articles:
    #             import numpy as np
    #             scores = [article['sentiment_score'] for article in filtered_articles]
    #             avg_score = np.mean(scores)
                
    #             # Determine average sentiment label
    #             if avg_score <= -0.7:
    #                 avg_label = "Very Negative"
    #             elif avg_score <= -0.4:
    #                 avg_label = "Negative"
    #             elif avg_score < -0.1:
    #                 avg_label = "Slightly Negative"
    #             elif avg_score <= 0.1:
    #                 avg_label = "Neutral"
    #             elif avg_score < 0.4:
    #                 avg_label = "Slightly Positive"
    #             elif avg_score < 0.7:
    #                 avg_label = "Positive"
    #             else:
    #                 avg_label = "Very Positive"
                
    #             return {
    #                 'article_count': len(filtered_articles),
    #                 'avg_sentiment_score': round(avg_score, 2),
    #                 'avg_sentiment_label': avg_label,
    #                 'score_distribution': {
    #                     'min': round(min(scores), 2),
    #                     'max': round(max(scores), 2),
    #                     'std': round(np.std(scores), 2)
    #                 },
    #                 'source_breakdown': {}
    #             }
            
    #         return None
            
    #     except Exception as e:
    #         print(f"❌ Error getting sentiment summary: {e}")
    #         return None
    def get_filtered_sentiment_summary(self, selected_region=None, selected_topic=None, selected_indicator=None):
        """Get sentiment summary for filtered articles - FIXED: No expansion needed"""
        try:
            filtered_sentiment_scores = []
            
            for source_name, df in self.news_data.items():
                for idx, row in df.iterrows():
                    try:
                        # Check region filter - FIXED: Direct region matching without expansion
                        region_match = True
                        if selected_region and selected_region != "All Regions":
                            article_region = str(row.get('Region', '')).strip()
                            if article_region and article_region != 'nan':
                                # Split regions and check if selected region is mentioned
                                individual_regions = [r.strip() for r in article_region.split(',')]
                                region_match = any(selected_region.lower() == region.lower() for region in individual_regions if region)
                            else:
                                region_match = False
                        
                        # Check topic filter (unchanged)
                        topic_match = True
                        if selected_topic and selected_topic != "All Topics":
                            topic_text = str(row.get('Topic Analysis', '')).strip()
                            topic_match = selected_topic.lower() in topic_text.lower() if topic_text else False
                        
                        # Check indicator filter (unchanged)
                        indicator_match = True
                        if selected_indicator and selected_indicator != "All Indicators":
                            topic_text = str(row.get('Topic Analysis', '')).strip()
                            if 'Indicators:' in topic_text:
                                indicators_part = topic_text.split('Indicators:')[1].strip()
                                indicator_match = selected_indicator.lower() in indicators_part.lower()
                            else:
                                indicator_match = False

                        # If filters match, collect sentiment data (ONE score per article)
                        if region_match and topic_match and indicator_match:
                            sentiment_score = row.get('Sentiment Score')
                            if pd.notna(sentiment_score):
                                try:
                                    score = float(sentiment_score)
                                    filtered_sentiment_scores.append(score)
                                except:
                                    continue
                                    
                    except Exception as e:
                        continue
            
            if filtered_sentiment_scores:
                import numpy as np
                avg_score = np.mean(filtered_sentiment_scores)
                
                # Determine average sentiment label
                if avg_score <= -0.7:
                    avg_label = "Very Negative"
                elif avg_score <= -0.4:
                    avg_label = "Negative"
                elif avg_score < -0.1:
                    avg_label = "Slightly Negative"
                elif avg_score <= 0.1:
                    avg_label = "Neutral"
                elif avg_score < 0.4:
                    avg_label = "Slightly Positive"
                elif avg_score < 0.7:
                    avg_label = "Positive"
                else:
                    avg_label = "Very Positive"
                
                return {
                    'article_count': len(filtered_sentiment_scores),
                    'avg_sentiment_score': round(avg_score, 2),
                    'avg_sentiment_label': avg_label,
                    'score_distribution': {
                        'min': round(min(filtered_sentiment_scores), 2),
                        'max': round(max(filtered_sentiment_scores), 2),
                        'std': round(np.std(filtered_sentiment_scores), 2)
                    }
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting sentiment summary: {e}")
            return None

    def filter_articles_by_region_topic_indicator_and_date(self, selected_region=None, selected_topic=None, selected_indicator=None, start_date=None, end_date=None, limit=None):
        filtered_articles = []

        print(f"🔍 Filtering: Region='{selected_region}', Topic='{selected_topic}', Indicator='{selected_indicator}', Dates='{start_date}' to '{end_date}'")

        for source_name, df in self.news_data.items():
            print(f"📊 Processing {source_name}: {len(df)} articles")
            
            for idx, row in df.iterrows():
                try:
                    # Check region filter
                    region_match = True
                    if selected_region and selected_region != "All Regions":
                        article_region = str(row.get('Region', '')).strip()
                        if article_region:
                            # Split comma-separated regions and check if selected region is in any of them
                            individual_regions = [r.strip() for r in article_region.split(',')]
                            region_match = any(selected_region.lower() == region.lower() for region in individual_regions)
                        else:
                            region_match = False
                    
                    # Check topic filter
                    topic_match = True
                    if selected_topic and selected_topic != "All Topics":
                        topic_text = str(row.get('Topic Analysis', '')).strip()
                        topic_match = selected_topic.lower() in topic_text.lower() if topic_text else False
                    
                    # Check indicator filter
                    indicator_match = True
                    if selected_indicator and selected_indicator != "All Indicators":
                        topic_text = str(row.get('Topic Analysis', '')).strip()
                        if 'Indicators:' in topic_text:
                            indicators_part = topic_text.split('Indicators:')[1].strip()
                            indicator_match = selected_indicator.lower() in indicators_part.lower()
                        else:
                            indicator_match = False
                    
                    # Check date filter (unchanged)
                    date_match = True
                    if start_date or end_date:
                        try:
                            article_date = pd.to_datetime(row.get('Published', ''), errors='coerce')
                            if pd.isna(article_date):
                                date_match = False
                            else:
                                article_date = article_date.date()
                                if start_date and article_date < start_date:
                                    date_match = False
                                if end_date and article_date > end_date:
                                    date_match = False
                        except:
                            date_match = False
                    
                    # If all filters match, add the article
                    if region_match and topic_match and indicator_match and date_match:
                        article = {
                            'title': str(row.get('Title', 'Untitled')),
                            'summary': str(row.get('Summary', 'No summary available')),
                            'published_date': str(row.get('Published', 'Unknown date')),
                            'impact_analysis': str(row.get('Impact Analysis', 'No impact analysis available')) if pd.notna(row.get('Impact Analysis')) else 'No impact analysis available',
                            'region': str(row.get('Region', 'Unknown region')),
                            'sentiment_score': float(row.get('Sentiment Score', 0)) if str(row.get('Sentiment Score', '')).replace('.', '').replace('-', '').isdigit() else 0.0,
                            'sentiment_label': str(row.get('Sentiment Label', 'Neutral')),
                            'link': str(row.get('URL', '')) if pd.notna(row.get('URL')) else None,
                            'topic_analysis': str(row.get('Topic Analysis', 'No topic analysis available')) if pd.notna(row.get('Topic Analysis')) else 'No topic analysis available',
                            'source': self.get_news_source_display_name(source_name),
                        }
                        filtered_articles.append(article)
                        
                        if limit and len(filtered_articles) >= limit:
                            break
                except Exception as e:
                    print(f"❌ Error processing article for filtering: {e}")
                    continue
            
            if limit and len(filtered_articles) >= limit:
                break
        
        print(f"📊 Final filtered articles: {len(filtered_articles)}")
        # Sort articles by date from oldest to newest
        if filtered_articles:
            try:
                # Convert published_date to datetime for proper sorting
                for article in filtered_articles:
                    try:
                        article['published_date_parsed'] = pd.to_datetime(article['published_date'])
                    except:
                        article['published_date_parsed'] = pd.to_datetime('1900-01-01')  # Fallback for invalid dates
                
                # Sort by parsed date (oldest first)
                filtered_articles.sort(key=lambda x: x['published_date_parsed'])
                
                # Remove the temporary parsing field
                for article in filtered_articles:
                    article.pop('published_date_parsed', None)
                    
                print(f"📅 Articles sorted by date (oldest to newest)")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not sort articles by date: {e}")

        return filtered_articles

    def get_available_sources(self):
        """Get list of available news sources"""
        return list(self.news_data.keys())
    
    def get_source_display_names(self):
        """Get display names for all sources"""
        return [self.get_news_source_display_name(source) for source in self.news_data.keys()]

# Example usage and testing
if __name__ == "__main__":
    """Test the news data visualizer"""
    print("🧪 Testing News Data Visualizer...")
    
    # Initialize the visualizer
    news_viz = NewsDataVisualizer()
    
    if news_viz.news_data:
        print(f"\n✅ Successfully loaded data from {len(news_viz.news_data)} sheets")
        
        # Test summary metrics
        metrics = news_viz.get_summary_metrics()
        if metrics:
            print(f"📊 Summary: {metrics['total_articles']} articles from {metrics['sources_count']} sources")
            print(f"📅 Latest update: {metrics['latest_update']}")
        
        # Test regions and topics
        regions = news_viz.get_all_regions()
        topics = news_viz.get_all_topics()
        print(f"🌍 Available regions: {regions}")
        print(f"🏷️ Available topics (first 5): {topics[:5]}")
        
        # Test filtering
        if regions and topics:
            filtered = news_viz.filter_articles_by_region_topic_indicator_and_date(
                selected_region=regions[0] if regions else None,
                selected_topic=topics[0] if topics else None,
                limit=3
            )
            print(f"🔍 Filtered articles: {len(filtered)}")
        
        print("\n🎉 All tests completed!")
    else:
        print("❌ No data loaded - check file path and format")
