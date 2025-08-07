import pandas as pd
import numpy as np
from datetime import datetime
import re
import os

def extract_topics(topic_analysis):
    """Extract topics from Topic_Analysis field after 'Topics:'"""
    if not topic_analysis or not isinstance(topic_analysis, str):
        return []
    
    # Look for "Topics:" and extract everything until line break
    topics_match = re.search(r'Topics:\s*([^\r\n]*?)(?=\r?\n|$)', topic_analysis)
    if not topics_match:
        return []
    
    topics_string = topics_match.group(1).strip()
    return [topic.strip() for topic in topics_string.split(',')]

def parse_regions(region_string):
    """Parse comma-separated regions"""
    if not region_string or not isinstance(region_string, str):
        return []
    return [region.strip() for region in region_string.split(',')]

def parse_date(date_string):
    """Parse date to YYYY-MM-DD format"""
    if not date_string:
        return None
    try:
        date_obj = pd.to_datetime(date_string)
        return date_obj.strftime('%Y-%m-%d')
    except:
        return None

def transform_news_to_country_topic_matrix(input_file='./news_analysis_output/master_news_analysis.xlsx', 
                                         output_file='./news_analysis_output/country_topic_sentiment.xlsx'):
    """
    Transform master news analysis data into country-topic sentiment matrix
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output Excel file
    """
    
    # Define target countries and topics
    target_countries = [
        "Singapore", "United States", "Euro Area", "United Kingdom", "China", 
        "Japan", "India", "Indonesia", "Malaysia", "Thailand", "Vietnam"
    ]

    target_topics = [
        "Monetary Policy", "Trade & Commerce", "Financial Markets", "Economic Growth",
        "Technology & Innovation", "Energy & Commodities", "Geopolitics & Economy",
        "Corporate & Business", "Real Estate & Housing", "Government & Fiscal Policy",
        "Labor & Employment", "Consumer & Retail"
    ]

    print("🚀 Starting news data transformation...")
    print(f"📖 Reading {input_file}...")

    try:
        # Read the Excel file and get all sheet names
        excel_file = pd.ExcelFile(input_file)
        sheet_names = excel_file.sheet_names
        
        # Filter out the Contents sheet
        data_sheets = [name for name in sheet_names if name != "Contents"]
        print(f"📊 Found {len(data_sheets)} data sheets: {data_sheets}")

        # Collect all articles from all sheets
        all_articles = []

        for sheet_name in data_sheets:
            print(f"🔄 Processing {sheet_name}...")
            
            # Read sheet starting from row 3 (0-indexed row 2) since headers are on row 3
            df = pd.read_excel(input_file, sheet_name=sheet_name, header=2)
            
            # Clean column names (remove any extra spaces)
            df.columns = df.columns.str.strip()
            
            print(f"   - Found {len(df)} articles")
            print(f"   - Columns: {list(df.columns)}")
            
            # Process each row
            for idx, row in df.iterrows():
                # Extract required fields
                regions = parse_regions(row.get('Region', ''))
                date = parse_date(row.get('Published', ''))
                topics = extract_topics(row.get('Topic Analysis', '') or row.get('Topic_Analysis', ''))
                sentiment = row.get('Sentiment Score', row.get('Sentiment_Score', np.nan))
                
                # Validate data
                if date and not pd.isna(sentiment) and regions and topics:
                    try:
                        sentiment_float = float(sentiment)
                        all_articles.append({
                            'date': date,
                            'regions': regions,
                            'topics': topics,
                            'sentiment': sentiment_float,
                            'source': sheet_name
                        })
                    except (ValueError, TypeError):
                        continue

        print(f"\n📈 Collected {len(all_articles)} total valid articles")

        # Create country-topic combinations for each article
        country_topic_data = []

        for article in all_articles:
            for region in article['regions']:
                # Only include regions that are in our target countries
                if region in target_countries:
                    for topic in article['topics']:
                        # Only include topics that are in our target topics
                        if topic in target_topics:
                            country_topic_data.append({
                                'date': article['date'],
                                'country': region,
                                'topic': topic,
                                'country_topic': f"{region}-{topic}",
                                'sentiment': article['sentiment'],
                                'source': article['source']
                            })

        print(f"🔄 Created {len(country_topic_data)} country-topic records")

        # Convert to DataFrame for easier manipulation
        df_records = pd.DataFrame(country_topic_data)
        
        # Group by date and country-topic combination, calculate averages
        grouped = df_records.groupby(['date', 'country_topic'])['sentiment'].mean().reset_index()
        
        # Get all unique dates and sort them
        unique_dates = sorted(df_records['date'].unique())
        print(f"📅 Processing {len(unique_dates)} unique dates from {unique_dates[0]} to {unique_dates[-1]}")

        # Create column headers
        column_headers = ['Date']
        for country in target_countries:
            for topic in target_topics:
                column_headers.append(f"{country}-{topic}")

        print(f"📋 Created {len(column_headers)} columns (1 date + {len(column_headers)-1} country-topic combinations)")

        # Create the matrix
        matrix_data = []
        
        for date in unique_dates:
            row = [date]
            
            # Get data for this date
            date_data = grouped[grouped['date'] == date]
            
            # Create a dictionary for quick lookup
            date_lookup = dict(zip(date_data['country_topic'], date_data['sentiment']))
            
            # Add sentiment values for each country-topic combination
            for country in target_countries:
                for topic in target_topics:
                    country_topic = f"{country}-{topic}"
                    
                    if country_topic in date_lookup:
                        # Round to 4 decimal places
                        sentiment_avg = round(date_lookup[country_topic], 4)
                        row.append(sentiment_avg)
                    else:
                        row.append('')  # Empty cell if no data
            
            matrix_data.append(row)

        # Create DataFrame with the matrix
        result_df = pd.DataFrame(matrix_data, columns=column_headers)
        
        print(f"🎯 Created matrix with {len(result_df)} rows and {len(result_df.columns)} columns")

        # Save to Excel file
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name='Country-Topic Sentiment', index=False)
        
        print(f"\n✅ SUCCESS! Created {output_file}")
        
        # Calculate and display statistics
        total_possible_cells = len(result_df) * (len(result_df.columns) - 1)
        filled_cells = result_df.iloc[:, 1:].count().sum()  # Count non-empty cells excluding date column
        
        print("\n📊 Transformation Statistics:")
        print(f"- Original articles processed: {len(all_articles)}")
        print(f"- Country-topic combinations created: {len(country_topic_data)}")
        print(f"- Date range: {unique_dates[0]} to {unique_dates[-1]} ({len(unique_dates)} days)")
        print(f"- Matrix dimensions: {len(result_df)} rows × {len(result_df.columns)} columns")
        print(f"- Data density: {filled_cells}/{total_possible_cells} cells filled ({filled_cells/total_possible_cells*100:.1f}%)")
        
        # Show sample data for verification
        print("\n🔍 Sample Matrix Data (first 5 rows, first 6 columns):")
        print(result_df.iloc[:5, :6].to_string(index=False))
        
        print(f"\n📁 Output file saved as: {output_file}")
        return result_df
        
    except Exception as e:
        print(f"❌ Error during transformation: {str(e)}")
        raise e

# Example usage
if __name__ == "__main__":
    # Run the transformation
    result = transform_news_to_country_topic_matrix()
    print("\n🎉 Transformation completed successfully!")