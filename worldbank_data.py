"""
Enhanced World GDP Coverage Calculator for Economic Dashboard
Calculates GDP coverage AND provides time series data for visualization
"""

import requests
import pandas as pd
from datetime import datetime

class WorldGDPCalculator:
    """World GDP calculator with time series data for visualization"""
    
    def __init__(self):
        self.base_url = "https://api.worldbank.org/v2/country"
        self.latest_year = None
        self.world_gdp = None
        
        # Map your dataset countries to World Bank country codes
        self.dataset_countries = {
            'us': 'USA',           # United States
            'japan': 'JPN',        # Japan  
            'uk': 'GBR',          # United Kingdom
            'singapore': 'SGP',    # Singapore
            'india': 'IND',       # India
            'indonesia': 'IDN',   # Indonesia
            'malaysia': 'MYS',    # Malaysia
            'thailand': 'THA',    # Thailand
            'vietnam': 'VNM',     # Vietnam
            'china': 'CHN',       # China
            'euro_area': 'XC'    # Euro Area (World Bank uses EMU for eurozone)
        }
    
    def fetch_world_gdp(self, year=None):
        """Fetch world total GDP from World Bank API (existing method)"""
        if year is None:
            year = datetime.now().year - 1  # Use previous year for complete data
            
        url = f"{self.base_url}/WLD/indicator/NY.GDP.MKTP.CD"
        params = {
            'format': 'json',
            'date': f'{year}:{year}',
            'per_page': 1000
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if len(data) > 1 and data[1] and data[1][0]['value']:
                world_gdp = float(data[1][0]['value'])
                self.world_gdp = world_gdp
                self.latest_year = year
                return world_gdp
            else:
                # Try previous year if current year not available
                return self.fetch_world_gdp(year - 1)
                
        except Exception as e:
            print(f"Error fetching world GDP: {e}")
            return None
    
    def fetch_countries_gdp(self, countries_dict, year):
        """Fetch GDP for specified countries (existing method)"""
        gdp_data = {}
        
        for country_name, country_code in countries_dict.items():
            url = f"{self.base_url}/{country_code}/indicator/NY.GDP.MKTP.CD"
            params = {
                'format': 'json',
                'date': f'{year}:{year}',
                'per_page': 1000
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if len(data) > 1 and data[1] and data[1][0]['value']:
                    gdp_value = float(data[1][0]['value'])
                    gdp_data[country_name] = {
                        'gdp_usd': gdp_value,
                        'country_code': country_code,
                        'year': year
                    }
                    print(f"✓ {country_name.title()}: ${gdp_value:,.0f}")
                else:
                    print(f"✗ {country_name.title()}: No data available for {year}")
                    
            except Exception as e:
                print(f"✗ Error fetching {country_name}: {e}")
        
        return gdp_data
    
    def calculate_coverage(self, use_cached_data=True, data_dir="./extracted_data"):
        """Calculate GDP coverage percentage with option to use cached data"""
        print("🌍 Calculating World GDP Coverage")
        print("=" * 50)
        
        countries_to_use = self.dataset_countries
        print(f"Countries: {', '.join(countries_to_use.keys())}")
        
        # Try to use cached data first if requested
        if use_cached_data:
            try:
                cached_data = self.load_world_bank_data_from_excel(data_dir)
                if cached_data:
                    return self._calculate_coverage_from_cached_data(cached_data)
                else:
                    print("💡 No cached data found, falling back to API...")
            except Exception as e:
                print(f"⚠️ Error loading cached data: {e}")
                print("💡 Falling back to API...")
        
        # Fallback to original API-based method
        return self._calculate_coverage_from_api()
    
    def _calculate_coverage_from_cached_data(self, cached_data):
        """Calculate coverage using cached World Bank data"""
        print("📁 Using cached World Bank data for coverage calculation")
        
        # Get the latest year from cached data
        latest_year = 0
        total_dataset_gdp = 0
        countries_gdp = {}
        
        for country_key, indicators in cached_data.items():
            if 'gdp' in indicators:
                gdp_df = indicators['gdp']
                if not gdp_df.empty:
                    # Sort by date and get the LATEST value (not first value)
                    gdp_df_sorted = gdp_df.sort_values('date')
                    latest_gdp_trillions = gdp_df_sorted['value'].iloc[-1]  # Last (latest) value
                    latest_date = gdp_df_sorted['date'].iloc[-1]
                    
                    # Convert back to actual USD (from trillions)
                    latest_gdp_usd = latest_gdp_trillions * 1e12
                    year = latest_date.year
                    
                    # Update latest year
                    latest_year = max(latest_year, year)
                    
                    # Add to total
                    total_dataset_gdp += latest_gdp_usd
                    
                    # Store country data
                    if country_key in self.dataset_countries:
                        wb_code = self.dataset_countries[country_key]
                        countries_gdp[country_key] = {
                            'gdp_usd': latest_gdp_usd,
                            'country_code': wb_code,
                            'year': year
                        }
                        print(f"✓ {country_key.title()}: ${latest_gdp_usd:,.0f} ({year})")
        
        if not countries_gdp:
            print("❌ No GDP data found in cached files")
            return None
        
        # Get world GDP for the latest year (still need API for this)
        print(f"\n🌐 Fetching World GDP for {latest_year}...")
        world_gdp = self.fetch_world_gdp(latest_year)
        
        if not world_gdp:
            print("❌ Could not fetch world GDP data")
            return None
        
        print(f"🌐 World GDP ({latest_year}): ${world_gdp:,.0f}")
        
        # Calculate coverage
        coverage_percentage = (total_dataset_gdp / world_gdp) * 100
        
        print(f"\n📈 Results:")
        print(f"Dataset GDP: ${total_dataset_gdp:,.0f}")
        print(f"World GDP: ${world_gdp:,.0f}")
        print(f"Coverage: {coverage_percentage:.1f}% of World GDP")
        
        return {
            'world_gdp': world_gdp,
            'dataset_gdp': total_dataset_gdp,
            'coverage_percent': coverage_percentage,
            'year': latest_year,
            'countries': countries_gdp,
            'country_count': len(countries_gdp),
            'data_source': 'cached'
        }

    def _calculate_coverage_from_api(self):
        """Original API-based coverage calculation"""
        # Get world GDP
        world_gdp = self.fetch_world_gdp()
        if not world_gdp:
            print("❌ Could not fetch world GDP data")
            return None
            
        print(f"🌐 World GDP ({self.latest_year}): ${world_gdp:,.0f}")
        print(f"\n📊 Fetching GDP data:")
        
        # Get country GDPs
        countries_gdp = self.fetch_countries_gdp(self.dataset_countries, self.latest_year)
        
        if not countries_gdp:
            print("❌ Could not fetch country GDP data")
            return None
        
        # Calculate coverage
        total_dataset_gdp = sum(data['gdp_usd'] for data in countries_gdp.values())
        coverage_percentage = (total_dataset_gdp / world_gdp) * 100
        
        print(f"\n📈 Results:")
        print(f"Dataset GDP: ${total_dataset_gdp:,.0f}")
        print(f"World GDP: ${world_gdp:,.0f}")
        print(f"Coverage: {coverage_percentage:.1f}% of World GDP")
        
        return {
            'world_gdp': world_gdp,
            'dataset_gdp': total_dataset_gdp,
            'coverage_percent': coverage_percentage,
            'year': self.latest_year,
            'countries': countries_gdp,
            'country_count': len(countries_gdp),
            'data_source': 'api'
        }

    def get_dashboard_metrics(self, use_cached_data=True, data_dir="./extracted_data"):
        """Get formatted metrics for dashboard display using cached data when available"""
        coverage_data = self.calculate_coverage(use_cached_data=use_cached_data, data_dir=data_dir)
        
        if not coverage_data:
            return None
            
        data_source_label = "📁 Cached" if coverage_data.get('data_source') == 'cached' else "🌐 Live API"
        
        return {
            'countries_count': coverage_data['country_count'],
            'coverage_percent': f"{coverage_data['coverage_percent']:.1f}%",
            'total_gdp_trillions': f"${coverage_data['dataset_gdp']/1e12:.1f}T",
            'data_year': coverage_data['year'],
            'description': f"Represents {coverage_data['coverage_percent']:.1f}% of global economic output",
            'data_source': data_source_label
        }
    
    # NEW METHODS FOR VISUALIZATION
    def fetch_time_series_data(self, countries_list, start_year=2000, indicators=None):
        """Fetch time series GDP and GDP per capita data for visualization"""
        if indicators is None:
            indicators = {
                'gdp': 'NY.GDP.MKTP.CD',      # GDP current USD
                'gdp_per_capita': 'NY.GDP.PCAP.CD'  # GDP per capita current USD
            }
        
        print(f"📊 Fetching time series data from {start_year}...")
        
        end_year = datetime.now().year
        date_range = f"{start_year}:{end_year}"
        
        visualization_data = {}
        
        for country_key in countries_list:
            if country_key not in self.dataset_countries:
                continue
            
            country_code = self.dataset_countries[country_key]
            country_data = {}
            
            for indicator_name, indicator_code in indicators.items():
                url = f"{self.base_url}/{country_code}/indicator/{indicator_code}"
                params = {
                    'format': 'json',
                    'date': date_range,
                    'per_page': 1000
                }
                
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if len(data) > 1 and data[1]:
                        # Convert to DataFrame
                        records = []
                        for item in data[1]:
                            if item['value'] is not None:
                                records.append({
                                    'date': pd.to_datetime(f"{item['date']}-01-01"),
                                    'value': float(item['value']),
                                    'country': country_key,
                                    'indicator': indicator_name
                                })
                        
                        if records:
                            df = pd.DataFrame(records)
                            
                            # Convert GDP to trillions for better readability
                            if indicator_name == 'gdp':
                                df['value'] = df['value'] / 1e12
                                df['unit'] = 'Trillions USD'
                            else:
                                df['unit'] = 'USD per capita'
                            
                            country_data[indicator_name] = df
                            print(f"   ✅ {country_key} {indicator_name}: {len(df)} records")
                        else:
                            print(f"   ❌ {country_key} {indicator_name}: No valid data")
                    else:
                        print(f"   ❌ {country_key} {indicator_name}: No data returned")
                        
                except Exception as e:
                    print(f"   ❌ Error fetching {country_key} {indicator_name}: {e}")
            
            if country_data:
                visualization_data[country_key] = country_data
        
        return visualization_data
    
    def get_visualization_data(self, countries_list, start_year=2000):
        """Main method to get all visualization data"""
        return self.fetch_time_series_data(countries_list, start_year)
    
    def save_world_bank_data_to_excel(self, countries_list, start_year=2000, output_dir="./extracted_data"):
        """Save World Bank GDP data to Excel file for caching"""
        import os
        from datetime import datetime
        
        print("💾 Saving World Bank GDP data to Excel...")
        
        # Fetch the data
        wb_data = self.fetch_time_series_data(countries_list, start_year)
        
        if not wb_data:
            print("❌ No World Bank data to save")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"world_bank_gdp_data_{timestamp}.xlsx"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data for Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Create a contents sheet
            contents_data = []
            
            for country_key, indicators in wb_data.items():
                for indicator_name, df in indicators.items():
                    sheet_name = f"{country_key}_{indicator_name}"
                    
                    # Save to sheet
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Add to contents
                    contents_data.append({
                        'Country': country_key,
                        'Indicator': indicator_name,
                        'Sheet': sheet_name,
                        'Records': len(df),
                        'Latest_Year': df['date'].max().year,
                        'Unit': df['unit'].iloc[0]
                    })
            
            # Save contents sheet
            contents_df = pd.DataFrame(contents_data)
            contents_df.to_excel(writer, sheet_name='Contents', index=False)
        
        print(f"✅ World Bank data saved to: {filepath}")
        return filepath
    
    def load_world_bank_data_from_excel(self, data_dir="./extracted_data"):
        """Load World Bank GDP data from latest Excel file"""
        import os
        import glob
        
        # Find the latest World Bank data file
        pattern = os.path.join(data_dir, "world_bank_gdp_data_*.xlsx")
        files = glob.glob(pattern)
        
        if not files:
            print("📁 No cached World Bank data found")
            return {}
        
        # Get the latest file
        latest_file = max(files, key=os.path.getmtime)
        print(f"📖 Loading World Bank data from: {os.path.basename(latest_file)}")
        
        try:
            # Load contents sheet to get structure
            contents_df = pd.read_excel(latest_file, sheet_name='Contents')
            
            wb_data = {}
            
            for _, row in contents_df.iterrows():
                country_key = row['Country']
                indicator_name = row['Indicator']
                sheet_name = row['Sheet']
                
                # Load the data
                df = pd.read_excel(latest_file, sheet_name=sheet_name)
                df['date'] = pd.to_datetime(df['date'])
                
                # Organize data
                if country_key not in wb_data:
                    wb_data[country_key] = {}
                
                wb_data[country_key][indicator_name] = df
                
            print(f"✅ Loaded World Bank data for {len(wb_data)} countries")
            return wb_data
            
        except Exception as e:
            print(f"❌ Error loading World Bank data: {e}")
            return {}

if __name__ == "__main__":
    calculator = WorldGDPCalculator()

    # One-time: regenerate cache with all countries
    all_countries = list(calculator.dataset_countries.keys())
    print(f"🔄 Regenerating cache with all countries: {all_countries}")
    calculator.save_world_bank_data_to_excel(all_countries, start_year=2000)

    print("🌍 GDP COVERAGE ANALYSIS:")
    metrics = calculator.get_dashboard_metrics()
    if metrics:
        print(f"Countries: {metrics['countries_count']}")
        print(f"Coverage: {metrics['coverage_percent']}")
        print(f"Total GDP: {metrics['total_gdp_trillions']}")
        print(f"Year: {metrics['data_year']}")
        print(f"Description: {metrics['description']}")