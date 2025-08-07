"""
Cross-Country Economic Data Visualizer - Enhanced with GDP Charts
Compare economic indicators across multiple countries using original naming
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import shared utilities
from visualization_config import (
    COUNTRIES,
    CHART_COLORS,
    DataLoader,
    DataProcessor,
    AnalysisUtils,
    ChartDisplayer,
    ChartStyler,
    NameFormatter,
    load_country_data
)

# Import enhanced GDP calculator with error handling
try:
    from worldbank_data import WorldGDPCalculator
    WB_AVAILABLE = True
except ImportError:
    WB_AVAILABLE = False
    print("⚠️ World Bank GDP calculator not available")

def detect_data_frequency_for_cross_country(df):
    """Detect data frequency for cross-country growth calculations"""
    if len(df) < 2:
        return "annual"
    
    median_days = df['date'].diff().median().days
    
    if median_days <= 35:
        return "monthly" 
    elif median_days <= 100:
        return "quarterly"
    else:
        return "annual"

def get_appropriate_growth_periods_cross_country(df, indicator):
    """Get growth periods based on actual data frequency for cross-country comparison"""
    frequency = detect_data_frequency_for_cross_country(df)
    
    if indicator in ['GDP', 'CPI']:
        periods_map = {"monthly": 12, "quarterly": 4, "annual": 1}
        return periods_map[frequency]
    elif indicator == 'Population':
        return 1  # Population is typically annual
    else:
        return 1

class CrossCountryVisualizer:
    """Cross-country economic data visualizer"""
    
    def __init__(self, data_dir="./extracted_data"):
        """Initialize with data directory"""
        self.data_dir = data_dir
        self.country_data = {}
        self.wb_data = {}  # World Bank data
        self.available_countries = []
        self.common_indicators = []
        
        print("🌍 Cross-Country Economic Visualizer")
        print("=" * 50)
        
        # Load all country data
        self._load_all_countries()
        self._identify_common_indicators()
        
        # Fetch World Bank data if available
        if WB_AVAILABLE:
            self._fetch_world_bank_data()
        else:
            print("⚠️ Skipping World Bank data (calculator not available)")
    
    def _load_all_countries(self):
        """Load data from all available countries"""
        print("📂 Loading country data...")
        
        available_countries = DataLoader.get_available_countries(self.data_dir)
        
        for country_key in available_countries:
            country_name = NameFormatter.format_country_name(country_key)
            
            # Load country data
            country_data = load_country_data(country_key, self.data_dir)
            
            if country_data:
                # Process data and convert to standard format
                processed_data = {}
                
                for indicator, data in country_data.items():
                    if not data.empty:
                        # Ensure date column is datetime
                        data['date'] = pd.to_datetime(data['date'])
                        # Sort by date
                        data = data.sort_values('date')
                        processed_data[indicator] = data
                
                if processed_data:
                    self.country_data[country_key] = processed_data
                    self.available_countries.append(country_key)
                    print(f"   ✅ {country_name}: {list(processed_data.keys())}")
                else:
                    print(f"   ❌ {country_name}: No valid data")
            else:
                print(f"   ❌ {country_name}: Could not load data")
        
        print(f"\n🎯 Loaded {len(self.available_countries)} countries")
    
    def _identify_common_indicators(self):
        """Identify indicators available across multiple countries"""
        indicator_counts = {}
        
        for country_key, country_data in self.country_data.items():
            for indicator in country_data.keys():
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
        
        # Keep indicators available in at least 2 countries
        self.common_indicators = [
            indicator for indicator, count in indicator_counts.items() 
            if count >= 2
        ]
        
        print(f"📈 Common indicators: {', '.join(self.common_indicators)}")
    
    def _fetch_world_bank_data(self):
        """Load World Bank data from cache, or fetch if not available"""
        try:
            calculator = WorldGDPCalculator()
            
            # Try to load from cache first
            self.wb_data = calculator.load_world_bank_data_from_excel(self.data_dir)
            
            if not self.wb_data:
                print("🌐 No cached World Bank data found")
                # For Streamlit, we don't want to do long API calls
                # Just continue without World Bank data
                print("   💡 Run the main pipeline to refresh World Bank data")
            else:
                print(f"🏦 World Bank data loaded from cache for {len(self.wb_data)} countries")
        except Exception as e:
            print(f"⚠️ World Bank data error: {e}")
            self.wb_data = {}
    
    def create_gdp_level_chart(self):
        """Create GDP level comparison chart"""
        if not self.wb_data:
            print("❌ No World Bank GDP data available")
            return None
        
        print("💰 Creating GDP Level Comparison...")
        
        fig = go.Figure()
        
        # Add GDP data for each country
        for country_key, indicators in self.wb_data.items():
            if 'gdp' not in indicators:
                continue
            
            gdp_df = indicators['gdp']
            country_color = ChartStyler.get_country_color(country_key)
            country_name = NameFormatter.format_country_name(country_key)
            
            fig.add_trace(go.Scatter(
                x=gdp_df['date'], 
                y=gdp_df['value'],
                mode='lines', 
                name=country_name,
                line=dict(color=country_color, width=3),
                hovertemplate=f'<b>{country_name}</b><br>Year: %{{x}}<br>GDP: $%{{y:.2f}}T<extra></extra>'
            ))
        
        # Apply styling
        ChartStyler.apply_default_layout(fig, title="💰 GDP Levels Comparison (World Bank)")
        fig.update_xaxes(title="Year")
        fig.update_yaxes(title="GDP (Trillions USD)")
        
        return fig
    
    def create_gdp_per_capita_chart(self):
        """Create GDP per capita comparison chart"""
        if not self.wb_data:
            print("❌ No World Bank data available")
            return None
        
        print("👥 Creating GDP Per Capita Comparison...")
        
        fig = go.Figure()
        
        # Add GDP per capita data for each country
        for country_key, indicators in self.wb_data.items():
            if 'gdp_per_capita' not in indicators:
                continue
            
            gdp_pc_df = indicators['gdp_per_capita']
            country_color = ChartStyler.get_country_color(country_key)
            country_name = NameFormatter.format_country_name(country_key)
            
            fig.add_trace(go.Scatter(
                x=gdp_pc_df['date'], 
                y=gdp_pc_df['value'],
                mode='lines', 
                name=country_name,
                line=dict(color=country_color, width=3),
                hovertemplate=f'<b>{country_name}</b><br>Year: %{{x}}<br>GDP per Capita: $%{{y:,.0f}}<extra></extra>'
            ))
        
        # Apply styling
        ChartStyler.apply_default_layout(fig, title="👥 GDP Per Capita Comparison (World Bank)")
        fig.update_xaxes(title="Year")
        fig.update_yaxes(title="GDP per Capita (USD)")
        
        return fig
    
    def create_overview_dashboard(self, selected_countries=None):
        """Create comprehensive overview dashboard"""
        if selected_countries is None:
            selected_countries = self.available_countries
        
        # Filter to available countries
        available_selected = [c for c in selected_countries if c in self.country_data]
        
        if not available_selected:
            print("❌ No data available for selected countries")
            return None
        
        print(f"🎨 Creating dashboard for: {[NameFormatter.format_country_name(c) for c in available_selected]}")
        
        # Create 3x2 subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "GDP Levels (Trillions USD)", 
                "GDP Per Capita (USD)",
                "GDP Growth Rates (YoY %)", 
                "Inflation Rates (YoY %)", 
                "Population Growth Rates (YoY %)", 
                "Interest Rates (%)"
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # Add World Bank data to subplots
        self._add_wb_subplot(fig, available_selected, 1, 1, 'gdp')
        self._add_wb_subplot(fig, available_selected, 1, 2, 'gdp_per_capita')
        
        # Add existing indicator data
        self._add_dashboard_subplot(fig, 'GDP', available_selected, 2, 1, periods=4, is_growth=True)
        self._add_dashboard_subplot(fig, 'CPI', available_selected, 2, 2, is_growth=True, is_inflation=True)
        self._add_dashboard_subplot(fig, 'Population', available_selected, 3, 1, periods=1, is_growth=True)
        self._add_dashboard_subplot(fig, 'Interest_Rate', available_selected, 3, 2, is_growth=False)
        
        # Apply styling
        country_names = [NameFormatter.format_country_name(c) for c in available_selected]
        ChartStyler.apply_default_layout(
            fig,
            title=f"📊 Economic Overview: {', '.join(country_names)}",
            height=1000
        )
        
        return fig
    
    def _add_wb_subplot(self, fig, countries, row, col, indicator_type):
        """Add World Bank data to subplot"""
        for country_key in countries:
            if country_key not in self.wb_data or indicator_type not in self.wb_data[country_key]:
                continue
            
            data_df = self.wb_data[country_key][indicator_type]
            country_color = ChartStyler.get_country_color(country_key)
            country_name = NameFormatter.format_country_name(country_key)
            
            fig.add_trace(
                go.Scatter(
                    x=data_df['date'], y=data_df['value'],
                    mode='lines', name=country_name,
                    line=dict(color=country_color, width=2),
                    showlegend=(row == 1 and col == 1),
                    legendgroup=country_key
                ), row=row, col=col
            )
    
    def _add_dashboard_subplot(self, fig, indicator, countries, row, col, periods=None, is_growth=False, is_inflation=False):
        """Add indicator data to subplot"""
        for country_key in countries:
            if indicator not in self.country_data[country_key]:
                continue
            
            data = self.country_data[country_key][indicator].copy()
            country_color = ChartStyler.get_country_color(country_key)
            country_name = NameFormatter.format_country_name(country_key)
            
            if is_growth:
                # Calculate appropriate growth periods
                if is_inflation:
                    growth_periods = AnalysisUtils.detect_inflation_periods(data, country_key)
                else:
                    growth_periods = get_appropriate_growth_periods_cross_country(data, indicator)
                
                data_with_growth = AnalysisUtils.calculate_growth_rates(data, periods=growth_periods)
                data_clean = data_with_growth.dropna(subset=['growth_rate'])
                
                if not data_clean.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=data_clean['date'], y=data_clean['growth_rate'],
                            mode='lines', name=country_name,
                            line=dict(color=country_color, width=2),
                            showlegend=False,
                            legendgroup=country_key
                        ), row=row, col=col
                    )
            else:
                # Level data (interest rates)
                if not data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=data['date'], y=data['value'],
                            mode='lines', name=country_name,
                            line=dict(color=country_color, width=2),
                            showlegend=False,
                            legendgroup=country_key
                        ), row=row, col=col
                    )
    
    def create_individual_chart(self, indicator):
        """Create detailed chart for specific indicator"""
        if indicator not in self.common_indicators:
            print(f"❌ {indicator} not available across countries")
            return None
        
        # Determine chart configuration
        chart_configs = {
            'GDP': ('GDP Growth Rates (YoY %)', 4, True, False),
            'CPI': ('Inflation Rates (YoY %)', None, True, True),
            'Interest_Rate': ('Interest Rates (%)', None, False, False),
            'Population': ('Population Growth Rates (YoY %)', 1, True, False),
            'Property_Price': ('Property Price Growth (YoY %)', 4, True, False)
        }
        
        if indicator not in chart_configs:
            title = f"{indicator} Comparison"
            periods, is_growth, is_inflation = 4, True, False
        else:
            title, periods, is_growth, is_inflation = chart_configs[indicator]
        
        fig = go.Figure()
        
        # Add data for each country
        for country_key in self.available_countries:
            if indicator not in self.country_data[country_key]:
                continue
            
            data = self.country_data[country_key][indicator].copy()
            country_color = ChartStyler.get_country_color(country_key)
            country_name = NameFormatter.format_country_name(country_key)
            
            if is_growth:
                # Calculate appropriate growth periods
                if is_inflation:
                    growth_periods = AnalysisUtils.detect_inflation_periods(data, country_key)
                else:
                    growth_periods = get_appropriate_growth_periods_cross_country(data, indicator)
                
                data_with_growth = AnalysisUtils.calculate_growth_rates(data, periods=growth_periods)
                data_clean = data_with_growth.dropna(subset=['growth_rate'])
                
                if not data_clean.empty:
                    fig.add_trace(go.Scatter(
                        x=data_clean['date'], y=data_clean['growth_rate'],
                        mode='lines', name=country_name,
                        line=dict(color=country_color, width=3)
                    ))
            else:
                # Level data
                if not data.empty:
                    fig.add_trace(go.Scatter(
                        x=data['date'], y=data['value'],
                        mode='lines', name=country_name,
                        line=dict(color=country_color, width=3)
                    ))
        
        # Apply styling
        y_label = "Growth Rate (%)" if is_growth else "Rate (%)"
        ChartStyler.apply_default_layout(fig, title=f"📊 {title}")
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title=y_label)
        
        # Add reference lines
        if is_growth:
            ChartStyler.add_reference_lines(fig, [0], line_color="red")
            if is_inflation:
                ChartStyler.add_reference_lines(fig, [2], line_color="green")
        
        return fig
    
    def create_summary_table(self):
        """Create summary table with latest indicators"""
        summary_data = []
        
        for country_key in self.available_countries:
            country_name = NameFormatter.format_country_name(country_key)
            country_summary = {'Country': country_name}
            
            # Get latest World Bank GDP data
            if country_key in self.wb_data:
                if 'gdp' in self.wb_data[country_key]:
                    # Sort by date to ensure we get the most recent value
                    gdp_data = self.wb_data[country_key]['gdp'].sort_values('date')
                    latest_gdp = gdp_data['value'].iloc[-1]  # Now this gets the newest data
                    country_summary['GDP (Trillions)'] = f"${latest_gdp:.2f}T"
                else:
                    country_summary['GDP (Trillions)'] = "N/A"
                
                if 'gdp_per_capita' in self.wb_data[country_key]:
                    # Sort by date to ensure we get the most recent value
                    gdp_pc_data = self.wb_data[country_key]['gdp_per_capita'].sort_values('date')
                    latest_gdp_pc = gdp_pc_data['value'].iloc[-1]  # Now this gets the newest data
                    country_summary['GDP per Capita'] = f"${latest_gdp_pc:,.0f}"
                else:
                    country_summary['GDP per Capita'] = "N/A"
            else:
                country_summary['GDP (Trillions)'] = "N/A"
                country_summary['GDP per Capita'] = "N/A"
            
            # Get latest values for each indicator
            for indicator, label in [
                ('GDP', 'GDP Growth'), 
                ('CPI', 'Inflation'), 
                ('Interest_Rate', 'Interest Rate'), 
                ('Population', 'Pop Growth')
            ]:
                if indicator in self.country_data[country_key]:
                    data = self.country_data[country_key][indicator]
                    
                    if not data.empty:
                        if indicator in ['GDP', 'CPI', 'Population']:
                            # Calculate growth rate with frequency detection
                            if indicator == 'CPI':
                                periods = AnalysisUtils.detect_inflation_periods(data, country_key)
                            else:
                                periods = get_appropriate_growth_periods_cross_country(data, indicator)
                            
                            growth_data = AnalysisUtils.calculate_growth_rates(data, periods=periods)
                            
                            if not growth_data.empty:
                                latest_growth = growth_data['growth_rate'].dropna()
                                if not latest_growth.empty:
                                    country_summary[label] = f"{latest_growth.iloc[-1]:.1f}%"
                                else:
                                    country_summary[label] = "N/A"
                            else:
                                country_summary[label] = "N/A"
                        else:
                            # Interest rate - use level
                            latest_value = data['value'].iloc[-1]
                            country_summary[label] = f"{latest_value:.2f}%"
                    else:
                        country_summary[label] = "N/A"
                else:
                    country_summary[label] = "N/A"
            
            summary_data.append(country_summary)
        
        return pd.DataFrame(summary_data) if summary_data else None
    
    def interactive_menu(self):
        """Interactive menu for cross-country analysis"""
        while True:
            print(f"\n🎯 CROSS-COUNTRY ANALYSIS OPTIONS")
            print("=" * 50)
            print("1. Overview Dashboard (enhanced with GDP charts)")
            print("2. GDP Level Comparison (World Bank)")
            print("3. GDP Per Capita Comparison (World Bank)")
            print("4. GDP Growth Comparison")
            print("5. Inflation Comparison")
            print("6. Interest Rates Comparison")
            print("7. Population Growth Comparison")
            print("8. Property Price Growth Comparison")
            print("9. Summary Table")
            print("10. Exit")
            
            choice = input(f"\nSelect option (1-10): ").strip()
            
            try:
                if choice == '1':
                    fig = self.create_overview_dashboard()
                    if fig:
                        ChartDisplayer.show_chart(fig, "cross_country_overview")
                    else:
                        print("❌ Could not create overview")
                
                elif choice == '2':
                    fig = self.create_gdp_level_chart()
                    if fig:
                        ChartDisplayer.show_chart(fig, "cross_country_gdp_levels")
                    else:
                        print("❌ Could not create GDP level chart")
                
                elif choice == '3':
                    fig = self.create_gdp_per_capita_chart()
                    if fig:
                        ChartDisplayer.show_chart(fig, "cross_country_gdp_per_capita")
                    else:
                        print("❌ Could not create GDP per capita chart")
                
                elif choice in ['4', '5', '6', '7', '8']:
                    indicators = ['GDP', 'CPI', 'Interest_Rate', 'Population', 'Property_Price']
                    indicator = indicators[int(choice) - 4]
                    
                    fig = self.create_individual_chart(indicator)
                    if fig:
                        ChartDisplayer.show_chart(fig, f"cross_country_{indicator.lower()}")
                    else:
                        print(f"❌ Could not create {indicator} chart")
                
                elif choice == '9':
                    summary_df = self.create_summary_table()
                    if summary_df is not None:
                        print(f"\n📊 CROSS-COUNTRY SUMMARY")
                        print("=" * 80)
                        print(summary_df.to_string(index=False))
                        print("=" * 80)
                    else:
                        print("❌ No summary data available")
                
                elif choice == '10':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-10.")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run(self):
        """Main execution function"""
        if not self.available_countries:
            print("❌ No country data available. Please check your data files.")
            return
        
        print(f"\n🎯 Ready for cross-country analysis:")
        print(f"   Countries: {', '.join([NameFormatter.format_country_name(c) for c in self.available_countries])}")
        print(f"   Common indicators: {', '.join(self.common_indicators)}")
        
        if self.wb_data:
            print(f"   World Bank GDP data: {len([c for c in self.wb_data if 'gdp' in self.wb_data[c]])} countries")
            print(f"   World Bank GDP per capita data: {len([c for c in self.wb_data if 'gdp_per_capita' in self.wb_data[c]])} countries")
        
        self.interactive_menu()

def main():
    """Main function"""
    print("🔗 CROSS-COUNTRY ECONOMIC VISUALIZER")
    print("=" * 50)
    
    try:
        visualizer = CrossCountryVisualizer()
        visualizer.run()
    except KeyboardInterrupt:
        print("\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Please check your data files and try again")

if __name__ == "__main__":
    main()
