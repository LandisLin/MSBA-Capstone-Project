"""
Cross-Country Economic Data Visualizer - Clean Version Using Shared Configuration
Compare economic indicators across multiple countries
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

class CrossCountryVisualizer:
    """Cross-country economic data visualizer"""
    
    def __init__(self, data_dir="./extracted_data"):
        """Initialize with data directory"""
        self.data_dir = data_dir
        self.country_data = {}
        self.available_countries = []
        self.common_indicators = []
        
        print("🌍 Cross-Country Economic Visualizer")
        print("=" * 50)
        
        # Load all country data
        self._load_all_countries()
        self._identify_common_indicators()
    
    def _load_all_countries(self):
        """Load data from all available countries"""
        print("📂 Loading country data...")
        
        available_countries = DataLoader.get_available_countries(self.data_dir)
        
        for country_key in available_countries:
            country_name = NameFormatter.format_country_name(country_key)
            print(f"   📄 Loading {country_name}...")
            
            try:
                country_data = load_country_data(country_key, self.data_dir)
                if country_data:
                    # Convert to cross-country format
                    standardized_data = self._standardize_country_data(country_data, country_key)
                    if standardized_data:
                        self.country_data[country_key] = standardized_data
                        self.available_countries.append(country_key)
                        print(f"   ✅ {country_name}: {len(standardized_data)} indicators")
                    else:
                        print(f"   ❌ {country_name}: No valid data")
                else:
                    print(f"   ❌ {country_name}: Failed to load")
                    
            except Exception as e:
                print(f"   ❌ {country_name}: Error - {e}")
        
        print(f"\n📊 Successfully loaded {len(self.available_countries)} countries")
    
    def _standardize_country_data(self, country_data, country_key):
        """Convert country data to cross-country standard format"""
        standardized = {}
        
        for indicator, df in country_data.items():
            try:
                # Create standardized format for cross-country comparison
                std_df = pd.DataFrame({
                    'date': df['date'],
                    'value': df['value'],
                    'country': country_key,
                    'indicator': indicator
                })
                
                standardized[indicator] = std_df.dropna().sort_values('date')
                
            except Exception as e:
                print(f"   ⚠️ Could not standardize {indicator}: {e}")
                continue
        
        return standardized
    
    def _identify_common_indicators(self):
        """Find indicators available across multiple countries"""
        if not self.country_data:
            return
        
        # Count indicators across countries
        indicator_counts = {}
        for country_data in self.country_data.values():
            for indicator in country_data.keys():
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
        
        # Keep indicators available in at least 2 countries
        self.common_indicators = [
            indicator for indicator, count in indicator_counts.items() 
            if count >= 2
        ]
        
        print(f"📈 Common indicators: {', '.join(self.common_indicators)}")
    
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
        
        # Create 2x2 subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "GDP Growth Rates (YoY %)", 
                "Inflation Rates (YoY %)", 
                "Population Growth Rates (YoY %)", 
                "Interest Rates (%)"
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Add data to subplots
        self._add_dashboard_subplot(fig, 'GDP', available_selected, 1, 1, periods=4, is_growth=True)
        self._add_dashboard_subplot(fig, 'CPI', available_selected, 1, 2, is_inflation=True)
        self._add_dashboard_subplot(fig, 'Population', available_selected, 2, 1, periods=1, is_growth=True)
        self._add_dashboard_subplot(fig, 'Interest_Rate', available_selected, 2, 2, is_growth=False)
        
        # Apply styling
        country_names = [NameFormatter.format_country_name(c) for c in available_selected]
        ChartStyler.apply_default_layout(
            fig,
            title=f"Economic Overview: {', '.join(country_names)}",
            height=800
        )
        
        # Add reference lines
        ChartStyler.add_reference_lines(fig, [0], row=1, col=1, line_color="red")
        ChartStyler.add_reference_lines(fig, [0, 2], row=1, col=2, line_color="red")
        ChartStyler.add_reference_lines(fig, [0], row=2, col=1, line_color="red")
        
        # Configure legend
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v", yanchor="top", y=1, 
                xanchor="left", x=1.02
            )
        )
        
        return fig
    
    def _add_dashboard_subplot(self, fig, indicator, countries, row, col, periods=4, is_growth=True, is_inflation=False):
        """Add data to dashboard subplot"""
        for country_key in countries:
            # Skip if country doesn't have this indicator
            if indicator not in self.country_data[country_key]:
                continue
            
            data = self.country_data[country_key][indicator].copy()
            country_color = ChartStyler.get_country_color(country_key)
            country_name = NameFormatter.format_country_name(country_key)
            
            if is_growth:
                # Calculate growth rates
                if is_inflation:
                    growth_periods = AnalysisUtils.detect_inflation_periods(data, country_key)
                else:
                    growth_periods = periods
                
                data_with_growth = AnalysisUtils.calculate_growth_rates(data, periods=growth_periods)
                data_clean = data_with_growth.dropna(subset=['growth_rate'])
                
                if not data_clean.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=data_clean['date'], y=data_clean['growth_rate'],
                            mode='lines', name=country_name,
                            line=dict(color=country_color, width=2),
                            showlegend=(row == 1 and col == 1),
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
                    growth_periods = periods
                
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
                            # Calculate growth rate
                            periods = AnalysisUtils.detect_inflation_periods(data, country_key) if indicator == 'CPI' else (4 if indicator == 'GDP' else 1)
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
            print("1. Overview Dashboard (all indicators)")
            print("2. GDP Growth Comparison")
            print("3. Inflation Comparison")
            print("4. Interest Rates Comparison")
            print("5. Population Growth Comparison")
            print("6. Property Price Growth Comparison")
            print("7. Summary Table")
            print("8. Exit")
            
            choice = input(f"\nSelect option (1-8): ").strip()
            
            try:
                if choice == '1':
                    fig = self.create_overview_dashboard()
                    if fig:
                        ChartDisplayer.show_chart(fig, "cross_country_overview")
                    else:
                        print("❌ Could not create overview")
                
                elif choice in ['2', '3', '4', '5', '6']:
                    indicators = ['GDP', 'CPI', 'Interest_Rate', 'Population', 'Property_Price']
                    indicator = indicators[int(choice) - 2]
                    
                    fig = self.create_individual_chart(indicator)
                    if fig:
                        ChartDisplayer.show_chart(fig, f"cross_country_{indicator.lower()}")
                    else:
                        print(f"❌ Could not create {indicator} chart")
                
                elif choice == '7':
                    summary_df = self.create_summary_table()
                    if summary_df is not None:
                        print(f"\n📊 CROSS-COUNTRY SUMMARY")
                        print("=" * 80)
                        print(summary_df.to_string(index=False))
                        print("=" * 80)
                    else:
                        print("❌ No summary data available")
                
                elif choice == '8':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-8.")
                    
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