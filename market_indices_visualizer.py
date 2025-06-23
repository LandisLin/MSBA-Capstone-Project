"""
Market Indices Visualizer - Clean Version Using Shared Configuration
Financial market data visualization with comprehensive analysis tools
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime

# Import shared utilities
from visualization_config import (
    MARKET_INDICES_PATTERN,
    DataLoader,
    DataProcessor,
    AnalysisUtils,
    ChartDisplayer,
    ChartStyler,
    NameFormatter,
    SourceUrlLoader
)

class MarketIndicesVisualizer:
    """Market indices data visualizer with comprehensive analysis tools"""
    
    def __init__(self, data_dir="./extracted_data"):
        """Initialize with data directory"""
        self.data_dir = data_dir
        self.market_data = {}
        self.available_indices = []
        
        print("📈 Market Indices Visualizer")
        print("=" * 40)
        
        # Load market data
        self._load_market_data()
    
    def _load_market_data(self):
        """Load market indices data from latest file"""
        latest_file = DataLoader.find_latest_file(MARKET_INDICES_PATTERN, self.data_dir)
        
        if not latest_file:
            print("❌ No market indices files found!")
            return
        
        print(f"📄 Loading: {latest_file.name}")
        
        try:
            excel_data = DataLoader.load_excel_sheets(str(latest_file))
            if not excel_data:
                return
            
            for sheet_name, df in excel_data.items():
                if sheet_name == 'Contents' or df.empty:
                    continue
                
                # Process market data
                processed_data = DataProcessor.process_market_data(df, sheet_name)
                if processed_data is not None and not processed_data.empty:
                    self.market_data[sheet_name] = processed_data
                    self.available_indices.append(sheet_name)
            
            print(f"✅ Loaded {len(self.available_indices)} market indices")
            for idx in self.available_indices:
                records = len(self.market_data[idx])
                date_range = f"{self.market_data[idx]['date'].min().strftime('%Y-%m')} to {self.market_data[idx]['date'].max().strftime('%Y-%m')}"
                print(f"   • {idx}: {records} records ({date_range})")
                
        except Exception as e:
            print(f"❌ Error loading market data: {e}")
    
    def create_market_overview(self, selected_indices=None):
        """Create comprehensive market overview dashboard"""
        if selected_indices is None:
            selected_indices = self.available_indices
        
        print(f"🎨 Creating market overview for: {', '.join(selected_indices)}")
        
        # Create 3x1 subplot layout
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                "Market Indices - Normalized Prices (Base 100)",
                "Monthly Returns (%)",
                "Year-to-Date Performance (%)"
            ],
            vertical_spacing=0.15,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Get colors
        colors = ChartStyler.get_qualitative_colors()
        
        # Plot 1: Normalized prices
        self._add_normalized_prices(fig, selected_indices, colors, 1)
        
        # Plot 2: Monthly returns
        self._add_monthly_returns(fig, selected_indices, colors, 2)
        
        # Plot 3: YTD performance
        self._add_ytd_performance(fig, selected_indices, colors, 3)
        
        # Apply styling
        ChartStyler.apply_default_layout(
            fig,
            title=f"📈 Market Indices Overview: {', '.join(selected_indices)}",
            height=1400
        )
        
        # Configure legend
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v", yanchor="top", y=0.98,
                xanchor="left", x=1.02, borderwidth=1
            )
        )
        
        return fig
    
    def _add_normalized_prices(self, fig, indices, colors, row):
        """Add normalized price chart (base 100)"""
        for i, index_name in enumerate(indices):
            if index_name in self.market_data:
                data = self.market_data[index_name].copy()
                
                # Normalize to base 100
                first_price = data['close'].iloc[0]
                data['normalized'] = (data['close'] / first_price) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=data['date'], y=data['normalized'],
                        mode='lines', name=NameFormatter.format_display_name(index_name),
                        line=dict(color=colors[i % len(colors)], width=2),
                        showlegend=(row == 1)
                    ), row=row, col=1
                )
        
        # Add base line at 100
        ChartStyler.add_reference_lines(fig, [100], row=row, col=1, line_color="gray")
    
    def _add_monthly_returns(self, fig, indices, colors, row):
        """Add monthly returns chart"""
        for i, index_name in enumerate(indices):
            if index_name in self.market_data:
                data = self.market_data[index_name].copy()
                
                # Calculate monthly returns
                data_with_returns = AnalysisUtils.calculate_returns(data.rename(columns={'close': 'value'}), periods=1)
                data_clean = data_with_returns.dropna(subset=['growth_rate'])
                
                if not data_clean.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=data_clean['date'], y=data_clean['growth_rate'],
                            mode='lines', name=NameFormatter.format_display_name(index_name),
                            line=dict(color=colors[i % len(colors)], width=2),
                            showlegend=False
                        ), row=row, col=1
                    )
        
        # Add zero line
        ChartStyler.add_reference_lines(fig, [0], row=row, col=1, line_color="red")
    
    def _add_ytd_performance(self, fig, indices, colors, row):
        """Add YTD performance bar chart"""
        ytd_data = []
        ytd_colors = []
        
        for i, index_name in enumerate(indices):
            if index_name in self.market_data:
                data = self.market_data[index_name]
                ytd_perf = AnalysisUtils.calculate_ytd_performance(data)
                
                if ytd_perf:
                    ytd_data.append({
                        'index': NameFormatter.format_display_name(index_name),
                        'ytd_return': ytd_perf['ytd_return']
                    })
                    # Color based on performance
                    ytd_colors.append('green' if ytd_perf['ytd_return'] >= 0 else 'red')
        
        if ytd_data:
            ytd_df = pd.DataFrame(ytd_data)
            
            fig.add_trace(
                go.Bar(
                    x=ytd_df['index'], y=ytd_df['ytd_return'],
                    name="YTD Performance",
                    marker_color=ytd_colors,
                    showlegend=False
                ), row=row, col=1
            )
            
            # Add zero line
            ChartStyler.add_reference_lines(fig, [0], row=row, col=1, line_color="black")
    
    def create_individual_chart(self, index_name, chart_type="price"):
        """Create detailed chart for individual index"""
        if index_name not in self.market_data:
            print(f"❌ Index {index_name} not found")
            return None
        
        data = self.market_data[index_name].copy()
        display_name = NameFormatter.format_display_name(index_name)
        
        fig = go.Figure()
        
        if chart_type == "price":
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=data['date'], y=data['close'],
                    mode='lines', name=f"{display_name} Price",
                    line=dict(width=3, color='blue')
                )
            )
            
            ChartStyler.apply_default_layout(
                fig,
                title=f"📈 {display_name} - Price Chart"
            )
            fig.update_yaxes(title="Price")
            
        elif chart_type == "returns":
            # Returns chart
            data_with_returns = AnalysisUtils.calculate_returns(data.rename(columns={'close': 'value'}), periods=1)
            data_clean = data_with_returns.dropna(subset=['growth_rate'])
            
            fig.add_trace(
                go.Scatter(
                    x=data_clean['date'], y=data_clean['growth_rate'],
                    mode='lines', name=f"{display_name} Returns",
                    line=dict(width=2, color='orange')
                )
            )
            
            ChartStyler.apply_default_layout(
                fig,
                title=f"📊 {display_name} - Monthly Returns (%)"
            )
            fig.update_yaxes(title="Monthly Returns (%)")
            ChartStyler.add_reference_lines(fig, [0], line_color="red")
        
        fig.update_xaxes(title="Date")
        return fig
    
    def create_correlation_matrix(self, selected_indices=None):
        """Create correlation matrix for selected indices"""
        if selected_indices is None:
            selected_indices = self.available_indices
        
        if len(selected_indices) < 2:
            print("❌ Need at least 2 indices for correlation analysis")
            return None
        
        print(f"📊 Creating correlation matrix for: {', '.join(selected_indices)}")
        
        # Prepare data for correlation
        correlation_data = {}
        
        for index_name in selected_indices:
            if index_name in self.market_data:
                data = self.market_data[index_name].copy()
                
                # Calculate returns for correlation
                data_with_returns = AnalysisUtils.calculate_returns(data.rename(columns={'close': 'value'}), periods=1)
                correlation_data[NameFormatter.format_display_name(index_name)] = data_with_returns.set_index('date')['growth_rate']
        
        if len(correlation_data) < 2:
            print("❌ Insufficient data for correlation analysis")
            return None
        
        # Create correlation DataFrame
        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        ChartStyler.apply_default_layout(
            fig,
            title="📊 Market Indices Correlation Matrix (Monthly Returns)",
            height=600
        )
        fig.update_xaxes(title="Market Index")
        fig.update_yaxes(title="Market Index")
        
        return fig
    
    def get_summary_data(self):
        """Get summary data for all indices"""
        summary_data = []
        
        # Load source URLs
        latest_file = DataLoader.find_latest_file(MARKET_INDICES_PATTERN, self.data_dir)
        source_urls = SourceUrlLoader.load_source_urls_from_excel(str(latest_file)) if latest_file else {}
        
        for index_name in self.available_indices:
            if index_name in self.market_data:
                data = self.market_data[index_name]
                display_name = NameFormatter.format_display_name(index_name)
                
                # Get source URL
                source_url = SourceUrlLoader.get_source_url_for_index(index_name, source_urls)
                
                summary_data.append({
                    'Index Name': display_name,
                    'Country': data['country'].iloc[0] if 'country' in data.columns else 'Unknown',
                    'Last Closing Date': data['date'].iloc[-1].strftime('%Y-%m-%d'),
                    'Last Closing Price': f"{data['close'].iloc[-1]:,.2f}",
                    'Records': len(data),
                    'Source URL': source_url
                })
        
        return summary_data
    
    def get_performance_summary_data(self):
        """Get performance summary with accurate cumulative returns"""
        summary_data = []
        
        for index_name in self.available_indices:
            if index_name in self.market_data:
                data = self.market_data[index_name].copy().sort_values('date')
                display_name = NameFormatter.format_display_name(index_name)
                
                # Latest values
                latest_price = data['close'].iloc[-1]
                latest_date = data['date'].iloc[-1]
                
                # Calculate cumulative returns (price changes over periods)
                def calculate_period_return(months_back):
                    target_date = latest_date - pd.DateOffset(months=months_back)
                    period_data = data[data['date'] <= target_date]
                    if len(period_data) > 0:
                        period_price = period_data['close'].iloc[-1]
                        return ((latest_price - period_price) / period_price) * 100
                    return 0
                
                # Calculate returns for different periods
                monthly_return = calculate_period_return(1)
                three_month_return = calculate_period_return(3)
                six_month_return = calculate_period_return(6)
                
                # YTD return
                current_year = latest_date.year
                ytd_data = data[data['date'].dt.year == current_year]
                if len(ytd_data) > 0:
                    ytd_start_price = ytd_data['close'].iloc[0]
                    ytd_return = ((latest_price - ytd_start_price) / ytd_start_price) * 100
                else:
                    ytd_return = 0
                
                summary_data.append({
                    'Index': display_name,
                    'Country': data['country'].iloc[0] if 'country' in data.columns else 'Unknown',
                    'Latest Price': f"{latest_price:,.2f}",
                    'Latest Date': latest_date.strftime('%Y-%m-%d'),
                    'Monthly Return (%)': f"{monthly_return:.2f}%",
                    '3-Month Return (%)': f"{three_month_return:.2f}%",
                    '6-Month Return (%)': f"{six_month_return:.2f}%",
                    'YTD Return (%)': f"{ytd_return:.2f}%"
                })
        
        return summary_data
    
    def print_summary_table(self):
        """Print summary table with latest values"""
        summary_data = self.get_summary_data()
        
        if summary_data:
            print(f"\n📊 MARKET INDICES SUMMARY")
            print("=" * 80)
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))
            print("=" * 80)
    
    def interactive_menu(self):
        """Interactive menu for market visualization options"""
        while True:
            print(f"\n📈 MARKET INDICES VISUALIZATION OPTIONS")
            print("=" * 50)
            print("1. Market Overview Dashboard")
            print("2. Individual Index - Price Chart")
            print("3. Individual Index - Returns Chart")
            print("4. Correlation Matrix")
            print("5. Show Summary Table")
            print("6. Show Performance Summary")
            print("7. Exit")
            
            choice = input(f"\nSelect option (1-7): ").strip()
            
            try:
                if choice == '1':
                    # Market Overview
                    fig = self.create_market_overview()
                    if fig:
                        ChartDisplayer.show_chart(fig, "market_overview")
                        print("✅ Market overview displayed!")
                    else:
                        print("❌ Could not create overview")
                
                elif choice in ['2', '3']:
                    # Individual Index Charts
                    print(f"\nAvailable indices:")
                    display_indices = [NameFormatter.format_display_name(idx) for idx in self.available_indices]
                    for i, display_name in enumerate(display_indices, 1):
                        print(f"{i}. {display_name}")
                    
                    try:
                        idx_choice = input(f"Select index (1-{len(display_indices)}): ").strip()
                        idx_num = int(idx_choice)
                        
                        if 1 <= idx_num <= len(display_indices):
                            index_name = self.available_indices[idx_num - 1]
                            chart_type = "price" if choice == '2' else "returns"
                            
                            fig = self.create_individual_chart(index_name, chart_type)
                            if fig:
                                ChartDisplayer.show_chart(fig, f"{index_name}_{chart_type}")
                                print(f"✅ {NameFormatter.format_display_name(index_name)} {chart_type} chart displayed!")
                            else:
                                print("❌ Could not create chart")
                        else:
                            print("❌ Invalid choice")
                    except ValueError:
                        print("❌ Invalid input")
                
                elif choice == '4':
                    # Correlation Matrix
                    fig = self.create_correlation_matrix()
                    if fig:
                        ChartDisplayer.show_chart(fig, "market_correlation")
                        print("✅ Correlation matrix displayed!")
                    else:
                        print("❌ Could not create correlation matrix")
                
                elif choice == '5':
                    # Summary Table
                    self.print_summary_table()
                
                elif choice == '6':
                    # Performance Summary
                    perf_data = self.get_performance_summary_data()
                    if perf_data:
                        print(f"\n📊 MARKET INDICES PERFORMANCE SUMMARY")
                        print("=" * 120)
                        perf_df = pd.DataFrame(perf_data)
                        print(perf_df.to_string(index=False))
                        print("=" * 120)
                    else:
                        print("❌ No performance data available")
                
                elif choice == '7':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-7.")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run(self):
        """Main execution function"""
        if not self.available_indices:
            print("❌ No market data available. Please check your data files.")
            return
        
        print(f"\n🎯 Ready for market indices analysis:")
        print(f"   Indices: {', '.join([NameFormatter.format_display_name(idx) for idx in self.available_indices])}")
        
        # Show summary first
        self.print_summary_table()
        
        # Start interactive menu
        self.interactive_menu()

def main():
    """Main function"""
    print("📈 MARKET INDICES VISUALIZER")
    print("=" * 40)
    
    try:
        visualizer = MarketIndicesVisualizer()
        visualizer.run()
    except KeyboardInterrupt:
        print("\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Please check your data files and try again")

if __name__ == "__main__":
    main()