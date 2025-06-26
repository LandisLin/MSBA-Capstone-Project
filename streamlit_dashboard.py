"""
Economic Data Dashboard - Streamlit
Interactive web dashboard for economic and market data visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Import shared utilities and visualization classes
from visualization_config import (
    COUNTRIES,
    MARKET_INDICES_PATTERN,
    DataLoader,
    AnalysisUtils,
    ChartStyler,
    NameFormatter,
    SourceUrlLoader,
    load_country_data,
    load_market_data
)

# Page configuration
st.set_page_config(
    page_title="Economic Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with fixed UI issues
def setup_css():
    """Setup improved CSS with enhanced animations and smooth transitions"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .market-ticker-card {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            background-color: #ffffff;
            color: #262730;
            border: 2px solid #e0e0e0;
            /* ENHANCED: Add smooth hover animations */
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: slideInFromTop 0.6s ease-out;
        }
        
        .market-ticker-card:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        [data-theme="dark"] .market-ticker-card {
            background-color: #262730 !important;
            color: #fafafa !important;
            border: 2px solid #404040 !important;
        }
        
        @media (prefers-color-scheme: dark) {
            .market-ticker-card {
                background-color: #262730;
                color: #fafafa;
                border: 2px solid #404040;
            }
        }
        
        .big-metric {
            text-align: center;
            padding: 1rem;
            /* ENHANCED: Add scale and hover animations */
            animation: scaleInMetric 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            transition: transform 0.3s ease;
        }
        
        .big-metric:hover {
            transform: scale(1.05);
        }
        
        .big-metric .metric-label {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }
        
        .big-metric .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .centered-title {
            text-align: center;
        }
        
        .last-updated {
            font-size: 0.9rem;
            color: #6c757d;
            font-style: italic;
            margin-bottom: 1rem;
        }
        
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1.5rem;
            color: var(--text-color);
        }
        
        .sidebar-nav-item {
            font-size: 1.1rem;
            font-weight: 500;
            margin: 0.5rem 0;
        }
        
        .sidebar-main-status {
            font-size: 1.4rem;
            font-weight: bold;
            margin: 1.5rem 0 1rem 0;
            color: var(--text-color);
        }
        
        .sidebar-section {
            font-size: 1.2rem;
            font-weight: 600;
            margin: 1rem 0 0.5rem 0;
            color: var(--text-color);
        }
        
        .sidebar-item {
            font-size: 0.95rem;
            margin: 0.2rem 0;
            padding-left: 1rem;
            color: var(--text-color);
        }
        
        .sidebar-divider {
            margin: 1.5rem 0;
            border-top: 1px solid var(--border-color, #e0e0e0);
        }
        
        /* ENHANCED: Better sidebar button spacing and animations */
        div[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] div[data-testid="stButton"] {
            margin: 0.4rem 0 !important;
        }
        
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            width: 100%;
            text-align: center !important;
            justify-content: center !important;
            padding: 0.75rem 1rem !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            /* ENHANCED: Smoother transition with better easing */
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: 1px solid #e0e0e0 !important;
            background-color: transparent !important;
            color: #666 !important;
            margin: 0 !important;
            transform: translateX(0) !important;
        }
        
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background-color: #f0f8ff !important;
            color: #007acc !important;
            border-color: #007acc !important;
            /* ENHANCED: More pronounced hover effect */
            transform: translateX(5px) scale(1.02) !important;
            box-shadow: 0 4px 15px rgba(0, 122, 204, 0.2) !important;
        }
        
        /* Active page styling through custom class */
        .sidebar-active-button {
            background: linear-gradient(90deg, #007acc, #005999) !important;
            color: white !important;
            font-weight: bold !important;
            border: 1px solid #007acc !important;
            box-shadow: 0 2px 8px rgba(0, 122, 204, 0.3) !important;
        }
        
        .sidebar-active-button:hover {
            background: linear-gradient(90deg, #005999, #007acc) !important;
            color: white !important;
            transform: none !important;
        }
        
        /* Dark theme support */
        [data-theme="dark"] div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background-color: transparent !important;
            color: #ccc !important;
            border-color: #404040 !important;
        }
        
        [data-theme="dark"] div[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background-color: #333 !important;
            color: #007acc !important;
        }
        
        /* ENHANCED: Much smoother page transitions with blur and scale effects */
        .page-content {
            animation: enhancedSlideIn 0.9s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            transform-origin: center;
        }
        
        @keyframes enhancedSlideIn {
            0% { 
                opacity: 0; 
                transform: translateY(25px) scale(0.95);
                filter: blur(3px);
            }
            30% {
                opacity: 0.4;
                transform: translateY(15px) scale(0.97);
                filter: blur(2px);
            }
            60% {
                opacity: 0.8;
                transform: translateY(5px) scale(0.99);
                filter: blur(1px);
            }
            100% { 
                opacity: 1; 
                transform: translateY(0) scale(1);
                filter: blur(0);
            }
        }
        
        /* ENHANCED: More sophisticated staggered content animations */
        .stMarkdown, .stDataFrame, .stPlotlyChart, .stSelectbox, .stMultiSelect, .stRadio {
            animation: fadeInContent 0.7s ease-out;
            animation-fill-mode: both;
            opacity: 0;
        }
        
        /* ENHANCED: More staggered delays for smoother sequential appearance */
        .stMarkdown:nth-child(1) { animation-delay: 0.1s; }
        .stMarkdown:nth-child(2) { animation-delay: 0.2s; }
        .stMarkdown:nth-child(3) { animation-delay: 0.3s; }
        .stMarkdown:nth-child(4) { animation-delay: 0.4s; }
        .stSelectbox { animation-delay: 0.5s; }
        .stMultiSelect { animation-delay: 0.6s; }
        .stRadio { animation-delay: 0.7s; }
        .stDataFrame { animation-delay: 0.8s; }
        .stPlotlyChart { animation-delay: 0.9s; }
        
        @keyframes fadeInContent {
            from { 
                opacity: 0; 
                transform: translateY(15px) scale(0.98);
                filter: blur(1px);
            }
            to { 
                opacity: 1; 
                transform: translateY(0) scale(1);
                filter: blur(0);
            }
        }
        
        /* ENHANCED: New animation types for different elements */
        @keyframes slideInFromTop {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes scaleInMetric {
            from {
                opacity: 0;
                transform: scale(0.8) rotate(-2deg);
            }
            50% {
                opacity: 0.7;
                transform: scale(1.05) rotate(1deg);
            }
            to {
                opacity: 1;
                transform: scale(1) rotate(0deg);
            }
        }
        
        @keyframes slideInFromLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideInFromRight {
            from {
                opacity: 0;
                transform: translateX(30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* ENHANCED: Smoother loading indicators with pulsing effect */
        .stSpinner {
            animation: enhancedPulse 2s ease-in-out infinite;
        }
        
        @keyframes enhancedPulse {
            0%, 100% { 
                opacity: 1; 
                transform: scale(1);
            }
            25% {
                opacity: 0.8;
                transform: scale(1.02);
            }
            50% { 
                opacity: 0.6; 
                transform: scale(1.05);
            }
            75% {
                opacity: 0.8;
                transform: scale(1.02);
            }
        }
        
        /* ENHANCED: Smooth animations for different column layouts */
        div[data-testid="column"]:nth-child(1) {
            animation: slideInFromLeft 0.8s ease-out;
        }
        
        div[data-testid="column"]:nth-child(2) {
            animation: slideInFromRight 0.8s ease-out;
            animation-delay: 0.1s;
            animation-fill-mode: both;
        }
        
        div[data-testid="column"]:nth-child(3) {
            animation: slideInFromLeft 0.8s ease-out;
            animation-delay: 0.2s;
            animation-fill-mode: both;
        }
        
        div[data-testid="column"]:nth-child(4) {
            animation: slideInFromRight 0.8s ease-out;
            animation-delay: 0.3s;
            animation-fill-mode: both;
        }
        
        /* ENHANCED: Smooth button animations throughout the app */
        div[data-testid="stButton"] > button {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        }
        
        /* ENHANCED: Smooth dataframe animations */
        div[data-testid="stDataFrame"] {
            animation: slideInFromBottom 0.8s ease-out;
            animation-delay: 0.4s;
            animation-fill-mode: both;
        }
        
        @keyframes slideInFromBottom {
            from {
                opacity: 0;
                transform: translateY(25px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* ENHANCED: Plotly chart smooth entrance */
        div[data-testid="stPlotlyChart"] {
            animation: scaleInChart 1s ease-out;
            animation-delay: 0.5s;
            animation-fill-mode: both;
        }
        
        @keyframes scaleInChart {
            from {
                opacity: 0;
                transform: scale(0.9) rotateX(10deg);
            }
            50% {
                opacity: 0.7;
                transform: scale(1.02) rotateX(5deg);
            }
            to {
                opacity: 1;
                transform: scale(1) rotateX(0deg);
            }
        }
    </style>
    """, unsafe_allow_html=True)

def detect_update_frequency(df, date_column='date'):
    """
    Detect update frequency based on time intervals between consecutive data points
    
    Args:
        df: DataFrame with date column
        date_column: Name of the date column (default: 'date')
    
    Returns:
        str: Detected frequency ('Daily', 'Weekly', 'Monthly', 'Quarterly', 'Annual', 'Irregular', 'Insufficient Data')
    """
    
    if df is None or len(df) < 2:
        return "Insufficient Data"
    
    if date_column not in df.columns:
        return "No Date Column"
    
    try:
        # Convert to datetime and sort
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_sorted = df_copy.sort_values(date_column)
        
        # Calculate differences between consecutive dates
        date_diffs = df_sorted[date_column].diff().dropna()
        
        if len(date_diffs) == 0:
            return "Insufficient Data"
        
        # Get median difference (more robust than mean for irregular data)
        median_diff = date_diffs.median()
        
        # Convert to days for easier comparison
        median_days = median_diff.total_seconds() / (24 * 3600)
        
        # Classify based on median interval
        if median_days <= 1.5:
            return "Daily"
        elif median_days <= 8:
            return "Weekly"
        elif median_days <= 35:  # ~1 month with some tolerance
            return "Monthly"
        elif median_days <= 100:  # ~3 months with tolerance
            return "Quarterly"
        elif median_days <= 400:  # ~1 year with tolerance
            return "Annual"
        else:
            return "Irregular"
            
    except Exception as e:
        return "Unknown"

# Add this function to create the enhanced sidebar
def create_enhanced_sidebar_navigation(data_sources):
    """Create enhanced sidebar with better spacing and centering"""
    
    st.sidebar.markdown('<div class="sidebar-header">📊 Navigation</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-nav-item">Select Page</div>', unsafe_allow_html=True)
    
    # Get current page
    current_page = st.session_state.get('selected_page', 'overview')
    
    # Define page options
    page_configs = [
        {'key': 'overview', 'icon': '🏠', 'title': 'Overview', 'description': 'Dashboard home'},
        {'key': 'country', 'icon': '🌍', 'title': 'Country Analysis', 'description': 'Economic indicators by country'},
        {'key': 'cross_country', 'icon': '🔗', 'title': 'Cross-Country', 'description': 'Compare across countries'},
        {'key': 'market_indices', 'icon': '📈', 'title': 'Market Indices', 'description': 'Global market analysis'}
    ]
    
    # Create navigation buttons with even spacing
    for page_config in page_configs:
        is_active = current_page == page_config['key']
        
        if is_active:
            # Active page - show as highlighted info box
            st.sidebar.info(f"**{page_config['icon']} {page_config['title']}**\n\n{page_config['description']}")
        else:
            # Inactive page - clickable button
            button_label = f"{page_config['icon']} {page_config['title']}"
            
            if st.sidebar.button(
                button_label,
                key=f"nav_{page_config['key']}",
                help=page_config['description'],
                use_container_width=True
            ):
                st.session_state.selected_page = page_config['key']
                if page_config['key'] != "country":
                    st.session_state.selected_country = None
                st.rerun()

# Add this function to wrap page content with smooth animations
def create_smooth_page_container():
    """Create smooth animated container for page content"""
    st.markdown('<div class="page-content">', unsafe_allow_html=True)

def close_smooth_page_container():
    """Close smooth animated page container"""
    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data
def load_data_sources():
    """Load all data sources with caching"""
    return {
        'countries': DataLoader.get_available_countries(),
        'market_available': DataLoader.find_latest_file(MARKET_INDICES_PATTERN) is not None,
        'country_files': {
            country: DataLoader.find_latest_country_file(country)
            for country in DataLoader.get_available_countries()
        }
    }

def get_extraction_time(file_path):
    """Get extraction time from Excel file"""
    if not file_path:
        return "Unknown"
    
    try:
        df = pd.read_excel(file_path, sheet_name='Contents', skiprows=2)
        if 'Last Extraction Time' in df.columns:
            times = df['Last Extraction Time'].dropna()
            if len(times) > 0:
                return times.iloc[0]
    except:
        pass
    
    return "Unknown"

def create_market_ticker(market_viz):
    """Create market ticker with improved card visibility"""
    if not market_viz or not market_viz.market_data:
        return
    
    st.subheader("📈 Today's Market Updates")
    
    ticker_data = []
    for index_name, data in market_viz.market_data.items():
        if len(data) < 2:
            continue
        
        latest_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        change_pct = ((latest_price - prev_price) / prev_price) * 100
        
        ticker_data.append({
            'name': NameFormatter.format_display_name(index_name),
            'price': latest_price,
            'change_pct': change_pct,
            'date': data['date'].iloc[-1].strftime('%Y-%m-%d')
        })
    
    if ticker_data:
        cols = st.columns(min(len(ticker_data), 4))
        
        for i, ticker in enumerate(ticker_data):
            with cols[i % 4]:
                change_color = "#28a745" if ticker['change_pct'] >= 0 else "#dc3545"
                symbol = "+" if ticker['change_pct'] >= 0 else ""
                
                st.markdown(f"""
                <div class="market-ticker-card">
                    <div style="font-weight: bold; font-size: 1rem; margin-bottom: 8px;">
                        {ticker['name']}
                    </div>
                    <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 5px;">
                        {ticker['price']:,.2f}
                    </div>
                    <div style="color: {change_color}; font-weight: bold; font-size: 1rem;">
                        ({symbol}{ticker['change_pct']:.2f}%)
                    </div>
                    <div style="font-size: 0.85rem; opacity: 0.7; margin-top: 5px;">
                        {ticker['date']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

def create_overview_page(data_sources):
    """Create main overview page with fixed navigation"""
    st.markdown('<h1 class="main-header">📊 Economic Data Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Enhanced metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="big-metric">
            <div class="metric-label">🌍 Countries</div>
            <div class="metric-value">{len(data_sources['countries'])}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        market_count = len(load_market_data().keys()) if data_sources['market_available'] else 0
        st.markdown(f"""
        <div class="big-metric">
            <div class="metric-label">📈 Market Indices</div>
            <div class="metric-value">{market_count}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="big-metric">
            <div class="metric-label">📊 Indicators</div>
            <div class="metric-value">5</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="big-metric">
            <div class="metric-label">📅 Data Coverage</div>
            <div class="metric-value">2000-2025</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Market ticker
    if data_sources['market_available']:
        try:
            from market_indices_visualizer import MarketIndicesVisualizer
            market_viz = MarketIndicesVisualizer()
            create_market_ticker(market_viz)
        except:
            pass
    
    st.markdown("---")
    
    # Quick access buttons - FIXED NAVIGATION
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Countries")
        for country in data_sources['countries']:
            country_name = NameFormatter.format_country_name(country)
            # FIXED: Properly navigate to country page AND set selected country
            if st.button(country_name, key=f"country_{country}", use_container_width=True):
                st.session_state.selected_page = "country"  # Navigate to country page
                st.session_state.selected_country = country  # Set selected country
                st.rerun()
    
    with col2:
        st.subheader("📈 Analysis Tools")
        if st.button("Cross-Country Comparison", use_container_width=True):
            st.session_state.selected_page = "cross_country"
            st.rerun()
        if st.button("Market Indices Analysis", use_container_width=True):
            st.session_state.selected_page = "market_indices"
            st.rerun()

def create_country_overview_chart(data, country_key):
    """Create country overview chart using shared utilities"""
    if not data:
        return None
    
    country_name = NameFormatter.format_country_name(country_key)
    
    # Create 4x2 subplot layout
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "GDP Level", "GDP Growth (%)",
            "CPI Level", "Inflation (%)",
            "Population", "Population Growth (%)",
            "Interest Rate (%)", "Property Price Index"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Helper to get main series
    def get_main_data(indicator):
        if indicator not in data:
            return None
        df = data[indicator]
        if 'series' in df.columns:
            main_series = df['series'].iloc[0]
            return df[df['series'] == main_series]
        return df
    
    # Add charts for each indicator
    indicators_config = [
        ('GDP', 1, 1, 4, 'blue', 'darkgreen'),
        ('CPI', 2, 1, 4, 'orange', 'red'),
        ('Population', 3, 1, 1, 'purple', 'darkviolet'),
        ('Interest_Rate', 4, 1, None, 'green', None),
        ('Property_Price', 4, 2, None, 'brown', None)
    ]
    
    for indicator, row, col, growth_periods, color, growth_color in indicators_config:
        indicator_data = get_main_data(indicator)
        
        if indicator_data is not None and not indicator_data.empty:
            # Level chart
            fig.add_trace(go.Scatter(
                x=indicator_data['date'], y=indicator_data['value'],
                mode='lines', name=indicator,
                line=dict(width=2, color=color)
            ), row=row, col=col)
            
            # Growth chart (if applicable)
            if growth_periods and col == 1 and growth_color:
                growth_data = AnalysisUtils.calculate_growth_rates(indicator_data, periods=growth_periods)
                growth_clean = growth_data.dropna(subset=['growth_rate'])
                
                if not growth_clean.empty:
                    fig.add_trace(go.Scatter(
                        x=growth_clean['date'], y=growth_clean['growth_rate'],
                        mode='lines', name=f'{indicator} Growth',
                        line=dict(width=2, color=growth_color)
                    ), row=row, col=2)
                    
                    # Add reference lines
                    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7, row=row, col=2)
                    if indicator == 'CPI':
                        fig.add_hline(y=2, line_dash="dash", line_color="green", opacity=0.7, row=row, col=2)
    
    # Apply styling
    ChartStyler.apply_default_layout(
        fig,
        title=f"{country_name} Economic Overview",
        height=1200
    )
    fig.update_layout(showlegend=False)
    
    return fig

def create_country_page(data_sources):
    """Create country analysis page with FIXED single-click country selection"""
    st.header("🏠 Country Economic Analysis")
    
    if not data_sources['countries']:
        st.warning("No country data available.")
        return
    
    # FIXED: Simplified country selection logic
    # Initialize selected_country if not exists
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = data_sources['countries'][0]
    
    # Ensure selected country is valid
    if st.session_state.selected_country not in data_sources['countries']:
        st.session_state.selected_country = data_sources['countries'][0]
    
    # Get current index
    try:
        current_index = data_sources['countries'].index(st.session_state.selected_country)
    except ValueError:
        current_index = 0
        st.session_state.selected_country = data_sources['countries'][0]
    
    # Country selection with automatic update
    selected_country = st.selectbox(
        "Select Country",
        options=data_sources['countries'],
        index=current_index,
        format_func=NameFormatter.format_country_name,
        key="country_selector_fixed"  # Different key to avoid conflicts
    )
    
    # FIXED: Only update session state if country actually changed
    if selected_country != st.session_state.selected_country:
        st.session_state.selected_country = selected_country
        st.rerun()  # Force immediate rerun when country changes
    
    # Now process the selected country
    if selected_country:
        # Show extraction time
        country_file = data_sources['country_files'].get(selected_country)
        if country_file:
            extraction_time = get_extraction_time(country_file)
            if extraction_time != "Unknown":
                try:
                    formatted_time = pd.to_datetime(extraction_time).strftime("%Y-%m-%d %H:%M")
                    st.markdown(f'<div class="last-updated">Last updated: {formatted_time}</div>', 
                              unsafe_allow_html=True)
                except:
                    st.markdown(f'<div class="last-updated">Last updated: {extraction_time}</div>', 
                              unsafe_allow_html=True)
        
        with st.spinner(f"Loading {NameFormatter.format_country_name(selected_country)} data..."):
            # Load country data
            country_data = load_country_data(selected_country)
            
            if not country_data:
                st.error(f"Failed to load data for {selected_country}")
                return
            
            # Load source URLs
            source_urls = SourceUrlLoader.load_source_urls_from_excel(country_file) if country_file else {}
            
            # Data summary
            st.subheader("📊 Data Summary")
            summary_data = []
            
            for indicator, df in country_data.items():
                if not df.empty:
                    date_range = f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"
                    source_url = source_urls.get(indicator, "N/A")
                    update_frequency = detect_update_frequency(df)  # NEW: Detect frequency
                    summary_data.append({
                        'Indicator': indicator,
                        'Records': str(len(df)),
                        'Update Frequency': update_frequency,
                        'Date Range': date_range,
                        'Source URL': source_url
                    })
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # Visualization options
            st.subheader("📈 Visualizations")
            viz_type = st.radio("Choose type:", ["Economic Overview", "Individual Indicator"], horizontal=True)
            
            if viz_type == "Economic Overview":
                st.markdown(f'<h3 class="centered-title">🌍 {NameFormatter.format_country_name(selected_country)} Economic Overview</h3>', 
                          unsafe_allow_html=True)
                
                try:
                    fig = create_country_overview_chart(country_data, selected_country)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not generate overview chart")
                except Exception as e:
                    st.error(f"Error creating overview: {e}")
            
            else:  # Individual Indicator
                selected_indicator = st.selectbox("Select Indicator", list(country_data.keys()))
                
                if selected_indicator:
                    df = country_data[selected_indicator]
                    
                    fig = go.Figure()
                    
                    # Handle multiple series
                    if 'series' in df.columns and len(df['series'].unique()) > 1:
                        colors = ChartStyler.get_qualitative_colors()
                        for i, series in enumerate(df['series'].unique()):
                            series_data = df[df['series'] == series]
                            fig.add_trace(go.Scatter(
                                x=series_data['date'], y=series_data['value'],
                                mode='lines', name=series,
                                line=dict(width=2, color=colors[i % len(colors)])
                            ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=df['date'], y=df['value'],
                            mode='lines', name=selected_indicator,
                            line=dict(width=3)
                        ))
                    
                    ChartStyler.apply_default_layout(
                        fig,
                        title=f"{NameFormatter.format_country_name(selected_country)} - {selected_indicator}"
                    )
                    fig.update_xaxes(title="Date")
                    fig.update_yaxes(title="Value")
                    
                    st.plotly_chart(fig, use_container_width=True)

def create_cross_country_page(data_sources):
    """Create cross-country comparison page"""
    st.header("🔗 Cross-Country Economic Comparison")
    
    if not data_sources['countries']:
        st.warning("Cross-country data not available.")
        return
    
    try:
        from cross_country_visualizer import CrossCountryVisualizer
        cross_viz = CrossCountryVisualizer()
        
        viz_type = st.radio("Choose type:", ["Overview Dashboard", "Individual Indicator", "Summary Table"], horizontal=True)
        
        if viz_type == "Overview Dashboard":
            st.markdown('<h3 class="centered-title">📊 Multi-Country Economic Overview</h3>', unsafe_allow_html=True)
            
            selected_countries = st.multiselect(
                "Select Countries (leave empty for all)",
                options=[NameFormatter.format_country_name(c) for c in cross_viz.available_countries],
                default=[NameFormatter.format_country_name(c) for c in cross_viz.available_countries]
            )
            
            # Convert display names back to internal keys
            selected_internal = []
            for display_name in selected_countries:
                for internal_key in cross_viz.available_countries:
                    if NameFormatter.format_country_name(internal_key) == display_name:
                        selected_internal.append(internal_key)
                        break
            
            if selected_internal:
                with st.spinner("Creating dashboard..."):
                    try:
                        fig = cross_viz.create_overview_dashboard(selected_internal)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        elif viz_type == "Individual Indicator":
            st.markdown('<h3 class="centered-title">📈 Cross-Country Indicator Comparison</h3>', unsafe_allow_html=True)
            
            selected_indicator = st.selectbox("Select Indicator", cross_viz.common_indicators)
            
            if selected_indicator:
                with st.spinner(f"Creating {selected_indicator} comparison..."):
                    try:
                        fig = cross_viz.create_individual_chart(selected_indicator)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        else:  # Summary Table
            st.subheader("📋 Latest Economic Indicators Summary")
            try:
                summary_df = cross_viz.create_summary_table()
                if summary_df is not None:
                    st.dataframe(summary_df, use_container_width=True)
                else:
                    st.warning("No summary data available")
            except Exception as e:
                st.error(f"Error: {e}")
                
    except ImportError:
        st.error("Cross-country visualizer not available")

def create_market_page(data_sources):
    """Create market indices page"""
    st.header("📈 Market Indices Analysis")
    
    if not data_sources['market_available']:
        st.warning("Market indices data not available.")
        return
    
    try:
        from market_indices_visualizer import MarketIndicesVisualizer
        market_viz = MarketIndicesVisualizer()
        
        # Show extraction time
        latest_file = DataLoader.find_latest_file(MARKET_INDICES_PATTERN)
        if latest_file:
            extraction_time = get_extraction_time(str(latest_file))
            if extraction_time != "Unknown":
                try:
                    formatted_time = pd.to_datetime(extraction_time).strftime("%Y-%m-%d %H:%M")
                    st.markdown(f'<div class="last-updated">Last updated: {formatted_time}</div>', unsafe_allow_html=True)
                except:
                    st.markdown(f'<div class="last-updated">Last updated: {extraction_time}</div>', unsafe_allow_html=True)
        
        # Market summary
        st.subheader("📊 Market Indices Summary")
        try:
            summary_data = market_viz.get_summary_data()
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        except Exception as e:
            st.error(f"Error creating summary: {e}")
        
        # Visualization options
        viz_type = st.radio("Choose type:", ["Market Overview", "Individual Index", "Correlation Analysis", "Performance Summary"], horizontal=True)
        
        if viz_type == "Market Overview":
            st.markdown('<h3 class="centered-title">📊 Market Indices Overview</h3>', unsafe_allow_html=True)
            
            selected_indices = st.multiselect(
                "Select Indices (leave empty for all)",
                options=[NameFormatter.format_display_name(idx) for idx in market_viz.available_indices],
                default=[NameFormatter.format_display_name(idx) for idx in market_viz.available_indices]
            )
            
            # Convert display names back to internal names
            selected_internal = []
            for display_name in selected_indices:
                for internal_name in market_viz.available_indices:
                    if NameFormatter.format_display_name(internal_name) == display_name:
                        selected_internal.append(internal_name)
                        break
            
            if selected_internal:
                with st.spinner("Creating overview..."):
                    try:
                        fig = market_viz.create_market_overview(selected_internal)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            st.info("📊 **Chart Explanation**: Price levels are normalized (base 100 = starting price) for performance comparison.")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        elif viz_type == "Individual Index":
            st.subheader("📈 Individual Index Analysis")
            
            selected_index_display = st.selectbox("Select Index", [NameFormatter.format_display_name(idx) for idx in market_viz.available_indices])
            
            # Find internal name
            selected_index = None
            for internal_name in market_viz.available_indices:
                if NameFormatter.format_display_name(internal_name) == selected_index_display:
                    selected_index = internal_name
                    break
            
            if selected_index:
                chart_type = st.radio("Chart Type:", ["Price Chart", "Returns Chart"], horizontal=True)
                chart_type_param = "price" if chart_type == "Price Chart" else "returns"
                
                with st.spinner(f"Creating {chart_type.lower()}..."):
                    try:
                        fig = market_viz.create_individual_chart(selected_index, chart_type_param)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        elif viz_type == "Correlation Analysis":
            st.markdown('<h3 class="centered-title">🔗 Market Indices Correlation</h3>', unsafe_allow_html=True)
            
            selected_indices_display = st.multiselect(
                "Select Indices for Correlation (minimum 2)",
                options=[NameFormatter.format_display_name(idx) for idx in market_viz.available_indices],
                default=[NameFormatter.format_display_name(idx) for idx in market_viz.available_indices]
            )
            
            # Convert to internal names
            selected_internal = []
            for display_name in selected_indices_display:
                for internal_name in market_viz.available_indices:
                    if NameFormatter.format_display_name(internal_name) == display_name:
                        selected_internal.append(internal_name)
                        break
            
            if len(selected_internal) >= 2:
                with st.spinner("Creating correlation matrix..."):
                    try:
                        fig = market_viz.create_correlation_matrix(selected_internal)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Please select at least 2 indices for correlation analysis")
        
        elif viz_type == "Performance Summary":
            st.markdown('<h3 class="centered-title">📊 Market Performance Summary</h3>', unsafe_allow_html=True)
            
            with st.spinner("Loading performance data..."):
                try:
                    performance_data = market_viz.get_performance_summary_data()
                    if performance_data:
                        perf_df = pd.DataFrame(performance_data)
                        st.dataframe(perf_df, use_container_width=True)
                        
                        st.info("""
                        📊 **Performance Explanation**:
                        - **Monthly Return**: Price change from 1 month ago
                        - **3-Month Return**: Price change from 3 months ago  
                        - **6-Month Return**: Price change from 6 months ago
                        - **YTD Return**: Price change since beginning of current year
                        """)
                        
                        # Download button
                        csv = perf_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Performance Data",
                            data=csv,
                            file_name=f"market_performance_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No performance data available")
                except Exception as e:
                    st.error(f"Error loading performance data: {e}")
                    
    except ImportError:
        st.error("Market indices visualizer not available")

def main():
    """Main Streamlit application with fixed navigation"""
    # Initialize session state
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = 'overview'
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = None
    
    # Setup improved CSS
    setup_css()
    
    # Load data sources
    with st.spinner("Loading data sources..."):
        data_sources = load_data_sources()
    
    # Create enhanced sidebar navigation
    create_enhanced_sidebar_navigation(data_sources)
    
    # Enhanced data status section
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-main-status">📊 Data Available</div>', unsafe_allow_html=True)
    
    countries_count = len(data_sources['countries'])
    if countries_count > 0:
        st.sidebar.markdown('<div class="sidebar-section">🌍 Countries</div>', unsafe_allow_html=True)
        for country in data_sources['countries']:
            country_display = NameFormatter.format_country_name(country)
            st.sidebar.markdown(f'<div class="sidebar-item">• {country_display}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.error("❌ No country data")
    
    if data_sources['market_available']:
        st.sidebar.markdown('<div class="sidebar-section">📈 Market Indices</div>', unsafe_allow_html=True)
        try:
            # Load market data to show available indices
            market_data = load_market_data()
            if market_data:
                for index_name in sorted(market_data.keys()):
                    display_name = NameFormatter.format_display_name(index_name)
                    st.sidebar.markdown(f'<div class="sidebar-item">• {display_name}</div>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown('<div class="sidebar-item">• Loading...</div>', unsafe_allow_html=True)
        except:
            st.sidebar.markdown('<div class="sidebar-item">• Market indices available</div>', unsafe_allow_html=True)
    else:
        st.sidebar.error("❌ No market data")
    
    # Render selected page
    page_key = st.session_state.selected_page
    
    # Create smooth animated container
    create_smooth_page_container()

    if page_key == "overview":
        create_overview_page(data_sources)
    elif page_key == "country":
        create_country_page(data_sources)
    elif page_key == "cross_country":
        create_cross_country_page(data_sources)
    elif page_key == "market_indices":
        create_market_page(data_sources)

    # Close smooth animated container
    close_smooth_page_container()

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.8rem;">'
        'Economic Data Dashboard | Sources: Singapore Data.gov.sg, FRED API, Yahoo Finance'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
