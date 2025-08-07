"""
Economic Data Dashboard - Streamlit (Fixed Navigation Version)
Interactive web dashboard for economic and market data visualization
Fixed: Country navigation and page state management
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
import hashlib
import os
import glob
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

# ============================================================================
# Cache
# ============================================================================
# 1. Add file hash-based caching for automatic cache invalidation
def get_file_hash(file_path):
    """Get hash of file for cache invalidation"""
    try:
        if not file_path or not os.path.exists(file_path):
            return None
        return hashlib.md5(Path(file_path).read_bytes()).hexdigest()
    except:
        return None

# 2. Enhanced country data caching with file hash
@st.cache_data
def load_country_data_cached(country_key, file_hash=None):
    """Load country data with caching based on file hash"""
    print(f"🔄 Loading {country_key} data (cache miss)")
    from visualization_config import load_country_data
    return load_country_data(country_key)

def get_country_data_with_caching(country_key, country_file):
    """Get country data with intelligent caching"""
    file_hash = get_file_hash(country_file) if country_file else None
    return load_country_data_cached(country_key, file_hash)

# 3. Enhanced market data caching
@st.cache_data
def load_market_data_cached(file_hash=None):
    """Load market data with caching based on file hash"""
    print(f"🔄 Loading market data (cache miss)")
    from market_indices_visualizer import MarketIndicesVisualizer
    return MarketIndicesVisualizer()

def get_market_visualizer_with_caching():
    """Get market visualizer with intelligent caching"""
    from visualization_config import DataLoader, MARKET_INDICES_PATTERN
    latest_file = DataLoader.find_latest_file(MARKET_INDICES_PATTERN)
    file_hash = get_file_hash(latest_file) if latest_file else None
    return load_market_data_cached(file_hash)

# 4. Enhanced news data caching
@st.cache_data
def load_news_data_cached(file_hash=None):
    """Load news data with caching based on file hash"""
    print(f"🔄 Loading news data (cache miss)")
    from news_visualizer import NewsDataVisualizer
    return NewsDataVisualizer()

def get_news_visualizer_with_caching():
    """Get news visualizer with intelligent caching"""
    news_file = "./news_analysis_output/master_news_analysis.xlsx"
    file_hash = get_file_hash(news_file) if os.path.exists(news_file) else None
    return load_news_data_cached(file_hash)

# 5. Cache expensive chart generation
@st.cache_data
def create_country_overview_chart_cached(country_key, data_hash=None):
    """Create country overview chart with caching"""
    print(f"🔄 Creating overview chart for {country_key} (cache miss)")
    # Load data fresh
    from visualization_config import load_country_data
    country_data = load_country_data(country_key)
    if not country_data:
        return None
    return create_country_overview_chart(country_data, country_key)

@st.cache_data
def create_market_overview_cached(selected_indices, data_hash=None):
    """Create market overview with caching"""
    print(f"🔄 Creating market overview (cache miss)")
    market_viz = get_market_visualizer_with_caching()
    return market_viz.create_market_overview(selected_indices)

# 6. Cache chart data preprocessing
@st.cache_data
def get_chart_data_cached(country_key, indicator, file_hash=None):
    """Cache processed chart data"""
    country_data = get_country_data_with_caching(country_key, None)
    if country_data and indicator in country_data:
        return country_data[indicator]
    return None

# ============================================================================
# 🎯 MEMORY OPTIMIZATION TECHNIQUES:

# 7. Session state caching for user selections
def init_session_cache():
    """Initialize session-level caching for user preferences"""
    if 'chart_cache' not in st.session_state:
        st.session_state.chart_cache = {}
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}

def get_from_session_cache(key):
    """Get data from session cache"""
    return st.session_state.data_cache.get(key)

def set_session_cache(key, value):
    """Store data in session cache"""
    st.session_state.data_cache[key] = value

# 8. Lazy loading for expensive operations
@st.cache_data
def get_expensive_analysis_cached(region, topic, indicator, data_hash=None):
    """Cache expensive analysis operations"""
    print(f"🔄 Running expensive analysis (cache miss)")
    # Expensive operations here
    return analysis_results

# 9. Cache clearing utilities
def clear_all_caches():
    """Clear all Streamlit caches"""
    st.cache_data.clear()
    if 'data_cache' in st.session_state:
        st.session_state.data_cache.clear()
    if 'chart_cache' in st.session_state:
        st.session_state.chart_cache.clear()
    st.success("🗑️ All caches cleared!")

# 10. Add cache status display
def show_cache_status():
    """Show cache status in sidebar"""
    with st.sidebar:
        st.markdown("### 🚀 Cache Status")
        
        # Check if data is cached
        cache_info = []
        
        try:
            # Check country data cache
            countries_cached = len([k for k in st.session_state.get('data_cache', {}).keys() if 'country' in k])
            cache_info.append(f"Countries: {countries_cached} cached")
        except:
            cache_info.append("Countries: Not cached")
        
        try:
            # Check market data cache
            market_cached = 'market_data' in st.session_state.get('data_cache', {})
            cache_info.append(f"Market: {'✅ Cached' if market_cached else '❌ Not cached'}")
        except:
            cache_info.append("Market: Not cached")
        
        for info in cache_info:
            st.text(info)
        
        if st.button("🗑️ Clear Cache"):
            clear_all_caches()

# ============================================================================
from worldbank_data import WorldGDPCalculator
def get_gdp_coverage_metrics():
    """Get GDP coverage metrics with detailed debugging"""
    try:
        print("🔍 DEBUG: Starting GDP coverage calculation...")
        
        calculator = WorldGDPCalculator()
        print("✅ DEBUG: Calculator created successfully")
        
        # Check if cached data exists
        import os
        import glob
        
        data_dir = "./extracted_data"
        pattern = os.path.join(data_dir, "world_bank_gdp_data_*.xlsx")
        files = glob.glob(pattern)
        
        print(f"🔍 DEBUG: Looking for files in pattern: {pattern}")
        print(f"🔍 DEBUG: Found {len(files)} World Bank files: {files}")
        
        if files:
            latest_file = max(files, key=os.path.getctime)
            print(f"✅ DEBUG: Latest file: {latest_file}")
            
            # Try to read the file
            import pandas as pd
            try:
                contents_df = pd.read_excel(latest_file, sheet_name='Contents')
                print(f"✅ DEBUG: Contents sheet loaded, shape: {contents_df.shape}")
                print(f"🔍 DEBUG: Contents columns: {contents_df.columns.tolist()}")
                print(f"🔍 DEBUG: First few rows:")
                print(contents_df.head())
            except Exception as read_error:
                print(f"❌ DEBUG: Error reading Excel file: {read_error}")
                return None
        else:
            print("❌ DEBUG: No World Bank cache files found")
            print("💡 DEBUG: You need to run the main pipeline first to generate cached data")
            return None
        
        # Try the actual calculation
        metrics = calculator.get_dashboard_metrics(
            use_cached_data=True, 
            data_dir=data_dir
        )
        
        if metrics:
            print("✅ DEBUG: GDP metrics calculated successfully")
            print(f"🔍 DEBUG: Metrics: {metrics}")
        else:
            print("❌ DEBUG: GDP metrics calculation returned None")
            
        return metrics
        
    except Exception as e:
        print(f"❌ DEBUG: GDP coverage error: {e}")
        import traceback
        print(f"🔍 DEBUG: Full traceback:")
        traceback.print_exc()
        return None

def calculate_quarters_from_actual_data(data_sources):
    """Calculate quarters from actual data files (more accurate)"""
    # Try to get actual date range from your data
    earliest_date = None
    latest_date = None
    
    # Check a sample country's data to get real date range
    if data_sources and data_sources.get('countries'):
        sample_country = data_sources['countries'][0]
        
        # Try to load sample data to get real date range
        try:
            from visualization_config import load_country_data
            country_data = load_country_data(sample_country)
            
            if country_data:
                all_dates = []
                
                # Collect all dates from all indicators
                for indicator, df in country_data.items():
                    if not df.empty and 'date' in df.columns:
                        all_dates.extend(df['date'].tolist())
                
                if all_dates:
                    import pandas as pd
                    all_dates = pd.to_datetime(all_dates)
                    earliest_date = all_dates.min()
                    latest_date = all_dates.max()
                    
                    # Calculate quarters between earliest and latest
                    years_diff = latest_date.year - earliest_date.year
                    months_diff = latest_date.month - earliest_date.month
                    
                    # Convert to quarters
                    total_quarters = (years_diff * 4) + (months_diff // 3)
                    
                    return total_quarters
        except:
            pass

def create_gdp_hero_banner(data_sources):
    """Eye-catching hero banner for GDP coverage"""
    gdp_metrics = get_gdp_coverage_metrics()
    
    if gdp_metrics:
        # Get values first
        coverage = str(gdp_metrics['coverage_percent'])
        total_gdp = str(gdp_metrics['total_gdp_trillions'])
        countries = str(gdp_metrics['countries_count'])
        year = str(gdp_metrics['data_year'])
        market_count = str(len(load_market_data().keys()) if data_sources.get('market_available') else 8)
        economies_count = str(len(data_sources.get('countries', [])) if data_sources else 11)
        quarters_count = calculate_quarters_from_actual_data(data_sources)
        
        # Calculate coverage percentage for chart
        coverage_num = float(coverage.replace('%', ''))
        
        import streamlit.components.v1 as components
        
        html_code = f"""
        <style>
        @keyframes pulse {{
            0% {{ transform: translate(-50%, -50%) scale(1); opacity: 1; }}
            100% {{ transform: translate(-50%, -50%) scale(1.1); opacity: 0; }}
        }}
        </style>
        
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 30px; margin: 1rem 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
            <div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 3rem; align-items: center; padding: 0 4rem;">
                <div>
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span style="font-size: 3rem; margin-right: 1rem;">🌍</span>
                        <span style="font-size: 4rem; font-weight: bold;">{coverage}</span>
                    </div>
                    <h2 style="margin: 0 0 1rem 0; font-size: 2rem;">of World GDP Coverage</h2>
                    <p style="margin: 0.5rem 0; font-size: 1.2rem; font-weight: bold;">{total_gdp} across {countries} major economies</p>
                    <p style="margin: 0.5rem 0; font-size: 1rem;">Latest global annual GDP data: {year}</p>
                    <p style="margin: 1rem 0 0 0; font-size: 1rem; font-style: italic; opacity: 0.9;">Comprehensive economic data from the world's largest markets <br /> (Quarter-level data is included in Region Analysis)</p>
                </div>
                
                <!-- Enhanced Center Donut Chart -->
               <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
                    <div style="position: relative; margin-bottom: 0;">
                        <svg width="250" height="250">
                            <!-- Outer glow effect -->
                            <defs>
                                <filter id="glow">
                                    <feGaussianBlur stdDeviation="5" result="coloredBlur"/>
                                    <feMerge> 
                                        <feMergeNode in="coloredBlur"/>
                                        <feMergeNode in="SourceGraphic"/>
                                    </feMerge>
                                </filter>
                                <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" style="stop-color:#4CAF50;stop-opacity:1" />
                                    <stop offset="50%" style="stop-color:#66BB6A;stop-opacity:1" />
                                    <stop offset="100%" style="stop-color:#81C784;stop-opacity:1" />
                                </linearGradient>
                            </defs>
                            <!-- Background circle with subtle shadow -->
                            <circle cx="125" cy="125" r="90" fill="none" stroke="rgba(255,255,255,0.2)" stroke-width="16"/>
                            <!-- Progress circle with gradient and glow -->
                            <circle cx="125" cy="125" r="90" fill="none" stroke="url(#progressGradient)" stroke-width="18" 
                                    stroke-dasharray="{coverage_num * 5.655} 565" stroke-dashoffset="0" transform="rotate(-90 125 125)" 
                                    style="transition: stroke-dasharray 0.8s ease-in-out; filter: url(#glow);" stroke-linecap="round"/>
                            <!-- Inner circle background -->
                            <circle cx="125" cy="125" r="65" fill="rgba(255,255,255,0.1)" stroke="rgba(255,255,255,0.3)" stroke-width="1"/>
                            <!-- Center text with shadow -->
                            <text x="125" y="118" text-anchor="middle" fill="white" font-size="28" font-weight="bold" style="text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{coverage}</text>
                            <text x="125" y="140" text-anchor="middle" fill="rgba(255,255,255,0.8)" font-size="14">Coverage</text>
                        </svg>
                        <!-- Pulsing effect -->
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 200px; height: 200px; border: 2px solid rgba(76, 175, 80, 0.3); border-radius: 50%; animation: pulse 2s infinite;"></div>
                    </div>
                    <!-- Subtitle -->
                    <div style="text-align: center; font-size: 1rem; font-weight: 400; opacity: 0.9;">
                        World GDP Coverage
                    </div>
                </div>
                
                <div style="margin-left: 4rem;">
                    <h3 style="margin: 0 0 1.5rem 0; font-size: 2rem;">📊 Dashboard Overview</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem 3rem;">
                        <div style="text-align: center;">
                            <div style="font-size: 1.4rem; margin-bottom: 0.3rem; opacity: 1;">🌍 Economies</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{economies_count}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.4rem; margin-bottom: 0.3rem; opacity: 1;">📈 Market Indices</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{market_count}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.4rem; margin-bottom: 0.3rem; opacity: 1;">📊 Indicators</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">5</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.4rem; margin-bottom: 0.3rem; opacity: 1;">📅 Data Coverage</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{quarters_count} Quarters</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        components.html(html_code, height=375, width=1500)

def detect_update_frequency(df, date_column='date'):
    """
    Detect update frequency based on time intervals between consecutive data points
    
    Args:
        df: DataFrame with date column
        date_column: Name of the date column (default: 'date')
    
    Returns:
        str: Detected frequency ('Daily', 'Weekly', 'Monthly', 'Bi-monthly', 'Quarterly', 'Annual', 'Irregular', 'Insufficient Data')
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
        elif median_days <= 70:  # ~2 months with tolerance (NEW)
            return "Bi-monthly"
        elif median_days <= 100:  # ~3 months with tolerance
            return "Quarterly"
        elif median_days <= 400:  # ~1 year with tolerance
            return "Annual"
        else:
            return "Irregular"
            
    except Exception as e:
        return "Unknown"

def detect_data_frequency_for_growth(df):
    """Detect if data is annual, quarterly, or monthly for growth calculations"""
    if len(df) < 2:
        return "annual"
    
    # Calculate median difference between consecutive dates
    df_sorted = df.copy().sort_values('date')
    median_days = df_sorted['date'].diff().median().days
    
    if median_days <= 35:
        return "monthly" 
    elif median_days <= 100:
        return "quarterly"
    else:
        return "annual"

def get_growth_periods_by_frequency(frequency, indicator):
    """Get appropriate growth periods based on data frequency and indicator type"""
    if indicator == 'GDP':
        return {
            "monthly": 12,     # YoY for monthly GDP data
            "quarterly": 4,    # YoY for quarterly GDP data  
            "annual": 1        # YoY for annual GDP data
        }.get(frequency, 1)
    elif indicator == 'CPI':
        return {
            "monthly": 12,     # YoY for monthly CPI data
            "quarterly": 4,    # YoY for quarterly CPI data
            "annual": 1        # YoY for annual CPI data  
        }.get(frequency, 1)
    else:
        return 1  # Default to 1 period for other indicators

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
        {'key': 'country', 'icon': '🌍', 'title': 'Region Analysis', 'description': 'Economic indicators by region'},
        {'key': 'cross_country', 'icon': '🔗', 'title': 'Cross-Region', 'description': 'Compare across regions'},
        {'key': 'market_indices', 'icon': '📈', 'title': 'Market Indices', 'description': 'Global market analysis'},
        {'key': 'news_analysis', 'icon': '📰', 'title': 'News Analysis', 'description': 'Economic news topic and sentiment'},
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

def get_sentiment_emoji(sentiment_score, sentiment_label):
    """Get emoji based on sentiment score and label"""
    try:
        score = float(sentiment_score)
        
        if score <= -0.7:
            return "😭"  # Very negative - crying
        elif score <= -0.4:
            return "😟"  # Negative - sad
        elif score < -0.1:
            return "🙁"  # Slightly negative - slightly sad
        elif score <= 0.1:
            return "😐"  # Neutral
        elif score < 0.4:
            return "🙂"  # Slightly positive - slightly happy
        elif score < 0.7:
            return "😊"  # Positive - happy
        else:
            return "😄"  # Very positive - very happy
    except:
        # Fallback based on label if score fails
        label_lower = str(sentiment_label).lower()
        if 'very negative' in label_lower:
            return "😭"
        elif 'negative' in label_lower:
            return "😟"
        elif 'positive' in label_lower:
            return "😊"
        else:
            return "😐"

def extract_topics_from_analysis(topic_analysis):
    """Extract clean topics from topic analysis text"""
    try:
        topic_text = str(topic_analysis)
        if 'Topics:' in topic_text:
            # Extract topics after "Topics:"
            topics_part = topic_text.split('Topics:')[1].split('Indicators:')[0] if 'Indicators:' in topic_text else topic_text.split('Topics:')[1]
            
            # Remove asterisks and clean formatting
            topics_part = topics_part.replace('*', '').strip()
            
            topics = [topic.strip() for topic in topics_part.split(',')]
            # Return first 5 topics to avoid clutter
            clean_topics = [t.replace('**', '').replace('*', '').strip() for t in topics if t and len(t.strip()) > 2][:5]
            clean_topics = [t for t in clean_topics if t]  # Remove empty strings
            return ", ".join(clean_topics)
    except:
        pass
    return "Not specified"

def extract_indicators_from_analysis(topic_analysis):
    """Extract indicators from topic analysis text"""
    try:
        topic_text = str(topic_analysis)
        if 'Indicators:' in topic_text:
            # Extract indicators after "Indicators:"
            indicators_part = topic_text.split('Indicators:')[1].strip()
            
            # Remove asterisks and clean formatting
            indicators_part = indicators_part.replace('*', '').strip()
            
            indicators = [indicator.strip() for indicator in indicators_part.split(',')]

            clean_indicators = [i.replace('**', '').replace('*', '').strip() for i in indicators if i and len(i.strip()) > 1]
            clean_indicators = [i for i in clean_indicators if i]  # Remove empty strings
            return ", ".join(clean_indicators)
    except:
        pass
    return "Not specified"

def create_sentiment_summary_display(news_viz, selected_region, selected_topic, selected_indicator=None):
    """Create sentiment summary display section"""
    
    # Get filtered sentiment summary
    summary = news_viz.get_filtered_sentiment_summary(selected_region, selected_topic, selected_indicator)

    if summary:
        # Create an attractive summary card
        sentiment_emoji = get_sentiment_emoji(summary['avg_sentiment_score'], summary['avg_sentiment_label'])
        
        # Determine card color based on sentiment - 7 tiers matching emoji classification
        score = summary['avg_sentiment_score']

        if score <= -0.7:
            # 😭 Very negative - dark red
            card_color = "#e8b4b8"
            border_color = "#721c24"
        elif score <= -0.4:
            # 😟 Negative - medium red
            card_color = "#f8d7da"
            border_color = "#dc3545"
        elif score < -0.1:
            # 🙁 Slightly negative - light red
            card_color = "#fdeced"
            border_color = "#f5a6a6"
        elif score <= 0.1:
            # 😐 Neutral - yellow/beige
            card_color = "#fff8e1"
            border_color = "#ffc107"
        elif score < 0.4:
            # 🙂 Slightly positive - light green
            card_color = "#f1f8e9"
            border_color = "#8bc34a"
        elif score < 0.7:
            # 😊 Positive - medium green
            card_color = "#d4edda"
            border_color = "#28a745"
        else:
            # 😄 Very positive - dark green
            card_color = "#c8e6c9"
            border_color = "#1b5e20"
        
        region_display = selected_region if selected_region and selected_region != "All Regions" else "All Regions"
        topic_display = selected_topic if selected_topic and selected_topic != "All Topics" else "All Topics"
        filter_display = selected_indicator if selected_indicator and selected_indicator != "All Indicators" else "All Indicators"

        filter_text = f"Region: {region_display} | Topic: {topic_display} | Indicator: {filter_display}"

        html_content = f'''
        <div style="background-color: {card_color}; border-left: 10px solid {border_color}; padding: 1rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 0.8rem;">
                <span style="font-size: 2rem; margin-right: 1rem;">{sentiment_emoji}</span>
                <div style="text-align: center;">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">{filter_text}</p>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #333;">{summary['article_count']}</div>
                    <div style="font-size: 0.9rem; color: #666;">Articles Found</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: {border_color};">{summary['avg_sentiment_score']}</div>
                    <div style="font-size: 0.9rem; color: #666;">Avg Score</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: bold; color: #333;">{summary['avg_sentiment_label']}</div>
                    <div style="font-size: 0.9rem; color: #666;">Overall Sentiment</div>
                </div>
            </div>
            <div style="margin-top: 0.8rem; font-size: 0.8rem; color: #666; text-align: center;">
                Score Range: {summary['score_distribution']['min']} to {summary['score_distribution']['max']} (σ = {summary['score_distribution']['std']})
            </div>
        </div>
        '''
        
        # CRITICAL: Make sure unsafe_allow_html=True is included
        st.markdown(html_content, unsafe_allow_html=True)
        
        return True
    else:
        st.info("🔍 **No articles found** matching your criteria. Try different filters.")
        return False

def create_overview_page(data_sources):
    """Create main overview page with fixed navigation"""
    st.markdown('<h1 class="main-header">Economic Data Dashboard</h1>', unsafe_allow_html=True)
    create_gdp_hero_banner(data_sources)
    # st.markdown("---")

    # # Enhanced metrics row
    # col1, col2, col3, col4 = st.columns(4)
    
    # with col1:
    #     st.markdown(f"""
    #     <div class="big-metric">
    #         <div class="metric-label">🌍 Economies</div>
    #         <div class="metric-value">{len(data_sources['countries'])}</div>
    #     </div>
    #     """, unsafe_allow_html=True)
        
    # with col2:
    #     market_count = len(load_market_data().keys()) if data_sources['market_available'] else 0
    #     st.markdown(f"""
    #     <div class="big-metric">
    #         <div class="metric-label">📈 Market Indices</div>
    #         <div class="metric-value">{market_count}</div>
    #     </div>
    #     """, unsafe_allow_html=True)
        
    # with col3:
    #     st.markdown(f"""
    #     <div class="big-metric">
    #         <div class="metric-label">📊 Indicators</div>
    #         <div class="metric-value">5</div>
    #     </div>
    #     """, unsafe_allow_html=True)
        
    # with col4:
    #     st.markdown(f"""
    #     <div class="big-metric">
    #         <div class="metric-label">📅 Data Coverage</div>
    #         <div class="metric-value">98 Quarters</div>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # st.markdown("---")

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
        st.subheader("🌍 Regions")
        for country in data_sources['countries']:
            country_name = NameFormatter.format_country_name(country)
            # FIXED: Properly navigate to country page AND set selected country
            if st.button(country_name, key=f"country_{country}", use_container_width=True):
                st.session_state.selected_page = "country"  # Navigate to country page
                st.session_state.selected_country = country  # Set selected country
                st.rerun()
    
    with col2:
        st.subheader("📈 Market Indices")
        if data_sources['market_available']:
            try:
                # Load market data to get available indices
                from market_indices_visualizer import MarketIndicesVisualizer
                market_viz = MarketIndicesVisualizer()
                available_indices = market_viz.available_indices
                
                # Display each market index as a button
                for index_name in available_indices:
                    # Format display name (same as used in market page)
                    display_name = NameFormatter.format_display_name(index_name)
                    
                    # Create button that navigates to market page with specific index
                    if st.button(display_name, key=f"market_{index_name}", use_container_width=True):
                        st.session_state.selected_page = "market_indices"
                        # Set session state to remember which index was selected
                        st.session_state.selected_market_index = index_name
                        # Set to Individual Index view and Price & Volume chart
                        st.session_state.market_viz_type = "Individual Index"
                        st.session_state.market_chart_type = "Price & Volume"
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Could not load market indices: {e}")
                # Fallback - show analysis tools if market data fails
                if st.button("Cross-Country Comparison", use_container_width=True):
                    st.session_state.selected_page = "cross_country"
                    st.rerun()
        else:
            # Fallback if no market data available - show analysis tools
            st.warning("Market data not available")
            if st.button("Cross-Country Comparison", use_container_width=True):
                st.session_state.selected_page = "cross_country"
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
        ('GDP', 1, 1, True, 'blue', 'darkgreen'),
        ('CPI', 2, 1, True, 'orange', 'red'),
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
                # NEW: Detect data frequency and use appropriate growth periods
                frequency = detect_data_frequency_for_growth(indicator_data)
                appropriate_periods = get_growth_periods_by_frequency(frequency, indicator)
                print(f"   📊 {indicator} detected as {frequency} data, using {appropriate_periods} periods for growth")

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
            country_data = get_country_data_with_caching(selected_country, country_file)
            
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
                    # Create hash from country data for cache invalidation
                    data_str = str(sorted(country_data.keys())) + str(len(country_data))
                    data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
                    fig = create_country_overview_chart_cached(selected_country, data_hash)
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
        market_viz = get_market_visualizer_with_caching()
        
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
        
        # UPDATED: Handle navigation from overview page
        # Check if we came from overview page with a specific index selected
        if 'selected_market_index' in st.session_state and st.session_state.selected_market_index:
            # Set default visualization type to Individual Index
            default_viz_type = st.session_state.get('market_viz_type', 'Individual Index')
            viz_options = ["Market Overview", "Individual Index", "Correlation Analysis", "Performance Summary"]
            default_index = viz_options.index(default_viz_type) if default_viz_type in viz_options else 1
        else:
            default_index = 0

        # Visualization options
        viz_type = st.radio("Choose type:", ["Market Overview", "Individual Index", "Correlation Analysis", "Performance Summary"], horizontal=True, index=default_index)
        
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
            
            # UPDATED: Check if we have a pre-selected index from overview
            if 'selected_market_index' in st.session_state and st.session_state.selected_market_index:
                # Find the display name for the pre-selected index
                selected_internal_default = st.session_state.selected_market_index
                selected_display_default = NameFormatter.format_display_name(selected_internal_default)
                
                # Get the index for the selectbox
                display_options = [NameFormatter.format_display_name(idx) for idx in market_viz.available_indices]
                try:
                    default_idx = display_options.index(selected_display_default)
                except ValueError:
                    default_idx = 0
                    
                # Clear the session state after using it
                st.session_state.selected_market_index = None
            else:
                default_idx = 0
            
            selected_index_display = st.selectbox("Select Index", 
                                                [NameFormatter.format_display_name(idx) for idx in market_viz.available_indices],
                                                index=default_idx)
            
            # Find internal name
            selected_index = None
            for internal_name in market_viz.available_indices:
                if NameFormatter.format_display_name(internal_name) == selected_index_display:
                    selected_index = internal_name
                    break
            
            if selected_index:
                # UPDATED: Check for pre-selected chart type from overview
                if 'market_chart_type' in st.session_state and st.session_state.market_chart_type:
                    default_chart_type = st.session_state.market_chart_type
                    # Clear the session state after using it
                    st.session_state.market_chart_type = None
                else:
                    default_chart_type = "Price Chart"
                
                chart_options = ["Price Chart", "Returns Chart", "Volume Chart", "Price & Volume"]
                try:
                    chart_default_idx = chart_options.index(default_chart_type)
                except ValueError:
                    chart_default_idx = 0
                
                chart_type = st.radio("Chart Type:", chart_options, horizontal=True, index=chart_default_idx)
                
                # Map to internal chart types
                chart_type_mapping = {
                    "Price Chart": "price",
                    "Returns Chart": "returns", 
                    "Volume Chart": "volume",
                    "Price & Volume": "price_volume"
                }
                chart_type_param = chart_type_mapping[chart_type]
                
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

def create_news_analysis_page():
    """Create News Analysis page with all improvements"""
    st.header("📰 News Analysis Dashboard")
    
    try:
        from news_visualizer import NewsDataVisualizer
        
        # Initialize with automatic file detection
        news_viz = get_news_visualizer_with_caching()
        
        # Check if file was found and data was loaded
        if not news_viz.news_file_path:
            st.error("No news analysis files found in ./news_analysis_output folder. Please ensure the agentic analysis has been run.")
            # st.info("Expected file pattern: ./news_analysis_output/agentic_analysis_*.xlsx")
            st.info("Expected file: ./news_analysis_output/master_news_analysis.xlsx")
            return
            
        if not news_viz.news_data:
            st.error("News analysis file found but no data could be loaded. Please check the file format.")
            st.info(f"Found file: {news_viz.news_file_path}")
            return
        
        # Summary metrics (3 columns)
        metrics = news_viz.get_summary_metrics()
        if metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📰 Total Articles", metrics['total_articles'])
            with col2:
                st.metric("📊 News Sources", metrics['sources_count'])
            with col3:
                st.metric("📅 Latest Update", metrics['latest_update'])
        
        st.markdown("---")

        st.subheader("📊 Article Distribution Analysis")
        col1, col2 = st.columns(2)

        with col1:
            country_dist_fig = news_viz.create_country_distribution_chart()
            if country_dist_fig:
                st.plotly_chart(country_dist_fig, use_container_width=True)
            else:
                st.info("No country distribution data available")

        with col2:
            source_dist_fig = news_viz.create_source_distribution_chart()
            if source_dist_fig:
                st.plotly_chart(source_dist_fig, use_container_width=True)
            else:
                st.info("No source distribution data available")

        # Charts section
        st.markdown("---")
        
        st.subheader("😊 Sentiment/Topic Analysis")
        # Chart tabs
        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
            "Regional Sentiment", "Sentiment by Source", "Sentiment Timeline", "Topic Analysis"
        ])
        
        with chart_tab1:
            regional_sentiment_fig = news_viz.create_regional_sentiment_chart()
            if regional_sentiment_fig:
                st.plotly_chart(regional_sentiment_fig, use_container_width=True)
                st.info("📊 **Chart shows average sentiment score by region** (-1=Very Negative, 0=Neutral, +1=Very Positive)")
            else:
                st.info("No regional sentiment data available for visualization")
        
        with chart_tab2:
            sentiment_fig = news_viz.create_sentiment_chart()
            if sentiment_fig:
                st.plotly_chart(sentiment_fig, use_container_width=True)
                st.info("📊 **Each source shows percentage distribution** to better compare sentiment patterns")
            else:
                st.info("No sentiment data available for visualization")
        
        with chart_tab3:
            # Add region, topic, and indicator selectors for timeline
            col1, col2, col3 = st.columns(3)
    
            with col1:
                available_regions = news_viz.get_all_regions()
                selected_timeline_region = st.selectbox(
                    "Select Region for Timeline",
                    options=["All Regions"] + available_regions,
                    index=0,
                    key="timeline_region_selector"
                )
            
            with col2:
                available_topics = news_viz.get_all_topics()
                selected_timeline_topic = st.selectbox(
                    "Select Topic for Timeline",
                    options=["All Topics"] + available_topics,
                    index=0,
                    key="timeline_topic_selector"
                )
            
            with col3:
                available_indicators = news_viz.get_all_indicators()  # New method needed
                selected_timeline_indicator = st.selectbox(
                    "Select Indicator for Timeline",
                    options=["All Indicators"] + available_indicators,
                    index=0,
                    key="timeline_indicator_selector"
                )

            timeline_region = None if selected_timeline_region == "All Regions" else selected_timeline_region
            timeline_topic = None if selected_timeline_topic == "All Topics" else selected_timeline_topic
            timeline_indicator = None if selected_timeline_indicator == "All Indicators" else selected_timeline_indicator
    
            st.markdown("##### 📊 Sentiment Summary by Region, Topic, and Indicator")
            has_data = create_sentiment_summary_display(news_viz, timeline_region, timeline_topic, timeline_indicator)
            st.markdown("") 
            st.markdown("##### 📈 Sentiment Trends Over Time by Region, Topic, and Indicator")

            # Check if topic breakdown is available
            topic_breakdown_available = (
                timeline_region and timeline_region != "All Regions" and
                (not timeline_topic or timeline_topic == "All Topics") and
                (not timeline_indicator or timeline_indicator == "All Indicators")
            )

            # Only show the toggle if topic breakdown is available
            show_topic_breakdown = False
            if topic_breakdown_available:
                show_topic_breakdown = st.toggle(
                    f"📊 Show topic breakdown for {timeline_region}",
                    value=False,
                    help="Break down sentiment by individual topics within the selected region"
                )
                
                if show_topic_breakdown:
                    st.info(f"📈 Showing individual topic trends within {timeline_region}")
            else:
                # Optionally show why the toggle is not available
                if timeline_region and timeline_region != "All Regions":
                    if timeline_topic and timeline_topic != "All Topics":
                        st.info("💡 Topic breakdown is only available when 'All Topics' is selected")
                    if timeline_indicator and timeline_indicator != "All Indicators":
                        st.info("💡 Topic breakdown is only available when 'All Indicators' is selected")

            if has_data:
                timeline_fig = news_viz.create_sentiment_timeline(
                    selected_region=timeline_region, 
                    selected_topic=timeline_topic,
                    selected_indicator=timeline_indicator,
                    show_topic_breakdown=show_topic_breakdown,
                    max_topics=5  # Show top 5 topics
                )
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)

                    # Dynamic info message based on toggle state
                    if timeline_region and not timeline_topic:
                        if show_topic_breakdown:
                            st.info(f"📊 **Topic Breakdown**: Individual topic trends within {timeline_region}")
                        else:
                            st.info(f"📈 **Overall Trend**: Combined sentiment for {timeline_region} across all topics")
                else:
                    st.info("No timeline data available for visualization")
            else:
                st.info("No timeline data available - try different filter criteria")
        
        with chart_tab4:
            topic_fig = news_viz.create_topic_analysis_chart()
            if topic_fig:
                st.plotly_chart(topic_fig, use_container_width=True)
            else:
                st.info("No topic data available for visualization")
        
        # Articles by Region and Topic section
        st.markdown("---")
        st.subheader("🔍 Articles by Region and Topic")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            available_regions = news_viz.get_all_regions()
            selected_region = st.selectbox(
                "Select Region",
                options=["None"] + available_regions,
                index=0
            )
            if selected_region == "None":
                selected_region = None

        with col2:
            available_topics = news_viz.get_all_topics()
            selected_topic = st.selectbox(
                "Select Topic",
                options=["None"] + available_topics,
                index=0
            )
            if selected_topic == "None":
                selected_topic = None

        with col3:
            available_indicators = news_viz.get_all_indicators()
            selected_indicator = st.selectbox(
                "Select Indicator",
                options=["None"] + available_indicators,
                index=0
            )
            if selected_indicator == "None":
                selected_indicator = None

        with col4:
            start_date = st.date_input(
                "Start Date (optional)",
                value=None,
                help="Leave blank for earliest date"
            )

        with col5:
            end_date = st.date_input(
                "End Date (optional)", 
                value=None,
                help="Leave blank for latest date"
            )

        # Filter and display articles
        if selected_region or selected_topic or selected_indicator or start_date or end_date:
            with st.spinner("Filtering articles..."):
                filtered_articles = news_viz.filter_articles_by_region_topic_indicator_and_date(
                    selected_region=selected_region,
                    selected_topic=selected_topic,
                    selected_indicator=selected_indicator,
                    start_date=start_date,
                    end_date=end_date,
                    limit=None  # Get all matching articles
                )
            
            if filtered_articles:
                filter_desc = []
                if selected_region: filter_desc.append(f"Region: {selected_region}")
                if selected_topic: filter_desc.append(f"Topic: {selected_topic}")
                if selected_indicator: filter_desc.append(f"Indicator: {selected_indicator}")
                if start_date or end_date:
                    date_range = f"Date: {start_date or 'earliest'} to {end_date or 'latest'}"
                    filter_desc.append(date_range)
                
                st.success(f"📊 Found **{len(filtered_articles)} articles** matching: {' | '.join(filter_desc)}")
                # st.success(f"📊 Found **{len(filtered_articles)} articles** matching your criteria")
                
                for i, article in enumerate(filtered_articles):
                    # FIXED: Remove icon and show full title
                    with st.expander(article['title']):  # Full title, no truncation, no icon
                        col_left, col_right = st.columns([2, 1])
                        
                        with col_left:
                            # st.markdown("**Summary:**")
                            st.markdown('<div style="font-size: 1.4rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem;">📄 Summary:</div>', unsafe_allow_html=True)
                            summary_text = article['summary'].replace('$', '\\$')
                            st.write(summary_text)

                            # st.markdown("**Impact Analysis:**")
                            st.markdown('<div style="font-size: 1.4rem; font-weight: bold; color: #ff7f0e; margin-bottom: 0.5rem; margin-top: 1rem;">📊 Impact Analysis:</div>', unsafe_allow_html=True)
                            st.markdown(article['impact_analysis'])
                            
                            if article['link']:
                                st.markdown(f"[🔗 Read Full Article]({article['link']})")
                        
                        with col_right:
                            # st.markdown("**Details:**")
                            st.markdown('<div style="font-size: 1.4rem; font-weight: bold; color: #28a745; margin-bottom: 0.5rem;">📋 Details:</div>', unsafe_allow_html=True)
                            st.write(f"📺 **Source:** {article['source']}")
                            st.write(f"📍 **Region:** {article['region']}")
                            st.write(f"📅 **Date:** {article['published_date']}")
                            # st.write(f"😊 **Sentiment:** {article['sentiment_label']} ({article['sentiment_score']:.2f})")

                            sentiment_emoji = get_sentiment_emoji(article['sentiment_score'], article['sentiment_label'])
                            st.write(f"{sentiment_emoji} **Sentiment:** {article['sentiment_label']} ({article['sentiment_score']:.2f})")

                            topics = extract_topics_from_analysis(article.get('topic_analysis', ''))
                            st.write(f"🏷️ **Topics:** {topics}")

                            indicators = extract_indicators_from_analysis(article.get('topic_analysis', ''))
                            st.write(f"📊 **Indicators:** {indicators}")
            else:
                st.warning("No articles found matching your criteria. Try different filters.")
        else:
            st.info("👆 Select region, topic, indicator, or date range above to see filtered articles")

        # Recent articles by source
        st.markdown("---")
        st.subheader("📋 Recent Articles by Source")

        source_names = news_viz.get_available_sources()
        display_names = news_viz.get_source_display_names()

        if source_names:
            tabs = st.tabs(display_names)
            
            for tab, source_name, display_name in zip(tabs, source_names, display_names):
                with tab:
                    articles = news_viz.get_detailed_articles(source_name, limit=10)
                    
                    if articles:
                        st.info(f"📊 **{len(articles)} recent articles** from {display_name} (latest first)")
                        
                        for article in articles:
                            # FIXED: Remove icon, show full title, consistent font size
                            with st.expander(article['title']):  # Full title, no truncation, no icon
                                col_left, col_right = st.columns([2, 1])
                                
                                with col_left:
                                    # st.markdown("**Summary:**")
                                    st.markdown('<div style="font-size: 1.4rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem;">📄 Summary:</div>', unsafe_allow_html=True)
                                    summary_text = article['summary'].replace('$', '\\$')
                                    st.markdown(summary_text)

                                    # st.markdown("**Impact Analysis:**")
                                    st.markdown('<div style="font-size: 1.4rem; font-weight: bold; color: #ff7f0e; margin-bottom: 0.5rem; margin-top: 1rem;">📊 Impact Analysis:</div>', unsafe_allow_html=True)
                                    st.markdown(article['impact_analysis'])

                                    if article['link']:
                                        st.markdown(f"[🔗 Read Full Article]({article['link']})")
                                
                                with col_right:
                                    # st.markdown("**Details:**")
                                    st.markdown('<div style="font-size: 1.4rem; font-weight: bold; color: #28a745; margin-bottom: 0.5rem;">📋 Details:</div>', unsafe_allow_html=True)
                                    st.write(f"📍 **Region:** {article['region']}")
                                    st.write(f"📅 **Date:** {article['published_date']}")
                                    # st.write(f"😊 **Sentiment:** {article['sentiment_label']} ({article['sentiment_score']:.2f})")

                                    sentiment_emoji = get_sentiment_emoji(article['sentiment_score'], article['sentiment_label'])
                                    st.write(f"{sentiment_emoji} **Sentiment:** {article['sentiment_label']} ({article['sentiment_score']:.2f})")

                                    topics = extract_topics_from_analysis(article.get('topic_analysis', ''))
                                    st.write(f"🏷️ **Topics:** {topics}")

                                    indicators = extract_indicators_from_analysis(article.get('topic_analysis', ''))
                                    st.write(f"📊 **Indicators:** {indicators}")
                    else:
                        st.info(f"No articles available for {display_name}")
        
    except ImportError:
        st.error("News data visualizer not available. Please ensure news_data_visualizer.py is in the same directory.")
    except Exception as e:
        st.error(f"Error loading news analysis: {e}")
        import traceback
        st.code(traceback.format_exc())

def main():
    """Main Streamlit application with fixed navigation"""
    # Initialize caching
    init_session_cache()

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
        st.sidebar.markdown('<div class="sidebar-section">🌍 Regions</div>', unsafe_allow_html=True)
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

    # Add cache status to sidebar
    show_cache_status()

    if page_key == "overview":
        create_overview_page(data_sources)
    elif page_key == "country":
        create_country_page(data_sources)
    elif page_key == "cross_country":
        create_cross_country_page(data_sources)
    elif page_key == "market_indices":
        create_market_page(data_sources)
    elif page_key == "news_analysis":
        create_news_analysis_page()

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
