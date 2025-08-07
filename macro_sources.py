"""
Enhanced Comprehensive Macroeconomic Data Sources for Major Economies
FIXED VERSION - API URL construction issue resolved
Focus: Singapore, US, EU, China, Hong Kong, Japan + Financial Market Indices
Priority: Official sources with API access + Enhanced coverage
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

import os
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')

@dataclass
class MacroDataSource:
    """Enhanced macroeconomic data source with comprehensive metadata"""
    name: str
    url: str
    country: str  # Singapore, US, EU, China, Hong Kong, Japan, Global
    data_type: str  # GDP, CPI, Interest_Rate, Population, Property_Price, Mortgage_Rate, Stock_Index
    source_type: str  # api, web_scraping, csv_download
    authority: str  # Central bank, government agency, etc.
    description: str = ""
    
    # Technical characteristics
    api_url: Optional[str] = None
    api_documentation: Optional[str] = None
    update_frequency: str = "quarterly"  # daily, weekly, monthly, quarterly, annually
    data_format: str = "json"  # json, csv, xml, html
    requires_api_key: bool = False
    extraction_complexity: str = "simple"  # simple, medium, complex
    
    # Data characteristics
    historical_data_available: bool = True
    real_time_data: bool = False
    data_quality_rating: int = 9  # 1-10 scale
    
    # Enhanced metadata
    backup_sources: List[str] = None
    api_rate_limits: Optional[str] = None
    authentication_method: Optional[str] = None
    data_coverage_start: Optional[str] = None
    
    # NEW: FRED series ID for dynamic URL construction
    fred_series_id: Optional[str] = None

    # NEW: World Bank API identifiers
    wb_series_id: Optional[str] = None
    wb_country_code: Optional[str] = None
    
    def __post_init__(self):
        if self.backup_sources is None:
            self.backup_sources = []
    
    def get_fred_api_url(self) -> Optional[str]:
        """Dynamically construct FRED API URL with current API key"""
        if not self.fred_series_id:
            return self.api_url
        
        # Get fresh API key from environment
        current_api_key = os.getenv('FRED_API_KEY')
        if not current_api_key:
            return None
            
        return f"https://api.stlouisfed.org/fred/series/observations?series_id={self.fred_series_id}&api_key={current_api_key}&file_type=json"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'name': self.name,
            'url': self.url,
            'country': self.country,
            'data_type': self.data_type,
            'source_type': self.source_type,
            'authority': self.authority,
            'description': self.description,
            'api_url': self.api_url,
            'api_documentation': self.api_documentation,
            'update_frequency': self.update_frequency,
            'data_format': self.data_format,
            'requires_api_key': self.requires_api_key,
            'extraction_complexity': self.extraction_complexity,
            'historical_data_available': self.historical_data_available,
            'real_time_data': self.real_time_data,
            'data_quality_rating': self.data_quality_rating,
            'backup_sources': self.backup_sources,
            'api_rate_limits': self.api_rate_limits,
            'authentication_method': self.authentication_method,
            'data_coverage_start': self.data_coverage_start,
            'fred_series_id': self.fred_series_id,
            'wb_series_id': self.wb_series_id,           # NEW
            'wb_country_code': self.wb_country_code      # NEW
        }

# =============================================================================
# SINGAPORE MACROECONOMIC DATA SOURCES (ENHANCED)
# =============================================================================

SINGAPORE_SOURCES = [
    # GDP Data
    MacroDataSource(
        name="Singapore GDP (in Chained 2015 Dollars), Quarterly",
        url="https://data.gov.sg/datasets/d_87a41037716428a042305df9b948ece2/view",
        country="Singapore",
        data_type="GDP",
        source_type="api",
        authority="Singapore Department of Statistics",
        description="Singapore GDP in chained 2015 dollars, quarterly data",
        api_url="https://data.gov.sg/api/action/datastore_search?resource_id=d_87a41037716428a042305df9b948ece2",
        api_documentation="https://data.gov.sg/developer",
        update_frequency="quarterly",
        data_format="json",
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1975",
        backup_sources=["https://www.singstat.gov.sg/find-data/search-by-theme/economy/national-accounts/latest-data"]
    ),
    
    # CPI Data
    MacroDataSource(
        name="Singapore Consumer Price Index (CPI), 2019 As Base Year, Quarterly",
        url="https://data.gov.sg/datasets/d_09f3ea075cdf660f722df3f743b1d796/view",
        country="Singapore",
        data_type="CPI",
        source_type="api",
        authority="Singapore Department of Statistics",
        description="Singapore CPI all items, quarterly data",
        api_url="https://data.gov.sg/api/action/datastore_search?resource_id=d_09f3ea075cdf660f722df3f743b1d796",
        update_frequency="quarterly",
        data_format="json",
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1961",
        backup_sources=["https://www.singstat.gov.sg/whats-new/latest-news/cpi-highlights"]
    ),
    
    # Interest Rates - FIXED: Now properly marked as API
    MacroDataSource(
        name="Singapore Interest Rates, Monthly",
        url="https://data.gov.sg/datasets/d_5fe5a4bb4a1ecc4d8a56a095832e2b24/view",
        country="Singapore",
        data_type="Interest_Rate",
        source_type="api",  # FIXED: Changed from web_scraping to api
        authority="Monetary Authority of Singapore",
        description="Singapore sovereign bond yields and interbank rates",
        api_url="https://data.gov.sg/api/action/datastore_search?resource_id=d_5fe5a4bb4a1ecc4d8a56a095832e2b24",
        update_frequency="monthly",
        data_format="json",  # FIXED: Changed from html to json
        extraction_complexity="simple",  # FIXED: Changed from medium to simple
        data_quality_rating=10,
        backup_sources=["https://data.gov.sg/search?q=interest+rates"]
    ),
    
    # Population
    MacroDataSource(
        name="Singapore Population",
        url="https://data.gov.sg/datasets/d_3d227e5d9fdec73f3bcadce671c333a6/view",
        country="Singapore",
        data_type="Population",
        source_type="api",
        authority="Singapore Department of Statistics",
        description="Singapore indicators on population and population structure",
        api_url="https://data.gov.sg/api/action/datastore_search?resource_id=d_3d227e5d9fdec73f3bcadce671c333a6",
        update_frequency="annually",
        data_format="json",
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1957",
        backup_sources=["https://www.singstat.gov.sg/find-data/search-by-theme/population/population-and-population-structure/latest-data"]
    ),
    
    # Property Price Index
    MacroDataSource(
        name="Singapore Private Residential Property Price Index (Base Quarter 2009-Q1 = 100)",
        url="https://data.gov.sg/datasets/d_97f8a2e995022d311c6c68cfda6d034c/view",
        country="Singapore",
        data_type="Property_Price",
        source_type="api",
        authority="Urban Redevelopment Authority",
        description="Singapore residential property price index",
        api_url="https://data.gov.sg/api/action/datastore_search?resource_id=d_97f8a2e995022d311c6c68cfda6d034c",
        update_frequency="quarterly",
        data_format="json",
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1975",
        backup_sources=[]
    ),
]

# =============================================================================
# CHINA MACROECONOMIC DATA SOURCES
# =============================================================================

CHINA_SOURCES = [
    # 1. GDP Data - Using IMF Data Explorer
    MacroDataSource(
        name="China Real GDP",
        url="https://data.imf.org/en/Data%20Explorer?datasetUrn=IMF.STA%3AQNEA%287.0.0%29&timeseriesName=CHN.B1GQ.Q.NSA.XDC.Q",
        country="China",
        data_type="GDP",
        source_type="api",
        authority="International Monetary Fund",
        description="China Real GDP, quarterly, not seasonally adjusted, LCU",
        api_url="https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/QNEA/%2B/CHN.B1GQ.Q.NSA.XDC.Q?dimensionAtObservation=TIME_PERIOD&attributes=dsd&measures=all&includeHistory=false",  # Simplified API endpoint
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=False,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="2011",
        backup_sources=[
            "https://fred.stlouisfed.org/series/CHNGDPNQDSMEI",  # FRED quarterly alternative
            "http://www.stats.gov.cn/english/PressRelease/"
        ]
    ),

    # 2. CPI Data - FRED OECD source
    MacroDataSource(
        name="China Consumer Price Index",
        url="https://fred.stlouisfed.org/series/CHNCPIALLMINMEI",
        country="China",
        data_type="CPI",
        source_type="api",
        authority="Federal Reserve Bank of St. Louis (OECD source)",
        description="China Consumer Price Index, all items, monthly",
        api_url=None,
        fred_series_id="CHNCPIALLMINMEI",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1993",
        backup_sources=["http://www.stats.gov.cn/english/"]
    ),

    # 3. Interest Rate - FRED OECD interbank rate
    MacroDataSource(
        name="China Interest Rate",
        url="https://fred.stlouisfed.org/series/IRSTCI01CNM156N",
        country="China",
        data_type="Interest_Rate",
        source_type="api",
        authority="Federal Reserve Bank of St. Louis (OECD source)",
        description="China interbank rate/call money rate, immediate rates (< 24 hours), monthly",
        api_url=None,
        fred_series_id="IRSTCI01CNM156N",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=8,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1990",
        backup_sources=[
            "https://fred.stlouisfed.org/series/INTDSRCNM193N",  # Discount rate alternative
            "http://www.pbc.gov.cn/en/3688110/index.html"  # People's Bank of China
        ]
    ),

    # 4. Population Data - FRED World Bank series
    MacroDataSource(
        name="China Population",
        url="https://fred.stlouisfed.org/series/POPTOTCNA647NWDB",
        country="China",
        data_type="Population",
        source_type="api",
        authority="Federal Reserve Bank of St. Louis (World Bank source)",
        description="China total population",
        api_url=None,
        fred_series_id="POPTOTCNA647NWDB",
        update_frequency="annually",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1960",
        backup_sources=["http://www.stats.gov.cn/english/"]
    ),

    # 5. Property Price Index - FRED BIS real property prices
    MacroDataSource(
        name="China Property Price Index",
        url="https://fred.stlouisfed.org/series/QCNR628BIS",
        country="China",
        data_type="Property_Price",
        source_type="api",
        authority="Federal Reserve Bank of St. Louis (BIS source)",
        description="China real residential property prices index, quarterly, not seasonally adjusted",
        api_url=None,
        fred_series_id="QCNR628BIS",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=8,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="2005",
        backup_sources=[
            "http://www.stats.gov.cn/english/",
            "https://fred.stlouisfed.org/series/QCNN628BIS"  # Nominal property prices alternative
        ]
    ),
]

# =============================================================================
# UNITED STATES MACROECONOMIC DATA SOURCES (FIXED)
# =============================================================================

US_SOURCES = [
    # GDP Data - FIXED: Use fred_series_id instead of f-string
    MacroDataSource(
        name="US Real GDP",
        url="https://fred.stlouisfed.org/series/GDPC1",
        country="US",
        data_type="GDP",
        source_type="api",
        authority="U.S. Bureau of Economic Analysis",
        description="US Real Gross Domestic Product, seasonally adjusted annual rate",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="GDPC1",  # FIXED: Add series ID for dynamic construction
        api_documentation="https://fred.stlouisfed.org/docs/api/fred/",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1947",
        backup_sources=["https://www.bea.gov/data/gdp/gross-domestic-product"]
    ),
    
    # CPI Data - FIXED
    MacroDataSource(
        name="US Consumer Price Index",
        url="https://fred.stlouisfed.org/series/CPIAUCSL",
        country="US",
        data_type="CPI",
        source_type="api",
        authority="U.S. Bureau of Labor Statistics",
        description="US Consumer Price Index for All Urban Consumers",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="CPIAUCSL",  # FIXED: Add series ID
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1947",
        backup_sources=["https://www.bls.gov/cpi/"]
    ),
    
    # Federal Funds Rate - FIXED
    MacroDataSource(
        name="US Federal Funds Rate",
        url="https://fred.stlouisfed.org/series/FEDFUNDS",
        country="US",
        data_type="Interest_Rate",
        source_type="api",
        authority="Board of Governors of the Federal Reserve System (US)",
        description="Federal funds effective rate",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="FEDFUNDS",  # FIXED: Add series ID
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1954",
        backup_sources=["https://www.federalreserve.gov/monetarypolicy/openmarket.htm"]
    ),
    
    # Population - FIXED
    MacroDataSource(
        name="US Population",
        url="https://fred.stlouisfed.org/series/POPTHM",
        country="US",
        data_type="Population",
        source_type="api",
        authority="U.S. Bureau of Economic Analysis",
        description="US Population, total in thousands",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="POPTHM",  # FIXED: Add series ID
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1952",
        backup_sources=["https://www.census.gov/topics/population/population-totals.html"]
    ),
    
    # # 30-Year Mortgage Rate - FIXED
    # MacroDataSource(
    #     name="US 30-Year Fixed Mortgage Rate",
    #     url="https://fred.stlouisfed.org/series/MORTGAGE30US",
    #     country="US",
    #     data_type="Mortgage_Rate",
    #     source_type="api",
    #     authority="Freddie Mac",
    #     description="30-Year Fixed Rate Mortgage Average in the United States",
    #     api_url=None,  # Will be constructed dynamically
    #     fred_series_id="MORTGAGE30US",  # FIXED: Add series ID
    #     update_frequency="weekly",
    #     data_format="json",
    #     requires_api_key=True,
    #     extraction_complexity="simple",
    #     data_quality_rating=10,
    #     api_rate_limits="120 requests per 60 seconds",
    #     authentication_method="API Key",
    #     data_coverage_start="1971",
    #     backup_sources=["https://www.freddiemac.com/pmms"]
    # ),
    
    # US Property Price Index - FIXED
    MacroDataSource(
        name="US Real Residential Property Price Index (BIS)",
        url="https://fred.stlouisfed.org/series/QUSR628BIS",
        country="US",
        data_type="Property_Price",
        source_type="api",
        authority="Bank for International Settlements",
        description="Real residential property prices for United States, deflated using CPI",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="QUSR628BIS",  # FIXED: Add series ID
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1970",
        backup_sources=[
            "https://www.bis.org/statistics/pp_detailed.htm",
            "https://www.spglobal.com/spdji/en/indices/indicators/sp-corelogic-case-shiller-us-national-home-price-nsa-index/"
        ]
    ),
]

# =============================================================================
# EUROPEAN UNION MACROECONOMIC DATA SOURCES (FIXED)
# =============================================================================

EU_SOURCES = [
    # GDP Data - FIXED
    MacroDataSource(
        name="Real Gross Domestic Product for Euro Area",
        url="https://fred.stlouisfed.org/series/CLVMNACSCAB1GQEA19",
        country="Euro Area",
        data_type="GDP",
        source_type="api",
        authority="European Central Bank",
        description="Eurozone Real GDP, quarterly, Chained 2010 Euros, seasonally adjusted",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="CLVMNACSCAB1GQEA19",  # FIXED: Add series ID
        update_frequency="quarterly",
        data_format="json",
        extraction_complexity="simple",  # FIXED: Changed from medium to simple
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1995",
        requires_api_key=True,
        backup_sources=["https://ec.europa.eu/eurostat/databrowser/view/teina011/default/table?lang=en"]
    ),
    
    # HICP (Harmonised Index of Consumer Prices) - FIXED
    MacroDataSource(
        name="Harmonized Index of Consumer Prices (HICP) for Euro Area",
        url="https://fred.stlouisfed.org/series/CP0000EZ19M086NEST",
        country="Euro Area",
        data_type="CPI",
        source_type="api",
        authority="Eurostat",
        description="Harmonized Index of Consumer Prices (HICP) for the Euro Area (19 Countries), monthly, not seasonally adjusted",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="CP0000EZ19M086NEST",  # FIXED: Add series ID
        update_frequency="monthly",
        data_format="json",
        extraction_complexity="simple",  # FIXED: Changed from medium to simple
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1996",
        requires_api_key=True,
        backup_sources=["https://ec.europa.eu/eurostat/databrowser/view/tec00118/default/table?lang=en"]
    ),
    
    # ECB Interest Rates - FIXED
    MacroDataSource(
        name="ECB Deposit Facility Rate",
        url="https://fred.stlouisfed.org/series/ECBDFR",
        country="Euro Area",
        data_type="Interest_Rate",
        source_type="api",
        authority="European Central Bank",
        description="ECB deposit facility rate",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="ECBDFR",  # FIXED: Add series ID
        update_frequency="daily",
        data_format="json",
        extraction_complexity="simple",  # FIXED: Changed from medium to simple
        real_time_data=True,
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1999",
        requires_api_key=True,
        backup_sources=["https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html"]
    ),
    
    # EU Population - FIXED
    MacroDataSource(
        name="Population, Total for Euro Area",
        url="https://fred.stlouisfed.org/series/SPPOPTOTLEMU",
        country="Euro Area",
        data_type="Population",
        source_type="api",
        authority="World Bank",
        description="Total population for the Euro Area, annual",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="SPPOPTOTLEMU",  # FIXED: Add series ID
        update_frequency="annually",
        data_format="json",
        extraction_complexity="simple",  # FIXED: Changed from medium to simple
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1960",
        requires_api_key=True
    ),
    
    # EU Property Price Index - FIXED
    MacroDataSource(
        name="Residential Property Price Index for the Euro Area",
        url="https://fred.stlouisfed.org/series/QXMR628BIS",
        country="Euro Area",
        data_type="Property_Price",
        source_type="api",
        authority="Bank for International Settlements (BIS)",
        description="Residential Property Price Index for the Euro Area, quarterly, not seasonally adjusted",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="QXMR628BIS",  # FIXED: Add series ID
        update_frequency="quarterly",
        data_format="json",
        extraction_complexity="simple",  # FIXED: Changed from medium to simple
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1975",
        requires_api_key=True
    ),
]

# =============================================================================
# UNITED KINGDOM MACROECONOMIC DATA SOURCES
# =============================================================================

UK_SOURCES = [
    # GDP Data
    MacroDataSource(
        name="UK Real GDP",
        url="https://fred.stlouisfed.org/series/NGDPRSAXDCGBQ",
        country="UK",
        data_type="GDP",
        source_type="api",
        authority="International Monetary Fund via FRED",
        description="UK Real GDP, quarterly, seasonally adjusted",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="NGDPRSAXDCGBQ",
        api_documentation="https://fred.stlouisfed.org/docs/api/fred/",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1955",
        backup_sources=["https://www.ons.gov.uk/economy/grossdomesticproductgdp"]
    ),
    
    # CPI Data
    MacroDataSource(
        name="UK Consumer Price Index",
        url="https://fred.stlouisfed.org/series/GBRCPIALLMINMEI",
        country="UK",
        data_type="CPI",
        source_type="api",
        authority="Organization for Economic Co-operation and Development via FRED",
        description="UK Consumer Price Index for All Items, monthly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="GBRCPIALLMINMEI",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1955",
        backup_sources=["https://www.ons.gov.uk/economy/inflationandpriceindices"]
    ),
    
    # Interest Rate
    MacroDataSource(
        name="UK Interest Rate",
        url="https://fred.stlouisfed.org/series/IRSTCI01GBM156N",
        country="UK",
        data_type="Interest_Rate",
        source_type="api",
        authority="Organization for Economic Co-operation and Development via FRED",
        description="UK immediate rates (< 24 hours), monthly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="IRSTCI01GBM156N",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1978",
        backup_sources=["https://www.bankofengland.co.uk/monetary-policy/the-interest-rate-bank-rate"]
    ),
    
    # Population
    MacroDataSource(
        name="UK Population",
        url="https://fred.stlouisfed.org/series/POPTOTGBA647NWDB",
        country="UK",
        data_type="Population",
        source_type="api",
        authority="World Bank via FRED",
        description="UK total population, annual",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="POPTOTGBA647NWDB",
        update_frequency="annually",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1960",
        backup_sources=["https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration"]
    ),
    
    # Property Price Index
    MacroDataSource(
        name="UK Residential Property Price Index",
        url="https://fred.stlouisfed.org/series/QGBR628BIS",
        country="UK",
        data_type="Property_Price",
        source_type="api",
        authority="Bank for International Settlements (BIS) via FRED",
        description="UK Residential Property Price Index, quarterly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="QGBR628BIS",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1968",
        backup_sources=["https://www.gov.uk/government/collections/uk-house-price-index-reports"]
    ),
]

# =============================================================================
# INDIA MACROECONOMIC DATA SOURCES
# =============================================================================

INDIA_SOURCES = [
    # GDP Data
    MacroDataSource(
        name="India Real GDP",
        url="https://fred.stlouisfed.org/series/NGDPRNSAXDCINQ",
        country="India",
        data_type="GDP",
        source_type="api",
        authority="International Monetary Fund via FRED",
        description="India Real GDP, quarterly, seasonally adjusted",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="NGDPRNSAXDCINQ",
        api_documentation="https://fred.stlouisfed.org/docs/api/fred/",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="2004",
        backup_sources=[]
    ),
    
    # CPI Data
    MacroDataSource(
        name="India Consumer Price Index",
        url="https://fred.stlouisfed.org/series/INDCPIALLMINMEI",
        country="India",
        data_type="CPI",
        source_type="api",
        authority="Organization for Economic Co-operation and Development via FRED",
        description="India Consumer Price Index for All Items, monthly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="INDCPIALLMINMEI",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1957",
        backup_sources=[]
    ),
    
    # Interest Rate
    MacroDataSource(
        name="India Interest Rate",
        url="https://fred.stlouisfed.org/series/IRSTCI01INM156N",
        country="India",
        data_type="Interest_Rate",
        source_type="api",
        authority="Organization for Economic Co-operation and Development via FRED",
        description="India immediate rates (< 24 hours), monthly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="IRSTCI01INM156N",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1968",
        backup_sources=["https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=3552"]
    ),
    
    # Population
    MacroDataSource(
        name="India Population",
        url="https://fred.stlouisfed.org/series/POPTOTINA647NWDB",
        country="India",
        data_type="Population",
        source_type="api",
        authority="World Bank via FRED",
        description="India total population, annual",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="POPTOTINA647NWDB",
        update_frequency="annually",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1960",
        backup_sources=["https://www.censusindia.gov.in/"]
    ),
    
    # Property Price Index
    MacroDataSource(
        name="India Residential Property Price Index",
        url="https://fred.stlouisfed.org/series/QINR628BIS",
        country="India",
        data_type="Property_Price",
        source_type="api",
        authority="Bank for International Settlements (BIS) via FRED",
        description="India Residential Property Price Index, quarterly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="QINR628BIS",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="2009",
        backup_sources=[]
    ),
]

# =============================================================================
# JAPAN MACROECONOMIC DATA SOURCES (FIXED)
# =============================================================================

JAPAN_SOURCES = [
    # GDP Data - FIXED
    MacroDataSource(
        name="Japan Real GDP",
        url="https://fred.stlouisfed.org/series/JPNRGDPEXP",
        country="Japan",
        data_type="GDP",
        source_type="api",
        authority="Federal Reserve Bank of St. Louis (OECD source)",
        description="Japan Real GDP, quarterly, seasonally adjusted",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="JPNRGDPEXP",  # FIXED: Add series ID
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1960",
        backup_sources=["https://www.esri.cao.go.jp/en/sna/data/sokuhou/files/2023/qe234_2/gdemenuja.html"]
    ),
    
    # CPI Data - FIXED
    MacroDataSource(
        name="Japan Consumer Price Index",
        url="https://fred.stlouisfed.org/series/JPNCPIALLMINMEI",
        country="Japan",
        data_type="CPI",
        source_type="api",
        authority="Federal Reserve Bank of St. Louis (OECD source)",
        description="Japan Consumer Price Index, all items, monthly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="JPNCPIALLMINMEI",  # FIXED: Add series ID
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1955",
        backup_sources=["https://www.stat.go.jp/english/data/cpi/"]
    ),
    
    # Bank of Japan Interest Rates
    MacroDataSource(
        name="Japan Call Money/Interbank Rate",
        url="https://fred.stlouisfed.org/series/IRSTCI01JPM156N",
        country="Japan",
        data_type="Interest_Rate",
        source_type="api",
        authority="Organization for Economic Co-operation and Development via FRED",
        description="Japan call money/interbank rate, immediate rates (< 24 hours)",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="IRSTCI01JPM156N",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1985",
        backup_sources=["https://www.stat-search.boj.or.jp/ssi/mtshtml/fm02_m_1_en.html"]
    ),
    
    # Japan Population - FIXED
    MacroDataSource(
        name="Japan Population",
        url="https://fred.stlouisfed.org/series/POPTOTJPA647NWDB",
        country="Japan",
        data_type="Population",
        source_type="api",
        authority="World Bank",
        description="Japan total population",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="POPTOTJPA647NWDB",  # FIXED: Add series ID
        update_frequency="annually",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1960",
        backup_sources=["https://www.stat.go.jp/english/data/jinsui/"]
    ),

    # Japan Property Price Index - FIXED
    MacroDataSource(
        name="Real Residential Property Prices for Japan",
        url="https://fred.stlouisfed.org/series/QJPR628BIS",
        country="Japan",
        data_type="Property_Price",
        source_type="api",
        authority="Bank for International Settlements (BIS)",
        description="Residential Property Price Index for Japan, quarterly, not seasonally adjusted",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="QJPR628BIS",  # FIXED: Add series ID
        update_frequency="quarterly",
        data_format="json",
        extraction_complexity="simple",  # FIXED: Changed from medium to simple
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1955",
        requires_api_key=True
    ),
]

# =============================================================================
# Neighboring Countries MACROECONOMIC DATA SOURCES
# =============================================================================

# =============================================================================
# INDONESIA MACROECONOMIC DATA SOURCES
# =============================================================================

INDONESIA_SOURCES = [
    # GDP Data
    MacroDataSource(
        name="Indonesia Real GDP",
        url="https://fred.stlouisfed.org/series/NGDPRSAXDCIDQ",
        country="Indonesia",
        data_type="GDP",
        source_type="api",
        authority="International Monetary Fund via FRED",
        description="Indonesia Real GDP, quarterly, seasonally adjusted",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="NGDPRSAXDCIDQ",
        api_documentation="https://fred.stlouisfed.org/docs/api/fred/",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="2000",
        backup_sources=["https://www.bps.go.id/en/statistics-table?subject=530"]
    ),
    
    # CPI Data
    MacroDataSource(
        name="Indonesia Consumer Price Index",
        url="https://fred.stlouisfed.org/series/IDNCPIALLMINMEI",
        country="Indonesia",
        data_type="CPI",
        source_type="api",
        authority="Organization for Economic Co-operation and Development via FRED",
        description="Indonesia Consumer Price Index for All Items, monthly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="IDNCPIALLMINMEI",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1968",
        backup_sources=["https://www.bps.go.id/subject/3/inflasi.html"]
    ),
    
    # Interest Rate
    MacroDataSource(
        name="Indonesia Interest Rate",
        url="https://fred.stlouisfed.org/series/IRSTCI01IDM156N",
        country="Indonesia",
        data_type="Interest_Rate",
        source_type="api",
        authority="Organization for Economic Co-operation and Development via FRED",
        description="Indonesia immediate rates (< 24 hours), monthly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="IRSTCI01IDM156N",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1990",
        backup_sources=[]
    ),
    
    # Population
    MacroDataSource(
        name="Indonesia Population",
        url="https://fred.stlouisfed.org/series/POPTOTIDA647NWDB",
        country="Indonesia",
        data_type="Population",
        source_type="api",
        authority="World Bank via FRED",
        description="Indonesia total population, annual",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="POPTOTIDA647NWDB",
        update_frequency="annually",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1960",
        backup_sources=["https://www.bps.go.id/subject/12/kependudukan.html"]
    ),
    
    # Property Price Index
    MacroDataSource(
        name="Indonesia Residential Property Price Index",
        url="https://fred.stlouisfed.org/series/QIDR628BIS",
        country="Indonesia",
        data_type="Property_Price",
        source_type="api",
        authority="Bank for International Settlements (BIS) via FRED",
        description="Indonesia Residential Property Price Index, quarterly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="QIDR628BIS",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="2002",
        backup_sources=[]
    ),
]

# =============================================================================
# MALAYSIA MACROECONOMIC DATA SOURCES
# =============================================================================

MALAYSIA_SOURCES = [
    # GDP Data
    MacroDataSource(
        name="Malaysia Real GDP (Quarterly)",
        url="https://data.gov.my/data-catalogue/gdp_qtr_real_sa",
        country="Malaysia",
        data_type="GDP",
        source_type="api",
        authority="Department of Statistics Malaysia",
        description="Malaysia Real GDP, quarterly data, seasonally adjusted",
        api_url="https://api.data.gov.my/data-catalogue?id=gdp_qtr_real_sa",
        api_documentation="https://developer.data.gov.my/",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=False,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="2015",
        backup_sources=[]
    ),
    
    # CPI Data
    MacroDataSource(
        name="Malaysia Consumer Price Index",
        url="https://data.gov.my/data-catalogue/cpi_headline",
        country="Malaysia",
        data_type="CPI",
        source_type="api",
        authority="Department of Statistics Malaysia",
        description="Malaysia CPI headline index, monthly data",
        api_url="https://api.data.gov.my/data-catalogue?id=cpi_headline",
        api_documentation="https://developer.data.gov.my/",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=False,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="2000",
        backup_sources=[]
    ),
    
    # Interest Rate Data - FIXED: Now properly marked as web_scraping
    MacroDataSource(
        name="Malaysia Overnight Policy Rate",
        url="https://financialmarkets.bnm.gov.my/data-download-opr?date_range=all_data",
        country="Malaysia",
        data_type="Interest_Rate",
        source_type="web_scraping",  # Using web scraping method
        authority="Bank Negara Malaysia",
        description="Malaysia overnight policy rate, irregular updates",
        api_url=None,  # No API available
        api_documentation=None,
        update_frequency="irregular",
        data_format="html",  # Web scraping returns HTML
        requires_api_key=False,
        extraction_complexity="medium",  # Web scraping is more complex
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="2004",
        backup_sources=[]
    ),
    
    # Population Data
    MacroDataSource(
        name="Malaysia Population",
        url="https://data.gov.my/data-catalogue/population_malaysia",
        country="Malaysia",
        data_type="Population",
        source_type="api",
        authority="Department of Statistics Malaysia",
        description="Malaysia total population, annual data",
        api_url="https://api.data.gov.my/data-catalogue?id=population_malaysia",
        api_documentation="https://developer.data.gov.my/",
        update_frequency="annually",
        data_format="json",
        requires_api_key=False,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1970",
        backup_sources=[]
    ),
    
    # Property Price Index
    MacroDataSource(
        name="Malaysia Real Residential Property Price Index (BIS)",
        url="https://fred.stlouisfed.org/series/QMYR628BIS",
        country="Malaysia",
        data_type="Property_Price",
        source_type="api",
        authority="Bank for International Settlements (BIS)",
        description="Real residential property prices for Malaysia, deflated using CPI",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="QMYR628BIS",  # FRED series ID for dynamic construction
        api_documentation="https://fred.stlouisfed.org/docs/api/fred/",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=10,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1988",
        backup_sources=["https://www.bis.org/statistics/pp_detailed.htm"]
    ),
]

# =============================================================================
# THAILAND MACROECONOMIC DATA SOURCES
# =============================================================================

THAILAND_SOURCES = [
    # GDP Data (IMF Database)
    MacroDataSource(
        name="Thailand Real GDP",
        url="https://data.imf.org/en/Data%20Explorer?datasetUrn=IMF.STA%3AQNEA%287.0.0%29&timeseriesName=THA.B1GQ.Q.SA.XDC.Q",
        country="Thailand",
        data_type="GDP",
        source_type="api",
        authority="International Monetary Fund",
        description="Thailand Real GDP, quarterly, seasonally adjusted",
        api_url="https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/QNEA/7.0.0/THA.B1GQ.Q.SA.XDC.Q?dimensionAtObservation=TIME_PERIOD&attributes=dsd&measures=all&includeHistory=false",
        api_documentation="https://data.imf.org/docs/",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=False,
        extraction_complexity="medium",
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="2003",
        backup_sources=["https://www.bot.or.th/english/statistics/econdata/data/"]
    ),
    
    # CPI Data (IMF Database)
    MacroDataSource(
        name="Thailand Consumer Price Index",
        url="https://data.imf.org/en/Data%20Explorer?datasetUrn=IMF.STA%3ACPI%284.0.0%29&timeseriesName=THA.CPI._T.IX.M",
        country="Thailand",
        data_type="CPI",
        source_type="api",
        authority="International Monetary Fund",
        description="Thailand Consumer Price Index, monthly",
        api_url="https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/CPI/+/THA.CPI._T.IX.M?dimensionAtObservation=TIME_PERIOD&attributes=dsd&measures=all&includeHistory=false",
        update_frequency="monthly",
        data_format="json",
        requires_api_key=False,
        extraction_complexity="medium",
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="2010",
        backup_sources=["https://www.bot.or.th/english/statistics/econdata/data/"]
    ),
    
    # Interest Rate (Bank of Thailand - Special Excel Download)
    MacroDataSource(
        name="Thailand Interest Rate (THOR)",
        url="https://app.bot.or.th/THOR/en",
        country="Thailand",
        data_type="Interest_Rate",
        source_type="excel_download",
        authority="Bank of Thailand",
        description="Thailand Historical Over-night Repo rate (THOR) from 2015, daily data",
        api_url="https://www.thaibma.or.th/api/thor/download-all",
        api_documentation="https://app.bot.or.th/THOR/en",
        update_frequency="daily",
        data_format="xlsx",
        requires_api_key=False,
        extraction_complexity="complex",  # Special handling required for Excel download
        real_time_data=True,
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="2015",
        backup_sources=[
            "https://app.bot.or.th/THOR/en",
            "https://www.bot.or.th/english/statistics/financialmarkets/interestrate/"
        ],
    ),
    
    # Population (FRED Database)
    MacroDataSource(
        name="Thailand Population",
        url="https://fred.stlouisfed.org/series/POPTOTTHA647NWDB",
        country="Thailand",
        data_type="Population",
        source_type="api",
        authority="World Bank via FRED",
        description="Thailand total population, annual",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="POPTOTTHA647NWDB",
        api_documentation="https://fred.stlouisfed.org/docs/api/fred/",
        update_frequency="annually",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1960",
        backup_sources=["https://www.nso.go.th/sites/2014en/Pages/Census/Population-and-Housing/The-2020-population-and-housing-census.aspx"]
    ),
    
    # Property Price Index (FRED Database)
    MacroDataSource(
        name="Thailand Residential Property Price Index",
        url="https://fred.stlouisfed.org/series/QTHR628BIS",
        country="Thailand",
        data_type="Property_Price",
        source_type="api",
        authority="Bank for International Settlements (BIS) via FRED",
        description="Thailand Residential Property Price Index, quarterly",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="QTHR628BIS",
        api_documentation="https://fred.stlouisfed.org/docs/api/fred/",
        update_frequency="quarterly",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1991",
        backup_sources=["https://www.bot.or.th/english/statistics/property/"]
    ),
]

# =============================================================================
# VIETNAM MACROECONOMIC DATA SOURCES
# =============================================================================

VIETNAM_SOURCES = [
    # GDP Data
    MacroDataSource(
        name="Vietnam Real GDP",
        url="https://databank.worldbank.org/reports.aspx?source=2&series=NY.GDP.MKTP.KN&country=VNM",
        country="Vietnam",
        data_type="GDP",
        source_type="api",
        authority="World Bank",
        description="Vietnam GDP at constant 2015 prices (Local Currency Unit), annual",
        wb_series_id="NY.GDP.MKTP.KN",
        wb_country_code="VNM",
        update_frequency="annually",
        data_format="json",
        requires_api_key=False,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1960",
        backup_sources=["https://fred.stlouisfed.org/series/VNMNGDPRPCPPPT"]
    ),
    
    # CPI Data
    MacroDataSource(
        name="Vietnam Consumer Price Index",
        url="https://databank.worldbank.org/reports.aspx?source=2&series=FP.CPI.TOTL&country=VNM",
        country="Vietnam",
        data_type="CPI",
        source_type="api",
        authority="World Bank",
        description="Vietnam Consumer Price Index (2010 = 100), annual",
        wb_series_id="FP.CPI.TOTL",
        wb_country_code="VNM",
        update_frequency="annually",
        data_format="json",
        requires_api_key=False,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1960",
        backup_sources=["https://fred.stlouisfed.org/series/FPCPITOTLZGVNM"]
    ),
    
    # Population
    MacroDataSource(
        name="Vietnam Population",
        url="https://fred.stlouisfed.org/series/POPTOTVNA647NWDB",
        country="Vietnam",
        data_type="Population",
        source_type="api",
        authority="World Bank via FRED",
        description="Vietnam total population, annual",
        api_url=None,  # Will be constructed dynamically
        fred_series_id="POPTOTVNA647NWDB",
        update_frequency="annually",
        data_format="json",
        requires_api_key=True,
        extraction_complexity="simple",
        data_quality_rating=9,
        api_rate_limits="120 requests per 60 seconds",
        authentication_method="API Key",
        data_coverage_start="1960",
        backup_sources=["https://www.gso.gov.vn/en/"]
    ),
    
    # Interest Rate - DROPPED for now due to data accessibility issues
    # MacroDataSource(
    #     name="Vietnam Interest Rate",
    #     url="https://dttktt.sbv.gov.vn/webcenter/portal/en/home/rm/ir/ibir",
    #     country="Vietnam",
    #     data_type="Interest_Rate", 
    #     source_type="manual_scraping",
    #     authority="State Bank of Vietnam",
    #     description="Vietnam interbank interest rates, daily",
    #     update_frequency="daily",
    #     data_format="html",
    #     extraction_complexity="very_high",
    #     data_quality_rating=8,
    #     notes="DROPPED: Daily data only, requires manual clicking for historical data"
    # ),
    
    # Property Price Index - NOT AVAILABLE
    # MacroDataSource(
    #     name="Vietnam Property Price Index",
    #     url="Not available",
    #     country="Vietnam", 
    #     data_type="Property_Price",
    #     source_type="not_available",
    #     authority="Not available",
    #     description="Vietnam does not publish official property price statistics",
    #     notes="BIS does not have Vietnam property price data. No official government property price index available."
    # ),
]

# =============================================================================
# MARKET INDICES SOURCES
# =============================================================================

MARKET_INDICES_SOURCES = [
    # S&P 500
    MacroDataSource(
        name="S&P 500 Index",
        url="https://finance.yahoo.com/quote/%5EGSPC/",
        country="US",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="S&P 500 stock market index, real-time daily data with OHLCV",
        api_url="^GSPC",  # yfinance symbol
        api_documentation="https://pypi.org/project/yfinance/",
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1985",
        backup_sources=["https://fred.stlouisfed.org/series/SP500"]
    ),
    
    # NASDAQ
    MacroDataSource(
        name="NASDAQ Composite Index",
        url="https://finance.yahoo.com/quote/%5EIXIC/",
        country="US",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="NASDAQ Composite Index, real-time daily data with OHLCV",
        api_url="^IXIC",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1971",
        backup_sources=["https://fred.stlouisfed.org/series/NASDAQCOM"]
    ),
    
    # Shanghai Composite
    MacroDataSource(
        name="Shanghai Composite Index",
        url="https://finance.yahoo.com/quote/000001.SS/",
        country="China",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="Shanghai Stock Exchange Composite Index, real-time data",
        api_url="000001.SS",  # yfinance symbol for Shanghai Composite
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1997",
        backup_sources=["https://www.sse.com.cn/assortment/stocks/overview/"]
    ),
    
    # Hang Seng Index
    MacroDataSource(
        name="Hang Seng Index",
        url="https://finance.yahoo.com/quote/%5EHSI/",
        country="Hong Kong",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="Hong Kong Hang Seng Index, real-time data",
        api_url="^HSI",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1986",
        backup_sources=["https://www.hsi.com.hk/eng"]
    ),
    
    # Nikkei 225
    MacroDataSource(
        name="Nikkei 225",
        url="https://finance.yahoo.com/quote/%5EN225/",
        country="Japan",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="Japan Nikkei 225 Stock Average, real-time data",
        api_url="^N225",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1965",
        backup_sources=["https://www.nikkei.com/nkave/"]
    ),
    
    # EURO STOXX 50
    MacroDataSource(
        name="EURO STOXX 50",
        url="https://finance.yahoo.com/quote/%5ESTOXX50E/",
        country="EU",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="EURO STOXX 50 Index, real-time data",
        api_url="^STOXX50E",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="2007",
        backup_sources=["https://www.stoxx.com/index-details?symbol=SX5E"]
    ),
    
    # Singapore STI
    MacroDataSource(
        name="Straits Times Index",
        url="https://finance.yahoo.com/quote/%5ESTI/",
        country="Singapore",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="Singapore Straits Times Index, real-time data",
        api_url="^STI",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1987",
        backup_sources=["https://www.sgx.com/indices/products/sti"]
    ),
    
    # VIX (Volatility Index)
    MacroDataSource(
        name="CBOE Volatility Index (VIX)",
        url="https://finance.yahoo.com/quote/%5EVIX/",
        country="US",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="CBOE Volatility Index (VIX), market fear indicator",
        api_url="^VIX",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1990",
        backup_sources=["https://fred.stlouisfed.org/series/VIXCLS", "https://www.cboe.com/tradable_products/vix/"]
    ),

    # FTSE 100 (UK)
    MacroDataSource(
        name="FTSE 100 Index",
        url="https://finance.yahoo.com/quote/%5EFTSE/",
        country="UK",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="FTSE 100 Index, UK's leading share index of top 100 companies",
        api_url="^FTSE",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=10,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1985",
        backup_sources=["https://www.londonstockexchange.com/indices/ftse-100"]
    ),

    # Sensex (India)
    MacroDataSource(
        name="BSE Sensex",
        url="https://finance.yahoo.com/quote/%5EBSESN/",
        country="India",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="BSE Sensex, India's benchmark stock market index",
        api_url="^BSESN",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1997",
        backup_sources=["https://www.bseindia.com/sensex/"]
    ),

    # JKSE (Indonesia)
    MacroDataSource(
        name="Jakarta Composite Index",
        url="https://finance.yahoo.com/quote/%5EJKSE/",
        country="Indonesia",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="Jakarta Composite Index (JKSE), Indonesia's main stock market index",
        api_url="^JKSE",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1990",
        backup_sources=["https://www.idx.co.id/en/market-data/indices/"]
    ),

    # KLCI (Malaysia)
    MacroDataSource(
        name="FTSE Bursa Malaysia KLCI",
        url="https://finance.yahoo.com/quote/%5EKLSE/",
        country="Malaysia",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="FTSE Bursa Malaysia KLCI, Malaysia's benchmark stock index",
        api_url="^KLSE",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1993",
        backup_sources=["https://www.bursamalaysia.com/market_information/equities_prices"]
    ),

    # SET (Thailand)
    MacroDataSource(
        name="SET Index",
        url="https://finance.yahoo.com/quote/%5ESET.BK/",
        country="Thailand",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="Stock Exchange of Thailand (SET) Index",
        api_url="^SET.BK",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=9,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="1996",
        backup_sources=["https://www.set.or.th/en/market/index"]
    ),

    # VN-Index (Vietnam)
    MacroDataSource(
        name="VN-Index",
        url="https://finance.yahoo.com/quote/%5EVNINDEX.VN/",
        country="Vietnam",
        data_type="Stock_Index",
        source_type="api",
        authority="Yahoo Finance via yfinance",
        description="VN-Index, Ho Chi Minh Stock Exchange main index",
        api_url="VNINDEX.VN",  # yfinance symbol
        update_frequency="daily",
        data_format="pandas_dataframe",
        requires_api_key=False,
        extraction_complexity="simple",
        real_time_data=True,
        data_quality_rating=8,
        api_rate_limits="No specific limits documented",
        authentication_method="No authentication required",
        data_coverage_start="Unknown",
        backup_sources=["https://www.hsx.vn/Modules/Listed/Web/SymbolList/en"]
    ),
]

# # =============================================================================
# # HONG KONG MACROECONOMIC DATA SOURCES (Keep unchanged for now)
# # =============================================================================

# HONG_KONG_SOURCES = [
#     # GDP Data
#     MacroDataSource(
#         name="Hong Kong GDP",
#         url="https://www.censtatd.gov.hk/en/scode490.html",
#         country="Hong Kong",
#         data_type="GDP",
#         source_type="csv_download",
#         authority="Census and Statistics Department",
#         description="Hong Kong GDP by expenditure component, quarterly",
#         update_frequency="quarterly",
#         data_format="csv",
#         extraction_complexity="medium",
#         data_quality_rating=9,
#         backup_sources=["https://fred.stlouisfed.org/series/HKNGDPNQDSMEI"]
#     ),
    
#     # CPI Data
#     MacroDataSource(
#         name="Hong Kong Consumer Price Index",
#         url="https://www.censtatd.gov.hk/en/scode520.html",
#         country="Hong Kong",
#         data_type="CPI",
#         source_type="csv_download",
#         authority="Census and Statistics Department",
#         description="Hong Kong Composite Consumer Price Index",
#         update_frequency="monthly",
#         data_format="csv",
#         extraction_complexity="medium",
#         data_quality_rating=9,
#         backup_sources=["https://fred.stlouisfed.org/series/HKGCPIALLMINMEI"]
#     ),
    
#     # Interest Rates
#     MacroDataSource(
#         name="Hong Kong Base Rate",
#         url="https://www.hkma.gov.hk/eng/market-data-and-statistics/monthly-statistical-bulletin/table/",
#         country="Hong Kong",
#         data_type="Interest_Rate",
#         source_type="web_scraping",
#         authority="Hong Kong Monetary Authority",
#         description="Hong Kong dollar base rate",
#         update_frequency="irregular",
#         data_format="html",
#         extraction_complexity="medium",
#         real_time_data=True,
#         data_quality_rating=9,
#         backup_sources=["https://fred.stlouisfed.org/series/INTHKGK01STM"]
#     ),
    
#     # Hong Kong Population
#     MacroDataSource(
#         name="Hong Kong Population",
#         url="https://www.censtatd.gov.hk/en/scode150.html",
#         country="Hong Kong",
#         data_type="Population",
#         source_type="csv_download",
#         authority="Census and Statistics Department",
#         description="Hong Kong total population",
#         update_frequency="annually",
#         data_format="csv",
#         extraction_complexity="medium",
#         data_quality_rating=9,
#         backup_sources=["https://fred.stlouisfed.org/series/SPPOPTOTALHKG"]
#     ),
    
#     # Hong Kong Property Price Index
#     MacroDataSource(
#         name="Hong Kong Property Price Index",
#         url="https://www.rvd.gov.hk/en/property_market_statistics/index.html",
#         country="Hong Kong",
#         data_type="Property_Price",
#         source_type="web_scraping",
#         authority="Rating and Valuation Department",
#         description="Hong Kong private domestic property price index",
#         update_frequency="monthly",
#         data_format="html",
#         extraction_complexity="medium",
#         data_quality_rating=9,
#         backup_sources=["https://fred.stlouisfed.org/series/QHKG628BIS"]
#     ),
# ]

# =============================================================================
# CONSOLIDATED DATA SOURCES
# =============================================================================

ENHANCED_MACROECONOMIC_DATA_SOURCES = {
    "singapore": SINGAPORE_SOURCES,
    "united_states": US_SOURCES,
    "european_union": EU_SOURCES,
    "china": CHINA_SOURCES,
    # "hong_kong": HONG_KONG_SOURCES,
    "united_kingdom": UK_SOURCES,      
    "india": INDIA_SOURCES,            
    "japan": JAPAN_SOURCES,
    "indonesia": INDONESIA_SOURCES,
    "malaysia": MALAYSIA_SOURCES,
    "thailand": THAILAND_SOURCES,
    "vietnam": VIETNAM_SOURCES,
    "market_indices": MARKET_INDICES_SOURCES,
}

# Enhanced Helper functions
def get_sources_by_country(country: str) -> List[MacroDataSource]:
    """Get all sources for a specific country"""
    country_map = {
        'singapore': 'singapore',
        'us': 'united_states',
        'usa': 'united_states',
        'united_states': 'united_states',
        'eu': 'european_union',
        'europe': 'european_union',
        'european_union': 'european_union',
        'china': 'china',
        'hong_kong': 'hong_kong',
        'hk': 'hong_kong',
        'japan': 'japan',
        'jp': 'japan',
        'uk': 'united_kingdom',           
        'united_kingdom': 'united_kingdom',  
        'india': 'india',                 
        'indonesia': 'indonesia',
        'malaysia': 'malaysia', 
        'thailand': 'thailand',
        'vietnam': 'vietnam',       
    }
    
    key = country_map.get(country.lower(), country.lower())
    return ENHANCED_MACROECONOMIC_DATA_SOURCES.get(key, [])

def get_sources_by_data_type(data_type: str) -> List[MacroDataSource]:
    """Get all sources for a specific data type"""
    sources = []
    for country_sources in ENHANCED_MACROECONOMIC_DATA_SOURCES.values():
        for source in country_sources:
            if source.data_type.lower() == data_type.lower():
                sources.append(source)
    return sources

def get_api_sources() -> List[MacroDataSource]:
    """Get all sources that have API access"""
    api_sources = []
    for country_sources in ENHANCED_MACROECONOMIC_DATA_SOURCES.values():
        for source in country_sources:
            if source.source_type == "api":
                api_sources.append(source)
    return api_sources

def get_high_quality_sources(min_rating: int = 9) -> List[MacroDataSource]:
    """Get sources with high data quality ratings"""
    high_quality = []
    for country_sources in ENHANCED_MACROECONOMIC_DATA_SOURCES.values():
        for source in country_sources:
            if source.data_quality_rating >= min_rating:
                high_quality.append(source)
    return high_quality

def get_sources_by_complexity(complexity: str) -> List[MacroDataSource]:
    """Get sources by extraction complexity"""
    complexity_sources = []
    for country_sources in ENHANCED_MACROECONOMIC_DATA_SOURCES.values():
        for source in country_sources:
            if source.extraction_complexity.lower() == complexity.lower():
                complexity_sources.append(source)
    return complexity_sources

def get_real_time_sources() -> List[MacroDataSource]:
    """Get sources that provide real-time data"""
    real_time = []
    for country_sources in ENHANCED_MACROECONOMIC_DATA_SOURCES.values():
        for source in country_sources:
            if source.real_time_data:
                real_time.append(source)
    return real_time

def get_all_macro_sources() -> List[MacroDataSource]:
    """Get all macroeconomic sources"""
    all_sources = []
    for country_sources in ENHANCED_MACROECONOMIC_DATA_SOURCES.values():
        all_sources.extend(country_sources)
    return all_sources

def get_sources_requiring_api_key() -> List[MacroDataSource]:
    """Get sources that require API keys"""
    api_key_sources = []
    for country_sources in ENHANCED_MACROECONOMIC_DATA_SOURCES.values():
        for source in country_sources:
            if source.requires_api_key:
                api_key_sources.append(source)
    return api_key_sources

# Enhanced summary statistics
def get_enhanced_sources_summary() -> Dict[str, Any]:
    """Get comprehensive summary of available sources"""
    all_sources = get_all_macro_sources()
    
    summary = {
        'total_sources': len(all_sources),
        'sources_by_country': {country: len(sources) for country, sources in ENHANCED_MACROECONOMIC_DATA_SOURCES.items()},
        'sources_by_data_type': {},
        'sources_by_source_type': {},
        'sources_by_complexity': {},
        'api_sources_count': len(get_api_sources()),
        'high_quality_sources_count': len(get_high_quality_sources()),
        'real_time_sources_count': len(get_real_time_sources()),
        'api_key_required_count': len(get_sources_requiring_api_key()),
        'coverage_statistics': {}
    }
    
    # Count by various dimensions
    data_types = {}
    source_types = {}
    complexities = {}
    authorities = {}
    
    for source in all_sources:
        data_types[source.data_type] = data_types.get(source.data_type, 0) + 1
        source_types[source.source_type] = source_types.get(source.source_type, 0) + 1
        complexities[source.extraction_complexity] = complexities.get(source.extraction_complexity, 0) + 1
        authorities[source.authority] = authorities.get(source.authority, 0) + 1
    
    summary['sources_by_data_type'] = data_types
    summary['sources_by_source_type'] = source_types
    summary['sources_by_complexity'] = complexities
    summary['top_authorities'] = dict(sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:5])
    
    return summary

# Priority source recommendations
def get_priority_sources() -> Dict[str, List[MacroDataSource]]:
    """Get priority sources for immediate implementation"""
    return {
        'immediate_priority': [
            # Singapore sources (all have good APIs)
            *[s for s in SINGAPORE_SOURCES if s.source_type == "api"],
            # US FRED sources (excellent APIs)
            *[s for s in US_SOURCES if s.source_type == "api"],
            # EU ECB sources (good APIs)
            *[s for s in EU_SOURCES if s.source_type == "api"]
        ],
        'secondary_priority': [
            # Market indices (Yahoo Finance APIs)
            *[s for s in MARKET_INDICES_SOURCES if s.data_quality_rating >= 8],
            # China and Hong Kong sources
            # *[s for s in CHINA_SOURCES + HONG_KONG_SOURCES if s.extraction_complexity != "complex"]
            # Japan sources
            *[s for s in JAPAN_SOURCES if s.extraction_complexity != "complex"]
        ],
        'future_implementation': [
            # Complex extraction sources
            *get_sources_by_complexity("complex")
        ]
    }

if __name__ == "__main__":
    # Print enhanced summary
    summary = get_enhanced_sources_summary()
    print("🌍 ENHANCED MACROECONOMIC DATA SOURCES SUMMARY (FIXED VERSION)")
    print("=" * 70)
    print(f"📊 Total Sources: {summary['total_sources']}")
    print(f"🔌 API Sources: {summary['api_sources_count']}")
    print(f"🔑 API Key Required: {summary['api_key_required_count']}")
    print(f"⭐ High Quality Sources (9+): {summary['high_quality_sources_count']}")
    print(f"⚡ Real-time Sources: {summary['real_time_sources_count']}")
    
    print(f"\n📍 Sources by Country:")
    for country, count in summary['sources_by_country'].items():
        print(f"   {country.replace('_', ' ').title()}: {count}")
    
    print(f"\n📈 Sources by Data Type:")
    for data_type, count in summary['sources_by_data_type'].items():
        print(f"   {data_type.replace('_', ' ')}: {count}")
    
    print(f"\n🔧 Sources by Type:")
    for source_type, count in summary['sources_by_source_type'].items():
        print(f"   {source_type.replace('_', ' ').title()}: {count}")
    
    print(f"\n⚙️ Sources by Complexity:")
    for complexity, count in summary['sources_by_complexity'].items():
        print(f"   {complexity.title()}: {count}")
    
    print(f"\n🏛️ Top Authorities:")
    for authority, count in summary['top_authorities'].items():
        print(f"   {authority}: {count}")
    
    # Show priority recommendations
    priority_sources = get_priority_sources()
    print(f"\n🎯 IMPLEMENTATION PRIORITY RECOMMENDATIONS:")
    print(f"   Immediate Priority: {len(priority_sources['immediate_priority'])} sources")
    print(f"   Secondary Priority: {len(priority_sources['secondary_priority'])} sources")
    print(f"   Future Implementation: {len(priority_sources['future_implementation'])} sources")
    
    # Show FRED API key status
    print(f"\n🔑 FRED API Configuration:")
    if FRED_API_KEY:
        print(f"   ✅ FRED API Key loaded: {FRED_API_KEY[:8]}...")
    else:
        print(f"   ❌ FRED API Key missing - add to .env file")
        print(f"   📝 Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
