#### This Streamlit dashboard has been deployed on Streamlit cloud, the link is as followed:  
#### https://capstone-project-nusmsba2025-landis.streamlit.app/
---
## 🎯 Quick Start

### 1. Macroeconomic and Market Data Pipeline
```bash
python main_runner.py
```
This will show an interactive menu for data pipeline.
Run the 2nd option "Everything Complete Pipeline" for the complete pipeline (Validation → Extraction → Cleaning → Dashboard).

### 2. News and Forum Data Pipeline
```bash
# News scraping and analysis
python news_scraper_selenium.py
python news_analyzer.py

# Forum scraping and analysis
python forum_scraper.py
python forum_analyzer.py
```

### 3. Launch the Dashboard (with existing data sheets)
```bash
streamlit run streamlit_dashboard.py
```
This opens the interactive web dashboard for data visualization.

(Note: You need to have your own FRED API key and put it in .env file with this format: FRED_API_KEY=...)

---

## 📂 Code Structure & Categories

### **🔧 Macroeconomic and Market Data Pipeline**
| File | Purpose | Entry Point |
|------|---------|-------------|
| `main_runner.py` | **Main entry point** - Interactive pipeline runner | ✅ **START HERE** |
| `main_pipeline.py` | Core validation logic (HTTP/API based) | Used by runner |
| `api_first_validator.py` | Smart API validation with yfinance support | Used by pipeline |
| `rule_based_extraction_tools.py` | Data extraction + Excel export | Used by runner |
| `consolidate_analysis.py` | Consolidates analysis across different data sources | Standalone |

### **📊 Data Sources & Configuration**
| File | Purpose | Description |
|------|---------|-------------|
| `macro_sources.py` | **Data source definitions** | Singapore, US, EU, Japan sources with API endpoints |
| `visualization_config.py` | Shared utilities for all visualizations | Colors, formatting, data processing |
| `worldbank_data.py` | World Bank data integration | Additional economic indicators |

### **🧹 Data Cleaning & Processing**
| File | Purpose | When to Use |
|------|---------|-------------|
| `singapore_data_cleaner.py` | Clean Singapore Excel files | After data extraction |
| `singapore_standardizer.py` | Convert Singapore data to vertical format | After cleaning |
| `fred_data_cleaner.py` | Clean US/EU/Japan data (filter to 2000+) | After data extraction |
| `table_data_scraper.py` | Generic table data extraction from web sources | As needed |

### **🌍 Country-Specific Data Extractors**
| File | Purpose | Coverage |
|------|---------|-----------|
| `malaysia_data_extractor.py` | Malaysia economic data extraction | GDP, CPI, interest rates |
| `thailand_data_extractor.py` | Thailand economic data extraction | GDP, CPI, population |
| `vietnam_data_extractor.py` | Vietnam economic data extraction | GDP, inflation, trade data |

### **📰 News & Sentiment Analysis**
| File | Purpose | Functionality |
|------|---------|---------------|
| `news_scraper_selenium.py` | Web scraping for news articles | Selenium-based scraping |
| `news_analyzer.py` | News content analysis and sentiment scoring | Text processing, sentiment analysis |
| `news_sentiment_standardization.py` | Standardize sentiment analysis output | Data normalization |

### **💬 Forum Analysis**
| File | Purpose | Functionality |
|------|---------|---------------|
| `forum_scraper.py` | Extract discussions from forum threads | Web scraping, content extraction |
| `forum_analyzer.py` | Analyze forum discussions for sentiment and topics by week | NLP, sentiment analysis |
| `forum_visualizer.py` | Visualize forum sentiment and trends | Charts, trend analysis |

### **🔮 Prediction & Testing**
| File | Purpose | Functionality |
|------|---------|---------------|
| `prediction_analysis_test.py` | Economic prediction modeling and testing | ML models, backtesting |

### **📈 Visualization & Analysis**
| File | Purpose | How to Run |
|------|---------|------------|
| `streamlit_dashboard.py` | **Main web dashboard** | `streamlit run streamlit_dashboard.py` |
| `dashboard.py` | Alternative dashboard loading | `python dashboard.py` |
| `cross_country_visualizer.py` | Multi-country economic comparison | `python cross_country_visualizer.py` |
| `market_indices_visualizer.py` | Financial market analysis | `python market_indices_visualizer.py` |
| `news_visualizer.py` | News analysis result visualization | `python news_visualizer.py` |
| `forum_visualizer.py` | Forum discussion analysis and visualization | `python forum_visualizer.py` |

---

## 📊 Data Sources Covered

### **Macroeconomic Data:**
- **Singapore** (via data.gov.sg API): GDP, CPI, Interest Rate, Population, Property Price
- **China** (via various sources): GDP, CPI, Industrial Production
- **United States** (via FRED API): GDP, CPI, Federal Funds Rate, Population, Mortgage Rate, Property Price
- **Euro Area** (via FRED API): GDP, HICP, ECB Rate, Population, Property Price
- **United Kingdom** (via various sources): GDP, CPI, Interest Rate
- **Japan** (via FRED API): GDP, CPI, Population, Property Price
- **Malaysia** (via custom extractor): GDP, CPI, Interest Rates
- **Thailand** (via custom extractor): GDP, CPI, Population
- **Vietnam** (via custom extractor): GDP, Inflation, Trade Data
- **India** (via various sources): GDP, CPI, Interest Rates
- **Indonesia** (via various sources): GDP, CPI, Exchange Rate

### **Financial Market Data:**
- **Global Stock Market Indices** (via Yahoo Finance):
  - **United States**: S&P 500, NASDAQ, VIX
  - **China**: Shanghai Composite
  - **Hong Kong**: Hang Seng 
  - **Japan**: Nikkei 225
  - **Euro Area**: Euro STOXX 50
  - **United Kingdom**: FTSE 100
  - **India**: BSE Sensex
  - **Singapore**: Straits Times
  - **Malaysia**: FTSE Bursa Malaysia KLCI
  - **Indonesia**: Jakarta Composite
  - **Thailand**: SET

### **News Data:**
- **BBC Business News**
- **Yahoo Finance Economic**
- **Yahoo Finance Stock Market**
- **Business Times Singapore**

### **Forum Data:**
- **HardwareZone**

### **Additional Macroeconomic Data:**
- **World Bank Data** (via API): Annual GDP (in Trillion USD), GDP per Capita(in USD)

---

## 📁 Expected Output

### **Excel Files (in `./extracted_data/`):**
- `cleaned_macro_data_[country]_[timestamp].xlsx` - Cleaned economic data by country
- `standardized_cleaned_macro_data_singapore_[timestamp].xlsx` - Standardized Singapore data
- `market_indices_data_[timestamp].xlsx` - Financial market data
- `world_bank_gdp_data_[timestamp].xlsx` - World Bank GDP data

### **Analysis Output (in `./news_analysis_output/` and `./forum_analysis_output/`):**
- `master_news_analysis.xlsx` - Consolidated news analysis outputs (include summary, region, topic, sentiment, and impact)
- `master_forum_analysis.xlsx` - Consolidated forum threads analysis outputs (include summary, topic, and sentiment)

### **News/Forum Scraping Output (will be in `./news_data/` and `./forum_data/` after running):**
- `{news_source}_{timestamp}.json` - Raw news article extraction for each news sources 
- `{thread_{id}_{timestamp}}.json` - Raw forum thread discussions extraction for each thread/forum

### **Excel Structure:**
- **Contents Sheet**: Summary with hyperlinks to data sheets
- **Individual Sheets**: Each economic indicator in separate sheet
- **Cleaned Data**: Proper date formatting, source URLs, metadata

---
