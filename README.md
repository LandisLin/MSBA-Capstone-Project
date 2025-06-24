## 🎯 Quick Start

### 1. Run the Main Pipeline
```bash
python main_runner.py
```
This will show an interactive menu for data pipeline.
Run the 2nd option "Everything Complete Pipeline" for the complete pipeline (Validation → Extraction → Cleaning → Dashboard).

### 2. Launch the Dashboard (with existing data sheets)
```bash
streamlit run streamlit_dashboard.py
```
This opens the interactive web dashboard for data visualization.

---

## 📂 Code Structure & Categories

### **🔧 Core Pipeline (Main Components)**
| File | Purpose | Entry Point |
|------|---------|-------------|
| `main_runner.py` | **Main entry point** - Interactive pipeline runner | ✅ **START HERE** |
| `main_pipeline.py` | Core validation logic (HTTP/API based) | Used by runner |
| `api_first_validator.py` | Smart API validation with yfinance support | Used by pipeline |
| `rule_based_extraction_tools.py` | **Data extraction engine** + Excel export | Used by runner |

### **📊 Data Sources & Configuration**
| File | Purpose | Description |
|------|---------|-------------|
| `macro_sources.py` | **Data source definitions** | Singapore, US, EU, Japan sources with API endpoints |
| `visualization_config.py` | Shared utilities for all visualizations | Colors, formatting, data processing |

### **🧹 Data Cleaning & Processing**
| File | Purpose | When to Use |
|------|---------|-------------|
| `singapore_data_cleaner.py` | Clean Singapore Excel files | After data extraction |
| `singapore_standardizer.py` | Convert Singapore data to vertical format | After cleaning |
| `fred_data_cleaner.py` | Clean US/EU/Japan data (filter to 2000+) | After data extraction |

### **📈 Visualization & Analysis**
| File | Purpose | How to Run |
|------|---------|------------|
| `streamlit_dashboard.py` | **Main web dashboard** | `streamlit run streamlit_dashboard.py` |
| `cross_country_visualizer.py` | Multi-country economic comparison | `python cross_country_visualizer.py` |
| `market_indices_visualizer.py` | Financial market analysis | `python market_indices_visualizer.py` |

---

## 📊 Data Sources Covered

### **Macroeconomic Data:**
- **Singapore** (via data.gov.sg API): GDP, CPI, Interest Rate, Population, Property Price
- **United States** (via FRED API): GDP, CPI, Federal Funds Rate, Population, Mortgage Rate, Property Price
- **Euro Area** (via FRED API): GDP, HICP, ECB Rate, Population, Property Price
- **Japan** (via FRED API): GDP, CPI, Population, Property Price

### **Financial Market Data:**
- **Global Market Indices** (via Yahoo Finance): S&P 500, NASDAQ, Hang Seng, Nikkei 225, Euro STOXX 50, Shanghai Composite, Singapore STI, VIX

---

## 📁 Expected Output

### **Excel Files (in `./extracted_data/`):**
- `macro_data_singapore_[timestamp].xlsx` - Singapore economic data
- `macro_data_us_[timestamp].xlsx` - US economic data  
- `macro_data_euro_area_[timestamp].xlsx` - EU economic data
- `macro_data_japan_[timestamp].xlsx` - Japan economic data
- `market_indices_data_[timestamp].xlsx` - Financial market data

### **Excel Structure:**
- **Contents Sheet**: Summary with hyperlinks to data sheets
- **Individual Sheets**: Each economic indicator in separate sheet
- **Cleaned Data**: Proper date formatting, source URLs, metadata

---

## 🛠 Technical Architecture

```
Data Collection Flow:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │ -> │   API Validator  │ -> │ Rule Extractor  │
│ (macro_sources) │    │(api_first_validator)│  │(rule_based_tools)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Visualization  │ <- │  Data Cleaning   │ <- │   Excel Files   │
│   (dashboard)   │    │   (cleaners)     │    │ (extracted_data)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Validation Strategy:**
1. **API-First**: Test API URLs with proper authentication
2. **yfinance Integration**: Market indices via Yahoo Finance
3. **Fallback Methods**: Main URL testing if API fails
4. **Rule-Based Method Selection**: Simple if/then logic based on source metadata

### **Data Processing:**
1. **Source-Specific Extractors**: Singapore API, FRED API, yfinance
2. **Unified Data Format**: Consistent column structure across all sources
3. **Metadata Preservation**: Source URLs, extraction times, data provenance
4. **Excel Integration**: Professional formatting with Contents navigation
