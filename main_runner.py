"""
Enhanced LLM-Free Runner with Integrated Data Cleaning and Dashboard Launch
Complete pipeline: Data Extraction → Data Cleaning → Standardization → Dashboard Launch
"""

import sys
import json
import subprocess
import time
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import pipeline components
from llm_free_pipeline import create_llm_free_pipeline

# Import extraction tools
try:
    from rule_based_extraction_tools import simple_rule_based_extractor
    EXTRACTION_AVAILABLE = True
    print("✅ Rule-based extraction tools loaded")
except ImportError:
    EXTRACTION_AVAILABLE = False
    print("❌ Extraction tools not available")

# Import data cleaning modules
try:
    from singapore_data_cleaner import clean_singapore_data_simple
    from singapore_standardizer import standardize_singapore_data
    from fred_data_cleaner import clean_fred_data_2000_plus
    CLEANING_AVAILABLE = True
    print("✅ Data cleaning tools loaded")
except ImportError:
    CLEANING_AVAILABLE = False
    print("❌ Data cleaning tools not available")

def load_macro_sources_only() -> List[Dict[str, Any]]:
    """Load ONLY macroeconomic sources (exclude market indices)"""
    try:
        from macro_sources import SINGAPORE_SOURCES, US_SOURCES, EU_SOURCES, JAPAN_SOURCES
        
        sources = []
        
        # Singapore macro sources
        for source in SINGAPORE_SOURCES:
            sources.append({
                'name': source.name,
                'url': source.url,
                'source_type': source.source_type,
                'data_type': source.data_type,
                'api_url': source.api_url,
                'country': 'Singapore'
            })

        # US macro sources
        for source in US_SOURCES:
            sources.append({
                'name': source.name,
                'url': source.url,
                'source_type': source.source_type,
                'data_type': source.data_type,
                'api_url': getattr(source, 'get_fred_api_url', lambda: source.api_url)(),
                'country': 'US'
            })
        
        # EU macro sources
        for source in EU_SOURCES:
            sources.append({
                'name': source.name,
                'url': source.url,
                'source_type': source.source_type,
                'data_type': source.data_type,
                'api_url': getattr(source, 'get_fred_api_url', lambda: source.api_url)(),
                'country': 'EU'
            })
        
        # Japan macro sources
        for source in JAPAN_SOURCES:
            sources.append({
                'name': source.name,
                'url': source.url,
                'source_type': source.source_type,
                'data_type': source.data_type,
                'api_url': getattr(source, 'get_fred_api_url', lambda: source.api_url)(),
                'country': 'Japan'
            })
        
        print(f"✅ Loaded {len(sources)} macro sources")
        return sources
        
    except ImportError:
        print("❌ macro_sources.py not found")
        return []
    except Exception as e:
        print(f"❌ Error loading macro sources: {e}")
        return []

def load_financial_sources_only() -> List[Dict[str, Any]]:
    """Load ONLY financial market sources (market indices)"""
    try:
        from macro_sources import MARKET_INDICES_SOURCES
        
        sources = []
        
        for source in MARKET_INDICES_SOURCES:
            sources.append({
                'name': source.name,
                'url': source.url,
                'source_type': source.source_type,
                'data_type': source.data_type,
                'api_url': source.api_url,
                'country': getattr(source, 'country', 'Global')
            })
        
        print(f"✅ Loaded {len(sources)} financial sources")
        return sources
        
    except ImportError:
        print("❌ macro_sources.py not found")
        return []
    except Exception as e:
        print(f"❌ Error loading financial sources: {e}")
        return []

def find_latest_files_by_pattern(data_dir: str = "./extracted_data"):
    """Find latest extraction files by pattern"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return {}
    
    patterns = {
        'singapore': 'macro_data_singapore_*.xlsx',
        'us': 'macro_data_us_*.xlsx', 
        'euro_area': 'macro_data_euro area_*.xlsx',
        'japan': 'macro_data_japan_*.xlsx',
        'market': 'market_indices_data_*.xlsx'
    }
    
    latest_files = {}
    
    for name, pattern in patterns.items():
        files = list(data_path.glob(pattern))
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            latest_files[name] = str(latest_file)
            print(f"🔍 {name}: Found {latest_file.name}")
    
    return latest_files

def clean_and_standardize_data(extracted_files: Dict[str, str]) -> Dict[str, str]:
    """
    Clean and standardize extracted data files
    
    Args:
        extracted_files: Dict mapping country/type to file paths
        
    Returns:
        Dict mapping country/type to cleaned file paths
    """
    
    print(f"\n🧹 DATA CLEANING AND STANDARDIZATION")
    print("=" * 60)
    
    cleaned_files = {}
    
    # Step 1: Clean Singapore data (2-step process)
    if 'singapore' in extracted_files:
        print(f"\n🇸🇬 SINGAPORE DATA PROCESSING")
        print("-" * 40)
        
        singapore_file = extracted_files['singapore']
        print(f"📄 Input: {Path(singapore_file).name}")
        
        # Step 1a: Clean Singapore data
        try:
            print("🔄 Step 1: Running singapore_data_cleaner...")
            cleaned_sg_file = clean_singapore_data_simple(singapore_file)
            
            if cleaned_sg_file:
                print(f"✅ Cleaned: {Path(cleaned_sg_file).name}")
                
                # Step 1b: Standardize Singapore data
                print("🔄 Step 2: Running singapore_standardizer...")
                standardized_sg_file = standardize_singapore_data(cleaned_sg_file)
                
                if standardized_sg_file:
                    print(f"✅ Standardized: {Path(standardized_sg_file).name}")
                    cleaned_files['singapore'] = standardized_sg_file
                else:
                    print("❌ Singapore standardization failed")
                    cleaned_files['singapore'] = cleaned_sg_file  # Use cleaned version
            else:
                print("❌ Singapore cleaning failed")
                
        except Exception as e:
            print(f"❌ Singapore processing error: {e}")
    
    # Step 2: Clean FRED data (US, EU, Japan)
    fred_countries = ['us', 'euro_area', 'japan']
    
    for country in fred_countries:
        if country in extracted_files:
            print(f"\n🌍 {country.upper().replace('_', ' ')} DATA PROCESSING")
            print("-" * 40)
            
            country_file = extracted_files[country]
            print(f"📄 Input: {Path(country_file).name}")
            
            try:
                print("🔄 Running fred_data_cleaner...")
                
                # Map country names for FRED cleaner
                country_name_map = {
                    'us': 'US',
                    'euro_area': 'Euro Area', 
                    'japan': 'Japan'
                }
                country_display_name = country_name_map[country]
                
                cleaned_file = clean_fred_data_2000_plus(country_file, country_display_name)
                
                if cleaned_file:
                    print(f"✅ Cleaned: {Path(cleaned_file).name}")
                    cleaned_files[country] = cleaned_file
                else:
                    print(f"❌ {country} cleaning failed")
                    
            except Exception as e:
                print(f"❌ {country} processing error: {e}")
    
    # Step 3: Market data (no cleaning needed)
    if 'market' in extracted_files:
        print(f"\n📈 MARKET DATA")
        print("-" * 40)
        print("ℹ️ Market data doesn't require cleaning")
        cleaned_files['market'] = extracted_files['market']
    
    return cleaned_files

def launch_streamlit_dashboard():
    """Launch Streamlit dashboard in browser"""
    
    print(f"\n🚀 LAUNCHING STREAMLIT DASHBOARD")
    print("=" * 50)
    
    dashboard_file = "streamlit_dashboard.py"
    
    # Check if dashboard file exists
    if not Path(dashboard_file).exists():
        print(f"❌ Dashboard file not found: {dashboard_file}")
        return False
    
    try:
        print(f"🌐 Starting Streamlit server...")
        print(f"📁 Dashboard: {dashboard_file}")
        print(f"🔗 URL: http://localhost:8501")
        print(f"⏳ Please wait for browser to open...")
        
        # Launch Streamlit (this will block until user stops it)
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_file
        ], check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n👋 Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error launching dashboard: {e}")
        return False

def run_complete_pipeline_with_cleaning(sources: List[Dict[str, Any]], workflow_name: str) -> Dict[str, Any]:
    """
    Run complete pipeline: Validation → Extraction → Cleaning → Dashboard Launch
    
    Args:
        sources: List of data sources to process
        workflow_name: Name for this workflow execution
        
    Returns:
        Dict with pipeline results
    """
    
    if not EXTRACTION_AVAILABLE or not CLEANING_AVAILABLE:
        print("❌ Required tools not available")
        return {"status": "failed", "error": "Required tools not available"}
    
    print(f"🚀 COMPLETE PIPELINE WITH CLEANING: {workflow_name}")
    print("Step 1: Validation → Step 2: Extraction → Step 3: Cleaning → Step 4: Dashboard")
    print("=" * 90)
    
    try:
        # Step 1: Data Validation
        print("📋 STEP 1: SOURCE VALIDATION")
        print("-" * 40)
        
        pipeline = create_llm_free_pipeline()
        validation_result = pipeline.run_source_validation(sources)
        
        if validation_result["status"] != "completed":
            print(f"❌ Validation failed: {validation_result.get('error', 'Unknown error')}")
            return validation_result
        
        working_sources = validation_result.get("successful_sources", 0)
        total_sources = validation_result.get("sources_processed", 0)
        
        print(f"✅ Validation completed: {working_sources}/{total_sources} sources working")
        
        # Step 2: Data Extraction
        print(f"\n🔧 STEP 2: DATA EXTRACTION")
        print("-" * 40)
        
        validation_outputs = validation_result.get("all_formatted_outputs", [])
        if not validation_outputs:
            print("❌ No validation outputs found for extraction")
            return {"status": "failed", "error": "No validation outputs"}
        
        combined_validation_results = "\n\n".join(validation_outputs)
        print(f"Processing validation results for {working_sources} working sources...")
        
        # Run extraction
        extraction_result_json = simple_rule_based_extractor._run(combined_validation_results)
        
        try:
            extraction_result = json.loads(extraction_result_json)
        except json.JSONDecodeError:
            print(f"❌ Could not parse extraction result")
            return {"status": "failed", "error": "Extraction result parsing failed"}
        
        if extraction_result.get("status") != "completed":
            print(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")
            return {"status": "failed", "error": f"Extraction failed: {extraction_result.get('error')}"}
        
        files_created = extraction_result.get("files_created", [])
        print(f"✅ Extraction completed: {len(files_created)} files created")
        
        # Step 3: Data Cleaning and Standardization
        print(f"\n🧹 STEP 3: DATA CLEANING AND STANDARDIZATION")
        print("-" * 40)
        
        # Find the latest extracted files
        latest_files = find_latest_files_by_pattern()
        
        if not latest_files:
            print("❌ No extracted files found for cleaning")
            return {"status": "failed", "error": "No files to clean"}
        
        # Clean and standardize data
        cleaned_files = clean_and_standardize_data(latest_files)
        
        if not cleaned_files:
            print("❌ Data cleaning failed")
            return {"status": "failed", "error": "Data cleaning failed"}
        
        print(f"✅ Data cleaning completed: {len(cleaned_files)} files processed")
        
        # Step 4: Launch Dashboard
        print(f"\n🌐 STEP 4: DASHBOARD LAUNCH")
        print("-" * 40)
        
        dashboard_launched = launch_streamlit_dashboard()
        
        # Return comprehensive result
        return {
            "status": "completed",
            "workflow_name": workflow_name,
            "validation_result": validation_result,
            "extraction_result": extraction_result,
            "raw_files_created": files_created,
            "cleaned_files_created": list(cleaned_files.values()),
            "working_sources": working_sources,
            "total_sources": total_sources,
            "dashboard_launched": dashboard_launched
        }
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

def run_extraction_only():
    """Run data extraction only (without cleaning or dashboard)"""
    
    print(f"🔧 EXTRACTION ONLY - ALL SOURCES")
    print("=" * 50)
    
    if not EXTRACTION_AVAILABLE:
        print("❌ Extraction tools not available")
        return None
    
    try:
        # Load all sources
        macro_sources = load_macro_sources_only()
        financial_sources = load_financial_sources_only()
        all_sources = macro_sources + financial_sources
        
        if not all_sources:
            print("❌ No sources loaded")
            return None
        
        print(f"📊 Processing {len(all_sources)} total sources")
        
        # Step 1: Validation
        print(f"\n📋 STEP 1: SOURCE VALIDATION")
        print("-" * 40)
        
        pipeline = create_llm_free_pipeline()
        validation_result = pipeline.run_source_validation(all_sources)
        
        if validation_result["status"] != "completed":
            print(f"❌ Validation failed: {validation_result.get('error', 'Unknown error')}")
            return None
        
        working_sources = validation_result.get("successful_sources", 0)
        print(f"✅ Validation completed: {working_sources} working sources")
        
        # Step 2: Extraction
        print(f"\n🔧 STEP 2: DATA EXTRACTION")
        print("-" * 40)
        
        validation_outputs = validation_result.get("all_formatted_outputs", [])
        combined_validation_results = "\n\n".join(validation_outputs)
        
        extraction_result_json = simple_rule_based_extractor._run(combined_validation_results)
        extraction_result = json.loads(extraction_result_json)
        
        if extraction_result.get("status") == "completed":
            files_created = extraction_result.get("files_created", [])
            print(f"✅ Extraction completed: {len(files_created)} files created")
            
            for file_path in files_created:
                print(f"   📄 {Path(file_path).name}")
            
            return files_created
        else:
            print(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return None

def run_cleaning_only():
    """Run cleaning and standardization on existing extracted files"""
    
    print(f"🧹 CLEANING EXISTING EXTRACTED FILES")
    print("=" * 50)
    
    if not CLEANING_AVAILABLE:
        print("❌ Cleaning tools not available")
        return None
    
    # Find latest extracted files
    latest_files = find_latest_files_by_pattern()
    
    if not latest_files:
        print("❌ No extracted files found")
        print("💡 Please run data extraction first")
        return None
    
    print(f"📄 Found {len(latest_files)} files to clean")
    
    # Clean and standardize
    cleaned_files = clean_and_standardize_data(latest_files)
    
    if cleaned_files:
        print(f"\n🎉 CLEANING COMPLETED!")
        print(f"📁 Processed files:")
        for country, file_path in cleaned_files.items():
            print(f"   • {country}: {Path(file_path).name}")
        return cleaned_files
    else:
        print("❌ Cleaning failed")
        return None

def run_dashboard_only():
    """Launch dashboard using existing cleaned data"""
    
    print(f"🌐 LAUNCHING DASHBOARD WITH EXISTING DATA")
    print("=" * 50)
    
    # Check if we have some cleaned data files
    data_dir = Path("./extracted_data")
    
    # Look for cleaned/standardized files
    cleaned_patterns = [
        "standardized_cleaned_macro_data_singapore_*.xlsx",
        "cleaned_macro_data_us_*.xlsx",
        "cleaned_macro_data_euro area_*.xlsx", 
        "cleaned_macro_data_japan_*.xlsx",
        "market_indices_data_*.xlsx"
    ]
    
    available_files = []
    for pattern in cleaned_patterns:
        files = list(data_dir.glob(pattern))
        available_files.extend(files)
    
    if not available_files:
        print("❌ No cleaned data files found")
        print("💡 Please run data extraction and cleaning first")
        return False
    
    print(f"📊 Found {len(available_files)} data files")
    for file in available_files:
        print(f"   • {file.name}")
    
    print(f"\n🚀 Launching dashboard...")
    return launch_streamlit_dashboard()

# =============================================================================
# WORKFLOW FUNCTIONS
# =============================================================================

def run_singapore_complete_pipeline():
    """Complete pipeline for Singapore sources only"""
    sources = load_macro_sources_only()
    singapore_sources = [s for s in sources if s['country'] == 'Singapore']
    
    if not singapore_sources:
        print("❌ No Singapore sources loaded")
        return None
    
    print(f"📊 Processing {len(singapore_sources)} Singapore sources")
    return run_complete_pipeline_with_cleaning(singapore_sources, "Singapore Complete Pipeline")

def run_everything_complete_pipeline():
    """Complete pipeline for all sources (macro + financial)"""
    macro_sources = load_macro_sources_only()
    financial_sources = load_financial_sources_only()
    all_sources = macro_sources + financial_sources
    
    if not all_sources:
        print("❌ No sources loaded")
        return None
    
    print(f"📊 Processing {len(all_sources)} total sources:")
    print(f"   Macro: {len(macro_sources)}")
    print(f"   Financial: {len(financial_sources)}")
    
    return run_complete_pipeline_with_cleaning(all_sources, "Complete Data Pipeline")

def check_system_status():
    """Check system status and available tools"""
    
    print("🔍 SYSTEM STATUS CHECK")
    print("=" * 50)
    
    status = {
        'extraction': EXTRACTION_AVAILABLE,
        'cleaning': CLEANING_AVAILABLE,
        'sources_available': False,
        'dashboard_available': Path("streamlit_dashboard.py").exists()
    }
    
    # Check extraction tools
    print(f"🔧 Extraction Tools: {'✅ Available' if status['extraction'] else '❌ Missing'}")
    
    # Check cleaning tools  
    print(f"🧹 Cleaning Tools: {'✅ Available' if status['cleaning'] else '❌ Missing'}")
    
    # Check dashboard
    print(f"🌐 Dashboard: {'✅ Available' if status['dashboard_available'] else '❌ Missing'}")
    
    # Check sources
    try:
        macro_sources = load_macro_sources_only()
        financial_sources = load_financial_sources_only()
        status['sources_available'] = len(macro_sources) > 0 or len(financial_sources) > 0
        
        print(f"📊 Data Sources:")
        print(f"   Macro sources: {len(macro_sources)}")
        print(f"   Financial sources: {len(financial_sources)}")
        
    except Exception as e:
        print(f"❌ Source loading failed: {e}")
    
    # Check existing data
    latest_files = find_latest_files_by_pattern()
    print(f"📁 Existing Files: {len(latest_files)} found")
    
    overall_status = all(status.values())
    print(f"\n🎯 Overall Status: {'✅ Ready' if overall_status else '❌ Issues detected'}")
    
    return status

def interactive_menu():
    """Enhanced interactive menu with extraction, cleaning and dashboard options"""
    
    while True:
        print(f"\n🚀 ENHANCED MACROECONOMIC DATA PIPELINE")
        print("Complete Workflow: Validation → Extraction → Cleaning → Dashboard")
        print("=" * 80)
        
        print("COMPLETE PIPELINES:")
        print("1. Singapore Complete Pipeline (Extract + Clean + Dashboard)")
        print("2. Everything Complete Pipeline (All sources + Clean + Dashboard)")
        print("")
        print("INDIVIDUAL STEPS:")
        print("3. Run Extraction Only (all sources)")
        print("4. Run Cleaning Only (on existing extracted files)")
        print("5. Launch Dashboard Only (using existing cleaned data)")
        print("")
        print("UTILITIES:")
        print("6. Check System Status")
        print("7. Exit")
        
        try:
            choice = input(f"\nSelect option (1-7): ").strip()

            if choice == '1':
                result = run_singapore_complete_pipeline()
                if result and result.get("status") == "completed":
                    print(f"\n🎉 Singapore pipeline completed successfully!")
                
            elif choice == '2':
                result = run_everything_complete_pipeline()
                if result and result.get("status") == "completed":
                    print(f"\n🎉 Complete pipeline finished successfully!")
                
            elif choice == '3':
                result = run_extraction_only()
                if result:
                    print(f"\n🎉 Data extraction completed!")
                    print(f"💡 Next: Run option 4 to clean the data")
                
            elif choice == '4':
                result = run_cleaning_only()
                if result:
                    print(f"\n🎉 Data cleaning completed!")
                    print(f"💡 Next: Run option 5 to launch dashboard")
                
            elif choice == '5':
                success = run_dashboard_only()
                if success:
                    print(f"\n🎉 Dashboard session completed!")
                
            elif choice == '6':
                check_system_status()
                
            elif choice == '7':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-7.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main entry point with enhanced pipeline"""
    
    print("🚀 ENHANCED MACROECONOMIC DATA PIPELINE")
    print("Complete Integration: Extraction → Cleaning → Standardization → Dashboard")
    print("=" * 90)
    
    # Command line interface
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            check_system_status()
        elif command == 'singapore':
            run_singapore_complete_pipeline()
        elif command == 'everything':
            run_everything_complete_pipeline()
        elif command == 'extract':
            run_extraction_only()
        elif command == 'clean':
            run_cleaning_only()
        elif command == 'dashboard':
            run_dashboard_only()
        elif command == 'test':
            print("🧪 Quick test with Singapore pipeline...")
            run_singapore_complete_pipeline()
        else:
            print(f"❌ Unknown command: {command}")
            print("\n📋 Available commands:")
            print("  status      - Check system status")
            print("  singapore   - Singapore complete pipeline")
            print("  everything  - Complete pipeline (all sources)")
            print("  extract     - Extract data only (all sources)")
            print("  clean       - Clean existing extracted files")
            print("  dashboard   - Launch dashboard with existing data")
            print("  test        - Quick test with Singapore")
    else:
        # Interactive menu
        interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        print("💡 Please check your configuration and try again")
