"""
Economic Data Dashboard - Simple Entry Point
"""

import streamlit as st

def main():
    st.set_page_config(
        page_title="Economic Data Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    try:
        # Try to import the main dashboard
        from streamlit_dashboard import main as dashboard_main
        dashboard_main()
    except ImportError as e:
        st.error(f"Import Error: {e}")
        st.info("Some modules are missing. Please check requirements.txt")
        
        # Show a simple fallback
        st.title("📊 Economic Data Dashboard")
        st.info("Dashboard is being set up. Please check back in a few minutes.")
        
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.title("📊 Economic Data Dashboard")
        st.info("There was an issue loading the dashboard. Please try refreshing the page.")

if __name__ == "__main__":
    main()
