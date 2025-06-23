"""
Economic Data Dashboard - Simple Entry Point
"""

import streamlit as st
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    st.set_page_config(
        page_title="Economic Data Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    try:
        from streamlit_dashboard import main as dashboard_main
        dashboard_main()
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
        st.title("📊 Economic Data Dashboard")
        st.info("Please check the app logs for more details.")

if __name__ == "__main__":
    main()
