"""
Main entry point for the VRP Route Optimizer Streamlit application.

Run with: streamlit run src/routing_optimizer/app/main.py
"""

import streamlit as st

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="Otimizador de Rotas - VRP",
    page_icon="🚚",
    layout="wide",
)

# Define pages with custom titles for sidebar
pages = [
    st.Page("pages/0_home.py", title="Página Inicial", icon="🏠"),
    st.Page("pages/1_upload.py", title="Endereços Destinos", icon="📁"),
    st.Page("pages/2_optimize.py", title="Otimização de Rotas", icon="⚙️"),
    st.Page("pages/3_results.py", title="Resultados", icon="🗺️"),
    st.Page("pages/4_instructions.py", title="Instruções", icon="📝"),
    st.Page("pages/5_experiments.py", title="Experimentos", icon="📊"),
]

# Create navigation and run selected page
pg = st.navigation(pages)
pg.run()
