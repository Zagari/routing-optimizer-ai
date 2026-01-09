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

# Create navigation
pg = st.navigation(pages)


# ============================================================
# Progress Stepper in Sidebar
# ============================================================
def render_progress_stepper():
    """Render workflow progress stepper in sidebar."""
    # Check completion states from session_state
    has_data = "original_df" in st.session_state or "geocoded_data" in st.session_state
    has_geocoded = "geocoded_data" in st.session_state
    has_routes = "routes" in st.session_state
    has_distance_matrix = "distance_matrix" in st.session_state
    has_experiments = "experiment_results" in st.session_state
    has_llm_interaction = "llm_interaction_done" in st.session_state

    # Define steps: (name, is_complete, is_available)
    steps = [
        ("Carregar Endereços", has_data, True),
        ("Geocodificar", has_geocoded, has_data),
        ("Otimizar Rotas", has_routes, has_geocoded),
        ("Ver Resultados", has_routes, has_routes),
        ("Gerar Instruções", has_llm_interaction, has_routes),  # Optional step
        ("Rodar Experimentos", has_experiments, has_distance_matrix),
    ]

    # Count completed steps (excluding optional "Ver Resultados" and "Gerar Instruções")
    core_steps = [steps[0], steps[1], steps[2]]
    completed_core = sum(1 for _, complete, _ in core_steps if complete)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Progresso")

    # Progress summary
    if completed_core == 3:
        st.sidebar.success("✅ Otimização alcançada!")
    else:
        st.sidebar.caption(f"Etapas principais: {completed_core}/3")

    # Render each step
    for i, (step_name, is_complete, is_available) in enumerate(steps, 1):
        if is_complete:
            # Completed step
            st.sidebar.markdown(
                f"<div style='color: #28a745; padding: 2px 0;'>"
                f"✅ {i}. {step_name}</div>",
                unsafe_allow_html=True,
            )
        elif is_available:
            # Available but not complete (current step to do)
            st.sidebar.markdown(
                f"<div style='color: #ffc107; padding: 2px 0;'>"
                f"⏳ {i}. {step_name}</div>",
                unsafe_allow_html=True,
            )
        else:
            # Not available yet
            st.sidebar.markdown(
                f"<div style='color: #6c757d; padding: 2px 0;'>"
                f"○ {i}. {step_name}</div>",
                unsafe_allow_html=True,
            )

    # Show hint for next step
    st.sidebar.markdown("---")
    if not has_data:
        st.sidebar.info("👉 Comece carregando um CSV na página **Endereços Destinos**")
    elif not has_geocoded:
        st.sidebar.info("👉 Clique em **Geocodificar** para processar os endereços")
    elif not has_routes:
        st.sidebar.info("👉 Vá para **Otimização de Rotas** e execute o algoritmo")
    elif not has_experiments:
        st.sidebar.caption("💡 Experimente comparar algoritmos na página **Experimentos**")


# Render the stepper
render_progress_stepper()

# Run selected page
pg.run()
