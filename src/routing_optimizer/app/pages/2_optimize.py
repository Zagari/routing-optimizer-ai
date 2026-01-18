"""
Page 2: Configure and run route optimization.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Show loading indicator immediately while heavy imports load
_loading_placeholder = st.empty()
_loading_placeholder.info("⏳ Carregando módulos de otimização...")

# Add app directory to path for local imports
_app_dir = Path(__file__).resolve().parent.parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from utils.osrm_batched import get_distance_matrix_batched

from routing_optimizer.genetic_algorithm.config import GAConfig
from routing_optimizer.genetic_algorithm.vrp import VRPSolver

# Clear loading indicator
_loading_placeholder.empty()

st.title("⚙️ Otimização de Rotas")


# =============================================================================
# Helper Functions
# =============================================================================
def go_to_tab(tab_name: str):
    """Navigate to a specific tab."""
    st.session_state["_optimize_tab"] = tab_name
    st.rerun()


# Initialize tab state
if "_optimize_tab" not in st.session_state:
    st.session_state["_optimize_tab"] = "config"

# Check if optimization just completed (for sidebar refresh)
if st.session_state.get("_optimization_just_completed"):
    del st.session_state["_optimization_just_completed"]
    st.balloons()

# Verificar se dados foram carregados
if "geocoded_data" not in st.session_state:
    st.warning("⚠️ Primeiro faça upload e geocodifique os dados na página **Endereços Destinos**.")
    st.info("Use o menu lateral para navegar até a página de Endereços Destinos.")
    st.stop()

geocoded_data = st.session_state["geocoded_data"]
names = st.session_state.get("names", [])

# Filtrar apenas endereços geocodificados com sucesso
successful = [r for r in geocoded_data if r.success]
coords = [(r.latitude, r.longitude) for r in successful]


# =============================================================================
# Tab Navigation
# =============================================================================
current_tab = st.session_state["_optimize_tab"]

tab_names = ["Configuração", "Otimização", "Resultados"]
tab_icons = ["⚙️", "🚀", "📊"]
tab_keys = ["config", "running", "results"]

# Determine which tabs are available
is_running = st.session_state.get("_optimization_running", False)
has_results = "routes" in st.session_state

tab_available = {
    "config": not is_running,  # Available if not running
    "running": is_running,  # Available only when running
    "results": has_results and not is_running,  # Available if has results and not running
}

# Clickable tab navigation
cols = st.columns(3)
for i, (name, icon, key) in enumerate(zip(tab_names, tab_icons, tab_keys)):
    with cols[i]:
        is_current = key == current_tab
        is_available = tab_available[key]

        if is_current:
            # Current tab - show as highlighted text
            st.markdown(f"**{icon} {name}**")
        elif is_available:
            # Available tab - show as clickable button
            if st.button(f"{icon} {name}", key=f"_nav_tab_{key}"):
                go_to_tab(key)
        else:
            # Not available - show as grayed out
            st.markdown(
                f"<span style='color: gray;'>{icon} {name}</span>",
                unsafe_allow_html=True,
            )

st.markdown("---")


# =============================================================================
# Tab: Configuração
# =============================================================================
if current_tab == "config":
    msg = f"✅ {len(coords)} locais prontos (1 depósito + {len(coords)-1} farmácias)"
    st.success(msg)

    st.markdown("---")

    # Configurações VRP
    st.subheader("Configurações do Problema")

    col1, col2 = st.columns(2)

    with col1:
        num_vehicles = st.slider(
            "Número de Veículos",
            min_value=1,
            max_value=min(10, len(coords) - 1),
            value=min(3, len(coords) - 1),
            help="Quantidade de veículos disponíveis para entrega",
        )

        capacity = st.slider(
            "Capacidade por Veículo",
            min_value=5,
            max_value=50,
            value=min(20, len(coords) - 1),
            help="Número máximo de paradas por veículo",
        )

    with col2:
        st.info(
            f"""
        **Resumo:**
        - Depósito: 1
        - Farmácias: {len(coords) - 1}
        - Veículos: {num_vehicles}
        - Capacidade: {capacity} paradas/veículo
        """
        )

    st.markdown("---")

    # Configurações do Algoritmo Genético
    st.subheader("Parâmetros do Algoritmo Genético")

    with st.expander("⚙️ Configurações Avançadas", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            population_size = st.slider(
                "Tamanho da População",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
                help="Número de soluções candidatas por geração",
            )

            max_epochs = st.slider(
                "Número de Gerações",
                min_value=100,
                max_value=10000,
                value=500,
                step=100,
                help="Número máximo de gerações do algoritmo (até 10.000)",
            )

            # Calcular valor padrão como 20% de max_epochs
            default_stagnation = max(1, int(max_epochs * 0.2))
            stagnation_threshold = st.slider(
                "Limite de Estagnação",
                min_value=10,
                max_value=max(100, max_epochs // 2),
                value=default_stagnation,
                step=10,
                help=f"Gerações sem melhoria para parar (padrão: 20% = {default_stagnation})",
            )

        with col2:
            mutation_prob = st.slider(
                "Probabilidade de Mutação",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Probabilidade de aplicar mutação em um indivíduo",
            )

            tournament_size = st.slider(
                "Tamanho do Torneio",
                min_value=2,
                max_value=10,
                value=5,
                help="Número de indivíduos no torneio de seleção",
            )

        # Opção de busca local (2-opt)
        st.markdown("---")
        is_large_dataset = len(coords) > 100
        default_local_search = not is_large_dataset  # Desabilitado por padrão para datasets grandes

        use_local_search = st.checkbox(
            "🔍 Usar Busca Local (2-opt)",
            value=default_local_search,
            help="Melhora a qualidade das rotas, mas é lento para datasets grandes (>100 locais)",
        )

        if is_large_dataset and use_local_search:
            st.warning(
                f"⚠️ Dataset grande ({len(coords)} locais). "
                "A busca local pode demorar muito. Considere desmarcar esta opção."
            )

    st.markdown("---")

    # Botão de otimização
    if st.button("🚀 Executar Otimização", type="primary", use_container_width=True):
        # Salvar configurações no session_state
        st.session_state["_opt_num_vehicles"] = num_vehicles
        st.session_state["_opt_capacity"] = capacity
        st.session_state["_opt_population_size"] = population_size
        st.session_state["_opt_max_epochs"] = max_epochs
        st.session_state["_opt_mutation_prob"] = mutation_prob
        st.session_state["_opt_tournament_size"] = tournament_size
        st.session_state["_opt_use_local_search"] = use_local_search
        st.session_state["_opt_stagnation_threshold"] = stagnation_threshold

        # Marcar que otimização está em andamento e ir para aba de execução
        st.session_state["_optimization_running"] = True
        go_to_tab("running")


# =============================================================================
# Tab: Otimização (em andamento)
# =============================================================================
elif current_tab == "running":
    # Scroll para o topo da página ao entrar nesta aba
    st.components.v1.html(
        """
        <script>
            // Aguarda um momento para garantir que a página renderizou
            setTimeout(function() {
                var mainSection = window.parent.document.querySelector('[data-testid="stMainBlockContainer"]');
                if (mainSection) {
                    mainSection.scrollIntoView({behavior: 'instant', block: 'start'});
                }
                // Fallback: tenta rolar a janela principal
                window.parent.scrollTo(0, 0);
            }, 100);
        </script>
        """,
        height=0,
    )

    # Recuperar configurações do session_state
    num_vehicles = st.session_state.get("_opt_num_vehicles", 3)
    capacity = st.session_state.get("_opt_capacity", 20)
    population_size = st.session_state.get("_opt_population_size", 200)
    max_epochs = st.session_state.get("_opt_max_epochs", 500)
    mutation_prob = st.session_state.get("_opt_mutation_prob", 0.6)
    tournament_size = st.session_state.get("_opt_tournament_size", 5)
    use_local_search = st.session_state.get("_opt_use_local_search", False)
    stagnation_threshold = st.session_state.get("_opt_stagnation_threshold", None)

    # ===== SEÇÃO PRINCIPAL: Progresso em tempo real (aparece primeiro) =====
    st.subheader("📊 Progresso da Otimização")

    # Barra de progresso no topo
    progress_bar = st.progress(0, text="Iniciando otimização...")

    # Gráfico grande primeiro, estatísticas menores ao lado
    col_chart, col_stats = st.columns([3, 1])

    with col_chart:
        chart_placeholder = st.empty()

    with col_stats:
        stats_placeholder = st.empty()

    # ===== SEÇÃO SECUNDÁRIA: Matriz de distâncias (colapsada) =====
    # Check if distance matrix already exists (loaded from saved dataset)
    if "distance_matrix" in st.session_state:
        distance_matrix = st.session_state["distance_matrix"]

        with st.status("Matriz de distâncias carregada!", expanded=False, state="complete") as status:
            n = distance_matrix.shape[0]
            st.write(f"✅ Matriz {n}x{n} carregada do dataset salvo")
            max_dist_km = distance_matrix.max() / 1000
            has_positive = (distance_matrix > 0).any()
            min_dist_km = distance_matrix[distance_matrix > 0].min() / 1000 if has_positive else 0
            st.write(f"📊 Distâncias: mín={min_dist_km:.1f}km, máx={max_dist_km:.1f}km")

    else:
        # Calculate new distance matrix
        with st.status("Calculando matriz de distâncias...", expanded=True) as status:
            st.write("Consultando OSRM para distâncias reais de estrada...")
            n_coords = len(coords)
            if n_coords > 100:
                num_batches = ((n_coords + 99) // 100) ** 2
                st.write(f"📦 {n_coords} locais requerem {num_batches} lotes (limite OSRM: 100)")

            start_time = time.time()

            try:
                # Progress bar with callback
                matrix_progress_bar = st.progress(0, text="Iniciando cálculo...")

                def update_progress(current, total):
                    pct = current / total
                    matrix_progress_bar.progress(pct, text=f"Processando lote {current}/{total}...")

                # Calculate distance matrix with batching support
                distance_matrix = get_distance_matrix_batched(
                    coords,
                    batch_size=100,
                    progress_callback=update_progress,
                )

                matrix_progress_bar.empty()  # Remove progress bar when done

                matrix_time = time.time() - start_time

                n = distance_matrix.shape[0]
                st.write(f"✅ Matriz {n}x{n} calculada em {matrix_time:.1f}s")

                # Stats
                max_dist_km = distance_matrix.max() / 1000
                has_positive = (distance_matrix > 0).any()
                min_dist_km = distance_matrix[distance_matrix > 0].min() / 1000 if has_positive else 0
                st.write(f"📊 Distâncias: mín={min_dist_km:.1f}km, máx={max_dist_km:.1f}km")

                # Save to dataset if available
                if "_current_dataset_name" in st.session_state:
                    from routing_optimizer.data.dataset_manager import DatasetManager

                    dataset_manager = DatasetManager()
                    try:
                        dataset_manager.save_distance_matrix(
                            st.session_state["_current_dataset_name"],
                            distance_matrix,
                        )
                        st.write("💾 Matriz salva no dataset para uso futuro")
                    except Exception as e:
                        st.warning(f"Não foi possível salvar matriz: {e}")

                status.update(label="Matriz de distâncias calculada!", state="complete")

            except Exception as e:
                st.error(f"Erro ao calcular matriz de distâncias: {e}")
                # Limpar flag de otimização em andamento
                if "_optimization_running" in st.session_state:
                    del st.session_state["_optimization_running"]
                go_to_tab("config")
                st.stop()

    # Track fitness history for real-time chart
    fitness_chart_data = []

    # Progress callback for real-time updates
    def on_progress(generation, total, best_fitness, best_routes):
        # Update progress bar
        pct = generation / total if total > 0 else 0

        # Handle initial state (before first generation completes)
        if best_routes is None or best_fitness == float("inf"):
            progress_bar.progress(0, text="Inicializando população...")
            stats_placeholder.markdown(
                f"""
                **Status:** Calculando primeira geração...

                **Locais:** {len(coords)}

                ⏳ Isso pode demorar para datasets grandes.
                """
            )
            return

        progress_bar.progress(pct, text=f"Geração {generation}/{total}")

        # Update stats
        num_routes = len([r for r in best_routes if r]) if best_routes else 0
        stats_placeholder.markdown(
            f"""
            **Geração:** {generation}/{total} ({pct*100:.0f}%)

            **Melhor Distância:** {best_fitness/1000:.1f} km

            **Rotas Ativas:** {num_routes}
            """
        )

        # Update fitness chart
        fitness_chart_data.append({"Geração": generation, "Fitness (km)": best_fitness / 1000})
        if len(fitness_chart_data) > 1:
            df = pd.DataFrame(fitness_chart_data)
            fig = px.line(
                df,
                x="Geração",
                y="Fitness (km)",
                title="Convergência do Algoritmo",
            )
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            chart_placeholder.plotly_chart(fig, width="stretch")

    config = GAConfig(
        population_size=population_size,
        mutation_probability=mutation_prob,
        max_epochs=max_epochs,
        tournament_size=tournament_size,
        local_search_elites_only=use_local_search,  # Só aplica se use_local_search=True
        local_search_rate=0.0 if not use_local_search else 0.1,  # Desabilita se False
        stagnation_threshold=stagnation_threshold,  # None = 20% de max_epochs
    )

    solver = VRPSolver(config)

    start_time = time.time()
    routes = solver.solve_with_distance_matrix(
        distance_matrix,
        num_vehicles=num_vehicles,
        capacity=capacity,
        progress_callback=on_progress,
        callback_interval=10,
    )
    optimization_time = time.time() - start_time

    # Clear progress indicators
    progress_bar.empty()
    stats_placeholder.empty()
    chart_placeholder.empty()

    st.success(f"✅ Otimização concluída em {optimization_time:.1f}s - {len(routes)} rotas geradas")

    # Calcular distância total
    total_distance = solver.get_total_distance(routes)

    # Salvar resultados no session_state
    st.session_state["routes"] = routes
    st.session_state["distance_matrix"] = distance_matrix
    st.session_state["total_distance"] = total_distance
    st.session_state["optimization_time"] = optimization_time
    st.session_state["fitness_history"] = solver.get_fitness_history()
    st.session_state["coords"] = coords

    # Limpar flag de otimização em andamento e ir para resultados
    if "_optimization_running" in st.session_state:
        del st.session_state["_optimization_running"]
    st.session_state["_optimization_just_completed"] = True
    go_to_tab("results")


# =============================================================================
# Tab: Resultados
# =============================================================================
elif current_tab == "results":
    # Scroll para o topo da página ao entrar nesta aba
    st.components.v1.html(
        """
        <script>
            setTimeout(function() {
                var mainSection = window.parent.document.querySelector('[data-testid="stMainBlockContainer"]');
                if (mainSection) {
                    mainSection.scrollIntoView({behavior: 'instant', block: 'start'});
                }
                window.parent.scrollTo(0, 0);
            }, 100);
        </script>
        """,
        height=0,
    )

    if "routes" not in st.session_state:
        st.info("Nenhuma otimização foi executada ainda.")
        if st.button("⚙️ Ir para Configuração"):
            go_to_tab("config")
    else:
        routes = st.session_state["routes"]
        distance_matrix = st.session_state["distance_matrix"]
        total_distance = st.session_state["total_distance"]
        optimization_time = st.session_state["optimization_time"]
        total_distance_km = total_distance / 1000

        # Exibir resumo
        st.subheader("📊 Resultado da Otimização")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Distância Total", f"{total_distance_km:.1f} km")
        with col2:
            st.metric("Rotas Geradas", len(routes))
        with col3:
            st.metric("Tempo de Execução", f"{optimization_time:.1f}s")

        # Detalhes das rotas
        st.markdown("---")
        st.subheader("Detalhes das Rotas")

        for i, route in enumerate(routes):
            if not route:
                continue

            # Calcular distância da rota
            route_distance = 0
            route_distance += distance_matrix[0, route[0]]  # Depósito -> primeira parada
            for j in range(len(route) - 1):
                route_distance += distance_matrix[route[j], route[j + 1]]
            route_distance += distance_matrix[route[-1], 0]  # Última parada -> depósito
            route_distance_km = route_distance / 1000

            with st.expander(f"🚚 Veículo {i+1}: {len(route)} paradas - {route_distance_km:.1f} km"):
                successful_names = [names[j] for j, r in enumerate(geocoded_data) if r.success]
                for j, stop_idx in enumerate(route):
                    if stop_idx < len(successful_names):
                        st.write(f"{j+1}. {successful_names[stop_idx]}")

        # Exibir histórico de fitness
        if "fitness_history" in st.session_state:
            st.markdown("---")
            st.subheader("📈 Evolução do Algoritmo")

            history = st.session_state["fitness_history"]
            df_history = pd.DataFrame(
                {
                    "Geração": range(len(history)),
                    "Fitness (km)": [f / 1000 for f in history],
                }
            )
            fig = px.line(
                df_history,
                x="Geração",
                y="Fitness (km)",
                title="Convergência do Algoritmo Genético",
            )
            st.plotly_chart(fig, width="stretch")

        # Navigation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗺️ Ver Mapa Completo", type="primary", use_container_width=True):
                st.switch_page("pages/3_results.py")
        with col2:
            if st.button("🔄 Nova Otimização", use_container_width=True):
                # Clear previous results
                for key in ["routes", "total_distance", "fitness_history", "optimization_time"]:
                    if key in st.session_state:
                        del st.session_state[key]
                go_to_tab("config")
