"""
Page 2: Configure and run route optimization.
"""

import time

import streamlit as st

from routing_optimizer.genetic_algorithm.config import GAConfig
from routing_optimizer.genetic_algorithm.vrp import VRPSolver
from routing_optimizer.routing.distance import OSRMDistanceMatrix

st.title("⚙️ Otimização de Rotas")

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
            max_value=2000,
            value=500,
            step=100,
            help="Número máximo de gerações do algoritmo",
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

st.markdown("---")

# Botão de otimização
if st.button("🚀 Executar Otimização", type="primary"):
    # Etapa 1: Calcular matriz de distâncias
    st.subheader("Executando Otimização...")

    with st.status("Calculando matriz de distâncias...", expanded=True) as status:
        st.write("Consultando OSRM para distâncias reais de estrada...")
        start_time = time.time()

        try:
            dm = OSRMDistanceMatrix()
            distance_matrix = dm.get_distance_matrix(coords)
            matrix_time = time.time() - start_time

            n = distance_matrix.shape[0]
            st.write(f"✅ Matriz {n}x{n} calculada em {matrix_time:.1f}s")

            # Estatísticas da matriz
            max_dist_km = distance_matrix.max() / 1000
            has_positive = (distance_matrix > 0).any()
            min_dist_km = distance_matrix[distance_matrix > 0].min() / 1000 if has_positive else 0
            st.write(f"📊 Distâncias: mín={min_dist_km:.1f}km, máx={max_dist_km:.1f}km")

            status.update(label="Matriz de distâncias calculada!", state="complete")

        except Exception as e:
            st.error(f"Erro ao calcular matriz de distâncias: {e}")
            st.stop()

    # Etapa 2: Executar Algoritmo Genético
    with st.status("Executando Algoritmo Genético...", expanded=True) as status:
        st.write(f"População: {population_size}, Gerações: {max_epochs}")

        config = GAConfig(
            population_size=population_size,
            mutation_probability=mutation_prob,
            max_epochs=max_epochs,
            tournament_size=tournament_size,
        )

        solver = VRPSolver(config)

        start_time = time.time()
        routes = solver.solve_with_distance_matrix(
            distance_matrix,
            num_vehicles=num_vehicles,
            capacity=capacity,
        )
        optimization_time = time.time() - start_time

        st.write(f"✅ Otimização concluída em {optimization_time:.1f}s")
        st.write(f"📊 {len(routes)} rotas geradas")

        status.update(label="Otimização concluída!", state="complete")

    # Calcular distância total
    total_distance = solver.get_total_distance(routes)
    total_distance_km = total_distance / 1000

    # Salvar resultados no session_state
    st.session_state["routes"] = routes
    st.session_state["distance_matrix"] = distance_matrix
    st.session_state["total_distance"] = total_distance
    st.session_state["optimization_time"] = optimization_time
    st.session_state["fitness_history"] = solver.get_fitness_history()
    st.session_state["coords"] = coords

    # Exibir resumo
    st.markdown("---")
    st.subheader("📊 Resultado da Otimização")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Distância Total", f"{total_distance_km:.1f} km")
    with col2:
        st.metric("Rotas Geradas", len(routes))
    with col3:
        st.metric("Tempo de Execução", f"{optimization_time:.1f}s")

    # Detalhes das rotas
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

    st.success("✅ Otimização concluída! Vá para a página **Resultados** para visualizar o mapa.")
    st.balloons()

# Exibir histórico de fitness se disponível
if "fitness_history" in st.session_state:
    st.markdown("---")
    st.subheader("📈 Evolução do Algoritmo")

    import pandas as pd
    import plotly.express as px

    history = st.session_state["fitness_history"]
    df_history = pd.DataFrame(
        {
            "Geração": range(len(history)),
            "Fitness (menor=melhor)": history,
        }
    )
    fig = px.line(
        df_history,
        x="Geração",
        y="Fitness (menor=melhor)",
        title="Convergência do Algoritmo Genético",
    )
    st.plotly_chart(fig)
