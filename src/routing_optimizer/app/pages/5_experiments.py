"""
Page 5: Run comparative experiments between VRP algorithms.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from routing_optimizer.experiments.runner import ExperimentRunner
from routing_optimizer.genetic_algorithm.config import GAConfig
from routing_optimizer.routing.distance import meters_to_km

st.title("📊 Experimentos e Comparativos")

# Check if experiments just completed (for sidebar refresh)
if st.session_state.get("_experiments_just_completed"):
    del st.session_state["_experiments_just_completed"]
    st.balloons()

st.markdown(
    """
Compare o desempenho do **Algoritmo Genético** com algoritmos baseline:
- **Random**: Alocação aleatória (pior caso)
- **Nearest Neighbor**: Heurística gulosa do vizinho mais próximo
- **Clarke-Wright**: Algoritmo clássico de savings para VRP
"""
)

# Verificar se otimização foi executada
if "distance_matrix" not in st.session_state:
    st.warning("⚠️ Primeiro execute a otimização na página **Otimização de Rotas**.")
    st.info("Use o menu lateral para navegar até a página de Otimização de Rotas.")
    st.stop()

distance_matrix = st.session_state["distance_matrix"]
n_locations = len(distance_matrix) - 1  # Excluindo depósito

st.success(f"✅ Matriz de distâncias carregada: {n_locations} farmácias + 1 depósito")

st.markdown("---")

# Configurações dos experimentos
st.subheader("⚙️ Configuração dos Experimentos")

col1, col2 = st.columns(2)

with col1:
    num_vehicles = st.slider(
        "Número de Veículos",
        min_value=1,
        max_value=min(10, n_locations),
        value=min(3, n_locations),
        help="Quantidade de veículos para todos os experimentos",
    )

    capacity = st.slider(
        "Capacidade por Veículo",
        min_value=5,
        max_value=50,
        value=min(20, n_locations),
        help="Número máximo de paradas por veículo",
    )

with col2:
    num_ga_configs = st.slider(
        "Número de Configurações do AG",
        min_value=1,
        max_value=5,
        value=3,
        help="Quantidade de configurações diferentes do AG para testar",
    )

    random_runs = st.slider(
        "Execuções do Random",
        min_value=1,
        max_value=10,
        value=5,
        help="Número de execuções para o baseline Random (usa o melhor)",
    )

st.markdown("---")

# Configurações do AG
st.subheader("🧬 Configurações do Algoritmo Genético")

ga_configs = []
cols = st.columns(num_ga_configs)

for i in range(num_ga_configs):
    with cols[i]:
        st.markdown(f"**Config {i + 1}**")
        pop = st.number_input(
            "População",
            min_value=50,
            max_value=500,
            value=100 + i * 100,
            step=50,
            key=f"pop_{i}",
        )
        epochs = st.number_input(
            "Gerações",
            min_value=100,
            max_value=2000,
            value=300 + i * 200,
            step=100,
            key=f"epochs_{i}",
        )
        mutation = st.slider(
            "Mutação",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.1,
            key=f"mut_{i}",
        )
        ga_configs.append(
            GAConfig(
                population_size=int(pop),
                max_epochs=int(epochs),
                mutation_probability=mutation,
            )
        )

st.markdown("---")

# Botão para executar experimentos
if st.button("🔬 Executar Experimentos", type="primary"):
    runner = ExperimentRunner(distance_matrix)

    with st.status("Executando experimentos...", expanded=True) as status:
        # Random
        st.write("🎲 Executando baseline Random...")
        random_result = runner.run_random(num_vehicles, capacity, random_runs)

        # Nearest Neighbor
        st.write("📍 Executando Nearest Neighbor...")
        nn_result = runner.run_nearest_neighbor(num_vehicles, capacity)

        # Clarke-Wright
        st.write("💰 Executando Clarke-Wright Savings...")
        cw_result = runner.run_clarke_wright(num_vehicles, capacity)

        # Genetic Algorithms
        ga_results = []
        for i, config in enumerate(ga_configs):
            st.write(
                f"🧬 Executando AG Config {i + 1} "
                f"(pop={config.population_size}, gerações={config.max_epochs})..."
            )
            result = runner.run_genetic_algorithm(
                num_vehicles, capacity, config, name_suffix=f"Config {i + 1}"
            )
            ga_results.append(result)

        status.update(label="Experimentos concluídos!", state="complete")

    # Combinar todos os resultados
    all_results = {
        random_result.algorithm: random_result,
        nn_result.algorithm: nn_result,
        cw_result.algorithm: cw_result,
    }
    for result in ga_results:
        all_results[result.algorithm] = result

    # Salvar no session_state
    st.session_state["experiment_results"] = all_results

    # Set flag and trigger rerun to update sidebar progress
    st.session_state["_experiments_just_completed"] = True
    st.rerun()

# Exibir resultados se disponíveis
if "experiment_results" in st.session_state:
    results = st.session_state["experiment_results"]

    st.markdown("---")
    st.subheader("📈 Resultados dos Experimentos")

    # Criar DataFrame com resultados
    df_results = pd.DataFrame([r.to_dict() for r in results.values()])
    df_results = df_results.sort_values("Distância Total (km)")

    # Tabela de resultados
    st.dataframe(df_results, width=900, hide_index=True)

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de barras - Distância
        fig_dist = px.bar(
            df_results,
            x="Algoritmo",
            y="Distância Total (km)",
            title="Comparativo de Distância Total",
            color="Algoritmo",
            text="Distância Total (km)",
        )
        fig_dist.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, width="stretch")

    with col2:
        # Gráfico de barras - Tempo
        fig_time = px.bar(
            df_results,
            x="Algoritmo",
            y="Tempo (s)",
            title="Tempo de Execução",
            color="Algoritmo",
            text="Tempo (s)",
        )
        fig_time.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_time.update_layout(showlegend=False)
        st.plotly_chart(fig_time, width="stretch")

    st.markdown("---")

    # Análise comparativa
    st.subheader("📊 Análise Comparativa")

    # Encontrar melhor e pior
    sorted_results = sorted(results.values(), key=lambda x: x.total_distance)
    best = sorted_results[0]
    worst = sorted_results[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🏆 Melhor Algoritmo",
            best.algorithm,
            f"{meters_to_km(best.total_distance):.1f} km",
        )

    with col2:
        improvement = (worst.total_distance - best.total_distance) / worst.total_distance * 100
        st.metric(
            "📉 Melhoria vs Pior",
            f"{improvement:.1f}%",
            f"{meters_to_km(worst.total_distance - best.total_distance):.1f} km",
        )

    with col3:
        # Comparar AG com melhor baseline
        baselines = ["Random", "Nearest Neighbor", "Clarke-Wright"]
        best_baseline = min(
            [r for r in results.values() if r.algorithm in baselines],
            key=lambda x: x.total_distance,
        )
        best_ga = min(
            [r for r in results.values() if "AG" in r.algorithm],
            key=lambda x: x.total_distance,
        )
        ga_vs_baseline = (
            (best_baseline.total_distance - best_ga.total_distance)
            / best_baseline.total_distance
            * 100
        )
        st.metric(
            "🧬 AG vs Melhor Baseline",
            f"{ga_vs_baseline:.1f}%",
            "melhor" if ga_vs_baseline > 0 else "pior",
        )

    st.markdown("---")

    # Ranking
    st.subheader("🏅 Ranking de Algoritmos")

    ranking_data = []
    for i, result in enumerate(sorted_results):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i + 1}º"
        ranking_data.append(
            {
                "Posição": medal,
                "Algoritmo": result.algorithm,
                "Distância (km)": f"{meters_to_km(result.total_distance):.1f}",
                "Tempo (s)": f"{result.execution_time:.2f}",
                "Rotas": result.num_routes,
            }
        )

    df_ranking = pd.DataFrame(ranking_data)
    st.dataframe(df_ranking, width=900, hide_index=True)

    st.markdown("---")

    # Evolução do AG
    st.subheader("📈 Evolução do Algoritmo Genético")

    ga_with_history = [r for r in results.values() if r.fitness_history]

    if ga_with_history:
        fig_evolution = go.Figure()

        for result in ga_with_history:
            fig_evolution.add_trace(
                go.Scatter(
                    y=result.fitness_history,
                    mode="lines",
                    name=result.algorithm,
                )
            )

        fig_evolution.update_layout(
            title="Convergência do Fitness ao Longo das Gerações",
            xaxis_title="Geração",
            yaxis_title="Fitness (menor = melhor)",
            legend_title="Configuração",
        )

        st.plotly_chart(fig_evolution, width="stretch")
    else:
        st.info("Nenhum histórico de evolução disponível.")

    st.markdown("---")

    # Exportar resultados
    st.subheader("📥 Exportar Resultados")

    col1, col2 = st.columns(2)

    with col1:
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📄 Baixar Resultados (CSV)",
            data=csv,
            file_name="experimentos_vrp.csv",
            mime="text/csv",
        )

    with col2:
        # Resumo em texto
        summary_text = f"""RELATÓRIO DE EXPERIMENTOS - VRP
================================

Configuração:
- Locais: {n_locations} farmácias + 1 depósito
- Veículos: {num_vehicles}
- Capacidade: {capacity} paradas/veículo

RANKING DE ALGORITMOS:
"""
        for i, result in enumerate(sorted_results):
            summary_text += f"""
{i + 1}. {result.algorithm}
   - Distância: {meters_to_km(result.total_distance):.1f} km
   - Tempo: {result.execution_time:.2f}s
   - Rotas: {result.num_routes}
"""

        summary_text += f"""
CONCLUSÃO:
- Melhor algoritmo: {best.algorithm}
- Melhoria sobre pior: {improvement:.1f}%
- AG vs melhor baseline: {ga_vs_baseline:.1f}%
"""

        st.download_button(
            label="📝 Baixar Relatório (TXT)",
            data=summary_text,
            file_name="relatorio_experimentos.txt",
            mime="text/plain",
        )

st.markdown("---")
st.info(
    "💡 Dica: Execute múltiplos experimentos com diferentes configurações "
    "do AG para encontrar os melhores parâmetros para seu problema."
)
