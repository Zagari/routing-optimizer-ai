"""
Page 3: Visualize optimized routes on map.
"""

import os

import streamlit as st

# Show loading indicator immediately while heavy imports load
_loading_placeholder = st.empty()
_loading_placeholder.info("⏳ Carregando módulos de visualização...")

import pandas as pd
from streamlit_folium import st_folium

from routing_optimizer.app.components.map_view import create_route_map
from routing_optimizer.routing.distance import meters_to_km

# Clear loading indicator
_loading_placeholder.empty()

# Configuração padrão para rotas reais (do .env)
DEFAULT_USE_REAL_ROADS = os.getenv("MAP_USE_REAL_ROADS", "false").lower() == "true"

st.title("🗺️ Resultados e Rotas")

# Verificar se otimização foi executada
if "routes" not in st.session_state:
    st.warning("⚠️ Primeiro execute a otimização na página **Otimização de Rotas**.")
    st.info("Use o menu lateral para navegar até a página de Otimização de Rotas.")
    st.stop()

# Recuperar dados do session_state
routes = st.session_state["routes"]
geocoded_data = st.session_state["geocoded_data"]
names = st.session_state.get("names", [])
distance_matrix = st.session_state["distance_matrix"]
total_distance = st.session_state["total_distance"]
optimization_time = st.session_state.get("optimization_time", 0)
coords = st.session_state.get("coords", [])

# Filtrar nomes dos locais geocodificados com sucesso
successful_names = [names[i] for i, r in enumerate(geocoded_data) if r.success]

# Métricas principais
st.subheader("📊 Resumo da Otimização")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Distância Total", f"{meters_to_km(total_distance):.1f} km")
with col2:
    st.metric("Veículos Utilizados", len([r for r in routes if r]))
with col3:
    st.metric("Total de Paradas", sum(len(r) for r in routes))
with col4:
    st.metric("Tempo de Otimização", f"{optimization_time:.1f}s")

st.markdown("---")

# Mapa com rotas
st.subheader("🗺️ Mapa das Rotas")

# Opção para mostrar rotas reais de estrada
col_map_opt1, col_map_opt2 = st.columns([1, 3])
with col_map_opt1:
    use_real_roads = st.checkbox(
        "🛣️ Mostrar rotas reais",
        value=DEFAULT_USE_REAL_ROADS,
        help="Busca o traçado real das estradas via OSRM. Pode demorar alguns segundos.",
    )
with col_map_opt2:
    if use_real_roads:
        st.info("⏳ Buscando rotas reais de estrada... Isso pode levar alguns segundos. Para resultado mais rápido, desmarque a opção ao lado.")

# Criar mapa
with st.spinner("Gerando mapa..."):
    m = create_route_map(
        coordinates=coords,
        routes=routes,
        labels=successful_names,
        depot_index=0,
        use_real_roads=use_real_roads,
    )

# Exibir mapa
st_folium(m, width=900, height=600)

st.markdown("---")

# Legenda das cores
st.subheader("🎨 Legenda")

colors = ["red", "blue", "green", "purple", "orange", "darkred", "darkblue", "darkgreen"]
legend_cols = st.columns(min(len(routes), 8))

for i, route in enumerate(routes):
    if route:
        with legend_cols[i % len(legend_cols)]:
            color = colors[i % len(colors)]
            route_distance = 0
            route_distance += distance_matrix[0, route[0]]
            for j in range(len(route) - 1):
                route_distance += distance_matrix[route[j], route[j + 1]]
            route_distance += distance_matrix[route[-1], 0]

            style = "background-color: {}; color: white; padding: 10px;".format(color)
            style += " border-radius: 5px; text-align: center;"
            html = f"""
            <div style="{style}">
                <b>Veículo {i+1}</b><br>
                {len(route)} paradas<br>
                {meters_to_km(route_distance):.1f} km
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

st.markdown("---")

# Detalhes das rotas
st.subheader("📋 Detalhes das Rotas")

for i, route in enumerate(routes):
    if not route:
        continue

    # Calcular distância da rota
    route_distance = 0
    route_distance += distance_matrix[0, route[0]]
    for j in range(len(route) - 1):
        route_distance += distance_matrix[route[j], route[j + 1]]
    route_distance += distance_matrix[route[-1], 0]

    label = f"🚚 Veículo {i+1}: {len(route)} paradas - {meters_to_km(route_distance):.1f} km"
    with st.expander(label, expanded=True):
        # Criar dataframe com detalhes da rota
        route_data = []

        # Depósito
        route_data.append(
            {
                "Ordem": 0,
                "Local": "DEPÓSITO (Partida)",
                "Distância até próximo (km)": meters_to_km(distance_matrix[0, route[0]]),
            }
        )

        # Paradas
        for j, stop_idx in enumerate(route):
            if stop_idx < len(successful_names):
                dist_to_next = 0
                if j < len(route) - 1:
                    dist_to_next = distance_matrix[route[j], route[j + 1]]
                else:
                    dist_to_next = distance_matrix[route[j], 0]

                route_data.append(
                    {
                        "Ordem": j + 1,
                        "Local": successful_names[stop_idx],
                        "Distância até próximo (km)": meters_to_km(dist_to_next),
                    }
                )

        # Retorno ao depósito
        route_data.append(
            {
                "Ordem": len(route) + 1,
                "Local": "DEPÓSITO (Retorno)",
                "Distância até próximo (km)": 0,
            }
        )

        df_route = pd.DataFrame(route_data)
        st.dataframe(df_route, width=900, hide_index=True)

st.markdown("---")

# Opções de exportação
st.subheader("📥 Exportar Resultados")

col1, col2 = st.columns(2)

with col1:
    # Exportar resumo como CSV
    summary_data = []
    for i, route in enumerate(routes):
        if not route:
            continue

        route_distance = 0
        route_distance += distance_matrix[0, route[0]]
        for j in range(len(route) - 1):
            route_distance += distance_matrix[route[j], route[j + 1]]
        route_distance += distance_matrix[route[-1], 0]

        for j, stop_idx in enumerate(route):
            if stop_idx < len(successful_names):
                summary_data.append(
                    {
                        "Veículo": i + 1,
                        "Ordem": j + 1,
                        "Local": successful_names[stop_idx],
                        "Latitude": coords[stop_idx][0],
                        "Longitude": coords[stop_idx][1],
                    }
                )

    df_export = pd.DataFrame(summary_data)
    csv = df_export.to_csv(index=False)

    st.download_button(
        label="📄 Baixar Rotas (CSV)",
        data=csv,
        file_name="rotas_otimizadas.csv",
        mime="text/csv",
    )

with col2:
    # Resumo em texto
    summary_text = f"""RESUMO DA OTIMIZAÇÃO
====================

Distância Total: {meters_to_km(total_distance):.1f} km
Veículos Utilizados: {len([r for r in routes if r])}
Total de Paradas: {sum(len(r) for r in routes)}
Tempo de Otimização: {optimization_time:.1f}s

ROTAS:
------
"""
    for i, route in enumerate(routes):
        if not route:
            continue

        route_distance = 0
        route_distance += distance_matrix[0, route[0]]
        for j in range(len(route) - 1):
            route_distance += distance_matrix[route[j], route[j + 1]]
        route_distance += distance_matrix[route[-1], 0]

        dist_km = meters_to_km(route_distance)
        summary_text += f"\nVeículo {i+1} ({len(route)} paradas - {dist_km:.1f} km):\n"
        for j, stop_idx in enumerate(route):
            if stop_idx < len(successful_names):
                summary_text += f"  {j+1}. {successful_names[stop_idx]}\n"

    st.download_button(
        label="📝 Baixar Resumo (TXT)",
        data=summary_text,
        file_name="resumo_otimizacao.txt",
        mime="text/plain",
    )

st.markdown("---")
st.info("💡 Dica: Na próxima página, gere instruções para motoristas com ChatGPT!")
