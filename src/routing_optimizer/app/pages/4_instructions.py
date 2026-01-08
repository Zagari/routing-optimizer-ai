"""
Page 4: Generate driver instructions and efficiency reports using LLM.
"""

import streamlit as st

from routing_optimizer.llm.openai_client import RouteAssistant
from routing_optimizer.routing.distance import meters_to_km

st.title("📝 Instruções e Relatórios com IA")

# Verificar se otimização foi executada
if "routes" not in st.session_state:
    st.warning("⚠️ Primeiro execute a otimização na página **Otimização de Rotas**.")
    st.info("Use o menu lateral para navegar até a página de Otimização de Rotas.")
    st.stop()

# Verificar API key
assistant = RouteAssistant()
if not assistant.is_configured():
    st.error("⚠️ OPENAI_API_KEY não configurada!")
    st.markdown(
        """
    Para usar esta funcionalidade, configure a variável de ambiente:

    ```bash
    export OPENAI_API_KEY="sua-chave-aqui"
    ```

    Ou crie um arquivo `.env` na raiz do projeto:
    ```
    OPENAI_API_KEY=sua-chave-aqui
    ```
    """
    )
    st.stop()

# Recuperar dados
routes = st.session_state["routes"]
geocoded_data = st.session_state["geocoded_data"]
names = st.session_state.get("names", [])
total_distance = st.session_state.get("total_distance", 0)
optimization_time = st.session_state.get("optimization_time", 0)
distance_matrix = st.session_state.get("distance_matrix")

# Filtrar nomes dos locais geocodificados com sucesso
successful_names = [names[i] for i, r in enumerate(geocoded_data) if r.success]
successful_geocoded = [r for r in geocoded_data if r.success]

st.success(f"✅ {len([r for r in routes if r])} rotas prontas para gerar instruções")

st.markdown("---")

# Seção 1: Instruções para Motoristas
st.subheader("🚚 Instruções para Motoristas")

st.markdown(
    """
Selecione um veículo para gerar instruções detalhadas de navegação.
O ChatGPT irá criar um guia personalizado para o motorista.
"""
)

# Filtrar rotas não vazias
valid_routes = [(i, route) for i, route in enumerate(routes) if route]

if not valid_routes:
    st.warning("Nenhuma rota disponível.")
else:
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_idx = st.selectbox(
            "Selecione o veículo:",
            range(len(valid_routes)),
            format_func=lambda x: f"Veículo {valid_routes[x][0] + 1} "
            f"({len(valid_routes[x][1])} paradas)",
        )

        vehicle_idx, selected_route = valid_routes[selected_idx]

        # Calcular distância da rota
        if distance_matrix is not None:
            route_distance = 0
            route_distance += distance_matrix[0, selected_route[0]]
            for j in range(len(selected_route) - 1):
                route_distance += distance_matrix[selected_route[j], selected_route[j + 1]]
            route_distance += distance_matrix[selected_route[-1], 0]
            st.metric("Distância da Rota", f"{meters_to_km(route_distance):.1f} km")

        st.metric("Paradas", len(selected_route))

    with col2:
        st.markdown("**Paradas desta rota:**")
        for j, stop_idx in enumerate(selected_route):
            if stop_idx < len(successful_names):
                st.write(f"{j+1}. {successful_names[stop_idx]}")

    if st.button("🤖 Gerar Instruções com ChatGPT", type="primary"):
        # Preparar endereços formatados
        route_addresses = []
        for stop_idx in selected_route:
            if stop_idx < len(successful_geocoded):
                addr = successful_geocoded[stop_idx].formatted_address
                route_addresses.append(addr)

        with st.spinner("Gerando instruções com ChatGPT..."):
            instructions = assistant.generate_driver_instructions(route_addresses, vehicle_idx + 1)

        # Mark LLM interaction as done
        st.session_state["llm_interaction_done"] = True

        st.markdown("---")
        st.markdown("### Instruções Geradas:")
        st.markdown(instructions)

        # Botão para download
        st.download_button(
            label="📄 Baixar Instruções",
            data=instructions,
            file_name=f"instrucoes_veiculo_{vehicle_idx + 1}.txt",
            mime="text/plain",
        )

st.markdown("---")

# Seção 2: Relatório de Eficiência
st.subheader("📊 Relatório de Eficiência")

st.markdown(
    """
Gere um relatório profissional sobre a eficiência da otimização,
incluindo métricas, análises e recomendações.
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Distância Total", f"{meters_to_km(total_distance):.1f} km")
with col2:
    st.metric("Veículos", len([r for r in routes if r]))
with col3:
    st.metric("Tempo de Otimização", f"{optimization_time:.1f}s")

if st.button("📈 Gerar Relatório de Eficiência"):
    # Preparar dados das rotas
    all_routes_addresses = []
    for route in routes:
        if route:
            route_addrs = []
            for stop_idx in route:
                if stop_idx < len(successful_geocoded):
                    route_addrs.append(successful_geocoded[stop_idx].formatted_address)
            all_routes_addresses.append(route_addrs)

    with st.spinner("Gerando relatório de eficiência..."):
        report = assistant.generate_efficiency_report(
            all_routes_addresses,
            total_distance=meters_to_km(total_distance),
            optimization_time=optimization_time,
        )

    # Mark LLM interaction as done
    st.session_state["llm_interaction_done"] = True

    st.markdown("---")
    st.markdown("### Relatório de Eficiência:")
    st.markdown(report)

    # Botão para download
    st.download_button(
        label="📄 Baixar Relatório",
        data=report,
        file_name="relatorio_eficiencia.txt",
        mime="text/plain",
    )

st.markdown("---")

# Seção 3: Chat sobre Rotas
st.subheader("💬 Perguntas sobre as Rotas")

st.markdown(
    """
Faça perguntas sobre as rotas otimizadas. O assistente irá responder
com base nos dados da otimização.
"""
)

# Preparar contexto para o chat
num_vehicles = len([r for r in routes if r])
total_stops = sum(len(r) for r in routes)
route_details = []
for i, route in enumerate(routes):
    if route:
        if distance_matrix is not None:
            route_dist = 0
            route_dist += distance_matrix[0, route[0]]
            for j in range(len(route) - 1):
                route_dist += distance_matrix[route[j], route[j + 1]]
            route_dist += distance_matrix[route[-1], 0]
            route_details.append(
                f"Veículo {i+1}: {len(route)} paradas, {meters_to_km(route_dist):.1f} km"
            )
        else:
            route_details.append(f"Veículo {i+1}: {len(route)} paradas")

routes_context = f"""
Resumo da Otimização:
- Veículos utilizados: {num_vehicles}
- Total de paradas: {total_stops}
- Distância total: {meters_to_km(total_distance):.1f} km
- Tempo de otimização: {optimization_time:.1f} segundos

Detalhes por veículo:
{chr(10).join(route_details)}

Localização: São Paulo, Brasil
Tipo de carga: Medicamentos especializados
"""

# Histórico do chat na sessão
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Input de pergunta
question = st.text_input(
    "Digite sua pergunta:",
    placeholder="Ex: Qual veículo tem a rota mais longa?",
)

if question:
    with st.spinner("Processando pergunta..."):
        answer = assistant.chat_about_routes(question, routes_context)

    # Mark LLM interaction as done
    st.session_state["llm_interaction_done"] = True

    # Adicionar ao histórico
    st.session_state["chat_history"].append({"question": question, "answer": answer})

# Exibir histórico
if st.session_state["chat_history"]:
    st.markdown("### Histórico de Perguntas:")
    for i, item in enumerate(reversed(st.session_state["chat_history"][-5:])):
        with st.expander(f"❓ {item['question']}", expanded=(i == 0)):
            st.markdown(item["answer"])

    if st.button("🗑️ Limpar Histórico"):
        st.session_state["chat_history"] = []
        st.rerun()
