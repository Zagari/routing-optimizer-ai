"""
Page 1: Upload CSV and geocode addresses.
"""

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from routing_optimizer.app.components.map_view import create_locations_map
from routing_optimizer.routing.geocoding import Geocoder

st.title("📁 Upload de Dados")

st.markdown(
    """
Carregue um arquivo CSV com os endereços das farmácias. O arquivo deve conter
uma coluna com o endereço completo.

**Formato esperado:**
- Coluna com nome `Endereco` ou selecione a coluna correta após o upload
"""
)

# Upload do arquivo
uploaded_file = st.file_uploader(
    "Selecione o arquivo CSV com endereços",
    type=["csv"],
    help="O arquivo deve conter uma coluna com endereços para geocodificação",
)

if uploaded_file is not None:
    # Ler CSV
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin-1")

    st.subheader("Preview dos Dados")
    st.dataframe(df.head(10), width=900)
    st.info(f"Total de registros: {len(df)}")

    # Selecionar coluna de endereço
    columns = list(df.columns)
    default_col = "Endereco" if "Endereco" in columns else columns[0]
    address_column = st.selectbox(
        "Selecione a coluna com os endereços:",
        columns,
        index=columns.index(default_col) if default_col in columns else 0,
    )

    # Selecionar coluna de nome/identificador (opcional)
    name_column = st.selectbox(
        "Selecione a coluna com o nome/identificador (opcional):",
        ["(Nenhum)"] + columns,
        index=0,
    )

    # Adicionar depósito
    st.subheader("Endereço do Depósito")
    depot_address = st.text_input(
        "Endereço do depósito (ponto de partida):",
        value="Av. Paulista, 1000, São Paulo, SP",
        help="Este será o ponto de partida e chegada das rotas",
    )

    # Botão de geocodificação
    if st.button("🌍 Geocodificar Endereços", type="primary"):
        geocoder = Geocoder()

        # Preparar lista de endereços (depósito + farmácias)
        addresses = [depot_address] + df[address_column].tolist()
        names = ["DEPÓSITO"]
        if name_column != "(Nenhum)":
            names.extend(df[name_column].tolist())
        else:
            names.extend([f"Farmácia {i+1}" for i in range(len(df))])

        msg = f"Geocodificando {len(addresses)} endereços (1 depósito + {len(df)} farmácias)..."
        st.info(msg)
        st.warning("Aguarde... Pode levar alguns minutos devido ao rate limiting.")

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        errors = []

        for i, addr in enumerate(addresses):
            status_text.text(f"[{i+1}/{len(addresses)}] Geocodificando: {addr[:50]}...")
            result = geocoder.geocode_address(addr)
            results.append(result)

            if not result.success:
                errors.append((i, addr, result.error))

            progress_bar.progress((i + 1) / len(addresses))

        status_text.text("Geocodificação concluída!")

        # Calcular taxa de sucesso
        success_rate = geocoder.get_success_rate(results)
        successful = [r for r in results if r.success]

        # Exibir resultados
        st.subheader("Resultado da Geocodificação")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Endereços", len(addresses))
        with col2:
            st.metric("Geocodificados", len(successful))
        with col3:
            st.metric("Taxa de Sucesso", f"{success_rate * 100:.1f}%")

        if errors:
            with st.expander(f"⚠️ {len(errors)} endereços não foram geocodificados"):
                for idx, addr, error in errors:
                    st.write(f"- **{idx}**: {addr[:50]}... - *{error}*")

        if successful:
            # Salvar no session_state
            st.session_state["geocoded_data"] = results
            st.session_state["original_df"] = df
            st.session_state["names"] = names
            st.session_state["depot_index"] = 0

            # Criar coordenadas e labels para locais geocodificados
            coords = [(r.latitude, r.longitude) for r in results if r.success]
            labels = [names[i] for i, r in enumerate(results) if r.success]

            st.success(f"✅ {len(successful)} endereços geocodificados com sucesso!")
            st.info("Próximo passo: Vá para a página **Otimização de Rotas** para executar.")

            # Exibir mapa com locais
            st.subheader("🗺️ Locais Geocodificados")
            m = create_locations_map(coords, labels)
            st_folium(m, width=800, height=500)

            # Tabela com coordenadas
            with st.expander("Ver coordenadas"):
                coord_df = pd.DataFrame(
                    [
                        {
                            "Nome": names[i],
                            "Endereço Original": r.original_address,
                            "Endereço Formatado": r.formatted_address[:80],
                            "Latitude": r.latitude,
                            "Longitude": r.longitude,
                        }
                        for i, r in enumerate(results)
                        if r.success
                    ]
                )
                st.dataframe(coord_df, width=900)

        else:
            st.error("Nenhum endereço foi geocodificado. Verifique os endereços.")

else:
    # Opção para carregar dados de exemplo
    st.markdown("---")
    st.subheader("Ou use os dados de exemplo")

    if st.button("📦 Carregar Dados de Exemplo"):
        try:
            # Tentar carregar o CSV de exemplo
            example_path = "data/farmacias_sp.csv"
            df = pd.read_csv(example_path, encoding="latin-1")
            st.session_state["example_loaded"] = True
            st.session_state["example_df"] = df
            st.success("Dados de exemplo carregados! Faça o upload do arquivo para geocodificar.")
            st.dataframe(df.head(10), width=900)
        except FileNotFoundError:
            st.error("Arquivo de exemplo não encontrado. Faça upload de um CSV.")
