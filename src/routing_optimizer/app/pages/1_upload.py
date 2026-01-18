"""
Page 1: Upload CSV, manage datasets, and geocode addresses.
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root before any other imports
_project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
load_dotenv(_project_root / ".env")

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from routing_optimizer.app.components.map_view import create_locations_map
from routing_optimizer.data.dataset_manager import DatasetManager
from routing_optimizer.routing.geocoding import Geocoder

# Initialize dataset manager (global for this page)
dataset_manager = DatasetManager()


# =============================================================================
# Helper Functions (must be defined before use)
# =============================================================================
def go_to_tab(tab_name: str):
    """Navigate to a specific tab."""
    st.session_state["_upload_tab"] = tab_name
    st.rerun()


def clear_temp_upload_state():
    """Clear temporary upload state."""
    keys_to_clear = [
        "_temp_df",
        "_temp_filename",
        "_temp_address_col",
        "_temp_name_col",
        "_temp_depot",
        "_geocoding_results",
        "_geocoding_errors",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def _save_and_continue(df, results, address_column, name_column, depot_address):
    """Save dataset and continue to map tab."""
    filename = st.session_state.get("_temp_filename", "dataset.csv")
    base_name = filename.rsplit(".", 1)[0]  # Remove extension

    # Generate unique name
    dataset_name = dataset_manager.generate_unique_name(base_name)

    # Build names list
    names = ["DEPÓSITO"]
    if name_column and name_column in df.columns:
        names.extend(df[name_column].tolist())
    else:
        names.extend([f"Farmácia {i+1}" for i in range(len(df))])

    # Save dataset
    dataset_manager.save_dataset(
        name=dataset_name,
        original_df=df,
        original_filename=filename,
        address_column=address_column,
        name_column=name_column,
        depot_address=depot_address,
        geocoded_results=results,
    )

    # Clear downstream data
    keys_to_clear = [
        "routes",
        "distance_matrix",
        "total_distance",
        "optimization_time",
        "fitness_history",
        "coords",
        "experiment_results",
        "llm_interaction_done",
        "chat_history",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Set session state
    st.session_state["geocoded_data"] = results
    st.session_state["original_df"] = df
    st.session_state["names"] = names
    st.session_state["depot_index"] = 0
    st.session_state["_current_dataset_name"] = dataset_name

    # Clear temp state
    clear_temp_upload_state()

    go_to_tab("map")


def _regeocode_corrected(df, original_results, address_column):
    """Re-geocode addresses after corrections."""
    corrected = st.session_state.get("_corrected_addresses", {})

    if not corrected:
        st.info("Nenhuma correção aplicada. Continuando com endereços originais.")
        _save_and_continue(
            df,
            original_results,
            st.session_state.get("_temp_address_col"),
            st.session_state.get("_temp_name_col"),
            st.session_state.get("_temp_depot"),
        )
        return

    st.subheader("🌍 Re-geocodificando endereços corrigidos...")

    geocoder = Geocoder()
    new_results = list(original_results)  # Copy original results

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (idx, new_addr) in enumerate(corrected.items()):
        status_text.text(f"[{i+1}/{len(corrected)}] Geocodificando: {new_addr[:50]}...")
        result = geocoder.geocode_address(new_addr)
        new_results[idx] = result
        progress_bar.progress((i + 1) / len(corrected))

    status_text.text("Re-geocodificação concluída!")

    # Count new successes
    new_success = sum(1 for r in new_results if r.success)
    old_success = sum(1 for r in original_results if r.success)

    st.success(f"✅ Geocodificação: {old_success} → {new_success} endereços")

    # Update results and continue
    st.session_state["_geocoding_results"] = new_results

    # Clear correction state
    for key in list(st.session_state.keys()):
        if key.startswith("_correction") or key.startswith("_suggestions"):
            del st.session_state[key]
    if "_corrected_addresses" in st.session_state:
        del st.session_state["_corrected_addresses"]

    # Save and continue
    _save_and_continue(
        df,
        new_results,
        st.session_state.get("_temp_address_col"),
        st.session_state.get("_temp_name_col"),
        st.session_state.get("_temp_depot"),
    )


def _render_correction_interface(df, results, errors, address_column):
    """Render the address correction interface."""
    from routing_optimizer.llm.openai_client import RouteAssistant

    assistant = RouteAssistant()

    if not assistant.is_configured():
        st.error("⚠️ OPENAI_API_KEY não configurada! O assistente não está disponível.")
        return

    # Get current error index
    correction_index = st.session_state.get("_correction_index", 0)

    if correction_index >= len(errors):
        # All corrections done
        st.success("✅ Todas as correções foram processadas!")

        # Re-geocode corrected addresses
        if st.button("🌍 Re-geocodificar endereços corrigidos", type="primary"):
            _regeocode_corrected(df, results, address_column)
        return

    # Current error to fix
    idx, failed_addr, error = errors[correction_index]

    st.markdown("---")
    st.subheader(f"🤖 Corrigindo endereço {correction_index + 1} de {len(errors)}")

    st.markdown(f"**Endereço original:** `{failed_addr}`")
    st.markdown(f"**Erro:** {error}")

    # Get suggestions (with caching)
    cache_key = f"_suggestions_{idx}"
    if cache_key not in st.session_state:
        with st.spinner("Consultando assistente..."):
            suggestions = assistant.suggest_address_corrections(failed_addr)
            st.session_state[cache_key] = suggestions

    suggestions = st.session_state[cache_key]

    if not suggestions:
        st.warning("O assistente não conseguiu sugerir correções para este endereço.")
        options = ["Ignorar este endereço"]
    else:
        st.markdown("**Sugestões do Assistente:**")
        options = suggestions + ["Ignorar este endereço"]

    selected = st.radio(
        "Escolha uma opção:",
        options,
        key=f"_correction_choice_{idx}",
    )

    if st.button("Aplicar e Próximo ➡️"):
        if selected != "Ignorar este endereço":
            # Update the address in temp state
            if "_corrected_addresses" not in st.session_state:
                st.session_state["_corrected_addresses"] = {}
            st.session_state["_corrected_addresses"][idx] = selected

        # Move to next
        st.session_state["_correction_index"] = correction_index + 1
        st.rerun()

    # Progress indicator
    st.progress((correction_index + 1) / len(errors))


# =============================================================================
# Page Content
# =============================================================================
st.title("📁 Endereços Destinos")

# Initialize tab state
if "_upload_tab" not in st.session_state:
    st.session_state["_upload_tab"] = "selection"  # selection, preview, geocoding, map

# =============================================================================
# Tab Navigation Display
# =============================================================================
current_tab = st.session_state["_upload_tab"]

tab_names = ["Seleção", "Preview", "Geocodificação", "Mapa"]
tab_icons = ["📂", "👁️", "🌍", "🗺️"]
tab_keys = ["selection", "preview", "geocoding", "map"]

# Determine which tabs are available based on data state
has_temp_df = "_temp_df" in st.session_state
has_original_df = "original_df" in st.session_state
has_any_df = has_temp_df or has_original_df  # Either temp or saved data
has_address_col = "_temp_address_col" in st.session_state
has_geocoded = "geocoded_data" in st.session_state
has_geocoding_results = "_geocoding_results" in st.session_state

tab_available = {
    "selection": True,  # Always available
    "preview": has_any_df,  # Available if any data loaded (temp or saved)
    "geocoding": has_geocoded or (has_temp_df and has_address_col),  # Available if geocoded or configured
    "map": has_geocoded,  # Available if geocoding completed and saved
}

# Clickable tab navigation
cols = st.columns(4)
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
# Tab 1: Selection
# =============================================================================
if current_tab == "selection":
    # Section 1: Saved Datasets
    st.subheader("📂 Datasets Salvos")

    datasets = dataset_manager.list_datasets()

    if datasets:
        # Create selection
        dataset_options = {
            d.name: f"{d.name} ({d.geocoded_count} locais geocodificados)"
            for d in datasets
        }

        selected_dataset = st.radio(
            "Selecione um dataset:",
            options=list(dataset_options.keys()),
            format_func=lambda x: dataset_options[x],
            key="_selected_dataset",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Usar Dataset Selecionado", type="primary"):
                # Load dataset into session state
                metadata, original_df, geocoded_results, names = (
                    dataset_manager.load_dataset(selected_dataset)
                )

                # Clear downstream data
                keys_to_clear = [
                    "routes",
                    "distance_matrix",
                    "total_distance",
                    "optimization_time",
                    "fitness_history",
                    "coords",
                    "experiment_results",
                    "llm_interaction_done",
                    "chat_history",
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]

                # Set session state
                st.session_state["geocoded_data"] = geocoded_results
                st.session_state["original_df"] = original_df
                st.session_state["names"] = names
                st.session_state["depot_index"] = 0
                st.session_state["_current_dataset_name"] = selected_dataset

                # Check for saved distance matrix
                saved_matrix = dataset_manager.load_distance_matrix(selected_dataset)
                if saved_matrix is not None:
                    st.session_state["distance_matrix"] = saved_matrix

                st.success(f"✅ Dataset '{selected_dataset}' carregado!")
                go_to_tab("map")

        with col2:
            if st.button("🗑️ Excluir Dataset"):
                if dataset_manager.delete_dataset(selected_dataset):
                    st.success(f"Dataset '{selected_dataset}' excluído.")
                    st.rerun()
    else:
        st.info(
            "Nenhum dataset salvo ainda. Faça upload de um arquivo ou use os dados de exemplo."
        )

    st.markdown("---")

    # Section 2: Upload New File
    st.subheader("📤 Upload de Novo Arquivo")

    uploaded_file = st.file_uploader(
        "Selecione o arquivo CSV com endereços",
        type=["csv"],
        help="O arquivo deve conter uma coluna com endereços para geocodificação",
    )

    if uploaded_file is not None:
        # Read CSV
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin-1")

        # Store in temp state
        st.session_state["_temp_df"] = df
        st.session_state["_temp_filename"] = uploaded_file.name

        st.success(f"Arquivo carregado: {len(df)} registros")
        if st.button("➡️ Continuar para Preview", type="primary"):
            go_to_tab("preview")

    st.markdown("---")

    # Section 3: Example Data
    st.subheader("📦 Dados de Exemplo")

    if st.button("Carregar Dados de Exemplo"):
        try:
            example_path = "data/0039-farmacias_SP.csv"
            df = pd.read_csv(example_path, encoding="latin-1")
            st.session_state["_temp_df"] = df
            st.session_state["_temp_filename"] = "0039-farmacias_SP.csv"
            st.success("Dados de exemplo carregados!")
            go_to_tab("preview")
        except FileNotFoundError:
            st.error("Arquivo de exemplo não encontrado.")

# =============================================================================
# Tab 2: Preview
# =============================================================================
elif current_tab == "preview":
    # Check if we have temp data or saved data
    has_temp_data = "_temp_df" in st.session_state
    has_saved_data = "original_df" in st.session_state

    if not has_temp_data and not has_saved_data:
        st.warning("Nenhum arquivo carregado. Volte para a aba de Seleção.")
        if st.button("⬅️ Voltar"):
            go_to_tab("selection")
        st.stop()

    # Use temp data if available, otherwise use saved data
    if has_temp_data:
        df = st.session_state["_temp_df"]
        filename = st.session_state.get("_temp_filename", "arquivo.csv")
        is_saved_dataset = False
    else:
        df = st.session_state["original_df"]
        filename = st.session_state.get("_current_dataset_name", "dataset salvo")
        is_saved_dataset = True

    st.subheader(f"Preview: {filename}")
    st.dataframe(df.head(10), width="stretch")
    st.info(f"Total de registros: {len(df)}")

    st.markdown("---")

    if is_saved_dataset:
        # Read-only view for saved datasets
        st.success("✅ Este é um dataset já geocodificado e salvo.")
        st.caption("Para ver os endereços geocodificados, vá para a aba **Mapa**.")
        st.caption("Para corrigir endereços com erro, vá para a aba **Geocodificação**.")

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Voltar para Seleção"):
                go_to_tab("selection")
        with col2:
            if st.button("🗺️ Ver Mapa", type="primary"):
                go_to_tab("map")
    else:
        # Configuration for new uploads
        columns = list(df.columns)
        default_addr_col = "Endereco" if "Endereco" in columns else columns[0]

        address_column = st.selectbox(
            "Selecione a coluna com os endereços:",
            columns,
            index=columns.index(default_addr_col) if default_addr_col in columns else 0,
            key="_temp_address_col_select",
        )

        name_column = st.selectbox(
            "Selecione a coluna com o nome/identificador (opcional):",
            ["(Nenhum)"] + columns,
            index=0,
            key="_temp_name_col_select",
        )

        depot_address = st.text_input(
            "Endereço do depósito (ponto de partida):",
            value="Av. Paulista, 1000, São Paulo, SP",
            help="Este será o ponto de partida e chegada das rotas",
            key="_temp_depot_input",
        )

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Voltar"):
                clear_temp_upload_state()
                go_to_tab("selection")

        with col2:
            if st.button("🌍 Geocodificar Endereços", type="primary"):
                # Store configuration
                st.session_state["_temp_address_col"] = address_column
                st.session_state["_temp_name_col"] = (
                    name_column if name_column != "(Nenhum)" else None
                )
                st.session_state["_temp_depot"] = depot_address
                go_to_tab("geocoding")

# =============================================================================
# Tab 3: Geocoding
# =============================================================================
elif current_tab == "geocoding":
    # Check if we have temp data or saved data to work with
    has_temp_data = "_temp_df" in st.session_state
    has_saved_data = "geocoded_data" in st.session_state and "original_df" in st.session_state

    if not has_temp_data and not has_saved_data:
        st.warning("Nenhum arquivo carregado. Volte para a aba de Seleção.")
        if st.button("⬅️ Voltar"):
            go_to_tab("selection")
        st.stop()

    # If coming back from Map tab, reconstruct temp state from saved data
    if not has_temp_data and has_saved_data:
        df = st.session_state["original_df"]
        results = st.session_state["geocoded_data"]

        # Reconstruct errors from saved geocoded data
        errors = []
        for i, r in enumerate(results):
            if not r.success:
                errors.append((i, r.original_address, r.error))

        st.session_state["_geocoding_results"] = results
        st.session_state["_geocoding_errors"] = errors

        # We don't have full temp state, but we can work with what we have
        if "_temp_address_col" not in st.session_state:
            # Try to find address column from original_df
            st.session_state["_temp_address_col"] = df.columns[0]
    else:
        df = st.session_state["_temp_df"]

    address_column = st.session_state.get("_temp_address_col", df.columns[0])
    name_column = st.session_state.get("_temp_name_col")
    depot_address = st.session_state.get(
        "_temp_depot", "Av. Paulista, 1000, São Paulo, SP"
    )

    # Check if geocoding already done
    if "_geocoding_results" not in st.session_state:
        # Run geocoding
        st.subheader("🌍 Geocodificando Endereços...")

        geocoder = Geocoder()
        addresses = [depot_address] + df[address_column].tolist()

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

        # Store results
        st.session_state["_geocoding_results"] = results
        st.session_state["_geocoding_errors"] = errors
        st.rerun()

    # Display results
    results = st.session_state["_geocoding_results"]
    errors = st.session_state["_geocoding_errors"]

    success_count = sum(1 for r in results if r.success)
    total_count = len(results)
    success_rate = success_count / total_count * 100 if total_count > 0 else 0

    st.subheader("📊 Resultado da Geocodificação")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Endereços", total_count)
    with col2:
        st.metric("Geocodificados", success_count)
    with col3:
        st.metric("Taxa de Sucesso", f"{success_rate:.1f}%")

    if errors:
        st.warning(f"⚠️ {len(errors)} endereços não foram geocodificados:")
        with st.expander("Ver endereços com erro", expanded=True):
            for idx, addr, error in errors:
                st.write(f"- **{idx}**: {addr[:60]}... - *{error}*")

        st.markdown("---")
        st.markdown("### O que deseja fazer?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Continuar com os endereços válidos", type="primary"):
                # Save and continue
                _save_and_continue(df, results, address_column, name_column, depot_address)

        with col2:
            if st.button("🤖 Usar Assistente para corrigir", type="primary"):
                st.session_state["_correction_mode"] = True
                st.session_state["_correction_index"] = 0
                st.rerun()

        # Correction mode
        if st.session_state.get("_correction_mode"):
            _render_correction_interface(df, results, errors, address_column)

    else:
        st.success("✅ Todos os endereços foram geocodificados com sucesso!")
        if st.button("➡️ Ver Mapa dos Destinos", type="primary"):
            _save_and_continue(df, results, address_column, name_column, depot_address)

# =============================================================================
# Tab 4: Map
# =============================================================================
elif current_tab == "map":
    if "geocoded_data" not in st.session_state:
        st.warning("Nenhum dado geocodificado disponível.")
        if st.button("⬅️ Voltar para Seleção"):
            go_to_tab("selection")
        st.stop()

    geocoded_data = st.session_state["geocoded_data"]
    names = st.session_state.get("names", [])
    dataset_name = st.session_state.get("_current_dataset_name", "")

    # Filter successful
    successful = [r for r in geocoded_data if r.success]
    coords = [(r.latitude, r.longitude) for r in successful]
    labels = [names[i] for i, r in enumerate(geocoded_data) if r.success]

    st.subheader("🗺️ Mapa dos Destinos")

    if dataset_name:
        st.success(f"✅ Dataset: **{dataset_name}** ({len(successful)} locais)")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric("Locais no Mapa", len(successful))
    with col2:
        st.metric("Depósito", "1")

    # Render map
    if coords:
        m = create_locations_map(coords, labels)
        st_folium(m, width=800, height=500)
    else:
        st.warning("Nenhum local para exibir no mapa.")

    st.markdown("---")
    st.success(
        "✅ Próximo passo: Vá para **Otimização de Rotas** para calcular as melhores rotas."
    )

    # Coordinates table
    with st.expander("Ver coordenadas"):
        coord_df = pd.DataFrame(
            [
                {
                    "Nome": names[i],
                    "Endereço": r.formatted_address[:80],
                    "Latitude": r.latitude,
                    "Longitude": r.longitude,
                }
                for i, r in enumerate(geocoded_data)
                if r.success
            ]
        )
        st.dataframe(coord_df, width="stretch")

    # Navigation buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Carregar Novo Dataset"):
            clear_temp_upload_state()
            go_to_tab("selection")
    with col2:
        if st.button("⚙️ Ir para Otimização de Rotas", type="primary", key="_go_optimize"):
            st.switch_page("pages/2_optimize.py")
