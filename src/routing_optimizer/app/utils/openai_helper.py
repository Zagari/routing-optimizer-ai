"""Helper functions for OpenAI API key management in Streamlit."""

import streamlit as st

from routing_optimizer.llm.openai_client import RouteAssistant
from routing_optimizer.utils.secrets import get_secret_from_aws, get_openai_api_key as get_key_from_sources

# Session state key for storing the API key
_SESSION_KEY = "_openai_api_key"


def get_openai_api_key() -> str | None:
    """Get OpenAI API key from multiple sources.

    Priority:
    1. Session state (user-provided via UI)
    2. AWS Secrets Manager (when running on EC2)
    3. Environment variable (OPENAI_API_KEY)

    Returns:
        API key if available, None otherwise.
    """
    # First check session state (user-provided)
    session_key = st.session_state.get(_SESSION_KEY)
    if session_key:
        return session_key

    # Then try AWS Secrets Manager and env var via secrets module
    return get_key_from_sources()


def get_assistant() -> RouteAssistant:
    """Get RouteAssistant instance using session state API key if available.

    Returns:
        RouteAssistant instance (may or may not be configured).
    """
    saved_key = get_openai_api_key()
    return RouteAssistant(api_key=saved_key)


def render_api_key_input(
    show_stop: bool = True,
    context_message: str = "",
) -> RouteAssistant | None:
    """Render API key input form and return configured assistant.

    This function handles multiple scenarios:
    1. Server has a configured key (AWS Secrets Manager or env var) - user can still use own key
    2. No server key - user must provide their own

    Args:
        show_stop: If True, calls st.stop() when no key is provided.
                   If False, returns None instead.
        context_message: Optional additional context to show the user.

    Returns:
        Configured RouteAssistant if key is valid, None if show_stop=False
        and no valid key provided.
    """
    import os

    # Check if there's a server-side key available (AWS or env)
    server_key = get_key_from_sources()

    if server_key:
        # Server has a configured key - show success and option to use own
        st.success("Chave do servidor configurada")

        use_own = st.checkbox(
            "Usar minha propria chave",
            value=False,
            help="Marque para usar sua propria API Key ao inves da configurada no servidor"
        )

        if use_own:
            api_key_input = st.text_input(
                "Sua API Key OpenAI:",
                type="password",
                placeholder="sk-...",
                help="Sua chave pessoal. So armazenada nesta sessao.",
                key="_openai_key_input",
            )

            if api_key_input:
                with st.spinner("Validando API Key..."):
                    test_assistant = RouteAssistant(api_key=api_key_input)
                    is_valid, error_msg = test_assistant.validate_api_key()

                if is_valid:
                    st.session_state[_SESSION_KEY] = api_key_input
                    st.success("Usando sua chave pessoal.")
                    st.rerun()
                else:
                    st.error(f"{error_msg}")

        return None

    # No server key available - user must provide their own
    st.warning("OPENAI_API_KEY nao configurada!")

    if context_message:
        st.markdown(context_message)

    st.markdown(
        """
        Para usar os recursos de IA, voce precisa de uma API Key da OpenAI.

        **Opcoes:**
        1. Configure no arquivo `.env` na raiz do projeto
        2. Ou digite abaixo para usar apenas nesta sessao

        [Obter API Key](https://platform.openai.com/api-keys)
        """
    )

    api_key_input = st.text_input(
        "API Key da OpenAI:",
        type="password",
        placeholder="sk-...",
        help="Sua chave sera usada apenas nesta sessao e nao sera salva no disco.",
        key="_openai_key_input",
    )

    if api_key_input:
        # Validate the key
        with st.spinner("Validando API Key..."):
            test_assistant = RouteAssistant(api_key=api_key_input)
            is_valid, error_msg = test_assistant.validate_api_key()

        if is_valid:
            st.session_state[_SESSION_KEY] = api_key_input
            st.success("API Key valida! Configurada para esta sessao.")
            st.rerun()
        else:
            st.error(f"{error_msg}")
            if show_stop:
                st.stop()
            return None
    else:
        st.info("Digite sua API Key acima para habilitar os recursos de IA.")
        if show_stop:
            st.stop()
        return None

    return None  # Should not reach here due to st.rerun()


def clear_api_key():
    """Clear the stored API key from session state."""
    if _SESSION_KEY in st.session_state:
        del st.session_state[_SESSION_KEY]
