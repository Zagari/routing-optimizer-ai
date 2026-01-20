"""Utility functions for the Streamlit app."""

# Only import OSRM batched at module level (lightweight)
from .osrm_batched import get_distance_matrix_batched

# Lazy imports for OpenAI helpers to avoid loading the SDK on every page
# These are only loaded when explicitly accessed
__all__ = [
    "get_distance_matrix_batched",
    "get_assistant",
    "get_openai_api_key",
    "render_api_key_input",
    "clear_api_key",
]


def __getattr__(name: str):
    """Lazy import for OpenAI-related helpers."""
    if name in ("get_assistant", "get_openai_api_key", "render_api_key_input", "clear_api_key"):
        from . import openai_helper
        return getattr(openai_helper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
