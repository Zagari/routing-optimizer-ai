"""
Tests for the LLM module (OpenAI integration).
"""

from unittest.mock import MagicMock, patch

import pytest

from routing_optimizer.llm.openai_client import RouteAssistant


class TestRouteAssistant:
    """Tests for RouteAssistant class."""

    def test_initialization_default(self):
        """Test default initialization."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            assistant = RouteAssistant()
            assert assistant.model == "gpt-4o-mini"
            assert assistant.api_key == "test-key"

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        assistant = RouteAssistant(api_key="custom-key", model="gpt-4")
        assert assistant.api_key == "custom-key"
        assert assistant.model == "gpt-4"

    def test_is_configured_with_key(self):
        """Test is_configured returns True when API key is set."""
        assistant = RouteAssistant(api_key="test-key")
        assert assistant.is_configured() is True

    def test_is_configured_without_key(self):
        """Test is_configured returns False when API key is not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove OPENAI_API_KEY if it exists
            import os

            original = os.environ.pop("OPENAI_API_KEY", None)
            try:
                assistant = RouteAssistant(api_key=None)
                assert assistant.is_configured() is False
            finally:
                if original:
                    os.environ["OPENAI_API_KEY"] = original

    @patch("routing_optimizer.llm.openai_client.OpenAI")
    def test_generate_driver_instructions_success(self, mock_openai_class):
        """Test successful driver instructions generation."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Instrucoes para o motorista..."
        mock_client.chat.completions.create.return_value = mock_response

        # Test
        assistant = RouteAssistant(api_key="test-key")
        result = assistant.generate_driver_instructions(
            route=["Endereco 1", "Endereco 2", "Endereco 3"],
            vehicle_id=1,
        )

        assert result == "Instrucoes para o motorista..."
        mock_client.chat.completions.create.assert_called_once()

        # Verify call parameters
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert call_args.kwargs["max_tokens"] == 1000
        assert len(call_args.kwargs["messages"]) == 2

    @patch("routing_optimizer.llm.openai_client.OpenAI")
    def test_generate_driver_instructions_error(self, mock_openai_class):
        """Test error handling in driver instructions generation."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        assistant = RouteAssistant(api_key="test-key")
        result = assistant.generate_driver_instructions(
            route=["Endereco 1"],
            vehicle_id=1,
        )

        assert "Erro ao gerar instrucoes" in result
        assert "API Error" in result

    @patch("routing_optimizer.llm.openai_client.OpenAI")
    def test_generate_efficiency_report_success(self, mock_openai_class):
        """Test successful efficiency report generation."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Relatorio de eficiencia..."
        mock_client.chat.completions.create.return_value = mock_response

        assistant = RouteAssistant(api_key="test-key")
        result = assistant.generate_efficiency_report(
            routes=[["Endereco 1", "Endereco 2"], ["Endereco 3"]],
            total_distance=50.5,
            optimization_time=2.3,
        )

        assert result == "Relatorio de eficiencia..."
        mock_client.chat.completions.create.assert_called_once()

        # Verify call parameters
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 1500

    @patch("routing_optimizer.llm.openai_client.OpenAI")
    def test_generate_efficiency_report_error(self, mock_openai_class):
        """Test error handling in efficiency report generation."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Network Error")

        assistant = RouteAssistant(api_key="test-key")
        result = assistant.generate_efficiency_report(
            routes=[["Endereco 1"]],
            total_distance=10.0,
            optimization_time=1.0,
        )

        assert "Erro ao gerar relatorio" in result
        assert "Network Error" in result

    @patch("routing_optimizer.llm.openai_client.OpenAI")
    def test_chat_about_routes_success(self, mock_openai_class):
        """Test successful chat about routes."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Resposta sobre as rotas..."
        mock_client.chat.completions.create.return_value = mock_response

        assistant = RouteAssistant(api_key="test-key")
        result = assistant.chat_about_routes(
            question="Qual a rota mais longa?",
            routes_context="3 veiculos, 15 paradas total",
        )

        assert result == "Resposta sobre as rotas..."
        mock_client.chat.completions.create.assert_called_once()

        # Verify call parameters
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 500

    @patch("routing_optimizer.llm.openai_client.OpenAI")
    def test_chat_about_routes_error(self, mock_openai_class):
        """Test error handling in chat about routes."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Timeout")

        assistant = RouteAssistant(api_key="test-key")
        result = assistant.chat_about_routes(
            question="Pergunta teste",
            routes_context="Contexto teste",
        )

        assert "Erro ao processar pergunta" in result
        assert "Timeout" in result

    @patch("routing_optimizer.llm.openai_client.OpenAI")
    def test_generate_driver_instructions_empty_route(self, mock_openai_class):
        """Test driver instructions with empty route."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Rota vazia"
        mock_client.chat.completions.create.return_value = mock_response

        assistant = RouteAssistant(api_key="test-key")
        result = assistant.generate_driver_instructions(route=[], vehicle_id=1)

        assert result == "Rota vazia"

    @patch("routing_optimizer.llm.openai_client.OpenAI")
    def test_generate_efficiency_report_empty_routes(self, mock_openai_class):
        """Test efficiency report with empty routes."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Sem rotas para analisar"
        mock_client.chat.completions.create.return_value = mock_response

        assistant = RouteAssistant(api_key="test-key")
        result = assistant.generate_efficiency_report(
            routes=[[], []],
            total_distance=0.0,
            optimization_time=0.1,
        )

        assert result == "Sem rotas para analisar"


@pytest.mark.integration
class TestRouteAssistantIntegration:
    """Integration tests for RouteAssistant (requires OPENAI_API_KEY)."""

    @pytest.fixture
    def assistant(self):
        """Create assistant if API key is available."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        return RouteAssistant()

    def test_generate_driver_instructions_real(self, assistant):
        """Test real driver instructions generation."""
        result = assistant.generate_driver_instructions(
            route=[
                "Av. Paulista, 1000, Sao Paulo",
                "Rua Augusta, 500, Sao Paulo",
                "Praca da Se, Sao Paulo",
            ],
            vehicle_id=1,
        )

        assert len(result) > 100  # Should have substantial content
        assert "Erro" not in result

    def test_generate_efficiency_report_real(self, assistant):
        """Test real efficiency report generation."""
        result = assistant.generate_efficiency_report(
            routes=[
                ["Endereco 1", "Endereco 2"],
                ["Endereco 3", "Endereco 4", "Endereco 5"],
            ],
            total_distance=45.5,
            optimization_time=3.2,
        )

        assert len(result) > 100
        assert "Erro" not in result

    def test_chat_about_routes_real(self, assistant):
        """Test real chat about routes."""
        result = assistant.chat_about_routes(
            question="Quantos veiculos estao sendo usados?",
            routes_context="3 veiculos, 15 paradas, 50km total",
        )

        assert len(result) > 20
        assert "Erro" not in result
