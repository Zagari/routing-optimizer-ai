"""
OpenAI client for generating driver instructions and efficiency reports.
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class RouteAssistant:
    """Assistant for generating route-related content using OpenAI GPT models.

    This class provides methods to generate:
    - Driver instructions for each vehicle route
    - Efficiency reports for the optimization
    - Answers to questions about routes

    Attributes:
        client: OpenAI client instance.
        model: Model to use for completions (default: gpt-4o-mini).
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the RouteAssistant.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model to use. If None, uses DEFAULT_MODEL.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[OpenAI] = None
        self.model = model or self.DEFAULT_MODEL

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate_driver_instructions(
        self,
        route: List[str],
        vehicle_id: int,
    ) -> str:
        """Generate detailed instructions for a driver.

        Args:
            route: List of addresses in order of visit.
            vehicle_id: Vehicle number for identification.

        Returns:
            Formatted instructions string in Portuguese.

        Raises:
            Exception: If API call fails.
        """
        addresses = "\n".join(f"{i+1}. {addr}" for i, addr in enumerate(route))

        system_prompt = """Voce e um assistente de logistica especializado em entregas \
de medicamentos. Gere instrucoes claras e objetivas para motoristas. Seja conciso mas \
inclua dicas uteis sobre o trajeto. Responda sempre em portugues brasileiro."""

        user_prompt = f"""Gere instrucoes de navegacao para o Veiculo {vehicle_id}.

Rota a seguir:
{addresses}

Inclua:
- Resumo da rota (origem, destino, numero de paradas)
- Tempo estimado total considerando transito medio
- Dicas de transito para horarios de pico em Sao Paulo
- Observacoes importantes para entrega de medicamentos (cuidados, documentacao)
- Sugestao de horario ideal para iniciar a rota"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Erro ao gerar instrucoes: {e}"

    def generate_efficiency_report(
        self,
        routes: List[List[str]],
        total_distance: float,
        optimization_time: float,
    ) -> str:
        """Generate an efficiency report for the optimization.

        Args:
            routes: List of routes, each containing addresses.
            total_distance: Total distance in kilometers.
            optimization_time: Time taken to optimize in seconds.

        Returns:
            Formatted report string in Portuguese.

        Raises:
            Exception: If API call fails.
        """
        route_summary = "\n".join(
            f"- Veiculo {i+1}: {len(r)} paradas" for i, r in enumerate(routes) if r
        )

        num_vehicles = len([r for r in routes if r])
        total_stops = sum(len(r) for r in routes)

        system_prompt = """Voce e um analista de logistica especializado em otimizacao \
de rotas. Gere relatorios profissionais e detalhados sobre eficiencia de entregas. \
Use dados concretos e metricas relevantes. Responda sempre em portugues brasileiro."""

        user_prompt = f"""Gere um relatorio de eficiencia para a otimizacao de rotas.

Dados da Otimizacao:
- Numero de veiculos utilizados: {num_vehicles}
- Total de paradas: {total_stops}
- Distribuicao por veiculo:
{route_summary}
- Distancia total percorrida: {total_distance:.2f} km
- Tempo de otimizacao do algoritmo: {optimization_time:.2f} segundos

Gere um relatorio contendo:
1. Resumo Executivo
2. Metricas de Eficiencia (distancia media por veiculo, paradas por veiculo)
3. Analise de Balanceamento de Carga
4. Recomendacoes de Melhoria
5. Comparacao com Benchmarks do Setor (distribuicao farmaceutica)"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1500,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Erro ao gerar relatorio: {e}"

    def chat_about_routes(
        self,
        question: str,
        routes_context: str,
    ) -> str:
        """Answer questions about the current routes.

        Args:
            question: User's question about the routes.
            routes_context: Context information about the routes.

        Returns:
            Answer string in Portuguese.

        Raises:
            Exception: If API call fails.
        """
        system_prompt = f"""Voce e um assistente especializado em rotas de entrega de \
medicamentos em Sao Paulo. Responda perguntas de forma util, precisa e em portugues.

Contexto das rotas atuais:
{routes_context}

Se a pergunta for sobre algo que nao esta no contexto, responda com base no seu \
conhecimento sobre logistica e distribuicao farmaceutica."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Erro ao processar pergunta: {e}"

    def is_configured(self) -> bool:
        """Check if the assistant is properly configured with an API key.

        Returns:
            True if API key is set, False otherwise.
        """
        return bool(self.api_key)
