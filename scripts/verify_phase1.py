"""
Script de verificação da Fase 1 - Algoritmo Genético para VRP.

Este script permite testar o solver VRP com dados de teste ou dados de um arquivo CSV.
"""

import sys
from pathlib import Path

import pandas as pd

from routing_optimizer.genetic_algorithm import GAConfig, VRPSolver
from routing_optimizer.routing.geocoding import Geocoder


def get_test_locations():
    """Retorna localizações de teste (coordenadas fixas)."""
    return [
        (-23.5505, -46.6333),  # Depósito (São Paulo - Centro)
        (-23.5614, -46.6560),  # Local 1 - Av. Paulista
        (-23.5489, -46.6388),  # Local 2 - República
        (-23.5575, -46.6619),  # Local 3 - Consolação
        (-23.5430, -46.6290),  # Local 4 - Luz
        (-23.5678, -46.6480),  # Local 5 - Bela Vista
    ]


def load_locations_from_csv(file_path: str) -> list:
    """
    Carrega endereços de um CSV e converte para coordenadas.

    Args:
        file_path: Caminho para o arquivo CSV com coluna 'Endereco'.

    Returns:
        Lista de tuplas (latitude, longitude).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    df = pd.read_csv(file_path)

    if "Endereco" not in df.columns:
        raise ValueError("O arquivo CSV deve conter uma coluna 'Endereco'")

    addresses = df["Endereco"].tolist()
    print(f"\nEncontrados {len(addresses)} endereços no arquivo.")

    # Geocodificar endereços
    print("Geocodificando endereços (isso pode levar alguns minutos)...")
    geocoder = Geocoder()

    def progress_callback(current, total):
        print(f"  Processando {current}/{total}...", end="\r")

    results = geocoder.geocode_batch(addresses, on_progress=progress_callback)
    print()  # Nova linha após o progresso

    # Filtrar resultados com sucesso
    locations = []
    failed = []
    for result in results:
        if result.success:
            locations.append((result.latitude, result.longitude))
        else:
            failed.append(f"  - {result.original_address}: {result.error}")

    if failed:
        print(f"\nAtenção: {len(failed)} endereço(s) não foram geocodificados:")
        for msg in failed[:5]:  # Mostrar apenas os 5 primeiros
            print(msg)
        if len(failed) > 5:
            print(f"  ... e mais {len(failed) - 5} endereço(s)")

    print(f"\n{len(locations)} localizações obtidas com sucesso.")
    return locations


def main():
    """Função principal do script."""
    print("=" * 60)
    print("Verificação da Fase 1 - Algoritmo Genético para VRP")
    print("=" * 60)

    # Perguntar ao usuário a origem dos dados
    print("\nEscolha a origem dos dados:")
    print("  1. Usar dados de teste (coordenadas fixas)")
    print("  2. Carregar de arquivo CSV (coluna 'Endereco')")

    while True:
        choice = input("\nDigite sua escolha (1 ou 2): ").strip()
        if choice in ("1", "2"):
            break
        print("Opção inválida. Por favor, digite 1 ou 2.")

    # Obter localizações conforme a escolha
    if choice == "1":
        print("\nUsando dados de teste...")
        locations = get_test_locations()
    else:
        file_path = input("\nDigite o caminho do arquivo CSV: ").strip()
        try:
            locations = load_locations_from_csv(file_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"\nErro: {e}")
            sys.exit(1)

    if len(locations) < 2:
        print("\nErro: São necessárias pelo menos 2 localizações.")
        sys.exit(1)

    print(f"\nTotal de localizações: {len(locations)}")
    print(f"  - Depósito: {locations[0]}")
    print(f"  - Pontos de entrega: {len(locations) - 1}")

    # Configurar o solver
    print("\n" + "-" * 60)
    print("Configurando o solver...")

    config = GAConfig(
        population_size=100,
        max_epochs=500,
        mutation_probability=0.6,
    )
    solver = VRPSolver(config)

    # Definir número de veículos
    num_vehicles = max(1, (len(locations) - 1) // 5)  # ~5 locais por veículo
    print(f"Número de veículos: {num_vehicles}")

    # Resolver
    print("\nExecutando algoritmo genético...")
    routes = solver.solve(
        locations=locations,
        num_vehicles=num_vehicles,
        capacity=50,
    )

    # Ver resultados
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"Rotas encontradas: {len(routes)}")
    for i, route in enumerate(routes):
        print(f"  Rota {i + 1}: {route}")
    print(f"\nDistância total: {solver.get_total_distance():.2f}")


if __name__ == "__main__":
    main()
