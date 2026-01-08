#!/usr/bin/env python3
"""
Script de verificacao manual da Fase 2.

Execute com: python scripts/verify_fase2.py
"""

import sys
import time

# Adicionar src ao path
sys.path.insert(0, "src")

from routing_optimizer.routing import Geocoder, OSRMDistanceMatrix, meters_to_km


def test_geocoding():
    """Teste 1: Geocodificacao de enderecos do CSV."""
    print("\n" + "=" * 60)
    print("TESTE 1: Geocodificacao de Enderecos")
    print("=" * 60)

    geocoder = Geocoder()

    # Enderecos de exemplo do CSV de farmacias
    test_addresses = [
        "Rua da Paz, 2150 - Chácara Santo Antônio, Sao Paulo",
        "Rua Adma Jafet, 91, Sao Paulo",
        "Rua Visconde de Porto Seguro, 391, Jardim dos Estados, Sao Paulo",
    ]

    print("\nGeocodificando 3 enderecos de teste...")
    print("(Aguarde ~3 segundos devido ao rate limiting)\n")

    results = []
    for i, addr in enumerate(test_addresses, 1):
        print(f"[{i}/3] Geocodificando: {addr[:50]}...")
        start = time.time()
        result = geocoder.geocode_address(addr)
        elapsed = time.time() - start

        if result.success:
            print(f"      OK: lat={result.latitude:.4f}, lon={result.longitude:.4f}")
            print(f"      Endereco formatado: {result.formatted_address[:60]}...")
        else:
            print(f"      ERRO: {result.error}")

        print(f"      Tempo: {elapsed:.2f}s")
        results.append(result)

    success_rate = sum(1 for r in results if r.success) / len(results)
    print(f"\nTaxa de sucesso: {success_rate * 100:.0f}%")

    return all(r.success for r in results), results


def test_distance_matrix(geocoded_results):
    """Teste 2: Matriz de distancias reais."""
    print("\n" + "=" * 60)
    print("TESTE 2: Matriz de Distancias OSRM")
    print("=" * 60)

    # Usar coordenadas dos enderecos geocodificados
    coords = [(r.latitude, r.longitude) for r in geocoded_results if r.success]

    if len(coords) < 2:
        print("ERRO: Precisa de pelo menos 2 enderecos geocodificados")
        return False

    print(f"\nCalculando matriz de distancias para {len(coords)} pontos...")

    dm = OSRMDistanceMatrix()
    try:
        matrix = dm.get_distance_matrix(coords)

        print("\nMatriz de Distancias (em km):")
        print("-" * 40)

        # Cabecalho
        header = "      " + "".join(f"  P{i+1}   " for i in range(len(coords)))
        print(header)

        # Linhas da matriz
        for i, row in enumerate(matrix):
            row_str = f"P{i+1}  "
            for val in row:
                km = meters_to_km(val)
                row_str += f"{km:7.1f} "
            print(row_str)

        print("-" * 40)

        # Verificar valores
        max_distance_km = meters_to_km(matrix.max())
        min_nonzero = meters_to_km(matrix[matrix > 0].min()) if (matrix > 0).any() else 0

        print(f"\nEstatisticas:")
        print(f"  - Maior distancia: {max_distance_km:.1f} km")
        print(f"  - Menor distancia (exceto 0): {min_nonzero:.1f} km")

        # Validar que as distancias fazem sentido para SP
        if max_distance_km > 500:
            print("\n⚠️  ATENCAO: Distancia maxima > 500km pode indicar problema")
        elif max_distance_km < 0.1:
            print("\n⚠️  ATENCAO: Distancias muito pequenas podem indicar problema")
        else:
            print("\n✓ Distancias parecem realistas para Sao Paulo")

        return True

    except Exception as e:
        print(f"ERRO: {e}")
        return False


def test_rate_limiting():
    """Teste 3: Verificar rate limiting."""
    print("\n" + "=" * 60)
    print("TESTE 3: Rate Limiting do Nominatim")
    print("=" * 60)

    geocoder = Geocoder()

    # Fazer 5 requests rapidos
    addresses = [
        "Rua Augusta, 100, Sao Paulo",
        "Av. Brasil, 500, Sao Paulo",
        "Rua Oscar Freire, 200, Sao Paulo",
        "Av. Reboucas, 1000, Sao Paulo",
        "Rua da Consolacao, 300, Sao Paulo",
    ]

    print(f"\nFazendo {len(addresses)} requests sequenciais...")
    print("(O rate limiter deve esperar ~1s entre cada)\n")

    start_total = time.time()
    errors_429 = 0

    for i, addr in enumerate(addresses, 1):
        start = time.time()
        result = geocoder.geocode_address(addr)
        elapsed = time.time() - start

        status = "OK" if result.success else f"ERRO: {result.error}"
        if "429" in str(result.error or ""):
            errors_429 += 1

        print(f"  [{i}/5] {elapsed:.2f}s - {status}")

    total_time = time.time() - start_total

    print(f"\nTempo total: {total_time:.1f}s")
    print(f"Tempo medio por request: {total_time / len(addresses):.1f}s")
    print(f"Erros 429 (rate limit): {errors_429}")

    if errors_429 == 0:
        print("\n✓ Rate limiting funcionando corretamente!")
        return True
    else:
        print("\n✗ Rate limiting pode precisar de ajustes")
        return False


def main():
    print("\n" + "=" * 60)
    print("  VERIFICACAO MANUAL - Fase 2: Modulo de Roteamento Real")
    print("=" * 60)

    # Teste 1: Geocodificacao
    geo_ok, geo_results = test_geocoding()

    # Teste 2: Matriz de distancias
    dist_ok = test_distance_matrix(geo_results) if geo_ok else False

    # Teste 3: Rate limiting
    rate_ok = test_rate_limiting()

    # Resumo
    print("\n" + "=" * 60)
    print("  RESUMO DA VERIFICACAO")
    print("=" * 60)

    print(f"\n  [{'✓' if geo_ok else '✗'}] Geocodificacao de enderecos")
    print(f"  [{'✓' if dist_ok else '✗'}] Matriz de distancias realistas")
    print(f"  [{'✓' if rate_ok else '✗'}] Rate limiting funcionando")

    if all([geo_ok, dist_ok, rate_ok]):
        print("\n" + "=" * 60)
        print("  ✓ Fase 2 VERIFICADA COM SUCESSO!")
        print("=" * 60 + "\n")
        return 0
    else:
        print("\n" + "=" * 60)
        print("  ✗ ALGUNS TESTES FALHARAM")
        print("  Verifique os erros acima antes de prosseguir")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
