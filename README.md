# Routing Optimizer - VRP com Algoritmo Genético

Sistema de otimização de rotas para distribuição de medicamentos utilizando Algoritmo Genético para resolver o Vehicle Routing Problem (VRP).

**Tech Challenge FIAP - Pós-graduação em IA para Devs - Fase 2**

## Funcionalidades

- Otimização de rotas para múltiplos veículos
- Suporte a restrições de capacidade
- Geocodificação de endereços (Nominatim)
- Cálculo de distâncias reais (OSRM)
- Interface web com Streamlit
- Visualização em mapas interativos (Folium)
- Geração de instruções com ChatGPT
- Comparativo com algoritmos baseline

## Requisitos

- Python 3.11+
- pip

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Zagari/routing-optimizer-ai.git
cd routing-optimizer-ai
```

2. Crie e ative o ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
# Instalação básica (apenas AG)
pip install -e .

# Instalação completa (todas as funcionalidades)
pip install -e ".[all]"

# Ou instale módulos específicos:
pip install -e ".[dev]"  # Ferramentas de desenvolvimento
pip install -e ".[web]"  # Interface Streamlit
pip install -e ".[geo]"  # Geocodificação
pip install -e ".[llm]"  # Integração ChatGPT
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite .env e adicione sua OPENAI_API_KEY
```

## Uso Rápido

### Usando o VRPSolver diretamente

```python
from routing_optimizer.genetic_algorithm import VRPSolver, GAConfig

# Configurar o solver
config = GAConfig(
    population_size=100,
    max_epochs=500,
    mutation_probability=0.6
)
solver = VRPSolver(config)

# Definir localizações (primeira é o depósito)
locations = [
    (0, 0),   # Depósito
    (1, 2),   # Local 1
    (3, 4),   # Local 2
    (5, 1),   # Local 3
    # ...
]

# Resolver
routes = solver.solve(
    locations=locations,
    num_vehicles=3,
    capacity=50
)

# Ver resultados
print(f"Rotas encontradas: {routes}")
print(f"Distância total: {solver.get_total_distance()}")
```

### Executando a Interface Web

```bash
streamlit run src/routing_optimizer/app/main.py
```

## Estrutura do Projeto

```
routing-optimizer-ai/
├── src/routing_optimizer/
│   ├── genetic_algorithm/   # Core do AG
│   │   ├── core.py          # Funções fundamentais
│   │   ├── vrp.py           # VRPSolver
│   │   └── config.py        # Configurações
│   ├── routing/             # Geocodificação e distâncias
│   ├── baselines/           # Algoritmos de comparação
│   ├── llm/                 # Integração com ChatGPT
│   └── app/                 # Interface Streamlit
├── tests/                   # Testes automatizados
├── data/                    # Dados de exemplo
└── infra/                   # Terraform para AWS
```

## Testes

```bash
# Executar todos os testes
pytest

# Com cobertura
pytest --cov=src/routing_optimizer

# Apenas testes unitários (sem integração)
pytest -m "not integration"
```

## Algoritmos Implementados

1. **Algoritmo Genético (AG)** - Nossa solução principal
2. **Random** - Baseline mínimo
3. **Nearest Neighbor** - Heurística gulosa
4. **Clarke-Wright Savings** - Clássico para VRP

## Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do Tech Challenge FIAP.
