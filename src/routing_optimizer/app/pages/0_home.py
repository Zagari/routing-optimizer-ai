"""
Home page with project overview and instructions.
"""

import streamlit as st

st.title("🚚 Otimizador de Rotas para Distribuição de Medicamentos")

st.markdown(
    """
### Tech Challenge FIAP - Fase 2

Este sistema otimiza rotas de entrega usando **Algoritmo Genético** para resolver
o **Vehicle Routing Problem (VRP)**.

---

#### Como usar:

1. **📁 Upload**: Carregue um CSV com endereços das farmácias
2. **⚙️ Otimizar**: Configure parâmetros e execute o algoritmo
3. **🗺️ Resultados**: Visualize rotas no mapa e obtenha instruções

---

#### Sobre o projeto:

O sistema utiliza:
- **Nominatim** (OpenStreetMap) para geocodificação de endereços
- **OSRM** para cálculo de distâncias reais de estrada
- **Algoritmo Genético** para otimização das rotas de entrega

---

Navegue pelas páginas no menu lateral para começar.
"""
)
