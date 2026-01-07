import random
import math
import numpy as np
from typing import List, Tuple, Dict

def calculate_distance(city1: Tuple[int, int], city2: Tuple[int, int]) -> float:
    """Calcula distância euclidiana entre duas cidades"""
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def calculate_route_distance(route: List[Tuple[int, int]]) -> float:
    """Calcula distância total de uma rota"""
    if len(route) < 2:
        return 0
    distance = 0
    for i in range(len(route) - 1):
        distance += calculate_distance(route[i], route[i + 1])
    # Retorna ao ponto inicial (depósito)
    distance += calculate_distance(route[-1], route[0])
    return distance

def calculate_fitness(individual: List[Tuple[int, int]]) -> float:
    """Calcula fitness básico (distância total)"""
    return calculate_route_distance(individual)

def calculate_fitness_vrp(routes: List[List[Dict]], 
                          depot: Tuple[int, int],
                          priorities: Dict[Tuple[int, int], int],
                          vehicle_capacity: float,
                          vehicle_autonomy: float,
                          penalty_weight: float = 1000) -> float:
    """
    Calcula fitness para VRP com múltiplas restrições
    
    Args:
        routes: Lista de rotas, cada rota é uma lista de dicionários com info das entregas
        depot: Coordenadas do depósito central
        priorities: Dicionário mapeando localização -> prioridade (1=crítico, 2=urgente, 3=normal)
        vehicle_capacity: Capacidade máxima de carga do veículo (kg)
        vehicle_autonomy: Autonomia máxima do veículo (distância em pixels)
        penalty_weight: Peso da penalidade por violação de restrições
    
    Returns:
        Fitness (menor é melhor)
    """
    total_distance = 0
    total_penalty = 0
    priority_penalty = 0
    
    for route in routes:
        if not route:
            continue
            
        route_distance = 0
        route_load = 0
        
        # Distância do depósito até primeira entrega
        if route:
            route_distance += calculate_distance(depot, route[0]['location'])
        
        # Distâncias entre entregas
        for i in range(len(route) - 1):
            route_distance += calculate_distance(route[i]['location'], route[i + 1]['location'])
            route_load += route[i]['weight']
        
        # Adiciona peso da última entrega
        if route:
            route_load += route[-1]['weight']
            # Distância de volta ao depósito
            route_distance += calculate_distance(route[-1]['location'], depot)
        
        total_distance += route_distance
        
        # Penalidade por excesso de capacidade
        if route_load > vehicle_capacity:
            total_penalty += (route_load - vehicle_capacity) * penalty_weight
        
        # Penalidade por excesso de autonomia
        if route_distance > vehicle_autonomy:
            total_penalty += (route_distance - vehicle_autonomy) * penalty_weight
        
        # Penalidade por ordem de prioridade
        # Entregas críticas devem ser feitas primeiro
        for i, delivery in enumerate(route):
            priority = priorities.get(delivery['location'], 3)
            # Quanto mais crítico (prioridade baixa) e mais tarde na rota, maior a penalidade
            if priority == 1:  # Crítico
                priority_penalty += i * 50
            elif priority == 2:  # Urgente
                priority_penalty += i * 20
    
    # Penalidade por número de veículos (queremos minimizar)
    num_vehicles_penalty = len([r for r in routes if r]) * 100
    
    fitness = total_distance + total_penalty + priority_penalty + num_vehicles_penalty
    
    return fitness

def generate_random_population(cities: List[Tuple[int, int]], 
                               population_size: int) -> List[List[Tuple[int, int]]]:
    """Gera população inicial aleatória para TSP simples"""
    population = []
    for _ in range(population_size):
        individual = cities.copy()
        random.shuffle(individual)
        population.append(individual)
    return population

def generate_random_population_vrp(deliveries: List[Dict],
                                   num_vehicles: int,
                                   population_size: int) -> List[List[List[Dict]]]:
    """
    Gera população inicial para VRP
    
    Args:
        deliveries: Lista de entregas (cada uma é um dicionário)
        num_vehicles: Número de veículos disponíveis
        population_size: Tamanho da população
    
    Returns:
        População de soluções VRP
    """
    population = []
    
    for _ in range(population_size):
        # Embaralha entregas
        shuffled = deliveries.copy()
        random.shuffle(shuffled)
        
        # Divide entre veículos
        routes = [[] for _ in range(num_vehicles)]
        for i, delivery in enumerate(shuffled):
            vehicle_idx = i % num_vehicles
            routes[vehicle_idx].append(delivery)
        
        population.append(routes)
    
    return population

def order_crossover(parent1: List, parent2: List) -> List:
    """Order Crossover (OX) para TSP"""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [None] * size
    child[start:end] = parent1[start:end]
    
    pointer = end
    for city in parent2[end:] + parent2[:end]:
        if city not in child:
            if pointer >= size:
                pointer = 0
            child[pointer] = city
            pointer += 1
    
    return child

def pmx_crossover(parent1: List, parent2: List) -> List:
    """Partially Mapped Crossover (PMX) para TSP"""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [None] * size
    child[start:end] = parent1[start:end]
    
    for i in range(start, end):
        if parent2[i] not in child:
            pos = i
            while start <= pos < end:
                pos = parent2.index(parent1[pos])
            child[pos] = parent2[i]
    
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]
    
    return child

def vrp_crossover(parent1: List[List[Dict]], 
                  parent2: List[List[Dict]]) -> List[List[Dict]]:
    """
    Crossover para VRP - combina rotas de dois pais
    """
    num_vehicles = len(parent1)
    
    # Flatten para obter todas as entregas
    all_deliveries_p1 = [d for route in parent1 for d in route]
    all_deliveries_p2 = [d for route in parent2 for d in route]
    
    # Ponto de corte
    cut_point = random.randint(1, len(all_deliveries_p1) - 1)
    
    # Pega primeira parte do parent1
    child_deliveries = all_deliveries_p1[:cut_point].copy()
    
    # Adiciona entregas do parent2 que não estão no filho
    child_locations = {d['location'] for d in child_deliveries}
    for delivery in all_deliveries_p2:
        if delivery['location'] not in child_locations:
            child_deliveries.append(delivery)
            child_locations.add(delivery['location'])
    
    # Redistribui entre veículos
    child_routes = [[] for _ in range(num_vehicles)]
    for i, delivery in enumerate(child_deliveries):
        vehicle_idx = i % num_vehicles
        child_routes[vehicle_idx].append(delivery)
    
    return child_routes

def mutate(individual: List, mutation_probability: float) -> List:
    """Mutação por troca (swap) para TSP"""
    if random.random() < mutation_probability:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def mutate_vrp(individual: List[List[Dict]], 
               mutation_probability: float) -> List[List[Dict]]:
    """
    Mutação para VRP - várias estratégias
    """
    if random.random() > mutation_probability:
        return individual
    
    mutation_type = random.choice(['swap_within', 'swap_between', 'reverse', 'relocate'])
    
    non_empty_routes = [i for i, route in enumerate(individual) if route]
    
    if not non_empty_routes:
        return individual
    
    if mutation_type == 'swap_within':
        # Troca duas entregas dentro da mesma rota
        route_idx = random.choice(non_empty_routes)
        if len(individual[route_idx]) >= 2:
            i, j = random.sample(range(len(individual[route_idx])), 2)
            individual[route_idx][i], individual[route_idx][j] = \
                individual[route_idx][j], individual[route_idx][i]
    
    elif mutation_type == 'swap_between':
        # Troca entregas entre duas rotas diferentes
        if len(non_empty_routes) >= 2:
            route1, route2 = random.sample(non_empty_routes, 2)
            if individual[route1] and individual[route2]:
                i = random.randint(0, len(individual[route1]) - 1)
                j = random.randint(0, len(individual[route2]) - 1)
                individual[route1][i], individual[route2][j] = \
                    individual[route2][j], individual[route1][i]
    
    elif mutation_type == 'reverse':
        # Inverte uma subsequência dentro de uma rota
        route_idx = random.choice(non_empty_routes)
        if len(individual[route_idx]) >= 2:
            i, j = sorted(random.sample(range(len(individual[route_idx])), 2))
            individual[route_idx][i:j+1] = reversed(individual[route_idx][i:j+1])
    
    elif mutation_type == 'relocate':
        # Move uma entrega de uma rota para outra
        if len(individual) >= 2:
            route1 = random.choice(non_empty_routes)
            route2 = random.randint(0, len(individual) - 1)
            if individual[route1]:
                delivery_idx = random.randint(0, len(individual[route1]) - 1)
                delivery = individual[route1].pop(delivery_idx)
                individual[route2].append(delivery)
    
    return individual

def sort_population(population: List, fitness_values: List[float]) -> Tuple[List, List[float]]:
    """Ordena população por fitness (menor é melhor)"""
    sorted_pairs = sorted(zip(population, fitness_values), key=lambda x: x[1])
    sorted_population = [ind for ind, _ in sorted_pairs]
    sorted_fitness = [fit for _, fit in sorted_pairs]
    return sorted_population, sorted_fitness

def tournament_selection(population: List, 
                        fitness_values: List[float], 
                        tournament_size: int = 3) -> List:
    """Seleção por torneio"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
    return population[winner_idx]

# Problemas padrão (mantido para compatibilidade)
default_problems = {}