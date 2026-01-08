"""
Core genetic algorithm functions for VRP optimization.

This module contains the fundamental operations for genetic algorithms:
- Distance calculations
- Fitness evaluation
- Population generation
- Selection, crossover, and mutation operators
"""

import math
import random
from typing import List, Tuple

import numpy as np


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        point1: First point as (x, y) or (lat, lon).
        point2: Second point as (x, y) or (lat, lon).

    Returns:
        Euclidean distance between the points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_route_distance(
    route: List[int],
    distance_matrix: np.ndarray,
    depot_index: int = 0,
) -> float:
    """Calculate total distance of a route using a distance matrix.

    Args:
        route: List of location indices (not including depot).
        distance_matrix: NxN matrix of distances between all locations.
        depot_index: Index of the depot in the distance matrix.

    Returns:
        Total distance including return to depot.
    """
    if not route:
        return 0.0

    total = 0.0
    # Depot to first stop
    total += distance_matrix[depot_index, route[0]]

    # Between stops
    for i in range(len(route) - 1):
        total += distance_matrix[route[i], route[i + 1]]

    # Last stop back to depot
    total += distance_matrix[route[-1], depot_index]

    return total


def calculate_routes_total_distance(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    depot_index: int = 0,
) -> float:
    """Calculate total distance of all routes.

    Args:
        routes: List of routes, each route is a list of location indices.
        distance_matrix: NxN matrix of distances between all locations.
        depot_index: Index of the depot in the distance matrix.

    Returns:
        Sum of all route distances.
    """
    return sum(calculate_route_distance(route, distance_matrix, depot_index) for route in routes)


def calculate_fitness_vrp(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    vehicle_capacity: float,
    demands: List[float],
    depot_index: int = 0,
    capacity_penalty: float = 1000.0,
    vehicle_penalty: float = 100.0,
) -> float:
    """Calculate fitness for VRP solution.

    Lower fitness is better. Fitness includes:
    - Total distance traveled
    - Penalty for exceeding vehicle capacity
    - Penalty for number of vehicles used

    Args:
        routes: List of routes, each route is a list of location indices.
        distance_matrix: NxN matrix of distances between all locations.
        vehicle_capacity: Maximum capacity per vehicle.
        demands: List of demands for each location (index 0 is depot with 0 demand).
        depot_index: Index of the depot in the distance matrix.
        capacity_penalty: Penalty multiplier for capacity violations.
        vehicle_penalty: Penalty per vehicle used.

    Returns:
        Fitness value (lower is better).
    """
    total_distance = 0.0
    total_penalty = 0.0

    for route in routes:
        if not route:
            continue

        # Calculate route distance
        total_distance += calculate_route_distance(route, distance_matrix, depot_index)

        # Check capacity
        route_demand = sum(demands[loc] for loc in route)
        if route_demand > vehicle_capacity:
            total_penalty += (route_demand - vehicle_capacity) * capacity_penalty

    # Penalty for number of vehicles
    num_vehicles_used = len([r for r in routes if r])
    total_penalty += num_vehicles_used * vehicle_penalty

    return total_distance + total_penalty


def generate_random_solution(
    num_locations: int,
    num_vehicles: int,
) -> List[List[int]]:
    """Generate a random VRP solution.

    Args:
        num_locations: Total number of locations (excluding depot).
        num_vehicles: Number of vehicles available.

    Returns:
        List of routes, each containing location indices.
    """
    # Create list of all locations (1 to num_locations, 0 is depot)
    locations = list(range(1, num_locations + 1))
    random.shuffle(locations)

    # Distribute among vehicles
    routes: List[List[int]] = [[] for _ in range(num_vehicles)]
    for i, loc in enumerate(locations):
        routes[i % num_vehicles].append(loc)

    return routes


def generate_random_population(
    num_locations: int,
    num_vehicles: int,
    population_size: int,
) -> List[List[List[int]]]:
    """Generate initial random population for VRP.

    Args:
        num_locations: Total number of locations (excluding depot).
        num_vehicles: Number of vehicles available.
        population_size: Number of individuals in population.

    Returns:
        Population of VRP solutions.
    """
    return [generate_random_solution(num_locations, num_vehicles) for _ in range(population_size)]


def tournament_selection(
    population: List[List[List[int]]],
    fitness_values: List[float],
    tournament_size: int = 3,
) -> List[List[int]]:
    """Select an individual using tournament selection.

    Args:
        population: List of solutions.
        fitness_values: Fitness value for each solution.
        tournament_size: Number of individuals in tournament.

    Returns:
        Selected individual (copy).
    """
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
    return [route.copy() for route in population[winner_idx]]


def vrp_crossover(
    parent1: List[List[int]],
    parent2: List[List[int]],
) -> List[List[int]]:
    """Perform crossover for VRP solutions.

    Uses a simple crossover that:
    1. Flattens both parents
    2. Takes first part from parent1
    3. Fills remaining from parent2 (preserving order, avoiding duplicates)
    4. Redistributes among vehicles

    Args:
        parent1: First parent solution.
        parent2: Second parent solution.

    Returns:
        Child solution.
    """
    num_vehicles = len(parent1)

    # Flatten parents
    flat_p1 = [loc for route in parent1 for loc in route]
    flat_p2 = [loc for route in parent2 for loc in route]

    if not flat_p1:
        return [route.copy() for route in parent2]

    # Crossover point
    cut_point = random.randint(1, len(flat_p1) - 1) if len(flat_p1) > 1 else 1

    # Build child
    child_locations = flat_p1[:cut_point].copy()
    child_set = set(child_locations)

    for loc in flat_p2:
        if loc not in child_set:
            child_locations.append(loc)
            child_set.add(loc)

    # Redistribute among vehicles
    child_routes: List[List[int]] = [[] for _ in range(num_vehicles)]
    for i, loc in enumerate(child_locations):
        child_routes[i % num_vehicles].append(loc)

    return child_routes


def mutate_vrp(
    individual: List[List[int]],
    mutation_probability: float,
) -> List[List[int]]:
    """Apply mutation to a VRP solution.

    Mutation types:
    - swap_within: Swap two locations within the same route
    - swap_between: Swap locations between two routes
    - reverse: Reverse a segment within a route
    - relocate: Move a location from one route to another

    Args:
        individual: VRP solution to mutate.
        mutation_probability: Probability of mutation.

    Returns:
        Mutated solution.
    """
    if random.random() > mutation_probability:
        return individual

    mutation_type = random.choice(["swap_within", "swap_between", "reverse", "relocate"])
    non_empty_routes = [i for i, route in enumerate(individual) if route]

    if not non_empty_routes:
        return individual

    if mutation_type == "swap_within":
        route_idx = random.choice(non_empty_routes)
        if len(individual[route_idx]) >= 2:
            i, j = random.sample(range(len(individual[route_idx])), 2)
            individual[route_idx][i], individual[route_idx][j] = (
                individual[route_idx][j],
                individual[route_idx][i],
            )

    elif mutation_type == "swap_between":
        if len(non_empty_routes) >= 2:
            route1, route2 = random.sample(non_empty_routes, 2)
            if individual[route1] and individual[route2]:
                i = random.randint(0, len(individual[route1]) - 1)
                j = random.randint(0, len(individual[route2]) - 1)
                individual[route1][i], individual[route2][j] = (
                    individual[route2][j],
                    individual[route1][i],
                )

    elif mutation_type == "reverse":
        route_idx = random.choice(non_empty_routes)
        if len(individual[route_idx]) >= 2:
            i, j = sorted(random.sample(range(len(individual[route_idx])), 2))
            individual[route_idx][i : j + 1] = reversed(individual[route_idx][i : j + 1])

    elif mutation_type == "relocate":
        if len(individual) >= 2:
            route1 = random.choice(non_empty_routes)
            route2 = random.randint(0, len(individual) - 1)
            if individual[route1]:
                loc_idx = random.randint(0, len(individual[route1]) - 1)
                location = individual[route1].pop(loc_idx)
                individual[route2].append(location)

    return individual


def sort_population(
    population: List[List[List[int]]],
    fitness_values: List[float],
) -> Tuple[List[List[List[int]]], List[float]]:
    """Sort population by fitness (ascending - lower is better).

    Args:
        population: List of solutions.
        fitness_values: Fitness value for each solution.

    Returns:
        Tuple of (sorted_population, sorted_fitness_values).
    """
    sorted_pairs = sorted(zip(population, fitness_values), key=lambda x: x[1])
    sorted_population = [ind for ind, _ in sorted_pairs]
    sorted_fitness = [fit for _, fit in sorted_pairs]
    return sorted_population, sorted_fitness
