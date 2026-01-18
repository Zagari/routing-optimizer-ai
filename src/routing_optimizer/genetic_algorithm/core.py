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


def nearest_neighbor_solution(
    distance_matrix: np.ndarray,
    num_vehicles: int,
) -> List[List[int]]:
    """Generate a VRP solution using nearest neighbor heuristic.

    Builds routes by repeatedly selecting the closest unvisited location.
    This typically produces better initial solutions than random allocation.

    Args:
        distance_matrix: NxN matrix of distances. Index 0 is depot.
        num_vehicles: Number of vehicles available.

    Returns:
        List of routes, each containing location indices.
    """
    n = len(distance_matrix)
    num_locations = n - 1
    unvisited = set(range(1, n))  # Exclude depot
    routes: List[List[int]] = []

    # Target stops per vehicle for balanced distribution
    target_stops = (num_locations + num_vehicles - 1) // num_vehicles

    for v in range(num_vehicles):
        if not unvisited:
            routes.append([])
            continue

        route: List[int] = []
        current = 0  # Start at depot

        while unvisited and len(route) < target_stops:
            # Find nearest unvisited location
            nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        routes.append(route)

    # Distribute any remaining locations
    if unvisited:
        remaining = list(unvisited)
        for i, loc in enumerate(remaining):
            route_idx = i % num_vehicles
            routes[route_idx].append(loc)

    return routes


def generate_hybrid_population(
    num_locations: int,
    num_vehicles: int,
    population_size: int,
    distance_matrix: np.ndarray,
    heuristic_ratio: float = 0.1,
) -> List[List[List[int]]]:
    """Generate hybrid population with heuristic and random solutions.

    Creates a population where a portion uses nearest neighbor heuristic
    (for good initial solutions) and the rest are random (for diversity).

    Args:
        num_locations: Total number of locations (excluding depot).
        num_vehicles: Number of vehicles available.
        population_size: Number of individuals in population.
        distance_matrix: NxN matrix of distances for nearest neighbor.
        heuristic_ratio: Fraction of population to initialize with heuristic (0.0 to 1.0).

    Returns:
        Population of VRP solutions.
    """
    population: List[List[List[int]]] = []

    # Number of heuristic solutions (at least 1 if ratio > 0)
    heuristic_count = max(1, int(population_size * heuristic_ratio)) if heuristic_ratio > 0 else 0

    # Generate heuristic solutions with variations
    base_solution = nearest_neighbor_solution(distance_matrix, num_vehicles)
    population.append([route.copy() for route in base_solution])

    for i in range(1, heuristic_count):
        # Create variation of the base solution
        variation = [route.copy() for route in base_solution]

        # Apply random swaps to create diversity
        for route in variation:
            if len(route) >= 2 and random.random() < 0.5:
                # Swap two random locations within the route
                idx1, idx2 = random.sample(range(len(route)), 2)
                route[idx1], route[idx2] = route[idx2], route[idx1]

        population.append(variation)

    # Fill rest with random solutions
    while len(population) < population_size:
        population.append(generate_random_solution(num_locations, num_vehicles))

    return population


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
    """Perform route-preserving crossover for VRP solutions.

    This crossover preserves complete routes from parents:
    1. Selects k complete routes from parent1
    2. Identifies locations already assigned
    3. Fills remaining locations maintaining parent2 order

    This approach preserves good route structures (e.g., geographic clusters)
    instead of destroying them through flattening.

    Args:
        parent1: First parent solution.
        parent2: Second parent solution.

    Returns:
        Child solution preserving route structure.
    """
    num_vehicles = len(parent1)

    # Find non-empty routes in parent1
    non_empty_p1 = [i for i, r in enumerate(parent1) if r]

    # Handle edge case: parent1 has no routes
    if not non_empty_p1:
        return [route.copy() for route in parent2]

    # Select random routes from parent1 to preserve (at least 1, at most n-1)
    max_to_keep = max(1, len(non_empty_p1) - 1)
    num_routes_to_keep = random.randint(1, max_to_keep)
    routes_to_keep = random.sample(non_empty_p1, min(num_routes_to_keep, len(non_empty_p1)))

    child: List[List[int]] = [[] for _ in range(num_vehicles)]
    used_locations: set = set()

    # Copy selected routes from parent1
    for idx in routes_to_keep:
        child[idx] = parent1[idx].copy()
        used_locations.update(parent1[idx])

    # Collect remaining locations in parent2 order
    remaining: List[int] = []
    for route in parent2:
        for loc in route:
            if loc not in used_locations:
                remaining.append(loc)

    # Distribute remaining locations to empty routes
    empty_routes = [i for i in range(num_vehicles) if not child[i]]

    if empty_routes:
        for i, loc in enumerate(remaining):
            route_idx = empty_routes[i % len(empty_routes)]
            child[route_idx].append(loc)
    else:
        # If no empty routes, distribute to existing routes round-robin
        for i, loc in enumerate(remaining):
            route_idx = i % num_vehicles
            child[route_idx].append(loc)

    return child


def _apply_single_mutation(individual: List[List[int]]) -> None:
    """Apply a single mutation to a VRP solution (in-place).

    Mutation types:
    - swap_within: Swap two locations within the same route
    - swap_between: Swap locations between two routes
    - reverse: Reverse a segment within a route
    - relocate: Move a location from one route to another

    Args:
        individual: VRP solution to mutate (modified in-place).
    """
    mutation_type = random.choice(["swap_within", "swap_between", "reverse", "relocate"])
    non_empty_routes = [i for i, route in enumerate(individual) if route]

    if not non_empty_routes:
        return

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


def mutate_vrp(
    individual: List[List[int]],
    mutation_probability: float,
    max_mutations: int = 1,
) -> List[List[int]]:
    """Apply mutation to a VRP solution.

    When mutation is triggered, applies 1 to max_mutations mutations.
    For large problems, multiple mutations help explore the solution space
    more effectively.

    Args:
        individual: VRP solution to mutate.
        mutation_probability: Probability of mutation.
        max_mutations: Maximum number of mutations to apply (1 to max_mutations).

    Returns:
        Mutated solution.
    """
    if random.random() > mutation_probability:
        return individual

    # Apply 1 to max_mutations mutations
    num_mutations = random.randint(1, max(1, max_mutations))
    for _ in range(num_mutations):
        _apply_single_mutation(individual)

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


def two_opt(
    route: List[int],
    distance_matrix: np.ndarray,
    depot_index: int = 0,
) -> List[int]:
    """Apply 2-opt local search to optimize a single route.

    2-opt iteratively reverses segments of the route to reduce total distance.
    It continues until no improvement is found (local optimum).

    This is particularly effective at removing "crossing" edges in routes.

    Args:
        route: List of location indices (not including depot).
        distance_matrix: NxN matrix of distances between all locations.
        depot_index: Index of the depot in the distance matrix.

    Returns:
        Optimized route (new list, original unchanged).
    """
    if len(route) < 3:
        return route.copy() if route else []

    improved = True
    best_route = route.copy()

    while improved:
        improved = False
        best_distance = calculate_route_distance(best_route, distance_matrix, depot_index)

        for i in range(len(best_route) - 1):
            for j in range(i + 2, len(best_route)):
                # Create new route with reversed segment between i+1 and j
                new_route = (
                    best_route[: i + 1]
                    + best_route[i + 1 : j + 1][::-1]
                    + best_route[j + 1 :]
                )

                new_distance = calculate_route_distance(
                    new_route, distance_matrix, depot_index
                )

                if new_distance < best_distance - 1e-9:  # Small epsilon for float comparison
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
                    break

            if improved:
                break

    return best_route


def apply_local_search(
    individual: List[List[int]],
    distance_matrix: np.ndarray,
    depot_index: int = 0,
) -> List[List[int]]:
    """Apply 2-opt local search to all routes in a VRP solution.

    Args:
        individual: VRP solution (list of routes).
        distance_matrix: NxN matrix of distances between all locations.
        depot_index: Index of the depot in the distance matrix.

    Returns:
        Optimized solution with improved routes (new list).
    """
    optimized: List[List[int]] = []
    for route in individual:
        if route:
            optimized.append(two_opt(route, distance_matrix, depot_index))
        else:
            optimized.append([])
    return optimized
