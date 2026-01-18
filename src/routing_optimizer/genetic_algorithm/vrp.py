"""
VRP Solver using Genetic Algorithm.

This module provides a high-level interface for solving the Vehicle Routing Problem
using a genetic algorithm approach.
"""

import random
from typing import Callable, List, Optional, Tuple

import numpy as np

from .config import GAConfig
from .core import (
    apply_local_search,
    calculate_distance,
    calculate_fitness_vrp,
    calculate_routes_total_distance,
    generate_hybrid_population,
    generate_random_population,
    mutate_vrp,
    sort_population,
    tournament_selection,
    vrp_crossover,
)


class VRPSolver:
    """Vehicle Routing Problem solver using Genetic Algorithm.

    This solver optimizes routes for multiple vehicles to visit a set of locations,
    minimizing total distance while respecting vehicle capacity constraints.

    Attributes:
        config: GAConfig instance with algorithm parameters.
        fitness_history: List of best fitness values per generation.
        best_solution: Best solution found during optimization.
        best_fitness: Fitness value of the best solution.
    """

    def __init__(self, config: Optional[GAConfig] = None):
        """Initialize the VRP solver.

        Args:
            config: GAConfig instance. If None, uses default configuration.
        """
        self.config = config or GAConfig()
        self.fitness_history: List[float] = []
        self.best_solution: Optional[List[List[int]]] = None
        self.best_fitness: Optional[float] = None
        self._distance_matrix: Optional[np.ndarray] = None
        self.converged: bool = False
        self.final_epoch: int = 0

    def solve(
        self,
        locations: List[Tuple[float, float]],
        num_vehicles: int,
        capacity: float,
        demands: Optional[List[float]] = None,
    ) -> List[List[int]]:
        """Solve VRP with Euclidean distances.

        Args:
            locations: List of (x, y) coordinates. First location is the depot.
            num_vehicles: Number of vehicles available.
            capacity: Maximum capacity per vehicle.
            demands: Demand at each location. If None, assumes 1.0 for each.

        Returns:
            List of routes. Each route is a list of location indices (1-based, 0 is depot).
        """
        # Build distance matrix from coordinates
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i, j] = calculate_distance(locations[i], locations[j])

        return self.solve_with_distance_matrix(
            distance_matrix=distance_matrix,
            num_vehicles=num_vehicles,
            capacity=capacity,
            demands=demands,
        )

    def solve_with_distance_matrix(
        self,
        distance_matrix: np.ndarray,
        num_vehicles: int,
        capacity: float,
        demands: Optional[List[float]] = None,
        progress_callback: Optional[
            Callable[[int, int, float, List[List[int]]], None]
        ] = None,
        callback_interval: int = 10,
    ) -> List[List[int]]:
        """Solve VRP with pre-computed distance matrix.

        Args:
            distance_matrix: NxN matrix of distances. Index 0 is the depot.
            num_vehicles: Number of vehicles available.
            capacity: Maximum capacity per vehicle.
            demands: Demand at each location. If None, assumes 1.0 for each.
            progress_callback: Function called every N generations with
                (current_generation, total_generations, best_fitness, best_routes).
            callback_interval: Interval of generations between callbacks (default: 10).

        Returns:
            List of routes. Each route is a list of location indices (1-based, 0 is depot).
        """
        self._distance_matrix = distance_matrix
        n = len(distance_matrix)
        num_locations = n - 1  # Exclude depot

        # Default demands
        if demands is None:
            demands = [0.0] + [1.0] * num_locations  # Depot has 0 demand

        # Reset state
        self.fitness_history = []
        self.best_solution = None
        self.best_fitness = float("inf")
        self.converged = False
        self.final_epoch = 0

        # Stagnation tracking for early convergence
        stagnation_counter = 0
        previous_best = float("inf")

        # Generate initial population
        if self.config.hybrid_initialization:
            population = generate_hybrid_population(
                num_locations=num_locations,
                num_vehicles=num_vehicles,
                population_size=self.config.population_size,
                distance_matrix=distance_matrix,
                heuristic_ratio=self.config.heuristic_ratio,
            )
        else:
            population = generate_random_population(
                num_locations=num_locations,
                num_vehicles=num_vehicles,
                population_size=self.config.population_size,
            )

        # Initial callback to show we're starting
        if progress_callback:
            progress_callback(0, self.config.max_epochs, float("inf"), None)

        # Evolution loop
        for epoch in range(self.config.max_epochs):
            # Evaluate fitness
            fitness_values = [
                calculate_fitness_vrp(
                    routes=individual,
                    distance_matrix=distance_matrix,
                    vehicle_capacity=capacity,
                    demands=demands,
                )
                for individual in population
            ]

            # Sort population
            population, fitness_values = sort_population(population, fitness_values)

            # Track best
            if fitness_values[0] < self.best_fitness:
                self.best_fitness = fitness_values[0]
                self.best_solution = [route.copy() for route in population[0]]

            self.fitness_history.append(fitness_values[0])
            self.final_epoch = epoch

            # Check for stagnation (early convergence)
            current_best = fitness_values[0]
            if abs(current_best - previous_best) < 1e-6:
                stagnation_counter += 1
                if stagnation_counter >= self.config.stagnation_threshold:
                    self.converged = True
                    break
            else:
                stagnation_counter = 0
                previous_best = current_best

            # Call progress callback
            if progress_callback and epoch % callback_interval == 0:
                progress_callback(
                    epoch, self.config.max_epochs, self.best_fitness, self.best_solution
                )

            # Create new population with elitism
            # Apply 2-opt only to elite individuals (much faster than all children)
            if self.config.local_search_elites_only:
                new_population = [
                    apply_local_search(individual, distance_matrix)
                    for individual in population[: self.config.elitism_count]
                ]
            else:
                new_population = [
                    [route.copy() for route in individual]
                    for individual in population[: self.config.elitism_count]
                ]

            # Generate offspring
            while len(new_population) < self.config.population_size:
                parent1 = tournament_selection(
                    population, fitness_values, self.config.tournament_size
                )
                parent2 = tournament_selection(
                    population, fitness_values, self.config.tournament_size
                )

                child = vrp_crossover(parent1, parent2)
                child = mutate_vrp(
                    child,
                    self.config.mutation_probability,
                    self.config.max_mutations_per_individual,
                )

                # Apply 2-opt with probability local_search_rate (if not using elites_only)
                if not self.config.local_search_elites_only:
                    if random.random() < self.config.local_search_rate:
                        child = apply_local_search(child, distance_matrix)

                new_population.append(child)

            population = new_population

        # Final callback at completion
        if progress_callback:
            progress_callback(
                self.final_epoch + 1,
                self.config.max_epochs,
                self.best_fitness,
                self.best_solution,
            )

        return self.best_solution or []

    def get_fitness_history(self) -> List[float]:
        """Get history of best fitness values per generation.

        Returns:
            List of fitness values.
        """
        return self.fitness_history.copy()

    def get_total_distance(self, routes: Optional[List[List[int]]] = None) -> float:
        """Calculate total distance of routes.

        Args:
            routes: Routes to calculate. If None, uses best_solution.

        Returns:
            Total distance of all routes.

        Raises:
            ValueError: If no routes provided and no solution has been computed.
        """
        if routes is None:
            routes = self.best_solution

        if routes is None:
            raise ValueError("No routes provided and no solution has been computed")

        if self._distance_matrix is None:
            raise ValueError("No distance matrix available. Run solve() first.")

        return calculate_routes_total_distance(routes, self._distance_matrix)

    def get_route_details(
        self,
        routes: Optional[List[List[int]]] = None,
    ) -> List[dict]:
        """Get detailed information about each route.

        Args:
            routes: Routes to analyze. If None, uses best_solution.

        Returns:
            List of dicts with route details (stops, distance, etc.).
        """
        if routes is None:
            routes = self.best_solution

        if routes is None:
            return []

        if self._distance_matrix is None:
            return []

        details = []
        for i, route in enumerate(routes):
            if not route:
                continue

            # Calculate route distance
            route_distance = 0.0
            route_distance += self._distance_matrix[0, route[0]]  # Depot to first
            for j in range(len(route) - 1):
                route_distance += self._distance_matrix[route[j], route[j + 1]]
            route_distance += self._distance_matrix[route[-1], 0]  # Last to depot

            details.append(
                {
                    "vehicle": i + 1,
                    "stops": len(route),
                    "locations": route.copy(),
                    "distance": route_distance,
                }
            )

        return details
