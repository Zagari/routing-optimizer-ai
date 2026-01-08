"""
Tests for the genetic algorithm module.
"""

import numpy as np
import pytest

from routing_optimizer.genetic_algorithm.config import GAConfig
from routing_optimizer.genetic_algorithm.core import (
    calculate_distance,
    calculate_fitness_vrp,
    calculate_route_distance,
    generate_random_population,
    generate_random_solution,
    mutate_vrp,
    sort_population,
    vrp_crossover,
)
from routing_optimizer.genetic_algorithm.vrp import VRPSolver


class TestGAConfig:
    """Tests for GAConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GAConfig()
        assert config.population_size == 200
        assert config.mutation_probability == 0.6
        assert config.max_epochs == 1000
        assert config.tournament_size == 5
        assert config.elitism_count == 2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GAConfig(
            population_size=100,
            mutation_probability=0.3,
            max_epochs=500,
        )
        assert config.population_size == 100
        assert config.mutation_probability == 0.3
        assert config.max_epochs == 500

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("GA_POPULATION_SIZE", "150")
        monkeypatch.setenv("GA_MUTATION_PROBABILITY", "0.5")
        config = GAConfig.from_env()
        assert config.population_size == 150
        assert config.mutation_probability == 0.5

    def test_invalid_population_size(self):
        """Test validation of population size."""
        with pytest.raises(ValueError, match="population_size must be at least 2"):
            GAConfig(population_size=1)

    def test_invalid_mutation_probability(self):
        """Test validation of mutation probability."""
        with pytest.raises(ValueError, match="mutation_probability must be between"):
            GAConfig(mutation_probability=1.5)

    def test_invalid_max_epochs(self):
        """Test validation of max epochs."""
        with pytest.raises(ValueError, match="max_epochs must be at least 1"):
            GAConfig(max_epochs=0)


class TestCoreFunctions:
    """Tests for core genetic algorithm functions."""

    def test_calculate_distance(self):
        """Test Euclidean distance calculation."""
        assert calculate_distance((0, 0), (3, 4)) == 5.0
        assert calculate_distance((0, 0), (0, 0)) == 0.0
        assert abs(calculate_distance((1, 1), (2, 2)) - 1.4142) < 0.001

    def test_calculate_route_distance(self):
        """Test route distance calculation with matrix."""
        # Simple 3x3 distance matrix
        matrix = np.array(
            [
                [0, 10, 20],
                [10, 0, 15],
                [20, 15, 0],
            ]
        )

        # Route: depot(0) -> 1 -> 2 -> depot(0)
        route = [1, 2]
        distance = calculate_route_distance(route, matrix, depot_index=0)
        # 0->1: 10, 1->2: 15, 2->0: 20 = 45
        assert distance == 45.0

    def test_calculate_route_distance_empty(self):
        """Test route distance with empty route."""
        matrix = np.array([[0, 10], [10, 0]])
        assert calculate_route_distance([], matrix) == 0.0

    def test_generate_random_solution(self):
        """Test random solution generation."""
        solution = generate_random_solution(num_locations=10, num_vehicles=3)

        # Should have 3 routes
        assert len(solution) == 3

        # All locations should be covered
        all_locations = set()
        for route in solution:
            all_locations.update(route)
        assert all_locations == set(range(1, 11))

    def test_generate_random_population(self):
        """Test population generation."""
        population = generate_random_population(
            num_locations=5,
            num_vehicles=2,
            population_size=10,
        )

        assert len(population) == 10

        # Each individual should cover all locations
        for individual in population:
            all_locations = set()
            for route in individual:
                all_locations.update(route)
            assert all_locations == set(range(1, 6))

    def test_sort_population(self):
        """Test population sorting by fitness."""
        population = [[[1, 2]], [[3, 4]], [[5]]]
        fitness = [100.0, 50.0, 75.0]

        sorted_pop, sorted_fit = sort_population(population, fitness)

        assert sorted_fit == [50.0, 75.0, 100.0]
        assert sorted_pop[0] == [[3, 4]]

    def test_vrp_crossover(self):
        """Test VRP crossover produces valid offspring."""
        parent1 = [[1, 2], [3, 4], [5]]
        parent2 = [[5, 4], [3, 2], [1]]

        child = vrp_crossover(parent1, parent2)

        # Should have same number of routes
        assert len(child) == len(parent1)

        # Should contain all locations exactly once
        all_locations = set()
        for route in child:
            for loc in route:
                assert loc not in all_locations
                all_locations.add(loc)
        assert all_locations == {1, 2, 3, 4, 5}

    def test_mutate_vrp(self):
        """Test VRP mutation produces valid solution."""
        individual = [[1, 2, 3], [4, 5], [6]]

        # Run mutation multiple times (it's probabilistic)
        for _ in range(10):
            mutated = mutate_vrp(individual.copy(), mutation_probability=1.0)

            # Should still have all locations
            all_locations = set()
            for route in mutated:
                all_locations.update(route)
            assert all_locations == {1, 2, 3, 4, 5, 6}


class TestVRPSolver:
    """Tests for VRPSolver class."""

    def test_solver_basic(self):
        """Test basic solver functionality."""
        locations = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]
        config = GAConfig(population_size=20, max_epochs=50)
        solver = VRPSolver(config)

        routes = solver.solve(
            locations=locations,
            num_vehicles=2,
            capacity=10,
        )

        # Should return valid routes
        assert len(routes) <= 2

        # All locations (except depot) should be visited
        all_visited = set()
        for route in routes:
            all_visited.update(route)
        assert all_visited == set(range(1, 5))

    def test_solver_with_distance_matrix(self):
        """Test solver with pre-computed distance matrix."""
        # 5x5 symmetric distance matrix
        np.random.seed(42)
        n = 5
        matrix = np.random.randint(10, 100, size=(n, n)).astype(float)
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        np.fill_diagonal(matrix, 0)

        config = GAConfig(population_size=20, max_epochs=50)
        solver = VRPSolver(config)

        routes = solver.solve_with_distance_matrix(
            distance_matrix=matrix,
            num_vehicles=2,
            capacity=10,
        )

        # Verify solution validity
        all_visited = set()
        for route in routes:
            all_visited.update(route)
        assert all_visited == set(range(1, n))

    def test_solver_fitness_history(self):
        """Test that fitness history is recorded."""
        locations = [(0, 0), (1, 0), (0, 1), (1, 1)]
        config = GAConfig(population_size=10, max_epochs=20)
        solver = VRPSolver(config)

        solver.solve(locations=locations, num_vehicles=2, capacity=5)

        history = solver.get_fitness_history()
        assert len(history) == 20
        # Generally fitness should not increase (we keep elites)
        assert history[-1] <= history[0] or len(history) < 5

    def test_solver_get_total_distance(self):
        """Test total distance calculation."""
        locations = [(0, 0), (3, 0), (0, 4)]  # 3-4-5 triangle
        config = GAConfig(population_size=10, max_epochs=10)
        solver = VRPSolver(config)

        solver.solve(locations=locations, num_vehicles=1, capacity=10)

        distance = solver.get_total_distance()
        # Single route visiting both locations and returning
        # Distance should be: depot->loc1->loc2->depot or depot->loc2->loc1->depot
        assert distance > 0

    def test_solver_get_route_details(self):
        """Test route details extraction."""
        locations = [(0, 0), (1, 0), (2, 0), (3, 0)]
        config = GAConfig(population_size=10, max_epochs=10)
        solver = VRPSolver(config)

        solver.solve(locations=locations, num_vehicles=2, capacity=5)

        details = solver.get_route_details()

        # Should have details for non-empty routes
        assert len(details) > 0

        for d in details:
            assert "vehicle" in d
            assert "stops" in d
            assert "locations" in d
            assert "distance" in d
            assert d["distance"] >= 0


class TestFitnessVRP:
    """Tests for VRP fitness calculation."""

    def test_fitness_basic(self):
        """Test basic fitness calculation."""
        matrix = np.array(
            [
                [0, 10, 20],
                [10, 0, 15],
                [20, 15, 0],
            ]
        )
        routes = [[1, 2]]
        demands = [0, 5, 5]  # Depot has 0 demand

        fitness = calculate_fitness_vrp(
            routes=routes,
            distance_matrix=matrix,
            vehicle_capacity=20,
            demands=demands,
        )

        # Distance: 0->1->2->0 = 10 + 15 + 20 = 45
        # No capacity violation
        # Vehicle penalty: 100 (1 vehicle)
        assert fitness == 45 + 100

    def test_fitness_capacity_violation(self):
        """Test fitness with capacity violation penalty."""
        matrix = np.array(
            [
                [0, 10],
                [10, 0],
            ]
        )
        routes = [[1]]
        demands = [0, 15]  # Location 1 has demand 15

        fitness_ok = calculate_fitness_vrp(
            routes=routes,
            distance_matrix=matrix,
            vehicle_capacity=20,
            demands=demands,
        )

        fitness_violation = calculate_fitness_vrp(
            routes=routes,
            distance_matrix=matrix,
            vehicle_capacity=10,  # Capacity too low
            demands=demands,
        )

        # Violation should add penalty
        assert fitness_violation > fitness_ok
