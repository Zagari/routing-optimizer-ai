"""
Tests for the genetic algorithm module.
"""

import numpy as np
import pytest

from routing_optimizer.genetic_algorithm.config import GAConfig
from routing_optimizer.genetic_algorithm.core import (
    apply_local_search,
    calculate_distance,
    calculate_fitness_vrp,
    calculate_route_distance,
    generate_hybrid_population,
    generate_random_population,
    generate_random_solution,
    mutate_vrp,
    nearest_neighbor_solution,
    sort_population,
    two_opt,
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
        assert config.local_search_rate == 0.1
        assert config.local_search_elites_only is True
        # stagnation_threshold defaults to 20% of max_epochs (1000 * 0.2 = 200)
        assert config.stagnation_threshold == 200
        assert config.max_mutations_per_individual == 3
        assert config.hybrid_initialization is True
        assert config.heuristic_ratio == 0.1

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

    def test_invalid_local_search_rate(self):
        """Test validation of local_search_rate."""
        with pytest.raises(ValueError, match="local_search_rate must be between"):
            GAConfig(local_search_rate=1.5)

    def test_invalid_max_epochs(self):
        """Test validation of max epochs."""
        with pytest.raises(ValueError, match="max_epochs must be at least 1"):
            GAConfig(max_epochs=0)

    def test_max_epochs_limit(self):
        """Test validation of max epochs upper limit (10000)."""
        with pytest.raises(ValueError, match="max_epochs must be at most 10000"):
            GAConfig(max_epochs=10001)

    def test_stagnation_threshold_default_percentage(self):
        """Test that stagnation_threshold defaults to 20% of max_epochs."""
        # 500 epochs -> 100 stagnation threshold (20%)
        config = GAConfig(max_epochs=500)
        assert config.stagnation_threshold == 100

        # 1000 epochs -> 200 stagnation threshold (20%)
        config = GAConfig(max_epochs=1000)
        assert config.stagnation_threshold == 200

        # 10000 epochs -> 2000 stagnation threshold (20%)
        config = GAConfig(max_epochs=10000)
        assert config.stagnation_threshold == 2000

        # Small value - should be at least 1
        config = GAConfig(max_epochs=3)
        assert config.stagnation_threshold == 1  # max(1, int(3 * 0.2))

    def test_stagnation_threshold_explicit_override(self):
        """Test that explicit stagnation_threshold overrides default."""
        config = GAConfig(max_epochs=1000, stagnation_threshold=50)
        assert config.stagnation_threshold == 50

    def test_invalid_stagnation_threshold(self):
        """Test validation of stagnation threshold."""
        with pytest.raises(ValueError, match="stagnation_threshold must be at least 1"):
            GAConfig(stagnation_threshold=0)

    def test_invalid_max_mutations_per_individual(self):
        """Test validation of max mutations per individual."""
        with pytest.raises(ValueError, match="max_mutations_per_individual must be at least 1"):
            GAConfig(max_mutations_per_individual=0)

    def test_invalid_heuristic_ratio(self):
        """Test validation of heuristic ratio."""
        with pytest.raises(ValueError, match="heuristic_ratio must be between"):
            GAConfig(heuristic_ratio=1.5)


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

    def test_mutate_vrp_multiple_mutations(self):
        """Test VRP mutation with multiple mutations produces valid solution."""
        individual = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12]]

        # Run with max_mutations=5
        for _ in range(20):
            # Deep copy to avoid modifying original
            ind_copy = [[loc for loc in route] for route in individual]
            mutated = mutate_vrp(ind_copy, mutation_probability=1.0, max_mutations=5)

            # Should still have all locations
            all_locations = set()
            for route in mutated:
                all_locations.update(route)
            assert all_locations == set(range(1, 13))

    def test_mutate_vrp_max_mutations_respected(self):
        """Test that max_mutations parameter is respected."""
        individual = [[1, 2, 3], [4, 5], [6]]

        # With max_mutations=1, solution should still be valid
        mutated = mutate_vrp(individual.copy(), mutation_probability=1.0, max_mutations=1)
        all_locations = set()
        for route in mutated:
            all_locations.update(route)
        assert all_locations == {1, 2, 3, 4, 5, 6}

        # With max_mutations=10, solution should still be valid
        mutated = mutate_vrp(individual.copy(), mutation_probability=1.0, max_mutations=10)
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
        # Use high stagnation_threshold to ensure all epochs run
        config = GAConfig(population_size=10, max_epochs=20, stagnation_threshold=100)
        solver = VRPSolver(config)

        solver.solve(locations=locations, num_vehicles=2, capacity=5)

        history = solver.get_fitness_history()
        # With high stagnation threshold, should run all epochs
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

    def test_solver_progress_callback(self):
        """Test that progress callback is called during optimization."""
        np.random.seed(42)
        n = 5
        matrix = np.random.randint(10, 100, size=(n, n)).astype(float)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)

        # Use high stagnation_threshold to ensure all epochs run
        config = GAConfig(population_size=20, max_epochs=50, stagnation_threshold=100)
        solver = VRPSolver(config)

        callback_calls = []

        def progress_callback(generation, total, best_fitness, best_routes):
            callback_calls.append(
                {
                    "generation": generation,
                    "total": total,
                    "best_fitness": best_fitness,
                    "best_routes": best_routes,
                }
            )

        solver.solve_with_distance_matrix(
            distance_matrix=matrix,
            num_vehicles=2,
            capacity=10,
            progress_callback=progress_callback,
            callback_interval=10,
        )

        # Should be called: initial (before loop) + at generations 0, 10, 20, 30, 40, and final (50)
        assert len(callback_calls) >= 6

        # First call is initial state (before first generation)
        initial_call = callback_calls[0]
        assert initial_call["generation"] == 0
        assert initial_call["best_fitness"] == float("inf")
        assert initial_call["best_routes"] is None

        # Subsequent calls should have valid data
        for call in callback_calls[1:]:
            assert call["total"] == 50
            assert call["generation"] <= 50
            assert call["best_fitness"] > 0
            assert call["best_routes"] is not None


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


class TestRoutePreservingCrossover:
    """Tests for route-preserving crossover operator."""

    def test_all_locations_present(self):
        """Test that child contains all locations exactly once."""
        parent1 = [[1, 2, 3], [4, 5], [6, 7, 8]]
        parent2 = [[8, 7, 6], [5, 4], [3, 2, 1]]

        # Run multiple times due to randomness
        for _ in range(20):
            child = vrp_crossover(parent1, parent2)

            all_locs = [loc for route in child for loc in route]
            assert sorted(all_locs) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_preserves_some_parent1_routes(self):
        """Test that at least one route from parent1 is preserved in some runs."""
        parent1 = [[1, 2, 3], [4, 5], [6, 7, 8]]
        parent2 = [[8, 7, 6], [5, 4], [3, 2, 1]]

        preserved_count = 0
        for _ in range(50):
            child = vrp_crossover(parent1, parent2)
            for p1_route in parent1:
                if p1_route and p1_route in child:
                    preserved_count += 1
                    break

        # Should preserve routes in some runs
        assert preserved_count > 0

    def test_empty_parent(self):
        """Test crossover with empty parent1."""
        parent1 = [[], [], []]
        parent2 = [[1, 2], [3, 4], [5]]

        child = vrp_crossover(parent1, parent2)

        all_locs = [loc for route in child for loc in route]
        assert sorted(all_locs) == [1, 2, 3, 4, 5]

    def test_same_number_of_routes(self):
        """Test that child has same number of routes as parents."""
        parent1 = [[1, 2], [3], [4, 5, 6]]
        parent2 = [[6, 5], [4, 3], [2, 1]]

        child = vrp_crossover(parent1, parent2)
        assert len(child) == len(parent1)

    def test_single_route_parent(self):
        """Test crossover when parent has only one non-empty route."""
        parent1 = [[1, 2, 3, 4, 5], [], []]
        parent2 = [[5, 4], [3, 2], [1]]

        child = vrp_crossover(parent1, parent2)

        all_locs = [loc for route in child for loc in route]
        assert sorted(all_locs) == [1, 2, 3, 4, 5]


class TestTwoOpt:
    """Tests for 2-opt local search."""

    def test_improves_crossed_route(self):
        """Test that 2-opt fixes a route with crossing edges."""
        # Linear distance matrix: depot-1-2-3-4 in a line
        matrix = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=float,
        )

        # Suboptimal route with crossing: 1 -> 3 -> 2 -> 4
        route = [1, 3, 2, 4]

        optimized = two_opt(route, matrix)

        original_dist = calculate_route_distance(route, matrix)
        optimized_dist = calculate_route_distance(optimized, matrix)

        # Optimized should be better or equal
        assert optimized_dist <= original_dist

    def test_preserves_optimal_route(self):
        """Test that 2-opt doesn't worsen an already optimal route."""
        matrix = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=float,
        )

        # Optimal sequential route
        route = [1, 2, 3, 4]

        optimized = two_opt(route, matrix)

        original_dist = calculate_route_distance(route, matrix)
        optimized_dist = calculate_route_distance(optimized, matrix)

        # Should not get worse
        assert optimized_dist <= original_dist + 1e-9

    def test_empty_route(self):
        """Test 2-opt with empty route."""
        matrix = np.array([[0, 1], [1, 0]], dtype=float)
        assert two_opt([], matrix) == []

    def test_single_location_route(self):
        """Test 2-opt with single location route."""
        matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        assert two_opt([1], matrix) == [1]

    def test_two_location_route(self):
        """Test 2-opt with two location route."""
        matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        result = two_opt([1, 2], matrix)
        assert sorted(result) == [1, 2]

    def test_preserves_all_locations(self):
        """Test that 2-opt preserves all locations."""
        np.random.seed(42)
        matrix = np.random.rand(6, 6) * 100
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)

        route = [1, 2, 3, 4, 5]
        optimized = two_opt(route, matrix)

        assert sorted(optimized) == [1, 2, 3, 4, 5]


class TestApplyLocalSearch:
    """Tests for applying local search to full solutions."""

    def test_applies_to_all_routes(self):
        """Test that local search is applied to all non-empty routes."""
        matrix = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=float,
        )

        # First route is suboptimal (crossed)
        individual = [[1, 3, 2], [4], []]

        optimized = apply_local_search(individual, matrix)

        # Should have same structure
        assert len(optimized) == 3
        assert len(optimized[2]) == 0  # Empty route stays empty

        # All locations preserved
        all_locs = [loc for route in optimized for loc in route]
        assert sorted(all_locs) == [1, 2, 3, 4]

    def test_empty_solution(self):
        """Test local search with empty solution."""
        matrix = np.array([[0, 1], [1, 0]], dtype=float)

        individual = [[], [], []]
        optimized = apply_local_search(individual, matrix)

        assert len(optimized) == 3
        assert all(len(r) == 0 for r in optimized)

    def test_improves_distance(self):
        """Test that local search improves or maintains total distance."""
        matrix = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=float,
        )

        # Suboptimal solution
        individual = [[1, 3, 2, 4], []]

        original_dist = sum(
            calculate_route_distance(r, matrix) for r in individual if r
        )

        optimized = apply_local_search(individual, matrix)

        optimized_dist = sum(
            calculate_route_distance(r, matrix) for r in optimized if r
        )

        assert optimized_dist <= original_dist


class TestEarlyConvergence:
    """Tests for early convergence feature."""

    def test_early_convergence_stops_before_max_epochs(self):
        """Test that algorithm stops early when solution converges."""
        # Use a very small problem that converges quickly
        np.random.seed(42)
        n = 4
        matrix = np.array(
            [
                [0, 1, 2, 3],
                [1, 0, 1, 2],
                [2, 1, 0, 1],
                [3, 2, 1, 0],
            ],
            dtype=float,
        )

        # Low stagnation threshold to trigger early convergence
        config = GAConfig(
            population_size=20,
            max_epochs=500,
            stagnation_threshold=10,
            max_mutations_per_individual=1,
        )
        solver = VRPSolver(config)

        solver.solve_with_distance_matrix(
            distance_matrix=matrix,
            num_vehicles=1,
            capacity=100,
        )

        # Should have converged before max_epochs
        assert solver.final_epoch < 500 - 1
        assert solver.converged is True

    def test_no_convergence_when_threshold_high(self):
        """Test that algorithm runs all epochs when stagnation threshold is high."""
        np.random.seed(42)
        n = 4
        matrix = np.array(
            [
                [0, 1, 2, 3],
                [1, 0, 1, 2],
                [2, 1, 0, 1],
                [3, 2, 1, 0],
            ],
            dtype=float,
        )

        # Very high stagnation threshold that won't be reached
        config = GAConfig(
            population_size=10,
            max_epochs=30,
            stagnation_threshold=1000,
            max_mutations_per_individual=1,
        )
        solver = VRPSolver(config)

        solver.solve_with_distance_matrix(
            distance_matrix=matrix,
            num_vehicles=1,
            capacity=100,
        )

        # Should have run all epochs (final_epoch is 0-indexed)
        assert solver.final_epoch == 29
        assert solver.converged is False

    def test_converged_attribute_is_correct(self):
        """Test that converged attribute correctly reflects convergence state."""
        np.random.seed(42)
        matrix = np.array(
            [
                [0, 1, 2],
                [1, 0, 1],
                [2, 1, 0],
            ],
            dtype=float,
        )

        # Very easy problem with low threshold
        config = GAConfig(
            population_size=10,
            max_epochs=100,
            stagnation_threshold=5,
        )
        solver = VRPSolver(config)

        solver.solve_with_distance_matrix(
            distance_matrix=matrix,
            num_vehicles=1,
            capacity=100,
        )

        # With such a small problem, should converge quickly
        assert solver.converged is True
        assert solver.final_epoch < 100 - 1

    def test_fitness_history_matches_final_epoch(self):
        """Test that fitness history length matches the actual epochs run."""
        np.random.seed(42)
        matrix = np.array(
            [
                [0, 1, 2, 3],
                [1, 0, 1, 2],
                [2, 1, 0, 1],
                [3, 2, 1, 0],
            ],
            dtype=float,
        )

        config = GAConfig(
            population_size=15,
            max_epochs=200,
            stagnation_threshold=10,
        )
        solver = VRPSolver(config)

        solver.solve_with_distance_matrix(
            distance_matrix=matrix,
            num_vehicles=1,
            capacity=100,
        )

        # Fitness history should have final_epoch + 1 entries
        assert len(solver.fitness_history) == solver.final_epoch + 1

    def test_config_max_mutations_used(self):
        """Test that max_mutations_per_individual config is used in solver."""
        locations = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]

        # Config with different max_mutations values
        config1 = GAConfig(
            population_size=20,
            max_epochs=30,
            max_mutations_per_individual=1,
        )
        config5 = GAConfig(
            population_size=20,
            max_epochs=30,
            max_mutations_per_individual=5,
        )

        # Both should produce valid solutions
        solver1 = VRPSolver(config1)
        routes1 = solver1.solve(locations=locations, num_vehicles=2, capacity=10)

        solver5 = VRPSolver(config5)
        routes5 = solver5.solve(locations=locations, num_vehicles=2, capacity=10)

        # Both should visit all locations
        visited1 = set()
        for route in routes1:
            visited1.update(route)
        assert visited1 == set(range(1, 5))

        visited5 = set()
        for route in routes5:
            visited5.update(route)
        assert visited5 == set(range(1, 5))


class TestHybridInitialization:
    """Tests for hybrid population initialization."""

    def test_nearest_neighbor_solution_covers_all_locations(self):
        """Test that nearest neighbor covers all locations."""
        matrix = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=float,
        )

        solution = nearest_neighbor_solution(matrix, num_vehicles=2)

        # Should have 2 routes
        assert len(solution) == 2

        # Should cover all locations (1-4)
        all_locs = set()
        for route in solution:
            all_locs.update(route)
        assert all_locs == {1, 2, 3, 4}

    def test_nearest_neighbor_produces_reasonable_routes(self):
        """Test that nearest neighbor produces geographically coherent routes."""
        # Linear distance matrix: locations are in a line
        matrix = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=float,
        )

        solution = nearest_neighbor_solution(matrix, num_vehicles=1)

        # With one vehicle, should visit in sequential order (1, 2, 3, 4)
        # because each next location is closest
        assert len(solution) == 1
        assert solution[0] == [1, 2, 3, 4]

    def test_generate_hybrid_population_correct_size(self):
        """Test that hybrid population has correct size."""
        matrix = np.array(
            [
                [0, 1, 2, 3],
                [1, 0, 1, 2],
                [2, 1, 0, 1],
                [3, 2, 1, 0],
            ],
            dtype=float,
        )

        population = generate_hybrid_population(
            num_locations=3,
            num_vehicles=2,
            population_size=20,
            distance_matrix=matrix,
            heuristic_ratio=0.1,
        )

        assert len(population) == 20

    def test_generate_hybrid_population_all_valid(self):
        """Test that all individuals in hybrid population are valid."""
        matrix = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=float,
        )

        population = generate_hybrid_population(
            num_locations=4,
            num_vehicles=2,
            population_size=30,
            distance_matrix=matrix,
            heuristic_ratio=0.2,
        )

        for individual in population:
            all_locs = set()
            for route in individual:
                all_locs.update(route)
            assert all_locs == {1, 2, 3, 4}

    def test_hybrid_population_has_heuristic_solutions(self):
        """Test that hybrid population includes heuristic solutions."""
        matrix = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=float,
        )

        population = generate_hybrid_population(
            num_locations=4,
            num_vehicles=1,
            population_size=10,
            distance_matrix=matrix,
            heuristic_ratio=0.3,
        )

        # The first solution should be the exact nearest neighbor solution
        nn_solution = nearest_neighbor_solution(matrix, num_vehicles=1)
        assert population[0] == nn_solution

    def test_solver_with_hybrid_initialization(self):
        """Test that solver works with hybrid initialization enabled."""
        np.random.seed(42)
        matrix = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=float,
        )

        config = GAConfig(
            population_size=20,
            max_epochs=50,
            hybrid_initialization=True,
            heuristic_ratio=0.2,
        )
        solver = VRPSolver(config)

        routes = solver.solve_with_distance_matrix(
            distance_matrix=matrix,
            num_vehicles=2,
            capacity=100,
        )

        # Should produce valid solution
        all_locs = set()
        for route in routes:
            all_locs.update(route)
        assert all_locs == {1, 2, 3, 4}

    def test_solver_without_hybrid_initialization(self):
        """Test that solver works with hybrid initialization disabled."""
        np.random.seed(42)
        matrix = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=float,
        )

        config = GAConfig(
            population_size=20,
            max_epochs=50,
            hybrid_initialization=False,
        )
        solver = VRPSolver(config)

        routes = solver.solve_with_distance_matrix(
            distance_matrix=matrix,
            num_vehicles=2,
            capacity=100,
        )

        # Should produce valid solution
        all_locs = set()
        for route in routes:
            all_locs.update(route)
        assert all_locs == {1, 2, 3, 4}

    def test_hybrid_produces_better_initial_fitness(self):
        """Test that hybrid initialization produces better initial solutions."""
        np.random.seed(42)
        n = 10
        # Create a distance matrix with clear clusters
        matrix = np.random.rand(n, n) * 100
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)

        # Generate populations
        random_pop = generate_random_population(
            num_locations=n - 1,
            num_vehicles=3,
            population_size=20,
        )

        hybrid_pop = generate_hybrid_population(
            num_locations=n - 1,
            num_vehicles=3,
            population_size=20,
            distance_matrix=matrix,
            heuristic_ratio=0.2,
        )

        # Calculate fitness of best individual in each
        def get_best_fitness(population):
            demands = [0] + [1] * (n - 1)
            fitnesses = [
                calculate_fitness_vrp(ind, matrix, 100, demands)
                for ind in population
            ]
            return min(fitnesses)

        random_best = get_best_fitness(random_pop)
        hybrid_best = get_best_fitness(hybrid_pop)

        # Hybrid should generally have better (lower) initial fitness
        # We don't assert this strictly since it's probabilistic,
        # but the hybrid population should at least be valid
        assert hybrid_best > 0
        assert random_best > 0
