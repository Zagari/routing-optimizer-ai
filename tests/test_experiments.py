"""
Tests for the experiments module.
"""

import numpy as np
import pytest

from routing_optimizer.experiments.runner import ExperimentResult, ExperimentRunner
from routing_optimizer.genetic_algorithm.config import GAConfig


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_creation(self):
        """Test creating an ExperimentResult."""
        result = ExperimentResult(
            algorithm="Test",
            total_distance=1000.0,
            execution_time=1.5,
            num_routes=3,
            routes=[[1, 2], [3, 4], [5]],
        )

        assert result.algorithm == "Test"
        assert result.total_distance == 1000.0
        assert result.execution_time == 1.5
        assert result.num_routes == 3
        assert len(result.routes) == 3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ExperimentResult(
            algorithm="Test",
            total_distance=1000.0,
            execution_time=1.5,
            num_routes=3,
            routes=[[1, 2], [3, 4], [5]],
        )

        d = result.to_dict()

        assert d["Algoritmo"] == "Test"
        assert d["Distância Total (m)"] == 1000.0
        assert d["Distância Total (km)"] == 1.0
        assert d["Tempo (s)"] == 1.5
        assert d["Núm. Rotas"] == 3
        assert d["Total Paradas"] == 5

    def test_fitness_history_optional(self):
        """Test that fitness_history is optional."""
        result = ExperimentResult(
            algorithm="Test",
            total_distance=1000.0,
            execution_time=1.5,
            num_routes=3,
            routes=[[1, 2]],
        )

        assert result.fitness_history is None

        result_with_history = ExperimentResult(
            algorithm="GA",
            total_distance=900.0,
            execution_time=5.0,
            num_routes=2,
            routes=[[1, 2], [3]],
            fitness_history=[1000, 950, 920, 900],
        )

        assert result_with_history.fitness_history == [1000, 950, 920, 900]


class TestExperimentRunner:
    """Tests for ExperimentRunner class."""

    @pytest.fixture
    def distance_matrix(self):
        """Create a test distance matrix."""
        np.random.seed(42)
        n = 10
        matrix = np.random.randint(1000, 10000, size=(n, n))
        np.fill_diagonal(matrix, 0)
        matrix = (matrix + matrix.T) // 2
        return matrix.astype(float)

    @pytest.fixture
    def runner(self, distance_matrix):
        """Create an ExperimentRunner instance."""
        return ExperimentRunner(distance_matrix)

    def test_init(self, runner, distance_matrix):
        """Test runner initialization."""
        assert runner.n_locations == len(distance_matrix)
        assert np.array_equal(runner.distance_matrix, distance_matrix)

    def test_run_random(self, runner):
        """Test running random baseline."""
        result = runner.run_random(num_vehicles=2, capacity=10, num_runs=3)

        assert result.algorithm == "Random"
        assert result.total_distance > 0
        assert result.execution_time >= 0
        assert result.num_routes <= 2

        # Verify all locations visited
        all_visited = []
        for route in result.routes:
            all_visited.extend(route)
        assert sorted(all_visited) == list(range(1, runner.n_locations))

    def test_run_nearest_neighbor(self, runner):
        """Test running nearest neighbor baseline."""
        result = runner.run_nearest_neighbor(num_vehicles=2, capacity=10)

        assert result.algorithm == "Nearest Neighbor"
        assert result.total_distance > 0
        assert result.execution_time >= 0

        # Verify all locations visited
        all_visited = []
        for route in result.routes:
            all_visited.extend(route)
        assert sorted(all_visited) == list(range(1, runner.n_locations))

    def test_run_clarke_wright(self, runner):
        """Test running Clarke-Wright baseline."""
        result = runner.run_clarke_wright(num_vehicles=2, capacity=10)

        assert result.algorithm == "Clarke-Wright"
        assert result.total_distance > 0
        assert result.execution_time >= 0

        # Verify all locations visited
        all_visited = []
        for route in result.routes:
            all_visited.extend(route)
        assert sorted(all_visited) == list(range(1, runner.n_locations))

    def test_run_genetic_algorithm(self, runner):
        """Test running genetic algorithm."""
        config = GAConfig(population_size=50, max_epochs=50)
        result = runner.run_genetic_algorithm(num_vehicles=2, capacity=10, config=config)

        assert "AG" in result.algorithm
        assert result.total_distance > 0
        assert result.execution_time >= 0
        assert result.fitness_history is not None
        assert len(result.fitness_history) > 0

        # Verify all locations visited
        all_visited = []
        for route in result.routes:
            all_visited.extend(route)
        assert sorted(all_visited) == list(range(1, runner.n_locations))

    def test_run_all(self, runner):
        """Test running all algorithms."""
        config = GAConfig(population_size=50, max_epochs=50)
        results = runner.run_all(
            num_vehicles=2,
            capacity=10,
            ga_configs=[config],
            random_runs=2,
        )

        # Should have 4 results: Random, NN, CW, GA
        assert len(results) == 4
        assert "Random" in results
        assert "Nearest Neighbor" in results
        assert "Clarke-Wright" in results

        # One GA result
        ga_results = [k for k in results.keys() if "AG" in k]
        assert len(ga_results) == 1

    def test_run_all_multiple_ga_configs(self, runner):
        """Test running with multiple GA configurations."""
        configs = [
            GAConfig(population_size=30, max_epochs=30),
            GAConfig(population_size=50, max_epochs=50),
        ]
        results = runner.run_all(
            num_vehicles=2,
            capacity=10,
            ga_configs=configs,
            random_runs=1,
        )

        # Should have 5 results: Random, NN, CW, 2x GA
        assert len(results) == 5

        ga_results = [k for k in results.keys() if "AG" in k]
        assert len(ga_results) == 2

    def test_run_all_without_random(self, runner):
        """Test running without random baseline."""
        config = GAConfig(population_size=30, max_epochs=30)
        results = runner.run_all(
            num_vehicles=2,
            capacity=10,
            ga_configs=[config],
            include_random=False,
        )

        assert "Random" not in results
        assert len(results) == 3  # NN, CW, GA

    def test_get_comparison_summary(self, runner):
        """Test comparison summary generation."""
        config = GAConfig(population_size=50, max_epochs=50)
        results = runner.run_all(
            num_vehicles=2,
            capacity=10,
            ga_configs=[config],
            random_runs=2,
        )

        summary = runner.get_comparison_summary(results)

        assert "melhor_algoritmo" in summary
        assert "pior_algoritmo" in summary
        assert "melhor_distancia_km" in summary
        assert "pior_distancia_km" in summary
        assert "melhoria_percentual" in summary
        assert "ranking" in summary

        # Melhoria should be positive
        assert summary["melhoria_percentual"] >= 0

        # Ranking should have all algorithms
        assert len(summary["ranking"]) == len(results)


class TestExperimentIntegration:
    """Integration tests for experiments."""

    @pytest.fixture
    def large_matrix(self):
        """Create a larger test distance matrix."""
        np.random.seed(123)
        n = 20
        matrix = np.random.randint(1000, 10000, size=(n, n))
        np.fill_diagonal(matrix, 0)
        matrix = (matrix + matrix.T) // 2
        return matrix.astype(float)

    def test_ga_beats_random_baseline(self, large_matrix):
        """Test that GA typically beats random baseline."""
        runner = ExperimentRunner(large_matrix)

        random_result = runner.run_random(num_vehicles=3, capacity=10, num_runs=5)

        config = GAConfig(population_size=100, max_epochs=200)
        ga_result = runner.run_genetic_algorithm(num_vehicles=3, capacity=10, config=config)

        # GA should produce shorter total distance than random
        assert ga_result.total_distance < random_result.total_distance

    def test_all_algorithms_produce_valid_solutions(self, large_matrix):
        """Test that all algorithms produce valid solutions."""
        runner = ExperimentRunner(large_matrix)
        n_locations = len(large_matrix)

        config = GAConfig(population_size=50, max_epochs=100)
        results = runner.run_all(
            num_vehicles=4,
            capacity=10,
            ga_configs=[config],
            random_runs=3,
        )

        for name, result in results.items():
            # Collect all visited locations
            all_visited = []
            for route in result.routes:
                all_visited.extend(route)

            # All locations except depot should be visited exactly once
            expected = list(range(1, n_locations))
            assert sorted(all_visited) == expected, f"Invalid solution for {name}"

            # No duplicates
            assert len(all_visited) == len(set(all_visited)), f"Duplicates in {name}"

    def test_execution_times_are_reasonable(self, large_matrix):
        """Test that execution times are within expected bounds."""
        runner = ExperimentRunner(large_matrix)

        config = GAConfig(population_size=50, max_epochs=100)
        results = runner.run_all(
            num_vehicles=3,
            capacity=10,
            ga_configs=[config],
            random_runs=3,
        )

        for name, result in results.items():
            # All algorithms should complete in under 30 seconds for this size
            assert result.execution_time < 30, f"{name} took too long"

            # Heuristics should be very fast
            if name in ["Nearest Neighbor", "Clarke-Wright"]:
                assert result.execution_time < 1, f"{name} should be fast"
