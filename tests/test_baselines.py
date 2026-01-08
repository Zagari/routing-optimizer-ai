"""
Tests for baseline VRP algorithms.
"""

import numpy as np
import pytest

from routing_optimizer.baselines.clarke_wright import ClarkeWrightSolver
from routing_optimizer.baselines.nearest_neighbor import NearestNeighborSolver
from routing_optimizer.baselines.random_solver import RandomSolver


class TestRandomSolver:
    """Tests for RandomSolver class."""

    @pytest.fixture
    def solver(self):
        """Create a RandomSolver instance."""
        return RandomSolver()

    @pytest.fixture
    def small_matrix(self):
        """Create a small test distance matrix."""
        return np.array(
            [
                [0, 10, 20, 30, 40],
                [10, 0, 15, 25, 35],
                [20, 15, 0, 10, 20],
                [30, 25, 10, 0, 15],
                [40, 35, 20, 15, 0],
            ]
        )

    def test_solve_returns_routes(self, solver, small_matrix):
        """Test that solve returns a list of routes."""
        routes = solver.solve(small_matrix, num_vehicles=2, capacity=10)

        assert isinstance(routes, list)
        assert len(routes) == 2

    def test_all_locations_visited(self, solver, small_matrix):
        """Test that all locations are visited exactly once."""
        routes = solver.solve(small_matrix, num_vehicles=2, capacity=10)

        all_visited = []
        for route in routes:
            all_visited.extend(route)

        # Should have locations 1-4 (excluding depot 0)
        assert sorted(all_visited) == [1, 2, 3, 4]

    def test_no_duplicates(self, solver, small_matrix):
        """Test that no location is visited twice."""
        routes = solver.solve(small_matrix, num_vehicles=2, capacity=10)

        all_visited = []
        for route in routes:
            all_visited.extend(route)

        assert len(all_visited) == len(set(all_visited))

    def test_respects_capacity(self, solver, small_matrix):
        """Test that routes respect vehicle capacity."""
        routes = solver.solve(small_matrix, num_vehicles=4, capacity=2)

        for route in routes:
            assert len(route) <= 2

    def test_calculate_total_distance(self, solver, small_matrix):
        """Test total distance calculation."""
        routes = [[1, 2], [3, 4]]
        distance = solver.calculate_total_distance(routes, small_matrix)

        # Route 1: 0->1 (10) + 1->2 (15) + 2->0 (20) = 45
        # Route 2: 0->3 (30) + 3->4 (15) + 4->0 (40) = 85
        # Total: 130
        assert distance == 130

    def test_empty_routes(self, solver, small_matrix):
        """Test with more vehicles than locations."""
        routes = solver.solve(small_matrix, num_vehicles=10, capacity=10)

        all_visited = []
        for route in routes:
            all_visited.extend(route)

        # Should still visit all locations
        assert sorted(all_visited) == [1, 2, 3, 4]


class TestNearestNeighborSolver:
    """Tests for NearestNeighborSolver class."""

    @pytest.fixture
    def solver(self):
        """Create a NearestNeighborSolver instance."""
        return NearestNeighborSolver()

    @pytest.fixture
    def small_matrix(self):
        """Create a small test distance matrix."""
        return np.array(
            [
                [0, 10, 20, 30, 40],
                [10, 0, 15, 25, 35],
                [20, 15, 0, 10, 20],
                [30, 25, 10, 0, 15],
                [40, 35, 20, 15, 0],
            ]
        )

    def test_solve_returns_routes(self, solver, small_matrix):
        """Test that solve returns a list of routes."""
        routes = solver.solve(small_matrix, num_vehicles=2, capacity=10)

        assert isinstance(routes, list)
        assert len(routes) <= 2

    def test_all_locations_visited(self, solver, small_matrix):
        """Test that all locations are visited exactly once."""
        routes = solver.solve(small_matrix, num_vehicles=2, capacity=10)

        all_visited = []
        for route in routes:
            all_visited.extend(route)

        assert sorted(all_visited) == [1, 2, 3, 4]

    def test_greedy_behavior(self, solver, small_matrix):
        """Test that the first route starts with nearest to depot."""
        routes = solver.solve(small_matrix, num_vehicles=2, capacity=10)

        # Nearest to depot (0) is location 1 (distance 10)
        assert routes[0][0] == 1

    def test_respects_capacity(self, solver, small_matrix):
        """Test that routes respect vehicle capacity."""
        routes = solver.solve(small_matrix, num_vehicles=4, capacity=2)

        for route in routes:
            assert len(route) <= 2

    def test_calculate_total_distance(self, solver, small_matrix):
        """Test total distance calculation."""
        routes = [[1, 2], [3, 4]]
        distance = solver.calculate_total_distance(routes, small_matrix)

        assert distance == 130  # Same as RandomSolver test


class TestClarkeWrightSolver:
    """Tests for ClarkeWrightSolver class."""

    @pytest.fixture
    def solver(self):
        """Create a ClarkeWrightSolver instance."""
        return ClarkeWrightSolver()

    @pytest.fixture
    def small_matrix(self):
        """Create a small test distance matrix."""
        return np.array(
            [
                [0, 10, 20, 30, 40],
                [10, 0, 15, 25, 35],
                [20, 15, 0, 10, 20],
                [30, 25, 10, 0, 15],
                [40, 35, 20, 15, 0],
            ]
        )

    def test_solve_returns_routes(self, solver, small_matrix):
        """Test that solve returns a list of routes."""
        routes = solver.solve(small_matrix, num_vehicles=2, capacity=10)

        assert isinstance(routes, list)
        assert len(routes) <= 2

    def test_all_locations_visited(self, solver, small_matrix):
        """Test that all locations are visited exactly once."""
        routes = solver.solve(small_matrix, num_vehicles=2, capacity=10)

        all_visited = []
        for route in routes:
            all_visited.extend(route)

        assert sorted(all_visited) == [1, 2, 3, 4]

    def test_no_duplicates(self, solver, small_matrix):
        """Test that no location is visited twice."""
        routes = solver.solve(small_matrix, num_vehicles=2, capacity=10)

        all_visited = []
        for route in routes:
            all_visited.extend(route)

        assert len(all_visited) == len(set(all_visited))

    def test_savings_calculation(self, solver, small_matrix):
        """Test that savings are calculated correctly."""
        savings = solver._calculate_savings(small_matrix)

        # Should have pairs (4 choose 2) = 6 pairs
        assert len(savings) == 6

        # Savings should be sorted descending
        for i in range(len(savings) - 1):
            assert savings[i][0] >= savings[i + 1][0]

    def test_respects_num_vehicles(self, solver, small_matrix):
        """Test that result respects number of vehicles."""
        routes = solver.solve(small_matrix, num_vehicles=2, capacity=10)
        assert len(routes) <= 2

        routes = solver.solve(small_matrix, num_vehicles=4, capacity=10)
        assert len(routes) <= 4

    def test_calculate_total_distance(self, solver, small_matrix):
        """Test total distance calculation."""
        routes = [[1, 2], [3, 4]]
        distance = solver.calculate_total_distance(routes, small_matrix)

        assert distance == 130


class TestBaselineComparison:
    """Integration tests comparing all baselines."""

    @pytest.fixture
    def distance_matrix(self):
        """Create a larger test distance matrix."""
        np.random.seed(42)
        n = 15
        matrix = np.random.randint(1000, 10000, size=(n, n))
        np.fill_diagonal(matrix, 0)
        # Make symmetric
        matrix = (matrix + matrix.T) // 2
        return matrix

    def test_all_solvers_produce_valid_routes(self, distance_matrix):
        """Test that all solvers produce valid routes."""
        solvers = [
            RandomSolver(),
            NearestNeighborSolver(),
            ClarkeWrightSolver(),
        ]

        for solver in solvers:
            routes = solver.solve(distance_matrix, num_vehicles=3, capacity=10)

            # Collect all visited locations
            all_visited = []
            for route in routes:
                all_visited.extend(route)

            # All locations (except depot) should be visited
            expected = list(range(1, len(distance_matrix)))
            assert sorted(all_visited) == expected, f"Failed for {type(solver).__name__}"

    def test_nearest_neighbor_beats_random_on_average(self, distance_matrix):
        """Test that NN generally produces better results than random."""
        random_solver = RandomSolver()
        nn_solver = NearestNeighborSolver()

        # Run random multiple times and average
        random_distances = []
        for _ in range(10):
            routes = random_solver.solve(distance_matrix, num_vehicles=3, capacity=10)
            dist = random_solver.calculate_total_distance(routes, distance_matrix)
            random_distances.append(dist)

        avg_random = sum(random_distances) / len(random_distances)

        # Run NN once (deterministic)
        routes = nn_solver.solve(distance_matrix, num_vehicles=3, capacity=10)
        nn_dist = nn_solver.calculate_total_distance(routes, distance_matrix)

        # NN should be better than average random
        assert nn_dist < avg_random

    def test_clarke_wright_produces_reasonable_results(self, distance_matrix):
        """Test that Clarke-Wright produces reasonable results."""
        cw_solver = ClarkeWrightSolver()
        random_solver = RandomSolver()

        cw_routes = cw_solver.solve(distance_matrix, num_vehicles=3, capacity=10)
        cw_dist = cw_solver.calculate_total_distance(cw_routes, distance_matrix)

        # Compare with worst random
        worst_random = 0
        for _ in range(10):
            routes = random_solver.solve(distance_matrix, num_vehicles=3, capacity=10)
            dist = random_solver.calculate_total_distance(routes, distance_matrix)
            worst_random = max(worst_random, dist)

        # CW should be better than worst random
        assert cw_dist < worst_random
