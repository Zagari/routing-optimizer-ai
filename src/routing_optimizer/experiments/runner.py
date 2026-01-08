"""
Experiment runner for comparing VRP algorithms.

This module orchestrates running multiple algorithms on the same problem
instance and collecting comparable results.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from routing_optimizer.baselines.clarke_wright import ClarkeWrightSolver
from routing_optimizer.baselines.nearest_neighbor import NearestNeighborSolver
from routing_optimizer.baselines.random_solver import RandomSolver
from routing_optimizer.genetic_algorithm.config import GAConfig
from routing_optimizer.genetic_algorithm.vrp import VRPSolver


@dataclass
class ExperimentResult:
    """Results from running a single algorithm.

    Attributes:
        algorithm: Name of the algorithm.
        total_distance: Total distance of all routes.
        execution_time: Time taken to solve in seconds.
        num_routes: Number of routes generated.
        routes: The actual routes (list of location indices per vehicle).
        fitness_history: For GA, the fitness evolution over epochs.
    """

    algorithm: str
    total_distance: float
    execution_time: float
    num_routes: int
    routes: List[List[int]]
    fitness_history: Optional[List[float]] = field(default=None)

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "Algoritmo": self.algorithm,
            "Distância Total (m)": self.total_distance,
            "Distância Total (km)": self.total_distance / 1000,
            "Tempo (s)": self.execution_time,
            "Núm. Rotas": self.num_routes,
            "Total Paradas": sum(len(r) for r in self.routes),
        }


class ExperimentRunner:
    """Runs comparative experiments between VRP algorithms.

    This class provides methods to run all baseline algorithms plus
    the Genetic Algorithm with various configurations on the same
    problem instance.

    Example:
        >>> runner = ExperimentRunner(distance_matrix)
        >>> results = runner.run_all(num_vehicles=3, capacity=15)
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.total_distance:.0f}m")
    """

    def __init__(self, distance_matrix: np.ndarray):
        """Initialize the experiment runner.

        Args:
            distance_matrix: NxN matrix of distances between locations.
                Index 0 should be the depot.
        """
        self.distance_matrix = distance_matrix
        self.n_locations = len(distance_matrix)

    def run_random(
        self,
        num_vehicles: int,
        capacity: int,
        num_runs: int = 5,
    ) -> ExperimentResult:
        """Run Random baseline (best of multiple runs).

        Args:
            num_vehicles: Number of vehicles.
            capacity: Vehicle capacity.
            num_runs: Number of random runs to try.

        Returns:
            Best result from multiple runs.
        """
        solver = RandomSolver()
        best_result = None
        best_distance = float("inf")
        total_time = 0.0

        for _ in range(num_runs):
            start = time.time()
            routes = solver.solve(self.distance_matrix, num_vehicles, capacity)
            elapsed = time.time() - start
            total_time += elapsed

            dist = solver.calculate_total_distance(routes, self.distance_matrix)
            if dist < best_distance:
                best_distance = dist
                best_result = routes

        return ExperimentResult(
            algorithm="Random",
            total_distance=best_distance,
            execution_time=total_time,
            num_routes=len([r for r in best_result if r]),
            routes=best_result,
        )

    def run_nearest_neighbor(
        self,
        num_vehicles: int,
        capacity: int,
    ) -> ExperimentResult:
        """Run Nearest Neighbor heuristic.

        Args:
            num_vehicles: Number of vehicles.
            capacity: Vehicle capacity.

        Returns:
            Experiment result.
        """
        solver = NearestNeighborSolver()

        start = time.time()
        routes = solver.solve(self.distance_matrix, num_vehicles, capacity)
        elapsed = time.time() - start

        dist = solver.calculate_total_distance(routes, self.distance_matrix)

        return ExperimentResult(
            algorithm="Nearest Neighbor",
            total_distance=dist,
            execution_time=elapsed,
            num_routes=len([r for r in routes if r]),
            routes=routes,
        )

    def run_clarke_wright(
        self,
        num_vehicles: int,
        capacity: int,
    ) -> ExperimentResult:
        """Run Clarke-Wright Savings algorithm.

        Args:
            num_vehicles: Number of vehicles.
            capacity: Vehicle capacity.

        Returns:
            Experiment result.
        """
        solver = ClarkeWrightSolver()

        start = time.time()
        routes = solver.solve(self.distance_matrix, num_vehicles, capacity)
        elapsed = time.time() - start

        dist = solver.calculate_total_distance(routes, self.distance_matrix)

        return ExperimentResult(
            algorithm="Clarke-Wright",
            total_distance=dist,
            execution_time=elapsed,
            num_routes=len([r for r in routes if r]),
            routes=routes,
        )

    def run_genetic_algorithm(
        self,
        num_vehicles: int,
        capacity: int,
        config: Optional[GAConfig] = None,
        name_suffix: str = "",
    ) -> ExperimentResult:
        """Run Genetic Algorithm.

        Args:
            num_vehicles: Number of vehicles.
            capacity: Vehicle capacity.
            config: GA configuration. Uses defaults if None.
            name_suffix: Optional suffix for algorithm name.

        Returns:
            Experiment result with fitness history.
        """
        config = config or GAConfig()
        solver = VRPSolver(config)

        start = time.time()
        routes = solver.solve_with_distance_matrix(
            self.distance_matrix,
            num_vehicles=num_vehicles,
            capacity=capacity,
        )
        elapsed = time.time() - start

        dist = solver.get_total_distance(routes)
        history = solver.get_fitness_history()

        name = f"AG (pop={config.population_size}, gerações={config.max_epochs})"
        if name_suffix:
            name = f"AG {name_suffix}"

        return ExperimentResult(
            algorithm=name,
            total_distance=dist,
            execution_time=elapsed,
            num_routes=len([r for r in routes if r]),
            routes=routes,
            fitness_history=history,
        )

    def run_all(
        self,
        num_vehicles: int,
        capacity: int,
        ga_configs: Optional[List[GAConfig]] = None,
        include_random: bool = True,
        random_runs: int = 5,
    ) -> Dict[str, ExperimentResult]:
        """Run all algorithms and collect results.

        Args:
            num_vehicles: Number of vehicles.
            capacity: Vehicle capacity.
            ga_configs: List of GA configurations to test.
            include_random: Whether to include random baseline.
            random_runs: Number of random runs for averaging.

        Returns:
            Dictionary mapping algorithm name to result.
        """
        results: Dict[str, ExperimentResult] = {}

        # Random baseline
        if include_random:
            results["Random"] = self.run_random(num_vehicles, capacity, random_runs)

        # Nearest Neighbor
        results["Nearest Neighbor"] = self.run_nearest_neighbor(num_vehicles, capacity)

        # Clarke-Wright
        results["Clarke-Wright"] = self.run_clarke_wright(num_vehicles, capacity)

        # Genetic Algorithm(s)
        ga_configs = ga_configs or [GAConfig()]
        for i, config in enumerate(ga_configs):
            suffix = f"Config {i + 1}" if len(ga_configs) > 1 else ""
            result = self.run_genetic_algorithm(num_vehicles, capacity, config, name_suffix=suffix)
            results[result.algorithm] = result

        return results

    def get_comparison_summary(self, results: Dict[str, ExperimentResult]) -> Dict[str, Dict]:
        """Generate comparison summary.

        Args:
            results: Results from run_all().

        Returns:
            Summary with rankings and improvements.
        """
        # Sort by distance
        sorted_results = sorted(results.values(), key=lambda x: x.total_distance)

        best = sorted_results[0]
        worst = sorted_results[-1]

        summary = {
            "melhor_algoritmo": best.algorithm,
            "melhor_distancia_km": best.total_distance / 1000,
            "pior_algoritmo": worst.algorithm,
            "pior_distancia_km": worst.total_distance / 1000,
            "melhoria_percentual": (
                (worst.total_distance - best.total_distance) / worst.total_distance * 100
            ),
            "ranking": [
                {
                    "posicao": i + 1,
                    "algoritmo": r.algorithm,
                    "distancia_km": r.total_distance / 1000,
                    "tempo_s": r.execution_time,
                }
                for i, r in enumerate(sorted_results)
            ],
        }

        return summary
