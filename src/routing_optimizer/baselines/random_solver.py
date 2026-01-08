"""
Random baseline solver for VRP.

This is the simplest baseline - randomly assigns locations to vehicles.
Used as a lower bound for comparison.
"""

import random
from typing import List

import numpy as np


class RandomSolver:
    """Baseline: random allocation of routes.

    This solver randomly shuffles all locations and distributes them
    evenly among the available vehicles. It serves as a worst-case
    baseline for comparison.
    """

    def solve(
        self,
        distance_matrix: np.ndarray,
        num_vehicles: int,
        capacity: int,
    ) -> List[List[int]]:
        """Solve VRP with random allocation.

        Args:
            distance_matrix: NxN matrix of distances between locations.
                Index 0 is the depot.
            num_vehicles: Number of available vehicles.
            capacity: Maximum number of stops per vehicle.

        Returns:
            List of routes, where each route is a list of location indices.
        """
        n = len(distance_matrix)
        locations = list(range(1, n))  # Exclude depot (0)
        random.shuffle(locations)

        # Distribute locations among vehicles respecting capacity
        routes: List[List[int]] = [[] for _ in range(num_vehicles)]
        vehicle_idx = 0

        for loc in locations:
            # Find a vehicle with capacity
            attempts = 0
            while len(routes[vehicle_idx]) >= capacity and attempts < num_vehicles:
                vehicle_idx = (vehicle_idx + 1) % num_vehicles
                attempts += 1

            routes[vehicle_idx].append(loc)
            vehicle_idx = (vehicle_idx + 1) % num_vehicles

        return routes

    def calculate_total_distance(
        self,
        routes: List[List[int]],
        distance_matrix: np.ndarray,
    ) -> float:
        """Calculate total distance of all routes.

        Args:
            routes: List of routes from solve().
            distance_matrix: NxN matrix of distances.

        Returns:
            Total distance in the same unit as the matrix.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            # Depot -> first stop
            total += distance_matrix[0, route[0]]
            # Between stops
            for i in range(len(route) - 1):
                total += distance_matrix[route[i], route[i + 1]]
            # Last stop -> depot
            total += distance_matrix[route[-1], 0]
        return total
