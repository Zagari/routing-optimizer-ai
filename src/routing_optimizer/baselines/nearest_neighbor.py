"""
Nearest Neighbor heuristic for VRP.

A greedy approach that always visits the closest unvisited location.
Simple but often produces reasonable solutions.
"""

from typing import List

import numpy as np


class NearestNeighborSolver:
    """Greedy heuristic: always go to the nearest neighbor.

    This solver builds routes by repeatedly selecting the closest
    unvisited location. It's a simple but effective heuristic that
    typically outperforms random allocation.
    """

    def solve(
        self,
        distance_matrix: np.ndarray,
        num_vehicles: int,
        capacity: int,
    ) -> List[List[int]]:
        """Solve VRP using nearest neighbor heuristic.

        Args:
            distance_matrix: NxN matrix of distances between locations.
                Index 0 is the depot.
            num_vehicles: Number of available vehicles.
            capacity: Maximum number of stops per vehicle.

        Returns:
            List of routes, where each route is a list of location indices.
        """
        n = len(distance_matrix)
        unvisited = set(range(1, n))  # Exclude depot
        routes: List[List[int]] = []

        for v in range(num_vehicles):
            if not unvisited:
                break

            route: List[int] = []
            current = 0  # Start at depot

            while unvisited and len(route) < capacity:
                # Find nearest unvisited location
                nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
                route.append(nearest)
                unvisited.remove(nearest)
                current = nearest

            routes.append(route)

        # If locations remain, distribute them to existing routes
        if unvisited:
            remaining = list(unvisited)
            for i, loc in enumerate(remaining):
                # Add to the route with fewest stops that has capacity
                for route in sorted(routes, key=len):
                    if len(route) < capacity:
                        route.append(loc)
                        break
                else:
                    # All routes at capacity, add to least loaded anyway
                    min_route = min(routes, key=len)
                    min_route.append(loc)

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
