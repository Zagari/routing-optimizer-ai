"""
Clarke-Wright Savings algorithm for VRP.

A classic constructive heuristic that merges routes based on savings.
Often produces high-quality solutions for VRP problems.
"""

from typing import Dict, List, Tuple

import numpy as np


class ClarkeWrightSolver:
    """Classic Clarke-Wright Savings algorithm for VRP.

    This algorithm starts with each location as its own route, then
    iteratively merges routes based on the "savings" achieved by
    combining them. It's a well-established heuristic that often
    produces good solutions.

    Reference:
        Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from
        a central depot to a number of delivery points. Operations Research.
    """

    def solve(
        self,
        distance_matrix: np.ndarray,
        num_vehicles: int,
        capacity: int,
    ) -> List[List[int]]:
        """Solve VRP using Clarke-Wright Savings algorithm.

        Args:
            distance_matrix: NxN matrix of distances between locations.
                Index 0 is the depot.
            num_vehicles: Number of available vehicles.
            capacity: Maximum number of stops per vehicle.

        Returns:
            List of routes, where each route is a list of location indices.
        """
        n = len(distance_matrix)
        if n <= 1:
            return []

        # Calculate savings for all pairs
        savings = self._calculate_savings(distance_matrix)

        # Initialize: each location is its own route
        routes: Dict[int, List[int]] = {i: [i] for i in range(1, n)}
        route_of: Dict[int, int] = {i: i for i in range(1, n)}

        # Track which locations are at route endpoints
        route_start: Dict[int, int] = {i: i for i in range(1, n)}
        route_end: Dict[int, int] = {i: i for i in range(1, n)}

        # Process savings in descending order
        for saving, i, j in savings:
            if saving <= 0:
                break

            ri, rj = route_of[i], route_of[j]

            # Skip if same route
            if ri == rj:
                continue

            # Skip if routes don't exist (already merged)
            if ri not in routes or rj not in routes:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            # Check capacity constraint
            if len(route_i) + len(route_j) > capacity:
                continue

            # Check if i and j are at route endpoints
            merged = self._try_merge(i, j, route_i, route_j, route_start, route_end, ri, rj)

            if merged is not None:
                # Update routes
                routes[ri] = merged
                del routes[rj]

                # Update route assignments
                for node in route_j:
                    route_of[node] = ri

                # Update endpoints
                route_start[ri] = merged[0]
                route_end[ri] = merged[-1]

        # Convert to list
        result = list(routes.values())

        # Merge smallest routes if we have too many
        while len(result) > num_vehicles:
            result.sort(key=len)
            if len(result[0]) + len(result[1]) <= capacity:
                merged = result[0] + result[1]
                result = [merged] + result[2:]
            else:
                # Can't merge within capacity, just combine anyway
                merged = result[0] + result[1]
                result = [merged] + result[2:]

        return result

    def _calculate_savings(self, distance_matrix: np.ndarray) -> List[Tuple[float, int, int]]:
        """Calculate savings for all location pairs.

        Saving(i,j) = distance(depot,i) + distance(depot,j) - distance(i,j)

        Args:
            distance_matrix: NxN distance matrix.

        Returns:
            List of (saving, i, j) tuples, sorted by saving descending.
        """
        n = len(distance_matrix)
        savings: List[Tuple[float, int, int]] = []

        for i in range(1, n):
            for j in range(i + 1, n):
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((s, i, j))

        # Sort by savings descending
        savings.sort(reverse=True, key=lambda x: x[0])
        return savings

    def _try_merge(
        self,
        i: int,
        j: int,
        route_i: List[int],
        route_j: List[int],
        route_start: Dict[int, int],
        route_end: Dict[int, int],
        ri: int,
        rj: int,
    ) -> List[int] | None:
        """Try to merge two routes at nodes i and j.

        Routes can only be merged if i and j are at the endpoints.

        Args:
            i, j: Nodes to merge at.
            route_i, route_j: The two routes.
            route_start, route_end: Endpoint tracking dicts.
            ri, rj: Route identifiers.

        Returns:
            Merged route if successful, None otherwise.
        """
        # Case 1: i is end of route_i, j is start of route_j
        if route_end[ri] == i and route_start[rj] == j:
            return route_i + route_j

        # Case 2: j is end of route_j, i is start of route_i
        if route_end[rj] == j and route_start[ri] == i:
            return route_j + route_i

        # Case 3: i is end of route_i, j is end of route_j
        if route_end[ri] == i and route_end[rj] == j:
            return route_i + route_j[::-1]

        # Case 4: i is start of route_i, j is start of route_j
        if route_start[ri] == i and route_start[rj] == j:
            return route_i[::-1] + route_j

        return None

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
