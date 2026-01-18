"""
Distance matrix calculation using OSRM (Open Source Routing Machine).

This module provides functionality to calculate real road distances
between geographic coordinates using the public OSRM API.
"""

import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests


class OSRMDistanceMatrix:
    """Calculate distance matrices using OSRM public API.

    OSRM provides real road distances (not straight-line) based on
    OpenStreetMap data.

    Note:
        The public OSRM server has usage limits. For production use,
        consider hosting your own OSRM instance.

    Example:
        >>> dm = OSRMDistanceMatrix()
        >>> coords = [(-23.5505, -46.6333), (-23.5614, -46.6558)]
        >>> matrix = dm.get_distance_matrix(coords)
        >>> print(f"Distance: {matrix[0, 1]} meters")
    """

    DEFAULT_BASE_URL = "http://router.project-osrm.org"

    # Maximum coordinates per batch to stay under OSRM limits
    # Public OSRM server has ~8KB URL limit and 10,000 durations limit
    DEFAULT_BATCH_SIZE = 100

    def __init__(
        self,
        base_url: Optional[str] = None,
        profile: str = "driving",
        timeout: int = 30,
        batch_size: Optional[int] = None,
    ):
        """Initialize the OSRM client.

        Args:
            base_url: OSRM server URL. Defaults to public server.
            profile: Routing profile (driving, walking, cycling).
            timeout: Request timeout in seconds.
            batch_size: Max coordinates per request. Defaults to 100.
                       Use larger values for self-hosted OSRM servers.
        """
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.profile = profile
        self.timeout = timeout
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE

    def get_distance_matrix(
        self,
        coordinates: List[Tuple[float, float]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """Calculate distance matrix between all coordinates.

        Uses batching to handle large coordinate sets that exceed
        OSRM server limits (~100 coords for public server).

        Args:
            coordinates: List of (latitude, longitude) tuples.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            NxN numpy array of distances in meters.

        Raises:
            ValueError: If OSRM returns an error.
            requests.RequestException: If network error occurs.
        """
        if len(coordinates) < 2:
            return np.zeros((len(coordinates), len(coordinates)))

        n = len(coordinates)

        # If within batch size, use simple single request
        if n <= self.batch_size:
            return self._fetch_table(coordinates, "distance")

        # Use batching for large coordinate sets
        return self._get_matrix_batched(coordinates, "distance", progress_callback)

    def _fetch_table(
        self,
        coordinates: List[Tuple[float, float]],
        annotation: str,
        sources: Optional[List[int]] = None,
        destinations: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Fetch a single table from OSRM.

        Args:
            coordinates: List of (latitude, longitude) tuples.
            annotation: "distance" or "duration".
            sources: Optional list of source indices.
            destinations: Optional list of destination indices.

        Returns:
            Matrix of distances or durations.
        """
        # OSRM expects longitude,latitude (reversed from typical lat,lon)
        coords_str = ";".join(f"{lon},{lat}" for lat, lon in coordinates)

        url = f"{self.base_url}/table/v1/{self.profile}/{coords_str}"
        params = {"annotations": annotation}

        if sources is not None:
            params["sources"] = ";".join(str(i) for i in sources)
        if destinations is not None:
            params["destinations"] = ";".join(str(i) for i in destinations)

        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()

        if data.get("code") != "Ok":
            error_msg = data.get("message", "Unknown OSRM error")
            raise ValueError(f"OSRM error: {error_msg}")

        result_key = "distances" if annotation == "distance" else "durations"
        result = data.get(result_key)
        if result is None:
            raise ValueError(f"OSRM response missing {result_key}")

        return np.array(result)

    def _get_matrix_batched(
        self,
        coordinates: List[Tuple[float, float]],
        annotation: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """Calculate matrix using batched requests.

        Divides coordinates into batches and assembles the full matrix
        from multiple OSRM requests.

        Args:
            coordinates: List of (latitude, longitude) tuples.
            annotation: "distance" or "duration".
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            NxN numpy array.
        """
        n = len(coordinates)
        full_matrix = np.zeros((n, n))

        # Calculate total batches for progress reporting
        num_row_batches = (n + self.batch_size - 1) // self.batch_size
        num_col_batches = (n + self.batch_size - 1) // self.batch_size
        total_batches = num_row_batches * num_col_batches
        current_batch = 0

        # Process in blocks
        for i_start in range(0, n, self.batch_size):
            i_end = min(i_start + self.batch_size, n)

            for j_start in range(0, n, self.batch_size):
                j_end = min(j_start + self.batch_size, n)

                # Get coordinates for this block
                source_coords = coordinates[i_start:i_end]
                dest_coords = coordinates[j_start:j_end]

                # Combine and fetch with sources/destinations indices
                combined_coords = source_coords + dest_coords
                num_sources = len(source_coords)
                num_dests = len(dest_coords)

                sources = list(range(num_sources))
                destinations = list(range(num_sources, num_sources + num_dests))

                # Fetch this block
                block = self._fetch_table(
                    combined_coords, annotation, sources, destinations
                )

                # Insert into full matrix
                full_matrix[i_start:i_end, j_start:j_end] = block

                current_batch += 1
                if progress_callback:
                    progress_callback(current_batch, total_batches)

                # Small delay to be nice to public server
                if self.base_url == self.DEFAULT_BASE_URL:
                    time.sleep(0.1)

        return full_matrix

    def get_duration_matrix(
        self,
        coordinates: List[Tuple[float, float]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """Calculate duration matrix between all coordinates.

        Uses batching to handle large coordinate sets that exceed
        OSRM server limits (~100 coords for public server).

        Args:
            coordinates: List of (latitude, longitude) tuples.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            NxN numpy array of durations in seconds.

        Raises:
            ValueError: If OSRM returns an error.
        """
        if len(coordinates) < 2:
            return np.zeros((len(coordinates), len(coordinates)))

        n = len(coordinates)

        # If within batch size, use simple single request
        if n <= self.batch_size:
            return self._fetch_table(coordinates, "duration")

        # Use batching for large coordinate sets
        return self._get_matrix_batched(coordinates, "duration", progress_callback)

    def get_route(
        self,
        coordinates: List[Tuple[float, float]],
        overview: str = "full",
    ) -> dict:
        """Get detailed route between coordinates.

        Args:
            coordinates: List of (latitude, longitude) tuples.
            overview: Level of detail for route geometry
                     ("full", "simplified", "false").

        Returns:
            Dict with route information including geometry and steps.

        Raises:
            ValueError: If OSRM returns an error.
        """
        coords_str = ";".join(f"{lon},{lat}" for lat, lon in coordinates)

        url = f"{self.base_url}/route/v1/{self.profile}/{coords_str}"
        params = {
            "overview": overview,
            "geometries": "geojson",
            "steps": "true",
        }

        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()

        if data.get("code") != "Ok":
            error_msg = data.get("message", "Unknown OSRM error")
            raise ValueError(f"OSRM error: {error_msg}")

        return data

    def get_route_distance(
        self,
        coordinates: List[Tuple[float, float]],
    ) -> float:
        """Get total distance for a route through coordinates.

        Args:
            coordinates: List of (latitude, longitude) tuples.

        Returns:
            Total distance in meters.
        """
        if len(coordinates) < 2:
            return 0.0

        route_data = self.get_route(coordinates, overview="false")

        if route_data.get("routes"):
            return route_data["routes"][0].get("distance", 0.0)

        return 0.0

    def get_route_duration(
        self,
        coordinates: List[Tuple[float, float]],
    ) -> float:
        """Get total duration for a route through coordinates.

        Args:
            coordinates: List of (latitude, longitude) tuples.

        Returns:
            Total duration in seconds.
        """
        if len(coordinates) < 2:
            return 0.0

        route_data = self.get_route(coordinates, overview="false")

        if route_data.get("routes"):
            return route_data["routes"][0].get("duration", 0.0)

        return 0.0


def meters_to_km(meters: float) -> float:
    """Convert meters to kilometers.

    Args:
        meters: Distance in meters.

    Returns:
        Distance in kilometers.
    """
    return meters / 1000.0


def seconds_to_minutes(seconds: float) -> float:
    """Convert seconds to minutes.

    Args:
        seconds: Duration in seconds.

    Returns:
        Duration in minutes.
    """
    return seconds / 60.0
