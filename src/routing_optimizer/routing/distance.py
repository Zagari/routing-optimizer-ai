"""
Distance matrix calculation using OSRM (Open Source Routing Machine).

This module provides functionality to calculate real road distances
between geographic coordinates using the public OSRM API.
"""

from typing import List, Optional, Tuple

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

    def __init__(
        self,
        base_url: Optional[str] = None,
        profile: str = "driving",
        timeout: int = 30,
    ):
        """Initialize the OSRM client.

        Args:
            base_url: OSRM server URL. Defaults to public server.
            profile: Routing profile (driving, walking, cycling).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.profile = profile
        self.timeout = timeout

    def get_distance_matrix(
        self,
        coordinates: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Calculate distance matrix between all coordinates.

        Args:
            coordinates: List of (latitude, longitude) tuples.

        Returns:
            NxN numpy array of distances in meters.

        Raises:
            ValueError: If OSRM returns an error.
            requests.RequestException: If network error occurs.
        """
        if len(coordinates) < 2:
            return np.zeros((len(coordinates), len(coordinates)))

        # OSRM expects longitude,latitude (reversed from typical lat,lon)
        coords_str = ";".join(f"{lon},{lat}" for lat, lon in coordinates)

        url = f"{self.base_url}/table/v1/{self.profile}/{coords_str}"
        params = {"annotations": "distance"}

        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()

        if data.get("code") != "Ok":
            error_msg = data.get("message", "Unknown OSRM error")
            raise ValueError(f"OSRM error: {error_msg}")

        # OSRM returns distances in meters
        distances = data.get("distances")
        if distances is None:
            raise ValueError("OSRM response missing distances")

        return np.array(distances)

    def get_duration_matrix(
        self,
        coordinates: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Calculate duration matrix between all coordinates.

        Args:
            coordinates: List of (latitude, longitude) tuples.

        Returns:
            NxN numpy array of durations in seconds.

        Raises:
            ValueError: If OSRM returns an error.
        """
        if len(coordinates) < 2:
            return np.zeros((len(coordinates), len(coordinates)))

        coords_str = ";".join(f"{lon},{lat}" for lat, lon in coordinates)

        url = f"{self.base_url}/table/v1/{self.profile}/{coords_str}"
        params = {"annotations": "duration"}

        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()

        if data.get("code") != "Ok":
            error_msg = data.get("message", "Unknown OSRM error")
            raise ValueError(f"OSRM error: {error_msg}")

        durations = data.get("durations")
        if durations is None:
            raise ValueError("OSRM response missing durations")

        return np.array(durations)

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
