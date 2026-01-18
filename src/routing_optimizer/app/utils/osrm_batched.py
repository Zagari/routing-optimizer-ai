"""
OSRM distance matrix calculation with batching support.

This module handles large coordinate sets by splitting requests into batches
to avoid OSRM server limits (~100 coordinates, 8KB URL limit).
"""

import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests


def get_distance_matrix_batched(
    coords: List[Tuple[float, float]],
    batch_size: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    base_url: str = "http://router.project-osrm.org",
    timeout: int = 30,
) -> np.ndarray:
    """Calculate distance matrix using OSRM with batching for large datasets.

    Args:
        coords: List of (latitude, longitude) tuples.
        batch_size: Max coordinates per request. Default 100 for public OSRM.
        progress_callback: Optional callback(current_batch, total_batches).
        base_url: OSRM server URL.
        timeout: Request timeout in seconds.

    Returns:
        NxN numpy array of distances in meters.

    Raises:
        ValueError: If OSRM returns an error.
        requests.RequestException: If network error occurs.
    """
    n = len(coords)

    if n < 2:
        return np.zeros((n, n))

    # Small dataset - single request
    if n <= batch_size:
        return _fetch_osrm_table(coords, base_url, timeout)

    # Large dataset - use batching
    full_matrix = np.zeros((n, n))
    num_batches = ((n + batch_size - 1) // batch_size) ** 2
    current_batch = 0

    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)

        for j_start in range(0, n, batch_size):
            j_end = min(j_start + batch_size, n)

            # Get coordinates for this block
            source_coords = coords[i_start:i_end]
            dest_coords = coords[j_start:j_end]

            # Fetch this block
            block = _fetch_osrm_table_block(
                source_coords, dest_coords, base_url, timeout
            )

            # Insert into full matrix
            full_matrix[i_start:i_end, j_start:j_end] = block

            current_batch += 1
            if progress_callback:
                progress_callback(current_batch, num_batches)

            # Small delay to be nice to public server
            if "router.project-osrm.org" in base_url:
                time.sleep(0.1)

    return full_matrix


def _fetch_osrm_table(
    coords: List[Tuple[float, float]],
    base_url: str,
    timeout: int,
) -> np.ndarray:
    """Fetch full distance table from OSRM (for small datasets)."""
    coords_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = f"{base_url}/table/v1/driving/{coords_str}"
    params = {"annotations": "distance"}

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    if data.get("code") != "Ok":
        raise ValueError(f"OSRM error: {data.get('message', 'Unknown')}")

    return np.array(data["distances"])


def _fetch_osrm_table_block(
    source_coords: List[Tuple[float, float]],
    dest_coords: List[Tuple[float, float]],
    base_url: str,
    timeout: int,
) -> np.ndarray:
    """Fetch a block of the distance matrix using sources/destinations."""
    # Combine coordinates
    combined = source_coords + dest_coords
    num_sources = len(source_coords)
    num_dests = len(dest_coords)

    coords_str = ";".join(f"{lon},{lat}" for lat, lon in combined)
    url = f"{base_url}/table/v1/driving/{coords_str}"
    params = {
        "annotations": "distance",
        "sources": ";".join(str(i) for i in range(num_sources)),
        "destinations": ";".join(str(i) for i in range(num_sources, num_sources + num_dests)),
    }

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    if data.get("code") != "Ok":
        raise ValueError(f"OSRM error: {data.get('message', 'Unknown')}")

    return np.array(data["distances"])
