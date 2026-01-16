"""
Map visualization components using Folium.
"""

import time
from typing import List, Optional, Tuple

import folium

from routing_optimizer.routing.distance import OSRMDistanceMatrix

# Cores para cada veículo/rota
ROUTE_COLORS = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "darkred",
    "darkblue",
    "darkgreen",
    "cadetblue",
    "pink",
]


def _get_route_geometry(
    osrm: OSRMDistanceMatrix,
    start_coord: Tuple[float, float],
    end_coord: Tuple[float, float],
) -> Optional[List[Tuple[float, float]]]:
    """
    Get real road geometry between two coordinates using OSRM.

    Args:
        osrm: OSRMDistanceMatrix instance
        start_coord: Starting coordinate (lat, lon)
        end_coord: Ending coordinate (lat, lon)

    Returns:
        List of coordinates forming the road path, or None if failed
    """
    try:
        route_data = osrm.get_route([start_coord, end_coord], overview="full")

        if route_data.get("routes") and len(route_data["routes"]) > 0:
            geometry = route_data["routes"][0].get("geometry")
            if geometry and geometry.get("coordinates"):
                # OSRM returns [lon, lat], we need [lat, lon] for Folium
                coords = geometry["coordinates"]
                return [(lat, lon) for lon, lat in coords]
    except Exception:
        pass

    return None


def _get_full_route_geometry(
    osrm: OSRMDistanceMatrix,
    route_coords: List[Tuple[float, float]],
    delay: float = 0.1,
) -> List[Tuple[float, float]]:
    """
    Get real road geometry for a full route (multiple waypoints).

    Args:
        osrm: OSRMDistanceMatrix instance
        route_coords: List of coordinates in order (depot -> stops -> depot)
        delay: Delay between API calls to respect rate limiting

    Returns:
        List of coordinates forming the complete road path
    """
    if len(route_coords) < 2:
        return route_coords

    full_geometry = []

    for i in range(len(route_coords) - 1):
        start = route_coords[i]
        end = route_coords[i + 1]

        segment_geometry = _get_route_geometry(osrm, start, end)

        if segment_geometry:
            # Add segment geometry (skip first point if not first segment to avoid duplicates)
            if i == 0:
                full_geometry.extend(segment_geometry)
            else:
                full_geometry.extend(segment_geometry[1:])
        else:
            # Fallback to straight line if OSRM fails
            if i == 0:
                full_geometry.append(start)
            full_geometry.append(end)

        # Small delay to respect rate limiting on public OSRM server
        if delay > 0 and i < len(route_coords) - 2:
            time.sleep(delay)

    return full_geometry


def create_route_map(
    coordinates: List[Tuple[float, float]],
    routes: List[List[int]],
    labels: Optional[List[str]] = None,
    depot_index: int = 0,
    zoom_start: int = 11,
    use_real_roads: bool = False,
) -> folium.Map:
    """
    Create a Folium map with routes visualized.

    Args:
        coordinates: List of (latitude, longitude) for each location
        routes: List of routes, where each route is a list of location indices
        labels: Optional list of labels for each location
        depot_index: Index of the depot (default 0)
        zoom_start: Initial zoom level
        use_real_roads: If True, fetch real road geometry from OSRM (slower but more accurate)

    Returns:
        Folium Map object
    """
    if not coordinates:
        # Return empty map centered on São Paulo
        return folium.Map(location=[-23.5505, -46.6333], zoom_start=zoom_start)

    # Calculate center of map
    center_lat = sum(c[0] for c in coordinates) / len(coordinates)
    center_lon = sum(c[1] for c in coordinates) / len(coordinates)

    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # Initialize OSRM client if using real roads
    osrm = OSRMDistanceMatrix() if use_real_roads else None

    # Add depot marker
    if depot_index < len(coordinates):
        depot_coord = coordinates[depot_index]
        depot_label = labels[depot_index] if labels else "Depósito"
        folium.Marker(
            location=depot_coord,
            popup=f"<b>DEPÓSITO</b><br>{depot_label}",
            icon=folium.Icon(color="black", icon="home"),
        ).add_to(m)

    # Add routes
    for route_idx, route in enumerate(routes):
        color = ROUTE_COLORS[route_idx % len(ROUTE_COLORS)]

        # Build full route (depot -> stops -> depot)
        route_coords = []

        # Start from depot
        route_coords.append(coordinates[depot_index])

        # Add each stop
        for stop_idx in route:
            if stop_idx < len(coordinates):
                stop_coord = coordinates[stop_idx]
                route_coords.append(stop_coord)

                # Add marker for this stop
                label = labels[stop_idx] if labels else f"Parada {stop_idx}"
                folium.Marker(
                    location=stop_coord,
                    popup=f"<b>Veículo {route_idx + 1}</b><br>Parada: {label}",
                    icon=folium.Icon(color=color, icon="info-sign"),
                ).add_to(m)

        # Return to depot
        route_coords.append(coordinates[depot_index])

        # Get geometry for the route line
        if use_real_roads and osrm and len(route_coords) > 1:
            # Fetch real road geometry
            line_coords = _get_full_route_geometry(osrm, route_coords)
        else:
            # Use straight lines
            line_coords = route_coords

        # Draw route line
        if len(line_coords) > 1:
            folium.PolyLine(
                locations=line_coords,
                color=color,
                weight=4 if use_real_roads else 3,
                opacity=0.8,
                popup=f"Veículo {route_idx + 1}: {len(route)} paradas",
            ).add_to(m)

    return m


def create_locations_map(
    coordinates: List[Tuple[float, float]],
    labels: Optional[List[str]] = None,
    zoom_start: int = 11,
) -> folium.Map:
    """
    Create a simple map showing all locations.

    Args:
        coordinates: List of (latitude, longitude) for each location
        labels: Optional list of labels for each location
        zoom_start: Initial zoom level

    Returns:
        Folium Map object
    """
    if not coordinates:
        return folium.Map(location=[-23.5505, -46.6333], zoom_start=zoom_start)

    # Calculate center
    center_lat = sum(c[0] for c in coordinates) / len(coordinates)
    center_lon = sum(c[1] for c in coordinates) / len(coordinates)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    for idx, coord in enumerate(coordinates):
        label = labels[idx] if labels else f"Local {idx}"
        folium.Marker(
            location=coord,
            popup=label,
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

    return m
