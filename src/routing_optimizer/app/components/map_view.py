"""
Map visualization components using Folium.
"""

from typing import List, Optional, Tuple

import folium

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


def create_route_map(
    coordinates: List[Tuple[float, float]],
    routes: List[List[int]],
    labels: Optional[List[str]] = None,
    depot_index: int = 0,
    zoom_start: int = 11,
) -> folium.Map:
    """
    Create a Folium map with routes visualized.

    Args:
        coordinates: List of (latitude, longitude) for each location
        routes: List of routes, where each route is a list of location indices
        labels: Optional list of labels for each location
        depot_index: Index of the depot (default 0)
        zoom_start: Initial zoom level

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

        # Draw route line
        if len(route_coords) > 1:
            folium.PolyLine(
                locations=route_coords,
                color=color,
                weight=3,
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
