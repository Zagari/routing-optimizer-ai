"""
Geocoding module using Nominatim (OpenStreetMap).

This module provides functionality to convert addresses to geographic coordinates.
"""

from dataclasses import dataclass
from typing import List, Optional

from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim


@dataclass
class GeocodedLocation:
    """Result of geocoding an address.

    Attributes:
        original_address: The original address that was geocoded.
        latitude: Latitude coordinate (or 0.0 if failed).
        longitude: Longitude coordinate (or 0.0 if failed).
        formatted_address: The formatted address returned by the geocoder.
        success: Whether geocoding was successful.
        error: Error message if geocoding failed.
    """

    original_address: str
    latitude: float
    longitude: float
    formatted_address: str
    success: bool = True
    error: Optional[str] = None


class Geocoder:
    """Geocoder using Nominatim (OpenStreetMap).

    Uses rate limiting to respect Nominatim's usage policy (1 request/second).

    Example:
        >>> geocoder = Geocoder()
        >>> result = geocoder.geocode_address("Av. Paulista, 1000, Sao Paulo")
        >>> print(f"Lat: {result.latitude}, Lon: {result.longitude}")
    """

    def __init__(
        self,
        user_agent: str = "routing-optimizer-fiap",
        min_delay_seconds: float = 1.0,
        max_retries: int = 3,
        timeout: int = 10,
    ):
        """Initialize the geocoder.

        Args:
            user_agent: User agent string for Nominatim API.
            min_delay_seconds: Minimum delay between requests (rate limiting).
            max_retries: Maximum number of retry attempts.
            timeout: Timeout in seconds for each request.
        """
        self.geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
        self.geocode = RateLimiter(
            self.geolocator.geocode,
            min_delay_seconds=min_delay_seconds,
            max_retries=max_retries,
            error_wait_seconds=2.0,
        )

    def geocode_address(
        self,
        address: str,
        state: str = "SP",
        country: str = "Brasil",
    ) -> GeocodedLocation:
        """Geocode a single address.

        Args:
            address: The address to geocode.
            state: State to append to address for better results.
            country: Country to append to address.

        Returns:
            GeocodedLocation with coordinates or error information.
        """
        full_address = f"{address}, {state}, {country}"

        try:
            location = self.geocode(full_address)

            if location:
                return GeocodedLocation(
                    original_address=address,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    formatted_address=location.address,
                    success=True,
                )
            else:
                return GeocodedLocation(
                    original_address=address,
                    latitude=0.0,
                    longitude=0.0,
                    formatted_address="",
                    success=False,
                    error="Endereco nao encontrado",
                )

        except GeocoderTimedOut:
            return GeocodedLocation(
                original_address=address,
                latitude=0.0,
                longitude=0.0,
                formatted_address="",
                success=False,
                error="Timeout ao geocodificar",
            )
        except GeocoderUnavailable as e:
            return GeocodedLocation(
                original_address=address,
                latitude=0.0,
                longitude=0.0,
                formatted_address="",
                success=False,
                error=f"Servico indisponivel: {str(e)}",
            )
        except Exception as e:
            return GeocodedLocation(
                original_address=address,
                latitude=0.0,
                longitude=0.0,
                formatted_address="",
                success=False,
                error=str(e),
            )

    def geocode_batch(
        self,
        addresses: List[str],
        state: str = "SP",
        country: str = "Brasil",
        on_progress: Optional[callable] = None,
    ) -> List[GeocodedLocation]:
        """Geocode a list of addresses.

        Args:
            addresses: List of addresses to geocode.
            state: State to append to addresses.
            country: Country to append to addresses.
            on_progress: Optional callback(current, total) for progress updates.

        Returns:
            List of GeocodedLocation results.
        """
        results = []
        total = len(addresses)

        for i, address in enumerate(addresses):
            result = self.geocode_address(address, state=state, country=country)
            results.append(result)

            if on_progress:
                on_progress(i + 1, total)

        return results

    def get_success_rate(self, results: List[GeocodedLocation]) -> float:
        """Calculate the success rate of geocoding results.

        Args:
            results: List of geocoding results.

        Returns:
            Success rate as a float between 0.0 and 1.0.
        """
        if not results:
            return 0.0
        successful = sum(1 for r in results if r.success)
        return successful / len(results)
