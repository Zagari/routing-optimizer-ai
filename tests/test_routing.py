"""
Tests for the routing module (geocoding and distance calculation).
"""

from unittest.mock import MagicMock, patch

import pytest

from routing_optimizer.routing.distance import (
    OSRMDistanceMatrix,
    meters_to_km,
    seconds_to_minutes,
)
from routing_optimizer.routing.geocoding import GeocodedLocation, Geocoder


class TestGeocodedLocation:
    """Tests for GeocodedLocation dataclass."""

    def test_successful_location(self):
        """Test creating a successful geocoded location."""
        loc = GeocodedLocation(
            original_address="Av. Paulista, 1000",
            latitude=-23.5614,
            longitude=-46.6558,
            formatted_address="Av. Paulista, 1000, Sao Paulo, SP, Brasil",
            success=True,
        )
        assert loc.success
        assert loc.latitude == -23.5614
        assert loc.longitude == -46.6558
        assert loc.error is None

    def test_failed_location(self):
        """Test creating a failed geocoded location."""
        loc = GeocodedLocation(
            original_address="Invalid Address XYZ",
            latitude=0.0,
            longitude=0.0,
            formatted_address="",
            success=False,
            error="Endereco nao encontrado",
        )
        assert not loc.success
        assert loc.error == "Endereco nao encontrado"


class TestGeocoder:
    """Tests for Geocoder class."""

    def test_geocoder_initialization(self):
        """Test geocoder initializes correctly."""
        geocoder = Geocoder(user_agent="test-agent")
        assert geocoder.geolocator is not None

    def test_geocode_address_success(self):
        """Test successful address geocoding."""
        geocoder = Geocoder()

        # Mock the geocode function (instance attribute)
        mock_location = MagicMock()
        mock_location.latitude = -23.5505
        mock_location.longitude = -46.6333
        mock_location.address = "Praca da Se, Sao Paulo, SP, Brasil"
        geocoder.geocode = MagicMock(return_value=mock_location)

        result = geocoder.geocode_address("Praca da Se, Sao Paulo")

        assert result.success
        assert result.latitude == -23.5505
        assert result.longitude == -46.6333

    def test_geocode_address_not_found(self):
        """Test geocoding when address is not found."""
        geocoder = Geocoder()
        geocoder.geocode = MagicMock(return_value=None)

        result = geocoder.geocode_address("Endereco Inexistente XYZ 12345")

        assert not result.success
        assert result.error == "Endereco nao encontrado"

    def test_geocode_batch(self):
        """Test batch geocoding."""
        geocoder = Geocoder()

        mock_location = MagicMock()
        mock_location.latitude = -23.5505
        mock_location.longitude = -46.6333
        mock_location.address = "Test Address"
        geocoder.geocode = MagicMock(return_value=mock_location)

        addresses = ["Address 1", "Address 2", "Address 3"]
        results = geocoder.geocode_batch(addresses)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_get_success_rate(self):
        """Test success rate calculation."""
        geocoder = Geocoder()

        results = [
            GeocodedLocation("A", -23.5, -46.6, "A", success=True),
            GeocodedLocation("B", -23.6, -46.7, "B", success=True),
            GeocodedLocation("C", 0, 0, "", success=False, error="Not found"),
            GeocodedLocation("D", -23.7, -46.8, "D", success=True),
        ]

        rate = geocoder.get_success_rate(results)
        assert rate == 0.75  # 3 out of 4

    def test_get_success_rate_empty(self):
        """Test success rate with empty results."""
        geocoder = Geocoder()
        assert geocoder.get_success_rate([]) == 0.0


class TestOSRMDistanceMatrix:
    """Tests for OSRMDistanceMatrix class."""

    def test_initialization(self):
        """Test OSRM client initializes correctly."""
        dm = OSRMDistanceMatrix()
        assert dm.base_url == "http://router.project-osrm.org"
        assert dm.profile == "driving"

    def test_initialization_custom(self):
        """Test OSRM client with custom settings."""
        dm = OSRMDistanceMatrix(
            base_url="http://custom-osrm.example.com",
            profile="walking",
            timeout=60,
        )
        assert dm.base_url == "http://custom-osrm.example.com"
        assert dm.profile == "walking"
        assert dm.timeout == 60

    @patch("requests.get")
    def test_get_distance_matrix(self, mock_get):
        """Test distance matrix calculation with mocked response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": "Ok",
            "distances": [
                [0, 1000, 2000],
                [1000, 0, 1500],
                [2000, 1500, 0],
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        dm = OSRMDistanceMatrix()
        coords = [
            (-23.5505, -46.6333),
            (-23.5614, -46.6558),
            (-23.5475, -46.6361),
        ]

        matrix = dm.get_distance_matrix(coords)

        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 0
        assert matrix[0, 1] == 1000
        assert matrix[1, 2] == 1500

    @patch("requests.get")
    def test_get_distance_matrix_error(self, mock_get):
        """Test handling of OSRM error response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": "InvalidQuery",
            "message": "Invalid coordinates",
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        dm = OSRMDistanceMatrix()
        coords = [(-23.5505, -46.6333), (-23.5614, -46.6558)]

        with pytest.raises(ValueError, match="OSRM error"):
            dm.get_distance_matrix(coords)

    def test_get_distance_matrix_single_point(self):
        """Test with single coordinate (edge case)."""
        dm = OSRMDistanceMatrix()
        coords = [(-23.5505, -46.6333)]

        matrix = dm.get_distance_matrix(coords)

        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 0

    @patch("requests.get")
    def test_get_duration_matrix(self, mock_get):
        """Test duration matrix calculation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": "Ok",
            "durations": [
                [0, 120, 240],
                [120, 0, 180],
                [240, 180, 0],
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        dm = OSRMDistanceMatrix()
        coords = [
            (-23.5505, -46.6333),
            (-23.5614, -46.6558),
            (-23.5475, -46.6361),
        ]

        matrix = dm.get_duration_matrix(coords)

        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == 120  # 2 minutes

    @patch("requests.get")
    def test_get_route(self, mock_get):
        """Test route retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": "Ok",
            "routes": [
                {
                    "distance": 5000,
                    "duration": 600,
                    "geometry": {"type": "LineString", "coordinates": []},
                }
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        dm = OSRMDistanceMatrix()
        coords = [(-23.5505, -46.6333), (-23.5614, -46.6558)]

        route = dm.get_route(coords)

        assert route["code"] == "Ok"
        assert route["routes"][0]["distance"] == 5000

    @patch("requests.get")
    def test_get_route_distance(self, mock_get):
        """Test route distance calculation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": "Ok",
            "routes": [{"distance": 7500, "duration": 900}],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        dm = OSRMDistanceMatrix()
        coords = [(-23.5505, -46.6333), (-23.5614, -46.6558)]

        distance = dm.get_route_distance(coords)

        assert distance == 7500


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_meters_to_km(self):
        """Test meters to kilometers conversion."""
        assert meters_to_km(1000) == 1.0
        assert meters_to_km(2500) == 2.5
        assert meters_to_km(0) == 0.0

    def test_seconds_to_minutes(self):
        """Test seconds to minutes conversion."""
        assert seconds_to_minutes(60) == 1.0
        assert seconds_to_minutes(90) == 1.5
        assert seconds_to_minutes(0) == 0.0


@pytest.mark.integration
class TestGeocoderIntegration:
    """Integration tests for Geocoder (requires internet)."""

    def test_geocode_real_address(self):
        """Test geocoding a real address."""
        geocoder = Geocoder()
        result = geocoder.geocode_address("Av. Paulista, 1000, Sao Paulo")

        assert result.success
        # Sao Paulo coordinates approximately
        assert -24 < result.latitude < -23
        assert -47 < result.longitude < -46

    def test_geocode_invalid_address(self):
        """Test geocoding an invalid address."""
        geocoder = Geocoder()
        result = geocoder.geocode_address("XYZXYZXYZ123456789 Nowhere")

        assert not result.success


@pytest.mark.integration
class TestOSRMIntegration:
    """Integration tests for OSRM (requires internet)."""

    def test_distance_matrix_real(self):
        """Test distance matrix with real coordinates."""
        dm = OSRMDistanceMatrix()
        coords = [
            (-23.5505, -46.6333),  # SP Centro
            (-23.5614, -46.6558),  # Paulista
            (-23.5475, -46.6361),  # Se
        ]

        matrix = dm.get_distance_matrix(coords)

        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 0  # Distance to self
        assert matrix[0, 1] > 0  # Real distance should be positive
        assert matrix[1, 0] > 0  # Matrix should not be all zeros

    def test_route_real(self):
        """Test route calculation with real coordinates."""
        dm = OSRMDistanceMatrix()
        coords = [
            (-23.5505, -46.6333),
            (-23.5614, -46.6558),
        ]

        route = dm.get_route(coords)

        assert route["code"] == "Ok"
        assert len(route["routes"]) > 0
        assert route["routes"][0]["distance"] > 0
