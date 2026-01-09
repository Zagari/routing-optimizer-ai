"""
Dataset persistence manager for routing optimizer.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from routing_optimizer.routing.geocoding import GeocodedLocation


@dataclass
class DatasetMetadata:
    """Metadata for a saved dataset."""

    name: str
    original_filename: str
    created_at: str
    address_column: str
    name_column: Optional[str]
    depot_address: str
    total_locations: int
    geocoded_count: int
    has_distance_matrix: bool = False


class DatasetManager:
    """Manages dataset persistence to disk."""

    def __init__(self, base_path: str = "data/datasets"):
        """Initialize dataset manager.

        Args:
            base_path: Base directory for storing datasets.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[DatasetMetadata]:
        """List all saved datasets.

        Returns:
            List of DatasetMetadata for each saved dataset.
        """
        datasets = []
        if not self.base_path.exists():
            return datasets

        for dataset_dir in self.base_path.iterdir():
            if dataset_dir.is_dir() and not dataset_dir.name.startswith("."):
                metadata = self._load_metadata(dataset_dir.name)
                if metadata:
                    datasets.append(metadata)
        return sorted(datasets, key=lambda d: d.created_at, reverse=True)

    def dataset_exists(self, name: str) -> bool:
        """Check if a dataset with given name exists."""
        return (self.base_path / name).exists()

    def generate_unique_name(self, base_name: str) -> str:
        """Generate unique dataset name from base name.

        Args:
            base_name: Base name (usually from filename without extension).

        Returns:
            Unique name, adding suffix if needed.
        """
        # Clean the base name
        clean_name = base_name.replace(" ", "_").lower()
        clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")

        if not self.dataset_exists(clean_name):
            return clean_name

        # Add numeric suffix
        counter = 2
        while self.dataset_exists(f"{clean_name}_{counter}"):
            counter += 1
        return f"{clean_name}_{counter}"

    def save_dataset(
        self,
        name: str,
        original_df: pd.DataFrame,
        original_filename: str,
        address_column: str,
        name_column: Optional[str],
        depot_address: str,
        geocoded_results: list[GeocodedLocation],
    ) -> Path:
        """Save a dataset to disk.

        Args:
            name: Dataset name (will be directory name).
            original_df: Original DataFrame from CSV.
            original_filename: Original filename.
            address_column: Name of address column.
            name_column: Name of name/identifier column (or None).
            depot_address: Depot address string.
            geocoded_results: List of geocoding results.

        Returns:
            Path to saved dataset directory.
        """
        dataset_dir = self.base_path / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save original CSV
        original_df.to_csv(dataset_dir / "original.csv", index=False)

        # Save geocoded results
        geocoded_data = []
        for result in geocoded_results:
            geocoded_data.append({
                "original_address": result.original_address,
                "latitude": result.latitude,
                "longitude": result.longitude,
                "formatted_address": result.formatted_address,
                "success": result.success,
                "error": result.error,
            })
        with open(dataset_dir / "geocoded.json", "w", encoding="utf-8") as f:
            json.dump(geocoded_data, f, ensure_ascii=False, indent=2)

        # Save metadata
        successful_count = sum(1 for r in geocoded_results if r.success)
        metadata = DatasetMetadata(
            name=name,
            original_filename=original_filename,
            created_at=datetime.now().isoformat(),
            address_column=address_column,
            name_column=name_column,
            depot_address=depot_address,
            total_locations=len(geocoded_results),
            geocoded_count=successful_count,
            has_distance_matrix=False,
        )
        with open(dataset_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)

        return dataset_dir

    def load_dataset(
        self, name: str
    ) -> tuple[DatasetMetadata, pd.DataFrame, list[GeocodedLocation], list[str]]:
        """Load a dataset from disk.

        Args:
            name: Dataset name.

        Returns:
            Tuple of (metadata, original_df, geocoded_results, names).

        Raises:
            FileNotFoundError: If dataset doesn't exist.
        """
        dataset_dir = self.base_path / name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {name}")

        # Load metadata
        metadata = self._load_metadata(name)
        if not metadata:
            raise FileNotFoundError(f"Metadata not found for dataset: {name}")

        # Load original CSV
        original_df = pd.read_csv(dataset_dir / "original.csv")

        # Load geocoded results
        with open(dataset_dir / "geocoded.json", "r", encoding="utf-8") as f:
            geocoded_data = json.load(f)

        geocoded_results = []
        for item in geocoded_data:
            geocoded_results.append(
                GeocodedLocation(
                    original_address=item["original_address"],
                    latitude=item["latitude"],
                    longitude=item["longitude"],
                    formatted_address=item["formatted_address"],
                    success=item["success"],
                    error=item.get("error"),
                )
            )

        # Build names list
        names = ["DEPÓSITO"]
        if metadata.name_column and metadata.name_column in original_df.columns:
            names.extend(original_df[metadata.name_column].tolist())
        else:
            names.extend([f"Farmácia {i+1}" for i in range(len(original_df))])

        return metadata, original_df, geocoded_results, names

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset.

        Args:
            name: Dataset name.

        Returns:
            True if deleted, False if not found.
        """
        dataset_dir = self.base_path / name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            return True
        return False

    def save_distance_matrix(self, name: str, matrix: np.ndarray) -> None:
        """Save distance matrix for a dataset.

        Args:
            name: Dataset name.
            matrix: Distance matrix as numpy array.
        """
        dataset_dir = self.base_path / name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {name}")

        np.save(dataset_dir / "distance_matrix.npy", matrix)

        # Update metadata
        metadata = self._load_metadata(name)
        if metadata:
            metadata.has_distance_matrix = True
            with open(dataset_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)

    def load_distance_matrix(self, name: str) -> Optional[np.ndarray]:
        """Load distance matrix for a dataset.

        Args:
            name: Dataset name.

        Returns:
            Distance matrix or None if not saved.
        """
        matrix_path = self.base_path / name / "distance_matrix.npy"
        if matrix_path.exists():
            return np.load(matrix_path)
        return None

    def _load_metadata(self, name: str) -> Optional[DatasetMetadata]:
        """Load metadata for a dataset."""
        metadata_path = self.base_path / name / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return DatasetMetadata(**data)
