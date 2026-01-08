"""
Data loading utilities for the routing optimizer.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_pharmacies_csv(
    file_path: str,
    address_column: str = "Endereco",
    name_column: str = "Unidade",
    city_column: Optional[str] = "Municipio",
) -> pd.DataFrame:
    """Load pharmacies data from CSV file.

    Args:
        file_path: Path to the CSV file.
        address_column: Name of the column containing addresses.
        name_column: Name of the column containing pharmacy names.
        city_column: Name of the column containing city names (optional).

    Returns:
        DataFrame with pharmacy data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    # Check required columns
    required_cols = [address_column, name_column]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def get_addresses_from_csv(
    file_path: str,
    address_column: str = "Endereco",
    city_column: Optional[str] = "Municipio",
    state: str = "SP",
) -> List[str]:
    """Extract full addresses from CSV file.

    Args:
        file_path: Path to the CSV file.
        address_column: Name of the column containing addresses.
        city_column: Name of the column containing city names.
        state: State abbreviation to append.

    Returns:
        List of full addresses.
    """
    df = load_pharmacies_csv(file_path, address_column=address_column)

    addresses = []
    for _, row in df.iterrows():
        addr = row[address_column]
        if city_column and city_column in df.columns:
            addr = f"{addr}, {row[city_column]}"
        addr = f"{addr}, {state}, Brasil"
        addresses.append(addr)

    return addresses
