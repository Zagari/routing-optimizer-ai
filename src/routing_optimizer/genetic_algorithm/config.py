"""
Configuration classes for the Genetic Algorithm.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GAConfig:
    """Configuration for the Genetic Algorithm.

    Attributes:
        population_size: Number of individuals in the population.
        mutation_probability: Probability of mutation (0.0 to 1.0).
        max_epochs: Maximum number of generations to run (up to 10000).
        tournament_size: Number of individuals in tournament selection.
        elitism_count: Number of best individuals to preserve each generation.
        stagnation_threshold: Generations without improvement to stop.
            If None, defaults to 20% of max_epochs.
    """

    population_size: int = 200
    mutation_probability: float = 0.6
    max_epochs: int = 1000
    tournament_size: int = 5
    elitism_count: int = 2
    local_search_rate: float = 0.1  # Apply 2-opt to only 10% of children
    local_search_elites_only: bool = True  # Apply 2-opt only to elite individuals
    stagnation_threshold: Optional[int] = None  # None = 20% of max_epochs
    max_mutations_per_individual: int = 3  # Maximum mutations per individual
    hybrid_initialization: bool = True  # Use nearest neighbor for part of initial population
    heuristic_ratio: float = 0.1  # Fraction of population using heuristic (10%)

    @classmethod
    def from_env(cls) -> "GAConfig":
        """Create configuration from environment variables.

        Environment variables:
            GA_POPULATION_SIZE: Population size (default: 200)
            GA_MUTATION_PROBABILITY: Mutation probability (default: 0.6)
            GA_MAX_EPOCHS: Maximum epochs (default: 1000, max: 10000)
            GA_TOURNAMENT_SIZE: Tournament size (default: 5)
            GA_STAGNATION_THRESHOLD: Generations without improvement to stop.
                If not set, defaults to 20% of GA_MAX_EPOCHS.

        Returns:
            GAConfig instance with values from environment or defaults.
        """
        # Get stagnation threshold from env, or None to use default (20% of max_epochs)
        stagnation_env = os.getenv("GA_STAGNATION_THRESHOLD")
        stagnation_threshold = int(stagnation_env) if stagnation_env else None

        return cls(
            population_size=int(os.getenv("GA_POPULATION_SIZE", 200)),
            mutation_probability=float(os.getenv("GA_MUTATION_PROBABILITY", 0.6)),
            max_epochs=int(os.getenv("GA_MAX_EPOCHS", 1000)),
            tournament_size=int(os.getenv("GA_TOURNAMENT_SIZE", 5)),
            stagnation_threshold=stagnation_threshold,
        )

    def __post_init__(self) -> None:
        """Validate configuration values and set defaults."""
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0.0 and 1.0")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if self.max_epochs > 10000:
            raise ValueError("max_epochs must be at most 10000")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")
        if self.elitism_count < 0:
            raise ValueError("elitism_count must be non-negative")
        if self.elitism_count >= self.population_size:
            raise ValueError("elitism_count must be less than population_size")
        if not 0.0 <= self.local_search_rate <= 1.0:
            raise ValueError("local_search_rate must be between 0.0 and 1.0")

        # Calculate stagnation_threshold as 20% of max_epochs if not explicitly set
        if self.stagnation_threshold is None:
            self.stagnation_threshold = max(1, int(self.max_epochs * 0.2))

        if self.stagnation_threshold < 1:
            raise ValueError("stagnation_threshold must be at least 1")
        if self.max_mutations_per_individual < 1:
            raise ValueError("max_mutations_per_individual must be at least 1")
        if not 0.0 <= self.heuristic_ratio <= 1.0:
            raise ValueError("heuristic_ratio must be between 0.0 and 1.0")
