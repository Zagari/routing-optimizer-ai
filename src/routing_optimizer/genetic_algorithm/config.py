"""
Configuration classes for the Genetic Algorithm.
"""

import os
from dataclasses import dataclass


@dataclass
class GAConfig:
    """Configuration for the Genetic Algorithm.

    Attributes:
        population_size: Number of individuals in the population.
        mutation_probability: Probability of mutation (0.0 to 1.0).
        max_epochs: Maximum number of generations to run.
        tournament_size: Number of individuals in tournament selection.
        elitism_count: Number of best individuals to preserve each generation.
    """

    population_size: int = 200
    mutation_probability: float = 0.6
    max_epochs: int = 1000
    tournament_size: int = 5
    elitism_count: int = 2

    @classmethod
    def from_env(cls) -> "GAConfig":
        """Create configuration from environment variables.

        Environment variables:
            GA_POPULATION_SIZE: Population size (default: 200)
            GA_MUTATION_PROBABILITY: Mutation probability (default: 0.6)
            GA_MAX_EPOCHS: Maximum epochs (default: 1000)
            GA_TOURNAMENT_SIZE: Tournament size (default: 5)

        Returns:
            GAConfig instance with values from environment or defaults.
        """
        return cls(
            population_size=int(os.getenv("GA_POPULATION_SIZE", 200)),
            mutation_probability=float(os.getenv("GA_MUTATION_PROBABILITY", 0.6)),
            max_epochs=int(os.getenv("GA_MAX_EPOCHS", 1000)),
            tournament_size=int(os.getenv("GA_TOURNAMENT_SIZE", 5)),
        )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0.0 and 1.0")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")
        if self.elitism_count < 0:
            raise ValueError("elitism_count must be non-negative")
        if self.elitism_count >= self.population_size:
            raise ValueError("elitism_count must be less than population_size")
