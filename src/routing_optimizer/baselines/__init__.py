"""
Baseline algorithms for VRP comparison.

This module provides simple baseline algorithms to compare against
the Genetic Algorithm solution.
"""

from routing_optimizer.baselines.clarke_wright import ClarkeWrightSolver
from routing_optimizer.baselines.nearest_neighbor import NearestNeighborSolver
from routing_optimizer.baselines.random_solver import RandomSolver

__all__ = ["RandomSolver", "NearestNeighborSolver", "ClarkeWrightSolver"]
