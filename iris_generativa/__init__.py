"""Iris Generativa com NatGenAI."""

from .data import load_iris_data, prepare_context
from .evolution import run_generator
from .evaluation import evaluate_discriminators
from .experiments import (
    crossfit_generator_experiment,
    polynomial_mutation_experiment,
    robustness_experiment,
    run_main_experiment,
    sensitivity_experiment,
)

__all__ = [
    "crossfit_generator_experiment",
    "evaluate_discriminators",
    "load_iris_data",
    "polynomial_mutation_experiment",
    "prepare_context",
    "robustness_experiment",
    "run_generator",
    "run_main_experiment",
    "sensitivity_experiment",
]
