"""Operadores de variação usados no experimento."""

from __future__ import annotations

import numpy as np
import pygad
from sklearn.neighbors import KernelDensity

from .config import POLY_ETA, POLY_MUTATION_PROB, SBX_ETA


def dummy_fitness(ga_instance, solution, solution_idx) -> float:
    """Fitness mínimo exigido pela instância auxiliar do PyGAD."""
    return 0.0


def build_pygad_operator_engine(
    ctx: dict,
    seed: int,
    n_features: int,
    eta: float = SBX_ETA,
) -> pygad.GA:
    """Cria a instância auxiliar que fornece SBX e Polynomial Mutation."""
    lower = float(np.min(ctx["global_low"]))
    upper = float(np.max(ctx["global_high"]))
    initial_population = np.zeros((2, n_features), dtype=float)

    return pygad.GA(
        num_generations=1,
        num_parents_mating=2,
        fitness_func=dummy_fitness,
        initial_population=initial_population,
        init_range_low=lower,
        init_range_high=upper,
        crossover_type="sbx",
        sbx_crossover_eta=eta,
        mutation_type="polynomial",
        polynomial_mutation_eta=POLY_ETA,
        mutation_probability=POLY_MUTATION_PROB,
        random_seed=seed,
        suppress_warnings=True,
    )


def pygad_sbx_child(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    engine: pygad.GA,
    ctx: dict,
) -> np.ndarray:
    """Gera um descendente usando o SBX nativo do PyGAD."""
    parents = np.vstack([parent_a, parent_b]).astype(float)
    child = np.asarray(
        engine.sbx_crossover(
            parents,
            offspring_size=(1, parents.shape[1]),
        )[0],
        dtype=float,
    )

    return np.clip(child, ctx["global_low"], ctx["global_high"])


def fit_gene_kdes(populations: dict[int, np.ndarray]) -> list[KernelDensity]:
    """Ajusta uma KDE gaussiana por gene usando o pool populacional."""
    unified = np.vstack(
        [populations[task] for task in sorted(populations)]
    )
    models = []

    for gene_idx in range(unified.shape[1]):
        values = unified[:, gene_idx].reshape(-1, 1)
        scale = float(np.std(values, ddof=1))
        bandwidth = max(
            scale * len(values) ** (-1.0 / 5.0),
            0.05,
        )
        models.append(
            KernelDensity(
                kernel="gaussian",
                bandwidth=bandwidth,
            ).fit(values)
        )

    return models


def obscan_kde_child(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    kde_models: list[KernelDensity],
    rng: np.random.Generator,
    ctx: dict,
) -> np.ndarray:
    """Gera um descendente OB-Scan escolhendo o gene parental mais denso."""
    child = np.empty_like(parent_a, dtype=float)

    for gene_idx, kde in enumerate(kde_models):
        candidates = np.array(
            [[parent_a[gene_idx]], [parent_b[gene_idx]]],
            dtype=float,
        )
        log_density = kde.score_samples(candidates)

        if np.isclose(log_density[0], log_density[1]):
            chosen = int(rng.integers(0, 2))
        else:
            chosen = int(np.argmax(log_density))

        child[gene_idx] = candidates[chosen, 0]

    return np.clip(child, ctx["global_low"], ctx["global_high"])
