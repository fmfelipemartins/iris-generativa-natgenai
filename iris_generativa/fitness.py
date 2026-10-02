"""Função de fitness generativo."""

from __future__ import annotations

import numpy as np

from .config import FITNESS_WEIGHTS, K_NEIGHBORS


def mean_k_distance(
    point: np.ndarray,
    reference: np.ndarray,
    k: int = K_NEIGHBORS,
) -> float:
    """Calcula a distância média aos k vizinhos mais próximos."""
    distances = np.linalg.norm(reference - point, axis=1)
    k_eff = min(k, len(distances))
    return float(np.partition(distances, k_eff - 1)[:k_eff].mean())


def coherence_score(point: np.ndarray, target_task: int, ctx: dict) -> float:
    """Estima a compatibilidade local do indivíduo com a tarefa atribuída."""
    local_distances = np.array(
        [
            mean_k_distance(point, ctx["real_by_task"][task])
            for task in ctx["tasks"]
        ],
        dtype=float,
    )
    temperature = max(
        float(np.median(list(ctx["plausibility_scales"].values()))),
        0.05,
    )
    shifted = -(local_distances - local_distances.min()) / temperature
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    target_position = ctx["tasks"].index(target_task)

    return float(probabilities[target_position])


def generative_fitness(
    point: np.ndarray,
    task: int,
    ctx: dict,
) -> tuple[float, float, float, float]:
    """Combina plausibilidade, novidade e coerência."""
    reference = ctx["real_by_task"][task]

    nearest_distance = float(
        np.linalg.norm(reference - point, axis=1).min()
    )
    local_distance = mean_k_distance(point, reference)

    plausibility_scale = ctx["plausibility_scales"][task]
    plausibility = float(
        np.exp(-0.5 * (local_distance / plausibility_scale) ** 2)
    )

    novelty_target = ctx["novelty_targets"][task]
    novelty_sigma = ctx["novelty_sigmas"][task]
    novelty = float(
        np.exp(
            -0.5
            * ((nearest_distance - novelty_target) / novelty_sigma) ** 2
        )
    )

    coherence = coherence_score(point, task, ctx)

    total = (
        FITNESS_WEIGHTS["plausibility"] * plausibility
        + FITNESS_WEIGHTS["novelty"] * novelty
        + FITNESS_WEIGHTS["coherence"] * coherence
    )

    return total, plausibility, novelty, coherence
