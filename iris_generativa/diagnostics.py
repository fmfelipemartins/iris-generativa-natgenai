"""Medições observacionais: não consomem o RNG nem alteram a seleção."""

import numpy as np
from sklearn.metrics import pairwise_distances
from .fitness import generative_fitness


def population_metrics(points, task, context):
    """Diversidade INTRAtarefa e fitness de uma população, inclusive candidatos crus."""
    points = np.asarray(points)
    distances = pairwise_distances(points)
    upper = distances[np.triu_indices(len(points), 1)]
    reference = context["real_by_task"][task]
    reference_distances = pairwise_distances(reference)
    reference_upper = reference_distances[np.triu_indices(len(reference), 1)]
    diversity = float(upper.mean()) if len(upper) else 0.0
    reference_diversity = float(reference_upper.mean()) if len(reference_upper) else 0.0
    # Diagnóstico não entra no contador do orçamento da busca.
    diagnostic_context = {
        k: v for k, v in context.items() if k != "_fitness_evaluations"
    }
    scores = np.asarray(
        [generative_fitness(p, task, diagnostic_context) for p in points]
    )
    return dict(
        n=len(points),
        diversity=diversity,
        diversity_ratio=diversity / max(reference_diversity, 1e-12),
        variance_trace=float(np.var(points, axis=0).sum()),
        fitness_mean=float(scores[:, 0].mean()),
        plausibility_mean=float(scores[:, 1].mean()),
        novelty_mean=float(scores[:, 2].mean()),
        coherence_mean=float(scores[:, 3].mean()),
        nearest_real_mean=float(
            pairwise_distances(points, reference).min(axis=1).mean()
        ),
    )
