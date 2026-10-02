"""Carga da Iris e preparação do contexto evolutivo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from .config import K_NEIGHBORS, N_NICHES


def load_iris_data() -> dict:
    """Carrega a Iris e devolve arrays, rótulos e tabela de apoio."""
    iris = load_iris(as_frame=True)
    x_real = iris.data.to_numpy(dtype=float)
    y_species = iris.target.to_numpy(dtype=int)
    feature_names = [name.replace(" (cm)", "") for name in iris.feature_names]
    species_names = iris.target_names.tolist()

    real_df = pd.DataFrame(x_real, columns=feature_names)
    real_df["species_id"] = y_species
    real_df["species"] = [species_names[label] for label in y_species]

    return {
        "x_real": x_real,
        "y_species": y_species,
        "feature_names": feature_names,
        "species_names": species_names,
        "real_df": real_df,
    }


def nearest_neighbor_profile(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula distância ao vizinho mais próximo e média dos k vizinhos."""
    distances = pairwise_distances(points, points)
    np.fill_diagonal(distances, np.inf)

    nearest = distances.min(axis=1)
    k = min(K_NEIGHBORS, max(1, len(points) - 1))
    mean_knn = np.sort(distances, axis=1)[:, :k].mean(axis=1)

    return nearest, mean_knn


def prepare_context(
    x_source: np.ndarray,
    y_source: np.ndarray,
    seed: int,
) -> dict:
    """Calibra escalas, alvos de novidade, nichos e limites da busca."""
    scaler = StandardScaler()
    z_source = scaler.fit_transform(x_source)

    tasks = sorted(np.unique(y_source).tolist())
    real_by_task = {task: z_source[y_source == task] for task in tasks}

    novelty_targets = {}
    novelty_sigmas = {}
    plausibility_scales = {}
    niche_models = {}
    niche_quotas = {}

    for task in tasks:
        task_points = real_by_task[task]
        nearest, mean_knn = nearest_neighbor_profile(task_points)

        novelty_targets[task] = float(np.median(nearest))
        robust_scale = float(
            np.subtract(*np.percentile(nearest, [75, 25])) / 1.349
        )
        novelty_sigmas[task] = max(
            robust_scale,
            0.25 * novelty_targets[task],
            0.05,
        )
        plausibility_scales[task] = max(float(np.median(mean_knn)), 0.05)

        n_clusters = min(N_NICHES, len(task_points))
        niche_model = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=20,
        )
        niche_labels = niche_model.fit_predict(task_points)

        niche_models[task] = niche_model
        niche_quotas[task] = np.bincount(
            niche_labels,
            minlength=n_clusters,
        )

    global_low = z_source.min(axis=0) - 0.50
    global_high = z_source.max(axis=0) + 0.50

    return {
        "scaler": scaler,
        "Z_real": z_source,
        "y_real": y_source.copy(),
        "tasks": tasks,
        "real_by_task": real_by_task,
        "novelty_targets": novelty_targets,
        "novelty_sigmas": novelty_sigmas,
        "plausibility_scales": plausibility_scales,
        "niche_models": niche_models,
        "niche_quotas": niche_quotas,
        "global_low": global_low,
        "global_high": global_high,
    }


def build_calibration_table(context: dict, species_names: list[str]) -> pd.DataFrame:
    """Organiza os parâmetros de calibração por espécie."""
    rows = [
        {
            "species": species_names[task],
            "novelty_target": context["novelty_targets"][task],
            "novelty_sigma": context["novelty_sigmas"][task],
            "plausibility_scale": context["plausibility_scales"][task],
        }
        for task in context["tasks"]
    ]

    return pd.DataFrame(rows).set_index("species")
