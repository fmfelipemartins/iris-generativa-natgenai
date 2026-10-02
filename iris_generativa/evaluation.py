"""Métricas generativas e teste discriminativo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier

from .config import CV_SEED


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def exact_duplicate_count(
    synthetic_x: np.ndarray,
    real_x: np.ndarray,
) -> int:
    """Conta sintéticos exatamente iguais a algum registro real."""
    return sum(
        any(
            np.array_equal(synthetic, real)
            for real in real_x
        )
        for synthetic in synthetic_x
    )


def species_coherence_accuracy(
    synthetic_x: np.ndarray,
    synthetic_species: np.ndarray,
    real_x: np.ndarray,
    real_species: np.ndarray,
) -> float:
    """Mede a coerência da espécie por classificação k-NN."""
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(real_x, real_species)
    predictions = model.predict(synthetic_x)

    return float(
        accuracy_score(synthetic_species, predictions)
    )


def mean_pairwise_diversity(points: np.ndarray) -> float:
    """Calcula a distância média entre pares distintos."""
    if len(points) < 2:
        return 0.0

    distances = pairwise_distances(points, points)
    upper = distances[
        np.triu_indices_from(distances, k=1)
    ]

    return float(upper.mean())


def ks_statistic_1d(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
) -> float:
    """Calcula a estatística KS empírica em uma dimensão."""
    values = np.sort(
        np.unique(np.concatenate([sample_a, sample_b]))
    )
    cdf_a = (
        np.searchsorted(
            np.sort(sample_a),
            values,
            side="right",
        )
        / len(sample_a)
    )
    cdf_b = (
        np.searchsorted(
            np.sort(sample_b),
            values,
            side="right",
        )
        / len(sample_b)
    )

    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_equal_1d(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
) -> float:
    """Calcula Wasserstein-1 para amostras de mesmo tamanho."""
    if len(sample_a) != len(sample_b):
        raise ValueError(
            "As amostras devem ter o mesmo tamanho nesta implementação."
        )

    return float(
        np.mean(
            np.abs(
                np.sort(sample_a) - np.sort(sample_b)
            )
        )
    )


def evaluate_discriminators(
    real_x: np.ndarray,
    real_species: np.ndarray,
    synthetic_x: np.ndarray,
    synthetic_species: np.ndarray,
    cv_seed: int = CV_SEED,
) -> pd.DataFrame:
    """Avalia Regressão Logística e Random Forest em Real × Sintético."""
    x_combined = np.vstack([real_x, synthetic_x])
    origin = np.concatenate(
        [
            np.ones(len(real_x), dtype=int),
            np.zeros(len(synthetic_x), dtype=int),
        ]
    )
    species_combined = np.concatenate(
        [real_species, synthetic_species]
    )
    strata = np.array(
        [
            f"{origin_label}_{species_label}"
            for origin_label, species_label
            in zip(origin, species_combined)
        ],
        dtype=str,
    )

    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=cv_seed,
    )
    splits = list(splitter.split(x_combined, strata))

    models = {
        "Regressão Logística": LogisticRegression(
            max_iter=2000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=cv_seed,
            n_jobs=-1,
        ),
    }

    rows = []

    for model_name, model in models.items():
        scores = cross_validate(
            model,
            x_combined,
            origin,
            cv=splits,
            scoring=SCORING,
            n_jobs=-1,
        )
        rows.append(
            {
                "model": model_name,
                **{
                    f"{metric}_mean": float(
                        scores[f"test_{metric}"].mean()
                    )
                    for metric in SCORING
                },
                **{
                    f"{metric}_std": float(
                        scores[f"test_{metric}"].std()
                    )
                    for metric in SCORING
                },
            }
        )

    return pd.DataFrame(rows).set_index("model")


def summarize_generation(
    main_runs: dict,
    context: dict,
    real_x: np.ndarray,
    real_species: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    """Resume cópia, coerência, novidade e diversidade por condição."""
    real_diversity = mean_pairwise_diversity(
        context["Z_real"]
    )
    rows = []

    for condition_name, result in main_runs.items():
        synthetic_z = result["synthetic_z"]
        synthetic_x = result["synthetic_x"]
        synthetic_species = result["synthetic_species"]

        nearest_to_real = pairwise_distances(
            synthetic_z,
            context["Z_real"],
        ).min(axis=1)
        synthetic_diversity = mean_pairwise_diversity(
            synthetic_z
        )

        rows.append(
            {
                "condition": condition_name,
                "duplicates": exact_duplicate_count(
                    synthetic_x,
                    real_x,
                ),
                "species_coherence": species_coherence_accuracy(
                    synthetic_x,
                    synthetic_species,
                    real_x,
                    real_species,
                ),
                "novelty_mean": float(
                    nearest_to_real.mean()
                ),
                "diversity": synthetic_diversity,
                "diversity_ratio": (
                    synthetic_diversity / real_diversity
                ),
            }
        )

    return (
        pd.DataFrame(rows).set_index("condition"),
        real_diversity,
    )


def build_distribution_table(
    main_runs: dict,
    real_x: np.ndarray,
    real_species: np.ndarray,
    feature_names: list[str],
    species_names: list[str],
    tasks: list[int],
) -> pd.DataFrame:
    """Calcula Wasserstein e KS por condição, espécie e atributo."""
    rows = []

    for condition_name, result in main_runs.items():
        for task in tasks:
            real_task = real_x[real_species == task]
            synthetic_task = result["synthetic_x"][
                result["synthetic_species"] == task
            ]

            for feature_idx, feature_name in enumerate(feature_names):
                real_values = real_task[:, feature_idx]
                synthetic_values = synthetic_task[:, feature_idx]

                rows.append(
                    {
                        "condition": condition_name,
                        "species": species_names[task],
                        "feature": feature_name,
                        "wasserstein": wasserstein_equal_1d(
                            real_values,
                            synthetic_values,
                        ),
                        "ks": ks_statistic_1d(
                            real_values,
                            synthetic_values,
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_discriminator_tables(
    main_runs: dict,
    real_x: np.ndarray,
    real_species: np.ndarray,
    cv_seed: int = CV_SEED,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Avalia os discriminadores e constrói a comparação resumida."""
    tables = {}
    rows = []

    for condition_name, result in main_runs.items():
        table = evaluate_discriminators(
            real_x,
            real_species,
            result["synthetic_x"],
            result["synthetic_species"],
            cv_seed=cv_seed,
        )
        tables[condition_name] = table

        rows.append(
            {
                "condition": condition_name,
                "logistic_accuracy": table.loc[
                    "Regressão Logística",
                    "accuracy_mean",
                ],
                "rf_accuracy": table.loc[
                    "Random Forest",
                    "accuracy_mean",
                ],
                "rf_roc_auc": table.loc[
                    "Random Forest",
                    "roc_auc_mean",
                ],
            }
        )

    comparison = pd.DataFrame(rows).set_index("condition")

    return tables, comparison
