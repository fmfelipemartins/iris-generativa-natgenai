"""Orquestração dos experimentos da Iris Generativa."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import (
    CV_SEED,
    MAIN_CONDITIONS,
    RMP,
    SBX_ETA,
    SEED,
)
from .data import prepare_context
from .evaluation import (
    evaluate_discriminators,
    exact_duplicate_count,
    mean_pairwise_diversity,
)
from .evolution import populations_to_synthetic, run_generator


def run_main_experiment(
    context: dict,
    seed: int = SEED,
) -> dict:
    """Executa SBX, OB-Scan e Híbrido com a mesma seed."""
    main_runs = {}

    for condition_name, p_sbx in MAIN_CONDITIONS.items():
        populations, history = run_generator(
            context,
            seed=seed,
            p_sbx=p_sbx,
        )
        synthetic_z, synthetic_x, synthetic_species = (
            populations_to_synthetic(
                populations,
                context,
            )
        )

        main_runs[condition_name] = {
            "populations": populations,
            "history": history,
            "synthetic_z": synthetic_z,
            "synthetic_x": synthetic_x,
            "synthetic_species": synthetic_species,
        }

    return main_runs


def robustness_experiment(
    context: dict,
    real_x: np.ndarray,
    real_species: np.ndarray,
    real_diversity: float,
    seeds=range(20),
) -> pd.DataFrame:
    """Repete as três condições nas mesmas vinte seeds."""
    rows = []

    for seed in seeds:
        for condition_name, p_sbx in MAIN_CONDITIONS.items():
            populations, _ = run_generator(
                context,
                seed=seed,
                p_sbx=p_sbx,
            )
            synthetic_z, synthetic_x, synthetic_species = (
                populations_to_synthetic(
                    populations,
                    context,
                )
            )
            table = evaluate_discriminators(
                real_x,
                real_species,
                synthetic_x,
                synthetic_species,
                cv_seed=CV_SEED,
            )

            rows.append(
                {
                    "seed": seed,
                    "condition": condition_name,
                    "rf_accuracy": table.loc[
                        "Random Forest",
                        "accuracy_mean",
                    ],
                    "rf_roc_auc": table.loc[
                        "Random Forest",
                        "roc_auc_mean",
                    ],
                    "diversity_ratio": (
                        mean_pairwise_diversity(synthetic_z)
                        / real_diversity
                    ),
                    "duplicates": exact_duplicate_count(
                        synthetic_x,
                        real_x,
                    ),
                }
            )

    return pd.DataFrame(rows)


def quick_condition_score(
    context: dict,
    real_x: np.ndarray,
    real_species: np.ndarray,
    real_diversity: float,
    seed: int,
    p_sbx: float,
    sbx_eta: float = SBX_ETA,
    rmp: float = RMP,
    moderated: bool = True,
) -> dict:
    """Executa uma condição e retorna métricas rápidas para ablações."""
    populations, _ = run_generator(
        context,
        seed=seed,
        p_sbx=p_sbx,
        rmp=rmp,
        moderated=moderated,
        sbx_eta=sbx_eta,
    )
    synthetic_z, synthetic_x, synthetic_species = (
        populations_to_synthetic(
            populations,
            context,
        )
    )
    table = evaluate_discriminators(
        real_x,
        real_species,
        synthetic_x,
        synthetic_species,
    )

    return {
        "rf_accuracy": table.loc[
            "Random Forest",
            "accuracy_mean",
        ],
        "rf_roc_auc": table.loc[
            "Random Forest",
            "roc_auc_mean",
        ],
        "diversity_ratio": (
            mean_pairwise_diversity(synthetic_z)
            / real_diversity
        ),
    }


def sensitivity_experiment(
    context: dict,
    real_x: np.ndarray,
    real_species: np.ndarray,
    real_diversity: float,
    seed: int = SEED,
) -> pd.DataFrame:
    """Executa as ablações de eta, RMP e seleção moderada."""
    rows = []

    for eta in [5.0, 30.0, 50.0]:
        rows.append(
            {
                "experiment": "SBX eta",
                "value": eta,
                **quick_condition_score(
                    context,
                    real_x,
                    real_species,
                    real_diversity,
                    seed,
                    p_sbx=1.0,
                    sbx_eta=eta,
                ),
            }
        )

    for rmp in [0.0, 0.20]:
        rows.append(
            {
                "experiment": "RMP",
                "value": rmp,
                **quick_condition_score(
                    context,
                    real_x,
                    real_species,
                    real_diversity,
                    seed,
                    p_sbx=0.5,
                    rmp=rmp,
                ),
            }
        )

    for moderated in [False, True]:
        rows.append(
            {
                "experiment": "Seleção moderada",
                "value": moderated,
                **quick_condition_score(
                    context,
                    real_x,
                    real_species,
                    real_diversity,
                    seed,
                    p_sbx=0.5,
                    moderated=moderated,
                ),
            }
        )

    return pd.DataFrame(rows)


def polynomial_mutation_experiment(
    context: dict,
    real_x: np.ndarray,
    real_species: np.ndarray,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa SBX com Polynomial Mutation como laboratório complementar."""
    populations, history = run_generator(
        context,
        seed=seed,
        p_sbx=1.0,
        use_polynomial_mutation=True,
    )
    _, synthetic_x, synthetic_species = (
        populations_to_synthetic(
            populations,
            context,
        )
    )
    results = evaluate_discriminators(
        real_x,
        real_species,
        synthetic_x,
        synthetic_species,
    )

    return results, history


def crossfit_generator_experiment(
    real_x: np.ndarray,
    real_species: np.ndarray,
) -> pd.DataFrame:
    """Avalia sintéticos contra registros reais excluídos da calibração."""
    outer = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=CV_SEED,
    )
    rows = []

    for fold, (train_idx, holdout_idx) in enumerate(
        outer.split(real_x, real_species),
        start=1,
    ):
        x_train = real_x[train_idx]
        y_train = real_species[train_idx]
        x_holdout = real_x[holdout_idx]
        y_holdout = real_species[holdout_idx]

        fold_context = prepare_context(
            x_train,
            y_train,
            SEED + fold,
        )

        for condition_name, p_sbx in MAIN_CONDITIONS.items():
            populations, _ = run_generator(
                fold_context,
                seed=SEED + fold,
                p_sbx=p_sbx,
                pop_size=10,
            )
            _, synthetic_x, synthetic_species = (
                populations_to_synthetic(
                    populations,
                    fold_context,
                )
            )

            combined_x = np.vstack(
                [x_holdout, synthetic_x]
            )
            origin = np.concatenate(
                [
                    np.ones(len(x_holdout), dtype=int),
                    np.zeros(len(synthetic_x), dtype=int),
                ]
            )
            species_combined = np.concatenate(
                [y_holdout, synthetic_species]
            )
            strata = np.array(
                [
                    f"{origin_label}_{species_label}"
                    for origin_label, species_label
                    in zip(origin, species_combined)
                ],
                dtype=str,
            )

            inner = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=CV_SEED + fold,
            )
            splits = list(
                inner.split(combined_x, strata)
            )

            model = RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=CV_SEED,
                n_jobs=-1,
            )
            scores = cross_validate(
                model,
                combined_x,
                origin,
                cv=splits,
                scoring={
                    "accuracy": "accuracy",
                    "roc_auc": "roc_auc",
                },
                n_jobs=-1,
            )

            rows.append(
                {
                    "fold": fold,
                    "condition": condition_name,
                    "accuracy": float(
                        scores["test_accuracy"].mean()
                    ),
                    "roc_auc": float(
                        scores["test_roc_auc"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)
