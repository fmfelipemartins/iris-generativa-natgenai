"""Protocolos ampliados com orçamento explícito, metadados e isolamento por fold."""

import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config
from .data import prepare_context
from .evaluation import (
    evaluate_discriminators,
    exact_duplicate_count,
    mean_pairwise_diversity,
)
from .generative import ALL_CONDITIONS, run_condition
from .progress import progress


def execution_settings(profile=None):
    """Perfil quick é validação funcional, não substitui os resultados completos."""
    profile = config.EXECUTION_PROFILE if profile is None else profile
    if profile not in {"quick", "full"}:
        raise ValueError("Perfil deve ser quick ou full.")
    quick = profile == "quick"
    return dict(
        profile=profile,
        generations=config.QUICK_GENERATIONS if quick else config.GENERATIONS,
        vae_epochs=config.QUICK_VAE_EPOCHS if quick else config.VAE_EPOCHS,
        benchmark_size=config.QUICK_BENCHMARK_SIZE if quick else config.BENCHMARK_SIZE,
        benchmark_generations=3 if quick else config.BENCHMARK_GENERATIONS,
        seeds=(0, 1) if quick else config.ROBUSTNESS_SEEDS,
        evaluation=dict(
            n_splits=2 if quick else 5,
            n_repeats=1 if quick else 5,
            n_estimators=32 if quick else 300,
            n_jobs=1,
        ),
    )


def run_metadata(settings):
    packages = ("numpy", "pandas", "scikit-learn", "pygad", "torch", "tqdm")
    return dict(
        python=platform.python_version(),
        versions={p: version(p) for p in packages},
        settings=settings,
        configuration={k: v for k, v in vars(config).items() if k.isupper()},
    )


def condition_options(condition, settings, **extra):
    options = dict(generations=settings["generations"], show_progress=False)
    if condition == "vae_eda":
        options["vae_epochs"] = settings["vae_epochs"]
    return options | extra


def run_suite(
    context, seed=config.SEED, conditions=None, settings=None, show_progress=None
):
    conditions = list(ALL_CONDITIONS if conditions is None else conditions)
    settings = execution_settings() if settings is None else settings
    results = {}
    for condition in progress(conditions, "Condições gerativas", show_progress):
        results[condition] = run_condition(
            context, seed, condition, **condition_options(condition, settings)
        )
        results[condition]["metadata"].update(
            profile=settings["profile"], run=run_metadata(settings)
        )
    return results


def robustness_suite(
    context, real_x, species, conditions=None, settings=None, show_progress=None
):
    settings = execution_settings() if settings is None else settings
    conditions = list(ALL_CONDITIONS if conditions is None else conditions)
    jobs = [(seed, c) for seed in settings["seeds"] for c in conditions]
    real_diversity = mean_pairwise_diversity(context["Z_real"])
    rows = []
    for seed, condition in progress(jobs, "Robustez: execuções", show_progress):
        result = run_condition(
            context, seed, condition, **condition_options(condition, settings)
        )
        table = evaluate_discriminators(
            real_x,
            species,
            result["synthetic_x"],
            result["synthetic_species"],
            cv_seed=config.CV_SEED,
            **settings["evaluation"],
        )
        rows.append(
            dict(
                seed=seed,
                condition=condition,
                profile=settings["profile"],
                rf_accuracy=table.loc["Random Forest", "accuracy_mean"],
                rf_roc_auc=table.loc["Random Forest", "roc_auc_mean"],
                diversity_ratio=mean_pairwise_diversity(result["synthetic_z"])
                / real_diversity,
                duplicates=exact_duplicate_count(result["synthetic_x"], real_x),
                generation_seconds=result["metadata"]["seconds"],
                fitness_evaluations=int(
                    result["history"].iloc[-1]["fitness_evaluations"]
                ),
            )
        )
    return pd.DataFrame(rows)


def crossfit_suite(real_x, species, conditions=None, settings=None, show_progress=None):
    """Apenas x_train alimenta contexto, inicialização e ajuste de TODOS os modelos."""
    settings = execution_settings() if settings is None else settings
    conditions = list(ALL_CONDITIONS if conditions is None else conditions)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.CV_SEED)
    rows = []
    splits = list(outer.split(real_x, species))
    jobs = [(fold, condition) for fold in range(1, 6) for condition in conditions]
    contexts = {}
    for fold, condition in progress(jobs, "Crossfit: fold e condição", show_progress):
        train, holdout = splits[fold - 1]
        if fold not in contexts:
            contexts[fold] = prepare_context(
                real_x[train], species[train], config.SEED + fold
            )
        context = contexts[fold]
        result = run_condition(
            context,
            config.SEED + fold,
            condition,
            **condition_options(condition, settings, pop_size=10),
        )
        combined = np.vstack((real_x[holdout], result["synthetic_x"]))
        origin = np.r_[
            np.ones(len(holdout), dtype=int),
            np.zeros(len(result["synthetic_x"]), dtype=int),
        ]
        labels = np.r_[species[holdout], result["synthetic_species"]]
        strata = np.array([f"{o}_{t}" for o, t in zip(origin, labels)])
        inner = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=config.CV_SEED + fold
        )
        model = RandomForestClassifier(
            n_estimators=settings["evaluation"]["n_estimators"],
            min_samples_leaf=2,
            random_state=config.CV_SEED,
            n_jobs=1,
        )
        scores = cross_validate(
            model,
            combined,
            origin,
            cv=list(inner.split(combined, strata)),
            scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"},
            n_jobs=1,
        )
        rows.append(
            dict(
                fold=fold,
                condition=condition,
                profile=settings["profile"],
                accuracy=float(scores["test_accuracy"].mean()),
                roc_auc=float(scores["test_roc_auc"].mean()),
                train_indices=tuple(train.tolist()),
                holdout_indices=tuple(holdout.tolist()),
                generation_seconds=result["metadata"]["seconds"],
            )
        )
    return pd.DataFrame(rows)


def iris_selection_ablation(
    context, real_x, species, settings=None, show_progress=None
):
    """Só duas políticas balanceadas na Iris; global pertence ao laboratório 2D."""
    settings = execution_settings() if settings is None else settings
    rows = []
    for policy in progress(
        ["task", "task_niching"], "Ablação Iris: seleção", show_progress
    ):
        result = run_condition(
            context,
            config.SEED,
            "Híbrido 50/50",
            **condition_options("Híbrido 50/50", settings, selection_policy=policy),
        )
        table = evaluate_discriminators(
            real_x,
            species,
            result["synthetic_x"],
            result["synthetic_species"],
            **settings["evaluation"],
        )
        rows.append(
            dict(
                policy=policy,
                profile=settings["profile"],
                rf_accuracy=table.loc["Random Forest", "accuracy_mean"],
                rf_roc_auc=table.loc["Random Forest", "roc_auc_mean"],
            )
        )
    return pd.DataFrame(rows)


def sensitivity_suite(context, real_x, species, settings=None, show_progress=None):
    """Sensibilidade pareada: uma alteração por execução, com orçamento do perfil."""
    settings = execution_settings() if settings is None else settings
    cases = [(f"SBX eta={eta}", "SBX", {"sbx_eta": eta}) for eta in (5, 30, 50)]
    cases += [(f"Híbrido RMP={rmp}", "Híbrido 50/50", {"rmp": rmp}) for rmp in (0, 0.2)]
    rows = []
    for label, condition, options in progress(cases, "Sensibilidade", show_progress):
        result = run_condition(
            context,
            config.SEED,
            condition,
            **condition_options(condition, settings, **options),
        )
        table = evaluate_discriminators(
            real_x,
            species,
            result["synthetic_x"],
            result["synthetic_species"],
            **settings["evaluation"],
        )
        rows.append(
            dict(
                case=label,
                profile=settings["profile"],
                rf_accuracy=table.loc["Random Forest", "accuracy_mean"],
                rf_roc_auc=table.loc["Random Forest", "roc_auc_mean"],
            )
        )
    return pd.DataFrame(rows)
