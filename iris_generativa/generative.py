"""Interface comum e ciclo EDA; cada chamada isola modelos e dados de ajuste."""

from time import perf_counter

import numpy as np
import pandas as pd

from . import config
from .evolution import (
    initialize_populations,
    populations_to_synthetic,
    run_generator,
    select_next_population,
)
from .fitness import generative_fitness
from .gaussian import GaussianGenerator
from .progress import progress
from .diagnostics import population_metrics


ALL_CONDITIONS = {
    **config.MAIN_CONDITIONS,
    **{name: name for name in config.GENERATIVE_METHODS},
}


def run_eda(
    ctx,
    seed,
    method,
    pop_size=config.POP_SIZE,
    generations=config.GENERATIONS,
    selection_policy=config.SELECTION_POLICY,
    parent_fraction=config.EDA_PARENT_FRACTION,
    vae_epochs=config.VAE_EPOCHS,
    warm_start=config.VAE_WARM_START,
    refit_every=config.VAE_REFIT_EVERY,
    show_progress=None,
    model_options=None,
    record_diagnostics=config.RECORD_DIAGNOSTICS,
):
    """Atualiza modelos nos melhores pais de cada tarefa e amostra uma nova geração."""
    if method not in config.GENERATIVE_METHODS or selection_policy not in {
        "task",
        "task_niching",
    }:
        raise ValueError("Método/política incompatível com protocolo balanceado Iris.")
    if (
        pop_size < 2
        or generations < 1
        or not 0 < parent_fraction <= 1
        or refit_every < 1
    ):
        raise ValueError("Orçamento ou fração parental inválidos.")
    ctx = dict(ctx)
    ctx["_fitness_evaluations"] = 0
    rng = np.random.default_rng(seed)
    populations = initialize_populations(ctx, pop_size, seed)
    options = dict(model_options or {})
    model, history, training, events = None, [], [], []
    diagnostic_rows, latent_rows = [], []
    birth_generations = {t: np.zeros(len(p), dtype=int) for t, p in populations.items()}
    final_candidates = {}
    started = perf_counter()
    for generation in progress(range(1, generations + 1), method, show_progress):
        selected = {}
        for task, points in populations.items():
            scores = [generative_fitness(x, task, ctx)[0] for x in points]
            count = max(2, int(np.ceil(parent_fraction * len(points))))
            selected[task] = points[np.argsort(scores)[-count:]]
        should_fit = (
            model is None or (generation - 1) % refit_every == 0 or method != "vae_eda"
        )
        if method == "vae_eda":
            from .vae import VAEGenerator

            if model is None or (should_fit and not warm_start):
                model = VAEGenerator(
                    selected[ctx["tasks"][0]].shape[1],
                    ctx["tasks"],
                    seed if warm_start else seed + generation - 1,
                    **options,
                )
            if should_fit:
                offset = len(model.training_history)
                model.fit(
                    selected,
                    epochs=vae_epochs,
                    show_progress=show_progress,
                    generation=generation,
                )
                training.extend(model.training_history[offset:])
        else:
            defaults = dict(
                own_weight=config.GAUSSIAN_OWN_WEIGHT,
                product_weight=config.GAUSSIAN_PRODUCT_WEIGHT,
                include_all=config.GAUSSIAN_INCLUDE_ALL_PRODUCT,
                covariance=config.GAUSSIAN_COVARIANCE,
                regularization=config.GAUSSIAN_REG,
            )
            model = GaussianGenerator(method, **(defaults | options)).fit(selected)
        if record_diagnostics and method == "vae_eda":
            latent_rows.extend(
                dict(generation=generation, **row) for row in model.diagnose(selected)
            )
        next_populations = {}
        clipped_count, survivor_count = 0, 0
        for task in ctx["tasks"]:
            raw, origins = model.sample(task, pop_size, rng)
            if not np.isfinite(raw).all():
                raise FloatingPointError("Amostragem produziu valores não finitos.")
            offspring = np.clip(raw, ctx["global_low"], ctx["global_high"])
            clipped = np.any(raw != offspring, axis=1)
            candidates = np.vstack((populations[task], offspring))
            indices = select_next_population(
                candidates,
                task,
                ctx,
                pop_size,
                rng,
                moderated=selection_policy == "task_niching",
                return_indices=True,
            )
            survivors = set(indices[indices >= pop_size] - pop_size)
            next_populations[task] = candidates[indices].copy()
            births = np.r_[birth_generations[task], np.full(pop_size, generation)]
            birth_generations[task] = births[indices]
            if record_diagnostics:
                for stage, stage_points in (
                    ("selected_parents", selected[task]),
                    ("raw_candidates", raw),
                    ("clipped_candidates", offspring),
                    ("survivors", next_populations[task]),
                ):
                    diagnostic_rows.append(
                        dict(
                            generation=generation,
                            task=task,
                            stage=stage,
                            **population_metrics(stage_points, task, ctx),
                        )
                    )
                if generation == generations:
                    final_candidates[task] = dict(
                        raw=raw.copy(), clipped=offspring.copy()
                    )
            clipped_count += int(clipped.sum())
            survivor_count += len(survivors)
            for index, origin in enumerate(origins):
                events.append(
                    dict(
                        generation=generation,
                        task=task,
                        candidate=index,
                        component="+".join(map(str, origin)),
                        survived=index in survivors,
                        clipped=bool(clipped[index]),
                    )
                )
        populations = next_populations
        scores = [
            generative_fitness(x, t, ctx)[0]
            for t, pop in populations.items()
            for x in pop
        ]
        history.append(
            dict(
                generation=generation,
                fitness_mean=float(np.mean(scores)),
                fitness_max=float(np.max(scores)),
                generated_sbx=0,
                generated_obscan=0,
                generated_model=pop_size * len(populations),
                survived_offspring=survivor_count,
                clipped=clipped_count,
                founder_fraction=float(
                    np.mean(np.concatenate(list(birth_generations.values())) == 0)
                ),
                fitness_evaluations=ctx["_fitness_evaluations"],
                elapsed_seconds=perf_counter() - started,
            )
        )
    table = pd.DataFrame(history)
    table.attrs["training_history"] = pd.DataFrame(training)
    table.attrs["offspring_events"] = pd.DataFrame(events)
    table.attrs["diagnostics"] = pd.DataFrame(diagnostic_rows)
    table.attrs["latent_diagnostics"] = pd.DataFrame(latent_rows)
    table.attrs["final_candidates"] = final_candidates
    return populations, table


def run_condition(context, seed, condition, **options):
    """Normaliza saídas; opções são explícitas e inválidas não são descartadas."""
    started = perf_counter()
    if condition in config.MAIN_CONDITIONS:
        populations, history = run_generator(
            context, seed, config.MAIN_CONDITIONS[condition], **options
        )
    elif condition in config.GENERATIVE_METHODS:
        populations, history = run_eda(context, seed, condition, **options)
    else:
        raise ValueError(f"Condição desconhecida: {condition}")
    z, x, species = populations_to_synthetic(populations, context)
    return dict(
        populations=populations,
        history=history,
        synthetic_z=z,
        synthetic_x=x,
        synthetic_species=species,
        training_history=history.attrs.get("training_history", pd.DataFrame()),
        offspring_events=history.attrs.get("offspring_events", pd.DataFrame()),
        diagnostics=history.attrs.get("diagnostics", pd.DataFrame()),
        latent_diagnostics=history.attrs.get("latent_diagnostics", pd.DataFrame()),
        final_candidates=history.attrs.get("final_candidates", {}),
        metadata=dict(
            seed=seed,
            condition=condition,
            options=options,
            seconds=perf_counter() - started,
        ),
    )
