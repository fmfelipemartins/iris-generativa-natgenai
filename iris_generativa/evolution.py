"""Motor evolutivo multitarefa da Iris Generativa."""

from __future__ import annotations

import random
from time import perf_counter

import numpy as np
import pandas as pd

from .config import (
    GENERATIONS,
    INITIAL_JITTER,
    POP_SIZE,
    RMP,
    SBX_ETA,
    TOURNAMENT_SIZE,
    SBX_MODE,
)
from .progress import progress
from .fitness import generative_fitness
from .operators import (
    build_pygad_operator_engine,
    fit_gene_kdes,
    obscan_kde_child,
    pygad_sbx_child,
)


def initialize_populations(
    ctx: dict,
    pop_size: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Inicializa uma população por tarefa a partir da base real padronizada."""
    rng = np.random.default_rng(seed)
    populations = {}

    for task in ctx["tasks"]:
        reference = ctx["real_by_task"][task]
        indices = rng.choice(
            len(reference),
            size=pop_size,
            replace=len(reference) < pop_size,
        )
        jitter = rng.normal(
            0.0,
            INITIAL_JITTER,
            size=(pop_size, reference.shape[1]),
        )
        populations[task] = np.clip(
            reference[indices] + jitter,
            ctx["global_low"],
            ctx["global_high"],
        )

    return populations


def tournament_parent(
    population: np.ndarray,
    task: int,
    ctx: dict,
    rng: np.random.Generator,
    tournament_size: int = TOURNAMENT_SIZE,
) -> np.ndarray:
    """Seleciona um pai por torneio."""
    indices = rng.choice(
        len(population),
        size=min(tournament_size, len(population)),
        replace=False,
    )
    scores = [generative_fitness(population[idx], task, ctx)[0] for idx in indices]
    winner = int(indices[int(np.argmax(scores))])

    return population[winner].copy()


def choose_parent_tasks(
    tasks: list[int],
    rng: np.random.Generator,
    rmp: float,
) -> tuple[int, int, int]:
    """Escolhe tarefas parentais e atribui a tarefa do descendente."""
    task_a = int(rng.choice(tasks))

    if len(tasks) > 1 and rng.random() < rmp:
        alternatives = [task for task in tasks if task != task_a]
        task_b = int(rng.choice(alternatives))
    else:
        task_b = task_a

    child_task = task_a if task_a == task_b else int(rng.choice([task_a, task_b]))

    return task_a, task_b, child_task


def select_next_population(
    candidates: np.ndarray,
    task: int,
    ctx: dict,
    pop_size: int,
    rng: np.random.Generator,
    moderated: bool = True,
    return_indices: bool = False,
) -> np.ndarray:
    """Seleciona a próxima população com ou sem moderação por nichos."""
    scores = np.array(
        [generative_fitness(point, task, ctx)[0] for point in candidates],
        dtype=float,
    )

    if not moderated:
        selected = np.argsort(scores)[-pop_size:]
        return selected if return_indices else candidates[selected].copy()

    model = ctx["niche_models"][task]
    quotas_real = ctx["niche_quotas"][task].astype(float)
    proportional = quotas_real / quotas_real.sum() * pop_size
    quotas = np.floor(proportional).astype(int)

    while quotas.sum() < pop_size:
        quotas[int(np.argmax(proportional - quotas))] += 1

    labels = model.predict(candidates)
    selected_indices = []

    for niche_idx, quota in enumerate(quotas):
        available = np.where(labels == niche_idx)[0]

        if len(available) == 0 or quota == 0:
            continue

        ranked = available[np.argsort(scores[available])[::-1]]
        selected_indices.extend(ranked[: min(quota, len(ranked))].tolist())

    selected_set = set(selected_indices)
    remaining = [idx for idx in np.argsort(scores)[::-1] if idx not in selected_set]
    selected_indices.extend(remaining[: pop_size - len(selected_indices)])
    selected_array = np.array(
        selected_indices[:pop_size],
        dtype=int,
    )

    return selected_array if return_indices else candidates[selected_array].copy()


def populations_to_synthetic(
    populations: dict[int, np.ndarray],
    ctx: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte as populações finais para os espaços padronizado e original."""
    synthetic_z = np.vstack([populations[task] for task in ctx["tasks"]])
    synthetic_x = ctx["scaler"].inverse_transform(synthetic_z)
    synthetic_species = np.concatenate(
        [np.full(len(populations[task]), task, dtype=int) for task in ctx["tasks"]]
    )

    return synthetic_z, synthetic_x, synthetic_species


def run_generator(
    ctx: dict,
    seed: int,
    p_sbx: float,
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    rmp: float = RMP,
    moderated: bool = True,
    use_polynomial_mutation: bool = False,
    sbx_eta: float = SBX_ETA,
    show_progress: bool | None = None,
    selection_policy: str | None = None,
    sbx_mode: str = SBX_MODE,
    tournament_size: int = TOURNAMENT_SIZE,  # Pressão seletiva explícita.
    record_diagnostics: bool = False,  # Observação opcional sem alterar RNG.
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Executa uma condição gerativa por um número fixo de gerações."""
    if selection_policy is not None:
        if selection_policy not in {"task", "task_niching"}:
            raise ValueError("Controle global disponível somente no laboratório 2D.")
        moderated = selection_policy == "task_niching"
    if tournament_size < 1 or not 0 <= rmp <= 1 or not 0 <= p_sbx <= 1:
        raise ValueError("Torneio ou probabilidades inválidos.")
    started = perf_counter()
    ctx = dict(ctx)
    ctx["_fitness_evaluations"] = 0
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)

    populations = initialize_populations(ctx, pop_size, seed)
    n_features = next(iter(populations.values())).shape[1]
    engine = build_pygad_operator_engine(
        ctx,
        seed,
        n_features=n_features,
        eta=sbx_eta,
    )

    if sbx_mode not in {"symmetric", "legacy_lower"}:
        raise ValueError("Modo SBX inválido.")
    engine.iris_sbx_mode = sbx_mode

    history = []
    births = {
        t: np.zeros(len(p), dtype=int) for t, p in populations.items()
    }  # Geração de origem.
    diagnostic_rows, events, final_candidates = [], [], {}  # Saídas observacionais.
    from .diagnostics import (
        population_metrics,
    )  # Importação evita dependência circular.

    for generation in progress(range(1, generations + 1), "Evolução", show_progress):
        kde_models = fit_gene_kdes(populations)
        offspring_by_task = {task: [] for task in ctx["tasks"]}

        generated_sbx = 0
        generated_obscan = 0
        total_offspring = pop_size * len(ctx["tasks"])

        for _ in range(total_offspring):
            task_a, task_b, child_task = choose_parent_tasks(
                ctx["tasks"],
                rng,
                rmp,
            )
            parent_a = tournament_parent(
                populations[task_a],
                task_a,
                ctx,
                rng,
                tournament_size=tournament_size,
            )
            parent_b = tournament_parent(
                populations[task_b],
                task_b,
                ctx,
                rng,
                tournament_size=tournament_size,
            )

            if rng.random() < p_sbx:
                child = pygad_sbx_child(
                    parent_a,
                    parent_b,
                    engine,
                    ctx,
                )
                generated_sbx += 1
            else:
                child = obscan_kde_child(
                    parent_a,
                    parent_b,
                    kde_models,
                    rng,
                    ctx,
                )
                generated_obscan += 1

            if use_polynomial_mutation:
                child = np.asarray(
                    engine.polynomial_mutation(child.reshape(1, -1))[0],
                    dtype=float,
                )
                child = np.clip(
                    child,
                    ctx["global_low"],
                    ctx["global_high"],
                )

            offspring_by_task[child_task].append(child)

        next_populations = {}
        survived_count = 0  # Descendentes aceitos nesta geração.

        for task in ctx["tasks"]:
            if offspring_by_task[task]:
                offspring = np.vstack(offspring_by_task[task])
                candidates = np.vstack([populations[task], offspring])
            else:
                candidates = populations[task].copy()

            indices = select_next_population(
                candidates,
                task,
                ctx,
                pop_size,
                rng,
                moderated=moderated,
                return_indices=True,
            )
            next_populations[task] = candidates[
                indices
            ].copy()  # Mesmos índices da seleção original.
            n_new = (
                len(candidates) - pop_size
            )  # Número variável por tarefa no crossover.
            origins = np.r_[
                births[task], np.full(n_new, generation)
            ]  # Mantém fundadores identificáveis.
            births[task] = origins[
                indices
            ]  # Acompanha somente indivíduos sobreviventes.
            survived = set(
                indices[indices >= pop_size] - pop_size
            )  # Índices dos novos aceitos.
            survived_count += len(survived)  # Renovação total.
            if record_diagnostics:
                for stage, points in (
                    ("candidates", candidates[pop_size:]),
                    ("survivors", next_populations[task]),
                ):
                    if len(points):
                        diagnostic_rows.append(
                            dict(
                                generation=generation,
                                task=task,
                                stage=stage,
                                **population_metrics(points, task, ctx),
                            )
                        )  # Sem RNG.
                events.extend(
                    dict(
                        generation=generation,
                        task=task,
                        candidate=i,
                        survived=i in survived,
                    )
                    for i in range(n_new)
                )  # Registro por proposta.
                if generation == generations:
                    final_candidates[task] = {
                        "clipped": candidates[pop_size:].copy()
                    }  # Operadores já aplicam limites internamente.

        populations = next_populations

        all_scores = [
            generative_fitness(point, task, ctx)[0]
            for task in ctx["tasks"]
            for point in populations[task]
        ]

        history.append(
            {
                "generation": generation,
                "fitness_evaluations": ctx["_fitness_evaluations"],
                "elapsed_seconds": perf_counter() - started,
                "fitness_mean": float(np.mean(all_scores)),
                "fitness_max": float(np.max(all_scores)),
                "generated_sbx": generated_sbx,
                "generated_obscan": generated_obscan,
                "survived_offspring": survived_count,  # Renovação observada.
                "founder_fraction": float(
                    np.mean(np.concatenate(list(births.values())) == 0)
                ),  # Não é objetivo de seleção.
            }
        )

    table = pd.DataFrame(history)  # Histórico mantém as colunas anteriores.
    table.attrs.update(
        diagnostics=pd.DataFrame(diagnostic_rows),
        offspring_events=pd.DataFrame(events),
        final_candidates=final_candidates,
    )  # Interface comum aos EDAs.
    return populations, table
