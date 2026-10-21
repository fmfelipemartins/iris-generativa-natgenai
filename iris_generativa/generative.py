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
    parent_policy="top",  # top mantém a referência; niching preserva cotas parentais.
    boundary_policy="clip",  # Alternativa experimental: reamostragem limitada.
    fixed_parents=False,  # Controle VAE: congela os pais da primeira geração.
    resample_attempts=5,  # Limite de novas tentativas por candidato inválido.
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
    if (
        parent_policy not in {"top", "niching"}
        or boundary_policy not in {"clip", "resample"}
        or resample_attempts < 0
    ):
        raise ValueError("Política experimental inválida.")
    ctx = dict(ctx)
    ctx["_fitness_evaluations"] = 0
    rng = np.random.default_rng(seed)
    populations = initialize_populations(ctx, pop_size, seed)
    options = dict(model_options or {})
    model, history, training, events = None, [], [], []
    diagnostic_rows, latent_rows = [], []
    birth_generations = {t: np.zeros(len(p), dtype=int) for t, p in populations.items()}
    final_candidates = {}
    fixed_training = None  # Mesmo tamanho parental e orçamento de atualizações.
    started = perf_counter()
    for generation in progress(range(1, generations + 1), method, show_progress):
        selected = {}
        for task, points in populations.items():
            scores = [generative_fitness(x, task, ctx)[0] for x in points]
            count = max(2, int(np.ceil(parent_fraction * len(points))))
            selected[task] = (
                points[np.argsort(scores)[-count:]]
                if parent_policy == "top"
                else select_next_population(points, task, ctx, count, rng)
            )  # Cotas nos nichos já existentes.
        if fixed_parents:
            if fixed_training is None:
                fixed_training = {t: p.copy() for t, p in selected.items()}  # Congela sem RNG adicional.
            selected = fixed_training  # Contraste isolado com pais evolutivos.
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
        clipped_count, survivor_count, resample_count = 0, 0, 0
        for task in ctx["tasks"]:
            raw, origins = model.sample(task, pop_size, rng)
            if not np.isfinite(raw).all():
                raise FloatingPointError("Amostragem produziu valores não finitos.")
            original_raw = (
                raw.copy()
            )  # Preserva o diagnóstico antes da correção de limites.
            resampled = 0  # Conta todas as propostas adicionais, inclusive rejeitadas.
            if boundary_policy == "resample":
                for attempt in range(resample_attempts):
                    invalid = np.any(
                        (raw < ctx["global_low"]) | (raw > ctx["global_high"]), axis=1
                    )  # Candidatos fora da caixa.
                    if not invalid.any():
                        break  # Não consome RNG quando todos estão dentro dos limites.
                    replacements, replacement_origins = model.sample(
                        task, int(invalid.sum()), rng
                    )  # Reamostra o modelo.
                    if not np.isfinite(replacements).all():
                        raise FloatingPointError("Reamostragem não finita.")
                    raw[invalid] = replacements  # Substitui apenas propostas inválidas.
                    for index, origin in zip(
                        np.flatnonzero(invalid), replacement_origins
                    ):
                        origins[index] = origin  # Mantém a origem do candidato efetivo.
                    resampled += int(invalid.sum())  # Registra o custo extra.
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
                    ("raw_candidates", original_raw),
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
                        raw=original_raw.copy(), clipped=offspring.copy()
                    )
            resample_count += resampled  # Propostas adicionais por geração.
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
                resampled=resample_count,  # Custo da política opcional de limites.
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
    context = dict(context)  # Não altera o contexto compartilhado entre experimentos.
    options = dict(options)  # Evita mutação das opções fornecidas pelo chamador.
    survivor_policy = options.pop("survivor_policy", "deterministic")  # Opção comum aos motores.
    elite = options.pop("survivor_elite_fraction", 0.1)  # Fração elitista dentro da cota.
    if survivor_policy not in {"deterministic", "rank"} or not 0 <= elite <= 1:
        raise ValueError("Política de sobrevivência inválida.")
    context["survivor_policy"] = survivor_policy  # Cópia isolada por execução.
    context["survivor_elite_fraction"] = elite  # Não altera o contexto de outras variantes.
    novelty_policy = options.pop("novelty_policy", "target")  # Alvo histórico ou intervalo.
    if novelty_policy not in {"target", "band"}:
        raise ValueError("Política de novidade inválida.")
    context["novelty_policy"] = novelty_policy  # Quantis calculados exclusivamente no treino.
    weights = options.pop("fitness_weights", None)  # Sobrescrita explícita do fitness.
    if weights is not None:
        if (
            set(weights) != set(config.FITNESS_WEIGHTS)
            or any(not np.isfinite(v) or v < 0 for v in weights.values())
            or not np.isclose(sum(weights.values()), 1)
        ):
            raise ValueError(
                "Pesos do fitness devem ser finitos, não negativos e somar 1."
            )
        context["fitness_weights"] = dict(weights)  # Isolamento por execução.
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
