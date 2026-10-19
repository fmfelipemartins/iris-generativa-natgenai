"""Laboratórios 2D: demonstrações probabilísticas e controles de seleção.

Gaussianas seguem os parâmetros do artigo. Anel/espiral são reconstruções
ilustrativas, não reprodução exata das figuras sem dados originais.
"""

import random

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler

from . import config
from .data import prepare_context
from .evolution import select_next_population
from .fitness import generative_fitness
from .gaussian import fit_gaussian, gaussian_product, sample_components
from .operators import (
    build_pygad_operator_engine,
    fit_gene_kdes,
    obscan_kde_child,
    pygad_sbx_child,
)
from .progress import progress


def make_scenario(name, size=config.BENCHMARK_SIZE, seed=config.SEED):
    rng = np.random.default_rng(seed)
    if size < 6:
        raise ValueError("Cenário requer ao menos seis pontos por tarefa.")
    if name == "anisotropic":
        return {
            0: rng.multivariate_normal([2, 8], [[0.2, 0], [0, 1]], size),
            1: rng.multivariate_normal([8, 2], [[1, 0], [0, 0.2]], size),
        }
    if name == "geometry":
        angle = rng.uniform(0, 2 * np.pi, size)
        ring = np.column_stack((np.cos(angle), np.sin(angle))) * (
            1 + rng.normal(0, 0.04, (size, 1))
        )
        angle = rng.uniform(0, 4 * np.pi, size)
        radius = 0.15 + angle / (4 * np.pi)
        spiral = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
        return {0: ring, 1: spiral + rng.normal(0, 0.025, (size, 2))}
    raise ValueError("Cenário desconhecido.")


def outside_hull(reference, points):
    hull = ConvexHull(reference)
    return float(
        np.mean(
            np.any(
                points @ hull.equations[:, :-1].T + hull.equations[:, -1] > 1e-9, axis=1
            )
        )
    )


def recombine_pool(populations, size, seed, p_sbx=0.5, eta=50):
    """Pool parental uniforme; nenhuma seleção por torneio nesta demonstração."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)
    tasks = sorted(populations)
    points = np.vstack([populations[t] for t in tasks])
    labels = np.concatenate([np.full(len(populations[t]), t) for t in tasks])
    span = np.maximum(np.ptp(points, axis=0), 1)
    ctx = {
        "global_low": points.min(axis=0) - span,
        "global_high": points.max(axis=0) + span,
    }
    engine = build_pygad_operator_engine(ctx, seed, 2, eta)
    kdes = fit_gene_kdes(populations)
    children, assigned, parents = [], [], []
    for _ in range(size):
        a, b = rng.integers(0, len(points), size=2)
        child = (
            pygad_sbx_child(points[a], points[b], engine, ctx)
            if rng.random() < p_sbx
            else obscan_kde_child(points[a], points[b], kdes, rng, ctx)
        )
        task = (
            labels[a] if labels[a] == labels[b] else rng.choice([labels[a], labels[b]])
        )
        children.append(child)
        assigned.append(task)
        parents.append((labels[a], labels[b]))
    return np.asarray(children), np.asarray(assigned), np.asarray(parents)


def run_benchmarks(
    size=config.BENCHMARK_SIZE,
    seed=config.SEED,
    vae_epochs=config.VAE_EPOCHS,
    show_progress=None,
):
    """Amostragem de uma geração; referência e sintéticos guardados para gráficos."""
    from .vae import VAEGenerator

    rows, samples = [], {}
    for scenario in progress(["geometry", "anisotropic"], "Cenários 2D", show_progress):
        parents = make_scenario(scenario, size, seed)
        combined = np.vstack(list(parents.values()))
        generated = {}
        for eta in (5, 50):
            generated[f"SBX eta={eta}"] = recombine_pool(
                parents, 2 * size, seed, 1, eta
            )[:2]
        generated["OB-Scan"] = recombine_pool(parents, 2 * size, seed, 0)[:2]
        generated["Híbrido"] = recombine_pool(parents, 2 * size, seed, 0.5)[:2]
        scaler = StandardScaler().fit(combined)
        scaled = {t: scaler.transform(x) for t, x in parents.items()}
        vae = VAEGenerator(2, [0, 1], seed).fit(scaled, vae_epochs, show_progress)
        vae_points = np.vstack([vae.sample(t, size)[0] for t in (0, 1)])
        generated["VAE condicional"] = (
            scaler.inverse_transform(vae_points),
            np.repeat([0, 1], size),
        )
        if scenario == "anisotropic":
            models = [fit_gaussian(parents[t]) for t in (0, 1)]
            rng = np.random.default_rng(seed)
            a = sample_components(models, [0.7, 0.3], size, rng)[0]
            b = sample_components(models, [0.5, 0.5], size, rng)[0]
            generated["Mistura gaussiana"] = np.vstack((a, b)), np.repeat([0, 1], size)
            mixed = models + [gaussian_product(models)]
            points = sample_components(mixed, [0.3, 0.3, 0.4], 2 * size, rng)[0]
            generated["Mistura + produto"] = points, np.repeat([0, 1], size)
        for method, (points, labels) in generated.items():
            rows.append(
                dict(
                    scenario=scenario,
                    method=method,
                    seed=seed,
                    outside_parent_hull=outside_hull(combined, points),
                    mean_displacement=float(
                        np.linalg.norm(points.mean(0) - combined.mean(0))
                    ),
                    covariance_trace=float(np.trace(np.cov(points, rowvar=False))),
                )
            )
        samples[scenario] = dict(
            parents=parents, generated=generated, vae_history=vae.history()
        )
    return pd.DataFrame(rows), samples


def select_policy(points, tasks, scores, context, policy, per_task, rng):
    """Mesmos scores por tarefa; global remove somente a reserva de vagas por tarefa."""
    if policy == "global":
        return np.argsort(scores)[-per_task * len(context["tasks"]) :]
    if policy not in {"task", "task_niching"}:
        raise ValueError("Política desconhecida.")
    selected = []
    for task in context["tasks"]:
        indices = np.flatnonzero(tasks == task)
        count = min(per_task, len(indices))
        if count == 0:
            continue
        if policy == "task":
            keep = np.argsort(scores[indices])[-count:]
        else:
            keep = select_next_population(
                points[indices],
                task,
                context,
                count,
                rng,
                moderated=True,
                return_indices=True,
            )
        selected.extend(indices[keep])
    return np.asarray(selected, dtype=int)


def selection_study(
    size=50,
    generations=config.BENCHMARK_GENERATIONS,
    seed=config.SEED,
    show_progress=None,
):
    """Trajetórias 2D e contrafactual das três políticas no MESMO conjunto candidato.

    Global pode extinguir tarefas: não se recriam indivíduos artificialmente.
    Fitness Iris é adaptada ao cenário; isto não reproduz o estudo aerodinâmico.
    """
    source = make_scenario("anisotropic", size, seed)
    x, y = np.vstack(list(source.values())), np.repeat([0, 1], size)
    context = prepare_context(x, y, seed)
    initial = {t: context["real_by_task"][t].copy() for t in context["tasks"]}
    trajectories, counterfactual = [], []
    policies = ("global", "task", "task_niching")
    for policy in progress(policies, "Políticas de seleção", show_progress):
        populations = {t: p.copy() for t, p in initial.items()}
        rng = np.random.default_rng(seed)
        for generation in range(1, generations + 1):
            children, child_tasks, origins = recombine_pool(
                populations, 2 * size, seed + generation
            )
            old = np.vstack(list(populations.values()))
            old_tasks = np.concatenate(
                [np.full(len(p), t) for t, p in populations.items()]
            )
            points, tasks = (
                np.vstack((old, children)),
                np.concatenate((old_tasks, child_tasks)),
            )
            scores = np.array(
                [
                    generative_fitness(p, int(t), context)[0]
                    for p, t in zip(points, tasks)
                ]
            )
            # Contrafactual compara decisões sem mudar os candidatos da trajetória corrente.
            for alternative in policies:
                keep = select_policy(
                    points, tasks, scores, context, alternative, size, rng
                )
                new = keep[keep >= len(old)] - len(old)
                counterfactual.append(
                    dict(
                        trajectory=policy,
                        generation=generation,
                        policy=alternative,
                        survivors=len(keep),
                        offspring_survivors=len(new),
                        cross_task_survivors=int(
                            np.sum(origins[new, 0] != origins[new, 1])
                        ),
                        task_0=int(np.sum(tasks[keep] == 0)),
                        task_1=int(np.sum(tasks[keep] == 1)),
                    )
                )
            keep = select_policy(points, tasks, scores, context, policy, size, rng)
            populations = {
                t: points[keep][tasks[keep] == t]
                for t in context["tasks"]
                if np.any(tasks[keep] == t)
            }
            trajectories.append(
                dict(
                    policy=policy,
                    generation=generation,
                    fitness_mean=float(scores[keep].mean()),
                    task_0=int(np.sum(tasks[keep] == 0)),
                    task_1=int(np.sum(tasks[keep] == 1)),
                )
            )
    return pd.DataFrame(trajectories), pd.DataFrame(counterfactual)


def plot_benchmarks(samples):
    import matplotlib.pyplot as plt

    figures = []
    for scenario, result in samples.items():
        methods = result["generated"]
        fig, axes = plt.subplots(
            int(np.ceil(len(methods) / 3)), 3, figsize=(12, 7), squeeze=False
        )
        all_points = np.vstack(
            [*result["parents"].values(), *[v[0] for v in methods.values()]]
        )
        low, high = all_points.min(0) - 0.2, all_points.max(0) + 0.2
        for ax, (method, (points, labels)) in zip(axes.flat, methods.items()):
            for task, parents in result["parents"].items():
                ax.scatter(*parents.T, color=f"C{task}", s=12, alpha=0.25)
                ax.scatter(
                    *points[labels == task].T, color=f"C{task}", s=12, marker="x"
                )
            ax.set(title=method, xlim=(low[0], high[0]), ylim=(low[1], high[1]))
        for ax in list(axes.flat)[len(methods) :]:
            ax.set_visible(False)
        fig.suptitle(f"{scenario}: pais transparentes; descendentes em x")
        fig.tight_layout()
        figures.append(fig)
    return figures
