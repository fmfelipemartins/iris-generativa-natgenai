"""Reproduz a auditoria e as ablações exploratórias, sem escolher defaults pelo holdout."""

from pathlib import Path
import json
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iris_generativa import config
from iris_generativa.data import load_iris_data, prepare_context
from iris_generativa.operators import build_pygad_operator_engine, pygad_sbx_child
from iris_generativa.generative import run_condition
from iris_generativa.extended_experiments import (
    execution_settings,
    run_metadata,
    crossfit_suite,
    sensitivity_suite,
    iris_selection_ablation,
)
from iris_generativa.evaluation import evaluate_discriminators, summarize_generation
from iris_generativa.benchmarks import run_benchmarks, selection_study, plot_benchmarks
from iris_generativa.progress import progress


@contextmanager
def output_lock(output):
    """Impede dois escritores na mesma pasta, inclusive entre fases distintas."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".review.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"Pasta em uso: {path}. Se houve interrupção abrupta, confirme que o processo terminou antes de remover o lock."
        ) from error
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def audit_sbx():
    rows = []
    for eta in (5, 50):
        for mode in ("legacy_lower", "symmetric"):
            for reverse in (False, True):
                ctx = dict(
                    global_low=np.array([-5.0, -10.0]),
                    global_high=np.array([5.0, 10.0]),
                )
                engine = build_pygad_operator_engine(ctx, 42, 2, eta)
                engine.iris_sbx_mode = mode
                a, b = np.array([-1.0, 2.0]), np.array([1.0, -2.0])
                if reverse:
                    a, b = b, a
                x = np.array([pygad_sbx_child(a, b, engine, ctx) for _ in range(10000)])
                rows.append(
                    dict(
                        eta=eta,
                        mode=mode,
                        reversed=reverse,
                        mean_x=x[:, 0].mean(),
                        mean_y=x[:, 1].mean(),
                        positive_x=(x[:, 0] > 0).mean(),
                        positive_y=(x[:, 1] > 0).mean(),
                    )
                )
    return pd.DataFrame(rows)


def sbx_job(job, settings, ctx, x, y):
    seed, condition, mode = job
    r = run_condition(
        ctx,
        seed,
        condition,
        generations=settings["generations"],
        sbx_mode=mode,
        show_progress=False,
    )
    scores = evaluate_discriminators(
        x, y, r["synthetic_x"], r["synthetic_species"], **settings["evaluation"]
    )
    summary, _ = summarize_generation({condition: r}, ctx, x, y)
    return dict(
        seed=seed,
        condition=condition,
        mode=mode,
        accuracy=scores.loc["Random Forest", "accuracy_mean"],
        auc=scores.loc["Random Forest", "roc_auc_mean"],
        **summary.iloc[0].to_dict(),
    )


def run(output, profile="full", phase="all", workers=1, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"COMPLETE_{phase}").unlink(missing_ok=True)
    settings = execution_settings(profile)
    metadata_path = output / f"metadata_{phase}.json"
    if resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["settings"] != json.loads(json.dumps(settings)) or previous[
            "configuration"
        ] != json.loads(json.dumps(run_metadata(settings)["configuration"])):
            raise ValueError("Resume exige o mesmo perfil, orçamento e configuração.")
    (
        output
        / (f"metadata_{phase}_resume.json" if resume else f"metadata_{phase}.json")
    ).write_text(json.dumps(run_metadata(settings), indent=2, ensure_ascii=False))
    audit_sbx().to_csv(output / "sbx_controlled.csv", index=False)
    data = load_iris_data()
    x, y = data["x_real"], data["y_species"]
    ctx = prepare_context(x, y, 42)
    if phase in ("all", "sbx"):
        # A correção afeta SBX e híbrido: repetir 20 seeds no full, com modos pareados.
        rows = []
        jobs = [
            (seed, c, m)
            for seed in settings["seeds"]
            for c in ("SBX", "Híbrido 50/50")
            for m in ("legacy_lower", "symmetric")
        ]
        checkpoint = output / "sbx_paired.csv"
        if resume and checkpoint.exists():
            rows = pd.read_csv(checkpoint).to_dict("records")
            done = {(r["seed"], r["condition"], r["mode"]) for r in rows}
            jobs = [job for job in jobs if job not in done]

        def save(row):
            rows.append(row)
            temporary = checkpoint.with_suffix(".tmp")
            pd.DataFrame(rows).sort_values(["seed", "condition", "mode"]).to_csv(
                temporary, index=False
            )
            temporary.replace(checkpoint)

        if workers == 1:
            for job in progress(jobs, "SBX: antes/depois"):
                save(sbx_job(job, settings, ctx, x, y))
        else:
            # Cada processo possui seu próprio RNG; nunca paralelizar essas funções em threads.
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(sbx_job, job, settings, ctx, x, y) for job in jobs
                ]
                for future in progress(
                    as_completed(futures), "SBX: antes/depois", total=len(futures)
                ):
                    save(future.result())
        expected = {
            (seed, c, m)
            for seed in settings["seeds"]
            for c in ("SBX", "Híbrido 50/50")
            for m in ("legacy_lower", "symmetric")
        }
        keys = [(r["seed"], r["condition"], r["mode"]) for r in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise RuntimeError("Checkpoint SBX incompleto ou com duplicatas.")
        pd.DataFrame(rows).groupby(["condition", "mode"])[["accuracy", "auc"]].agg(
            ["mean", "std"]
        ).to_csv(output / "sbx_summary.csv")
    if phase in ("all", "vae"):
        # Quatro variantes OFAT predefinidas; nunca usar crossfit para escolher hiperparâmetros.
        variants = {
            "reference": {},
            "beta_0.1": {"model_options": {"beta": 0.1}},
            "latent_8": {"model_options": {"latent": 8}},
            "parents_1.0": {"parent_fraction": 1.0},
        }
        rows = []
        for seed in progress(config.VAE_ABLATION_SEEDS, "VAE: seeds exploratórias"):
            for label, options in variants.items():
                r = run_condition(
                    ctx,
                    seed,
                    "vae_eda",
                    generations=settings["generations"],
                    vae_epochs=settings["vae_epochs"],
                    show_progress=False,
                    **options,
                )
                stem = f"vae_{label}_seed{seed}"
                for key in (
                    "history",
                    "training_history",
                    "diagnostics",
                    "latent_diagnostics",
                ):
                    r[key].to_csv(output / f"{stem}_{key}.csv", index=False)
                z = np.vstack(
                    [r["final_candidates"][t]["clipped"] for t in ctx["tasks"]]
                )
                candidates = ctx["scaler"].inverse_transform(z)
                for stage, points in (
                    ("last_candidates", candidates),
                    ("survivors", r["synthetic_x"]),
                ):
                    scores = evaluate_discriminators(
                        x, y, points, r["synthetic_species"], **settings["evaluation"]
                    )
                    rows.append(
                        dict(
                            seed=seed,
                            variant=label,
                            stage=stage,
                            accuracy=scores.loc["Random Forest", "accuracy_mean"],
                            auc=scores.loc["Random Forest", "roc_auc_mean"],
                            founder_fraction=r["history"].iloc[-1].founder_fraction,
                        )
                    )
                pd.DataFrame(rows).to_csv(output / "vae_ablation.csv", index=False)
    if phase in ("all", "benchmarks"):
        crossfit_suite(
            x, y, conditions=["SBX", "Híbrido 50/50"], settings=settings
        ).to_csv(output / "crossfit_corrected.csv", index=False)
        metrics, samples = run_benchmarks(
            size=settings["benchmark_size"], vae_epochs=settings["vae_epochs"]
        )
        metrics.to_csv(output / "benchmarks.csv", index=False)
        for i, fig in enumerate(plot_benchmarks(samples)):
            fig.savefig(output / f"benchmark_{i}.png", dpi=120, bbox_inches="tight")
        a, b = selection_study(
            size=settings["benchmark_size"],
            generations=settings["benchmark_generations"],
        )
        a.to_csv(output / "selection_trajectories.csv", index=False)
        b.to_csv(output / "selection_decisions.csv", index=False)
    if phase in ("all", "controls"):
        sensitivity_suite(ctx, x, y, settings).to_csv(
            output / "sensitivity_corrected.csv", index=False
        )
        iris_selection_ablation(ctx, x, y, settings).to_csv(
            output / "iris_selection_corrected.csv", index=False
        )
        polynomial = run_condition(
            ctx,
            config.SEED,
            "SBX",
            generations=settings["generations"],
            use_polynomial_mutation=True,
            show_progress=False,
        )
        evaluate_discriminators(
            x,
            y,
            polynomial["synthetic_x"],
            polynomial["synthetic_species"],
            **settings["evaluation"],
        ).to_csv(output / "polynomial_corrected.csv")
    (output / f"COMPLETE_{phase}").write_text(
        "Todos os protocolos deste script concluídos.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--output", default="results/review_full_20261019")
    parser.add_argument(
        "--phase",
        choices=["all", "sbx", "vae", "benchmarks", "controls"],
        default="all",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with output_lock(args.output):
        run(args.output, args.profile, args.phase, args.workers, args.resume)
