"""Testes metodológicos: referências analíticas, isolamento e compatibilidade."""

import importlib.util
import subprocess

import numpy as np
import pytest

from iris_generativa.data import load_iris_data, prepare_context
from iris_generativa.evolution import run_generator
from iris_generativa.gaussian import (
    GaussianGenerator,
    fit_gaussian,
    gaussian_product,
    sample_components,
)
from iris_generativa.generative import run_condition
from iris_generativa.benchmarks import select_policy, selection_study, run_benchmarks


@pytest.fixture
def iris():
    data = load_iris_data()
    return data, prepare_context(data["x_real"], data["y_species"], 42)


def test_normalized_product_analytic():
    models = [
        (np.array([0.0]), np.array([[4.0]])),
        (np.array([2.0]), np.array([[4.0]])),
    ]
    mu, cov = gaussian_product(models)
    np.testing.assert_allclose(mu, [1.0])
    np.testing.assert_allclose(cov, [[2.0]])
    points, _ = sample_components([(mu, cov)], [1.0], 40000, np.random.default_rng(3))
    assert abs(points.mean() - 1) < 0.03
    assert abs(points.var() - 2) < 0.05


def test_constant_covariance_and_invalid_weights():
    mu, cov = fit_gaussian(np.ones((4, 3)))
    assert np.linalg.eigvalsh(cov).min() > 0
    with pytest.raises(ValueError):
        sample_components([(mu, cov)], [0.4], 4, np.random.default_rng(0))


def test_all_product_components():
    model = GaussianGenerator("gaussian_product", include_all=True).fit(
        {i: np.random.default_rng(i).normal(size=(15, 4)) for i in range(3)}
    )
    assert len(model.components) == 7  # Três individuais, três pares e um triplo.
    x, origins = model.sample(1, 200, np.random.default_rng(7))
    assert x.shape == (200, 4) and np.isfinite(x).all()
    assert any(len(o) > 1 for o in origins)


def test_original_operators_regression(iris, tmp_path):
    _, ctx = iris
    source = subprocess.check_output(
        ["git", "show", "ff2659f:iris_generativa/evolution.py"], text=True
    )
    path = tmp_path / "baseline.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("iris_generativa.baseline", path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    from iris_generativa.operators import pygad_sbx_child

    def legacy_child(a, b, engine, context):
        engine.iris_sbx_mode = "legacy_lower"
        return pygad_sbx_child(a, b, engine, context)

    baseline.pygad_sbx_child = legacy_child
    for probability in (0.0, 0.5, 1.0):
        old, old_history = baseline.run_generator(
            ctx, 42, probability, pop_size=10, generations=2
        )
        new, history = run_generator(
            ctx,
            42,
            probability,
            pop_size=10,
            generations=2,
            show_progress=False,
            sbx_mode="legacy_lower",
        )
        for task in old:
            np.testing.assert_array_equal(new[task], old[task])
        np.testing.assert_array_equal(history[old_history.columns], old_history)


@pytest.mark.parametrize("method", ["gaussian_mixture", "gaussian_product", "vae_eda"])
def test_new_generators_balanced_and_repeatable(iris, method):
    _, ctx = iris
    options = dict(pop_size=10, generations=2, show_progress=False)
    if method == "vae_eda":
        options.update(vae_epochs=2)
    a = run_condition(ctx, 9, method, **options)
    b = run_condition(ctx, 9, method, **options)
    np.testing.assert_array_equal(a["synthetic_z"], b["synthetic_z"])
    assert a["synthetic_z"].shape == (30, 4)
    assert np.all(np.bincount(a["synthetic_species"]) == 10)
    assert np.isfinite(a["synthetic_z"]).all()
    assert (a["synthetic_z"] >= ctx["global_low"]).all()
    assert (a["synthetic_z"] <= ctx["global_high"]).all()
    assert "_fitness_evaluations" not in ctx
    assert len(a["offspring_events"]) == 60


def test_vae_learns_and_preserves_global_rng():
    import torch
    from iris_generativa.vae import VAEGenerator

    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    model = VAEGenerator(2, [0, 1], 42, latent=2, hidden=16, learning_rate=0.01)
    assert torch.equal(before, torch.get_rng_state())
    populations = {0: np.tile([-1.0, -1.0], (20, 1)), 1: np.tile([1.0, 1.0], (20, 1))}
    model.fit(populations, epochs=40)
    history = model.history()
    assert history.tail(5).reconstruction.mean() < history.head(5).reconstruction.mean()
    assert torch.equal(before, torch.get_rng_state())
    assert np.isfinite(model.sample(0, 10)[0]).all()


def test_progress_does_not_change_results(iris):
    _, ctx = iris
    a, _ = run_generator(ctx, 42, 0.5, pop_size=10, generations=1, show_progress=False)
    b, _ = run_generator(ctx, 42, 0.5, pop_size=10, generations=1, show_progress=True)
    for task in a:
        np.testing.assert_array_equal(a[task], b[task])


def test_selection_budget_and_global_loss(iris):
    _, ctx = iris
    points = np.vstack([ctx["real_by_task"][t][:10] for t in range(3)])
    labels = np.repeat([0, 1, 2], 10)
    scores = np.arange(30.0)
    rng = np.random.default_rng(42)
    global_keep = select_policy(points, labels, scores, ctx, "global", 3, rng)
    assert np.all(labels[global_keep] == 2)
    for policy in ("task", "task_niching"):
        keep = select_policy(points, labels, scores, ctx, policy, 3, rng)
        np.testing.assert_array_equal(np.bincount(labels[keep]), [3, 3, 3])


def test_benchmark_and_selection_outputs():
    table, samples = run_benchmarks(size=12, vae_epochs=1, show_progress=False)
    assert len(table) == 12 and set(samples) == {"geometry", "anisotropic"}
    assert table.outside_parent_hull.between(0, 1).all()
    trajectory, counterfactual = selection_study(
        size=12, generations=2, show_progress=False
    )
    assert len(trajectory) == 6 and len(counterfactual) == 18
    assert np.all(trajectory.task_0 + trajectory.task_1 == 24)


def test_crossfit_excludes_holdout(iris, monkeypatch):
    from iris_generativa import extended_experiments as module

    data, _ = iris
    observed = []
    original = module.run_condition

    def capture(context, seed, condition, **options):
        observed.append((context["scaler"].mean_.copy(), len(context["Z_real"])))
        return original(context, seed, condition, **options)

    monkeypatch.setattr(module, "run_condition", capture)
    settings = module.execution_settings("quick")
    settings.update(generations=1, vae_epochs=1)
    result = module.crossfit_suite(
        data["x_real"],
        data["y_species"],
        conditions=["gaussian_mixture", "vae_eda"],
        settings=settings,
        show_progress=False,
    )
    assert len(result) == 10
    for index, row in result.iterrows():
        assert set(row.train_indices).isdisjoint(row.holdout_indices)
        assert observed[index][1] == 120
        np.testing.assert_allclose(
            observed[index][0], data["x_real"][list(row.train_indices)].mean(0)
        )
    holdouts = [
        set(r.holdout_indices)
        for _, r in result[result.condition == "vae_eda"].iterrows()
    ]
    assert len(set.union(*holdouts)) == 150
