"""Propriedades do SBX e neutralidade das medições gerativas."""

import numpy as np
import pytest
from iris_generativa.operators import build_pygad_operator_engine, pygad_sbx_child
from iris_generativa.data import load_iris_data, prepare_context
from iris_generativa.generative import run_condition


def sample(mode, a, b, size=6000, seed=19):
    ctx = dict(global_low=np.array([-5.0, -10.0]), global_high=np.array([5.0, 10.0]))
    engine = build_pygad_operator_engine(ctx, seed, 2, 5)
    engine.iris_sbx_mode = mode
    return np.array([pygad_sbx_child(a, b, engine, ctx) for _ in range(size)])


def test_sbx_symmetry_parent_order_and_reflection():
    a, b = np.array([-1.0, 2.0]), np.array([1.0, -2.0])
    x = sample("symmetric", a, b)
    np.testing.assert_array_equal(x, sample("symmetric", b, a))
    assert np.all(np.abs(x.mean(0)) < 0.08)
    assert np.all(np.abs((x > 0).mean(0) - 0.5) < 0.025)
    # Simetria conjunta: a seleção do ramo é independente por gene.
    assert np.mean((x[:, 0] > 0) & (x[:, 1] > 0)) == pytest.approx(0.25, abs=0.025)
    assert np.all(sample("legacy_lower", a, b, 100) < 0)


def test_sbx_equal_parents_bounds_and_repeatability():
    a = np.array([1.0, -2.0])
    np.testing.assert_array_equal(sample("symmetric", a, a, 3), np.tile(a, (3, 1)))
    x = sample("symmetric", np.array([-5.0, -10.0]), np.array([5.0, 10.0]), 100)
    assert np.all(x >= [-5.0, -10.0]) and np.all(x <= [5.0, 10.0])
    np.testing.assert_array_equal(
        x, sample("symmetric", np.array([-5.0, -10.0]), np.array([5.0, 10.0]), 100)
    )


def test_diagnostics_leave_vae_rng_and_population_unchanged():
    data = load_iris_data()
    context = prepare_context(data["x_real"], data["y_species"], 42)
    opts = dict(pop_size=10, generations=2, vae_epochs=2, show_progress=False)
    a = run_condition(context, 42, "vae_eda", record_diagnostics=False, **opts)
    b = run_condition(context, 42, "vae_eda", record_diagnostics=True, **opts)
    np.testing.assert_array_equal(a["synthetic_z"], b["synthetic_z"])
    np.testing.assert_array_equal(a["training_history"], b["training_history"])
    assert len(b["diagnostics"]) == 24
    assert len(b["latent_diagnostics"]) == 6
    assert b["history"].founder_fraction.between(0, 1).all()
    assert set(b["diagnostics"].stage) == {
        "selected_parents",
        "raw_candidates",
        "clipped_candidates",
        "survivors",
    }
