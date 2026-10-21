"""Contratos de seleção, isolamento e orçamento da iteração 2.5."""
import sys  # Importação dos executores sem executar CLI.
from pathlib import Path  # Raiz do repositório.
import numpy as np  # Comparações exatas.
import pytest  # Falhas esperadas.
from iris_generativa.data import load_iris_data, prepare_context  # Dados e calibração.
from iris_generativa.generative import run_condition  # Caminho público das opções.
from iris_generativa.evolution import select_next_population  # Cotas e índices.
from iris_generativa.vae import VAEGenerator  # Captura dos pais efetivamente ajustados.


def context():
    data = load_iris_data()  # Base fixa.
    return prepare_context(data["x_real"], data["y_species"], 42)  # Sem estado entre testes.


def test_rank_preserves_quotas_unique_indices_and_reproducibility():
    ctx = context()  # Todos os nichos representados pelos dados reais.
    points = ctx["real_by_task"][0]  # Uma tarefa.
    reference = select_next_population(points, 0, ctx, 25, np.random.default_rng(7), return_indices=True)
    ctx["survivor_policy"] = "rank"  # Ativa a variante explicitamente.
    a = select_next_population(points, 0, ctx, 25, np.random.default_rng(7), return_indices=True)
    b = select_next_population(points, 0, ctx, 25, np.random.default_rng(7), return_indices=True)
    labels = ctx["niche_models"][0].predict(points)  # Mesmo nicho para todas as políticas.
    np.testing.assert_array_equal(np.bincount(labels[a]), np.bincount(labels[reference]))  # Cotas mantidas.
    np.testing.assert_array_equal(a, b)  # Determinismo por seed.
    assert len(set(a)) == 25 and not np.array_equal(a, reference)  # Sem reposição; seleção distinta.


def test_fixed_parent_training_keeps_initial_inputs_and_update_budget(monkeypatch):
    ctx = context()  # Compartilha somente referência de treino.
    recorded = []  # Captura os argumentos de fit.
    original = VAEGenerator.fit  # Treinamento verdadeiro.
    def capture(self, populations, **options):
        recorded.append({t: p.copy() for t, p in populations.items()})  # Não altera os pais.
        return original(self, populations, **options)  # Mantém o otimizador.
    monkeypatch.setattr(VAEGenerator, "fit", capture)  # Observação da fronteira de ajuste.
    opts = dict(pop_size=10, generations=3, vae_epochs=2, show_progress=False, record_diagnostics=True)
    fixed = run_condition(ctx, 7, "vae_eda", fixed_parents=True, **opts)  # Controle congelado.
    for parents in recorded[1:]:
        for task in ctx["tasks"]:
            np.testing.assert_array_equal(parents[task], recorded[0][task])  # Mesmos pontos em toda geração.
    recorded.clear()  # Separa observações da referência.
    evolving = run_condition(ctx, 7, "vae_eda", **opts)  # Orçamento idêntico.
    assert fixed["training_history"].iloc[-1].optimizer_steps == evolving["training_history"].iloc[-1].optimizer_steps


def test_options_do_not_leak_and_default_rng_is_preserved():
    ctx = context()  # Dicionário reutilizado de propósito.
    opts = dict(pop_size=10, generations=2, show_progress=False)
    a = run_condition(ctx, 2, "SBX", **opts)  # Referência.
    run_condition(ctx, 2, "SBX", survivor_policy="rank", novelty_policy="band", **opts)  # Experimento.
    b = run_condition(ctx, 2, "SBX", survivor_policy="deterministic", novelty_policy="target", **opts)
    np.testing.assert_array_equal(a["synthetic_x"], b["synthetic_x"])  # Nenhum resíduo experimental.
    assert "novelty_policy" not in ctx and "survivor_policy" not in ctx  # Contexto isolado.
    with pytest.raises(ValueError):
        run_condition(ctx, 2, "SBX", survivor_policy="unknown", **opts)  # Falha explícita.


def test_protocol_has_fixed_bounded_catalog():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))  # Importa driver sem main.
    from refine_25 import protocol, catalog  # Catálogo congelado.
    assert protocol("full")["max_jobs"] == 725  # Limite principal acordado, inferior a 800.
    for family in ["SBX", "vae_eda"]:
        assert len(catalog(family, 40, 30)) == 5  # Referência mais quatro contrastes.
