"""Testes dos riscos da iteração: isolamento, condicionamento e avaliação."""

import importlib.util  # Carrega o executor sem iniciar a CLI.
from pathlib import Path  # Localiza o script no repositório.
import numpy as np  # Comparações numéricas.
import pytest  # Asserções e fixtures.
from iris_generativa.data import (
    load_iris_data,
    prepare_context,
)  # Contexto de teste realista.
from iris_generativa.generative import (
    run_condition,
)  # API que deve isolar configurações.
from iris_generativa.refinement import (
    balanced_sample,
    candidate_catalog,
    FAMILIES,
)  # Contratos do protocolo.
from iris_generativa.gaussian import GaussianGenerator  # Produtos condicionados.
from iris_generativa.vae import VAEGenerator  # Treinamento com warmup contínuo.


def context():
    data = load_iris_data()  # Fonte fixa.
    return prepare_context(
        data["x_real"], data["y_species"], 42
    )  # Referência independente por teste.


def test_target_product_origins_and_reference():
    populations = {
        t: np.random.default_rng(t).normal(t * 2, 1, (20, 4)) for t in range(3)
    }  # Espécies separadas.
    model = GaussianGenerator(
        "gaussian_product", target_products=True, product_weight=0.4
    ).fit(populations)  # Variante proposta.
    _, origins = model.sample(
        1, 1000, np.random.default_rng(42)
    )  # Amostra representativa.
    assert all(1 in group for group in origins)  # Nenhum componente sem espécie-alvo.
    assert any(len(group) == 1 for group in origins) and any(
        len(group) == 2 for group in origins
    )  # Mistura normalizada de ambos os tipos.
    original = GaussianGenerator("gaussian_product").fit(
        populations
    )  # Referência uniforme mantida.
    a, _ = original.sample(0, 20, np.random.default_rng(7))  # Primeira tarefa.
    b, _ = original.sample(
        2, 20, np.random.default_rng(7)
    )  # Mesma distribuição na referência.
    np.testing.assert_array_equal(a, b)  # A mudança é opt-in.


def test_kl_warmup_continues_between_fits():
    model = VAEGenerator(
        2, [0, 1], 42, latent=2, hidden=8, batch_size=4, beta=1, kl_warmup_steps=8
    )  # Duas atualizações por época.
    points = {0: np.zeros((4, 2)), 1: np.ones((4, 2))}  # Treino pequeno.
    model.fit(points, epochs=1)  # Atualizações 1 e 2.
    assert model.history().iloc[-1].effective_beta == pytest.approx(
        0.25
    )  # Ainda na rampa.
    model.fit(points, epochs=3)  # Atualizações 3 até 8.
    assert (
        model.optimizer_steps == 8 and model.history().iloc[-1].effective_beta == 1
    )  # Sem reiniciar rampa entre gerações.


def test_fitness_override_does_not_leak_and_diagnostics_are_neutral():
    ctx = context()  # Compartilhado de propósito para detectar vazamento.
    opts = dict(pop_size=10, generations=2, show_progress=False)  # Orçamento funcional.
    baseline = run_condition(ctx, 7, "SBX", **opts)  # Referência sem diagnóstico.
    observed = run_condition(
        ctx, 7, "SBX", record_diagnostics=True, **opts
    )  # Medição sem RNG.
    np.testing.assert_array_equal(
        baseline["synthetic_x"], observed["synthetic_x"]
    )  # Mesma população.
    run_condition(
        ctx,
        7,
        "SBX",
        fitness_weights=dict(plausibility=1, novelty=0, coherence=0),
        **opts,
    )  # Alteração isolada.
    repeated = run_condition(
        ctx, 7, "SBX", **opts
    )  # Volta ao padrão sem recarregar módulos.
    np.testing.assert_array_equal(
        baseline["synthetic_x"], repeated["synthetic_x"]
    )  # Sem resíduos no contexto.
    assert "fitness_weights" not in ctx  # Contexto original preservado.
    assert (
        observed["history"].founder_fraction.between(0, 1).all()
    )  # Proveniência válida.
    assert (
        len(observed["diagnostics"]) == 12
    )  # Três tarefas, dois estágios, duas gerações.


def test_balanced_sampling_and_catalog_budget():
    points, labels = (
        np.arange(600).reshape(150, 4),
        np.repeat([0, 1, 2], 50),
    )  # População completa.
    x, y = balanced_sample(
        points, labels, {0: 10, 1: 10, 2: 10}, 42
    )  # Amostra de avaliação.
    assert x.shape == (30, 4) and np.all(
        np.bincount(y) == 10
    )  # Origens balanceadas no cross-fitting.
    assert (
        len(np.unique(x[:, 0])) == 30 and len(points) == 150
    )  # Sem reposição nem redução do tamanho evolutivo.
    assert all(
        len(candidate_catalog(c)) <= 8 for c in FAMILIES
    )  # Busca não explode silenciosamente.


def test_executor_holdout_is_not_used_to_fit(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "refine_runner", Path(__file__).parents[1] / "scripts" / "refine_experiments.py"
    )  # Importa sem main.
    module = importlib.util.module_from_spec(spec)  # Namespace independente.
    spec.loader.exec_module(module)  # Funções do executor.
    data = load_iris_data()  # Dados conhecidos para conferir índices.
    valid = np.r_[0:10, 50:60, 100:110]  # Dez por tarefa.
    train = np.setdiff1d(np.arange(150), valid)  # Complemento disjunto.
    seen = []  # Captura exclusivamente a entrada do ajuste.
    original = module.prepare_context  # Implementação real.

    def capture(x, y, seed):
        seen.append(x.copy())  # Auditoria do treino.
        return original(x, y, seed)  # Não altera a função.

    monkeypatch.setattr(
        module, "prepare_context", capture
    )  # Espia a fronteira de dados.
    opts = module.options_for(
        "gaussian_mixture", "reference", generations=1, epochs=1
    )  # Orçamento curto, população integral.
    job = dict(
        condition="gaussian_mixture",
        variant="reference",
        train=train.tolist(),
        valid=valid.tolist(),
        seed=1,
        context_seed=42,
        sample_seed=70001,
        stage="external",
        outer=0,
        options=opts,
        trees=8,
        calibrations=2,
        discriminators=False,
    )  # Job explícito.
    result = module.evaluate_job(job)  # Executa geração e métricas.
    np.testing.assert_array_equal(
        seen[0], data["x_real"][train]
    )  # Holdout não orienta scaler, fitness nem nichos.
    assert (
        len(seen) == 1 and len(result["sample"]["x"]) == 30
    )  # Sem reajuste em dados externos.
    assert (
        np.isfinite(result["loss"]) and "candidate_metrics" in result
    )  # Avalia ambos os estágios.
    job["train"] = job["train"] + [int(valid[0])]  # Injeta vazamento deliberado.
    with pytest.raises(ValueError, match="Vazamento"):
        module.evaluate_job(job)  # Deve falhar antes de ajustar qualquer modelo.


def test_species_guard_rejects_hidden_deterioration():
    spec = importlib.util.spec_from_file_location(
        "guard_runner", Path(__file__).parents[1] / "scripts" / "refine_experiments.py"
    )  # Importa somente a regra de decisão.
    module = importlib.util.module_from_spec(spec)  # Namespace isolado.
    spec.loader.exec_module(module)  # Não executa a CLI.
    keys = module.QUALITY_KEYS  # Mesmo conjunto de perdas do protocolo.
    baseline_tasks = [
        dict(task=t, **{k: 1.0 for k in keys}) for t in range(3)
    ]  # Referência por espécie.
    variant_tasks = [
        dict(task=t, **{k: 0.8 for k in keys}) for t in range(3)
    ]  # Melhoria média em várias dimensões.
    variant_tasks[2]["covariance_error"] = (
        1.5  # Deterioração acima da margem em uma espécie.
    )

    def row(name, tasks, loss):
        metrics = {
            key: float(np.mean([t[key] for t in tasks])) for key in keys
        }  # Médias consistentes com as espécies.
        metrics["exact_duplicate_rate"] = 0.0  # Nenhuma cópia exata no controle.
        return dict(
            spec={"variant": name},
            loss=loss,
            metrics=metrics,
            candidate_metrics=metrics,
            by_task=tasks,
            candidate_by_task=tasks,
            scale={k: 1.0 for k in keys},
        )  # Estrutura mínima da decisão.

    result = module.choose_guarded(
        [row("reference", baseline_tasks, 1.0), row("variant", variant_tasks, 0.8)]
    )  # Média melhora, mas guardas por tarefa falham.
    assert result["selected"] == "reference"  # Não mascara dano local com média global.


def test_resampling_has_budget_and_clipping_fallback(monkeypatch):
    ctx = context()  # Limites reais usados pela geração.
    calls = []  # Número de propostas por chamada ao modelo.

    def outside(self, task, size, rng):
        calls.append(size)  # Inclui amostras iniciais e tentativas adicionais.
        return np.tile(ctx["global_high"] + 100, (size, 1)), [
            (task,)
        ] * size  # Modelo deliberadamente fora da caixa.

    monkeypatch.setattr(
        GaussianGenerator, "sample", outside
    )  # Força a condição limite sem depender de sorte.
    result = run_condition(
        ctx,
        5,
        "gaussian_mixture",
        pop_size=5,
        generations=1,
        boundary_policy="resample",
        resample_attempts=2,
        show_progress=False,
    )  # Duas tentativas por tarefa.
    assert sum(calls) == 45  # Quinze iniciais e trinta propostas adicionais.
    assert (
        result["history"].iloc[0].resampled == 30
    )  # Custo extra explicitamente registrado.
    assert result["history"].iloc[0].clipped == 15  # Fallback após esgotar o orçamento.
    assert np.all(
        result["synthetic_z"] <= ctx["global_high"]
    )  # Não deixa candidatos inválidos sobreviverem fora dos limites.
