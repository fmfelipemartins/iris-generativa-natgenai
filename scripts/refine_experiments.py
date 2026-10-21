"""Refinamento retomável: controles, triagem, confirmação e avaliação aninhada."""

import argparse  # Interface explícita de execução.
from concurrent.futures import (
    ProcessPoolExecutor,
)  # Paralelismo entre unidades independentes.
from contextlib import contextmanager  # Garante liberação do lock ao sair.
from hashlib import sha256  # Identidade do código, dados e configuração.
from importlib.metadata import version  # Registra o ambiente efetivamente carregado.
import json  # Manifestos e checkpoints legíveis.
import os  # Locks exclusivos e substituição atômica.
from pathlib import Path  # Caminhos portáveis Windows/Linux.
import platform  # Identificação do Python e do sistema.
import sys  # Execução direta a partir da raiz do repositório.
from time import perf_counter  # Custo observado de cada unidade experimental.

ROOT = Path(__file__).resolve().parents[1]  # Raiz do projeto.
sys.path.insert(0, str(ROOT))  # Permite invocação sem instalação editável.
import numpy as np  # Estatísticas e índices.
import pandas as pd  # Exportações tabulares.
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
)  # Divisões exclusivas para seleção e teste.
from threadpoolctl import (
    threadpool_limits,
)  # Evita sobreposição de threads entre processos.
from iris_generativa import config  # Hiperparâmetros comentados e centralizados.
from iris_generativa.data import (
    load_iris_data,
    prepare_context,
)  # Fonte e contexto ajustados no treino.
from iris_generativa.generative import run_condition  # Seis condições existentes.
from iris_generativa.progress import progress  # Progresso por unidade concluída.
from iris_generativa.refinement import (
    FAMILIES,
    QUALITY_KEYS,
    candidate_catalog,
    options_for,
    balanced_sample,
    calibration,
    quality_metrics,
    quality_loss,
    discriminators,
    initialization,
)  # Protocolo congelado.


def write_json(path, value):
    """Substitui arquivos somente após escrita completa."""
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )  # Mesmo filesystem da saída final.
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )  # Rejeita NaN silencioso.
    temporary.replace(path)  # Commit atômico do checkpoint.


@contextmanager
def exclusive_lock(directory):
    """Não remove automaticamente locks de outro processo ou sessão."""
    path = directory / ".refinement.lock"  # Um escritor por pasta.
    descriptor = os.open(
        path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
    )  # Falha se já existir.
    os.write(
        descriptor, str(os.getpid()).encode()
    )  # PID facilita diagnóstico de interrupção.
    os.close(descriptor)  # O lock persiste até finally.
    try:
        yield  # Execução protegida.
    finally:
        path.unlink()  # Remove apenas o lock criado por esta execução.


def budgets(profile):
    """Quick testa funcionamento; full é o protocolo científico predefinido."""
    if profile == "quick":
        return dict(
            screen_generations=2,
            generations=3,
            epochs=2,
            screen_seeds=[0],
            confirm_seeds=[0],
            final_seeds=[10],
            inner_folds=2,
            outer_folds=2,
            trees=16,
            calibrations=3,
        )  # Redução explícita para teste.
    return dict(
        screen_generations=config.REFINEMENT_SCREEN_GENERATIONS,
        generations=config.GENERATIONS,
        epochs=config.VAE_EPOCHS,
        screen_seeds=list(config.REFINEMENT_SCREEN_SEEDS),
        confirm_seeds=list(config.REFINEMENT_CONFIRM_SEEDS),
        final_seeds=list(config.REFINEMENT_FINAL_SEEDS),
        inner_folds=config.REFINEMENT_INNER_FOLDS,
        outer_folds=config.REFINEMENT_OUTER_FOLDS,
        trees=300,
        calibrations=config.REFINEMENT_CALIBRATIONS,
    )  # Seeds finais distintas das usadas na seleção.


def evaluate_job(spec):
    """Unidade independente e determinística, suficiente para retomar a busca."""
    with threadpool_limits(limits=1):
        import torch  # Ambiente de treinamento registrado mesmo nas condições não neurais.

        torch.set_num_threads(1)  # Custo comparável sem oversubscription.
        started = perf_counter()  # Tempo inclui contexto, geração e métricas.
        data = load_iris_data()  # Dataset fixo identificado no manifesto.
        x, y = data["x_real"], data["y_species"]  # Arrays originais.
        train, valid = (
            np.asarray(spec["train"]),
            np.asarray(spec["valid"]),
        )  # Índices absolutos auditáveis.
        if np.intersect1d(train, valid).size:
            raise ValueError("Vazamento entre treino e avaliação.")
        ctx = prepare_context(
            x[train], y[train], spec["context_seed"]
        )  # Mesmos nichos para referência e variantes.
        counts = {
            int(t): min(int(np.sum(y[valid] == t)), len(ctx["real_by_task"][t]) // 2)
            for t in ctx["tasks"]
        }  # Origens balanceadas.
        evaluation_x, evaluation_y, evaluation_indices = balanced_sample(
            x[valid], y[valid], counts, spec["sample_seed"] + 1, return_indices=True
        )  # Mesmo tamanho da calibração real-real.
        if spec["variant"] == "initialization":
            result = initialization(ctx, spec["seed"])  # Controle sem evolução.
        elif spec["variant"] == "direct_gaussian":
            from iris_generativa.gaussian import GaussianGenerator  # Controle sem evolução.
            model = GaussianGenerator("gaussian_mixture", own_weight=1.0).fit(ctx["real_by_task"])
            rng = np.random.default_rng(spec["seed"])  # Amostragem independente e reproduzível.
            points = np.vstack([model.sample(t, 50, rng)[0] for t in ctx["tasks"]])
            result = dict(synthetic_x=ctx["scaler"].inverse_transform(points),
                          synthetic_species=np.repeat(ctx["tasks"], 50))  # Sem seleção ou clipping.
        else:
            result = run_condition(
                ctx, spec["seed"], spec["condition"], **spec["options"]
            )  # Opções totalmente materializadas.
        sample_x, sample_y = balanced_sample(
            result["synthetic_x"],
            result["synthetic_species"],
            counts,
            spec["sample_seed"],
        )  # RNG independente.
        scale = calibration(
            ctx, counts, spec["context_seed"] + 1900, spec["calibrations"]
        )  # Somente treino calibra escalas.
        metrics, by_task = quality_metrics(
            evaluation_x, evaluation_y, sample_x, sample_y, ctx
        )  # Holdout não orienta geração.
        record = dict(
            spec=spec,
            metrics=metrics,
            by_task=by_task,
            scale=scale,
            loss=quality_loss(metrics, scale),
            seconds=perf_counter() - started,
        )  # Dados suficientes para reanálise.
        record["evaluation_indices"] = valid[
            evaluation_indices
        ].tolist()  # Subconjunto reservado que efetivamente entrou nas métricas.
        record["sample"] = dict(
            x=sample_x.tolist(), species=sample_y.tolist()
        )  # Preserva os registros efetivamente avaliados.
        if spec["discriminators"]:
            record["discriminators"] = discriminators(
                evaluation_x,
                evaluation_y,
                sample_x,
                sample_y,
                spec["context_seed"],
                spec["trees"],
            )  # Avaliação independente da seleção.
        if "history" in result:
            record["history"] = result["history"].to_dict(
                orient="records"
            )  # Trajetória completa sem attrs volumosos.
            for key in ("diagnostics", "latent_diagnostics", "training_history"):
                record[key] = result[key].to_dict(
                    orient="records"
                )  # Diagnósticos de concentração e treinamento.
            events = result[
                "offspring_events"
            ]  # Origem dos componentes quando disponível.
            if len(events) and "component" in events:
                record["components"] = (
                    events.groupby(["task", "component"], sort=True)
                    .agg(
                        proposals=("survived", "size"),
                        survived=("survived", "sum"),
                        clipped=("clipped", "sum"),
                    )
                    .reset_index()
                    .to_dict(orient="records")
                )  # Desperdício por componente.
            candidate_rows = []  # Avaliação dos candidatos com mesmo tamanho da população final.
            for task, values in result["final_candidates"].items():
                points = values[
                    "clipped"
                ]  # Crossover registra depois dos limites internos.
                candidate_rows.append(
                    (task, points)
                )  # Mantém o tamanho original auditável.
            if not candidate_rows or not all(
                len(points) >= counts[task] for task, points in candidate_rows
            ):
                raise ValueError(
                    "Candidatos insuficientes para avaliação balanceada; não omitir seu diagnóstico."
                )
            if candidate_rows and all(
                len(points) >= counts[task] for task, points in candidate_rows
            ):
                cz = np.vstack(
                    [points for _, points in candidate_rows]
                )  # Candidatos antes da seleção.
                cy = np.concatenate(
                    [np.full(len(points), task) for task, points in candidate_rows]
                )  # Rótulos da tarefa atribuída.
                cx, cy = balanced_sample(
                    ctx["scaler"].inverse_transform(cz), cy, counts, spec["sample_seed"]
                )  # Mesmo orçamento de avaliação.
                record["candidate_metrics"], record["candidate_by_task"] = (
                    quality_metrics(evaluation_x, evaluation_y, cx, cy, ctx)
                )  # Não confunde sobreviventes com qualidade do modelo.
        if "candidate_metrics" in record:
            record["survivor_loss"] = record[
                "loss"
            ]  # Separação explícita dos dois estágios.
            record["candidate_loss"] = quality_loss(
                record["candidate_metrics"], scale
            )  # Não recompensa apenas retenção da inicialização.
            record["loss"] = 0.5 * (
                record["survivor_loss"] + record["candidate_loss"]
            )  # Peso igual predefinido para candidatos e sobreviventes.
        if spec.get("research_evaluation", False):
            from sklearn.ensemble import RandomForestClassifier  # Utilidade secundária, sem tuning.
            from sklearn.metrics import balanced_accuracy_score  # Mesma avaliação por espécie.
            utility = {}  # Treina em populações completas; teste real reservado comum.
            for source, fit_x, fit_y in (
                ("synthetic", result["synthetic_x"], result["synthetic_species"]),
                ("real", x[train], y[train]),
            ):
                classifier = RandomForestClassifier(n_estimators=spec["trees"], random_state=731, n_jobs=1)
                classifier.fit(fit_x, fit_y)  # Nunca ajusta com os rótulos do teste.
                prediction = classifier.predict(x[valid])  # Somente mensuração.
                utility[source] = float(balanced_accuracy_score(y[valid], prediction))
                utility[source + "_by_task"] = {str(t): float(np.mean(prediction[y[valid] == t] == t)) for t in ctx["tasks"]}
            record["utility"] = utility  # Não participa de choose_guarded.
            proximity = []  # Contextualiza quase-cópias com reais reservados.
            for task in ctx["tasks"]:
                real = ctx["real_by_task"][task]  # Referência exclusivamente de treino.
                distances = np.linalg.norm(real[:, None] - real[None, :], axis=2)
                np.fill_diagonal(distances, np.inf)  # Não conta o próprio registro.
                threshold = max(float(np.quantile(distances.min(axis=1), 0.05)), 1e-12)
                held = ctx["scaler"].transform(x[valid][y[valid] == task])  # Scaler congelado.
                nearest = np.linalg.norm(held[:, None] - real[None, :], axis=2).min(axis=1)
                proximity.append(dict(task=task, threshold=threshold, near_copy_rate=float(np.mean(nearest < threshold)), mean_distance=float(nearest.mean())))
            record["heldout_proximity"] = proximity  # Diagnóstico, não filtro de seleção.
        record["seconds"] = (
            perf_counter() - started
        )  # Inclui avaliação discriminativa quando habilitada.
        return record  # Falhas não viram resultados aparentemente completos.


class Runner:
    """Executa jobs com cache vinculado ao manifesto imutável."""

    def __init__(self, directory, budget, workers):
        self.directory, self.budget, self.workers = (
            directory,
            budget,
            workers,
        )  # Configuração do executor.
        self.jobs = directory / "jobs"  # Resultados individuais atômicos.
        self.jobs.mkdir(exist_ok=True)  # Não remove checkpoints existentes.

    def batch(self, specs, description):
        paths = [
            self.jobs
            / (sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() + ".json")
            for s in specs
        ]  # Identidade inclui todos os parâmetros e índices.
        missing = [
            (s, p) for s, p in zip(specs, paths) if not p.exists()
        ]  # Retoma somente jobs ausentes.
        if missing:
            if self.workers == 1:
                for spec, path in progress(missing, description, True):
                    write_json(
                        path, evaluate_job(spec)
                    )  # Salva imediatamente após cada job.
            else:
                with ProcessPoolExecutor(max_workers=self.workers) as pool:
                    outputs = pool.map(
                        evaluate_job, [s for s, _ in missing]
                    )  # Processos não compartilham modelos nem RNG.
                    for (_, path), output in zip(
                        progress(missing, description, True), outputs
                    ):
                        write_json(
                            path, output
                        )  # Apenas o processo pai escreve checkpoints.
        results = [
            json.loads(path.read_text(encoding="utf-8")) for path in paths
        ]  # Ordem original preservada.
        if any(r["spec"] != s for r, s in zip(results, specs)):
            raise ValueError("Checkpoint incompatível com o job solicitado.")
        return results  # Resultados completos ou erro explícito.

    def spec(
        self,
        condition,
        variant,
        train,
        valid,
        seed,
        context_seed,
        stage,
        outer,
        options=None,
    ):
        b = self.budget  # Orçamento congelado da execução.
        opts = (
            options_for(
                condition,
                variant,
                b["screen_generations"] if stage == "screen" else b["generations"],
                b["epochs"],
            )
            if variant != "initialization" and options is None
            else {}
        )  # Controle ou variante composta não exige catálogo.
        if options is not None:
            opts = options  # Variante composta da fase de refinamento.
        return dict(
            condition=condition,
            variant=variant,
            train=[int(i) for i in train],
            valid=[int(i) for i in valid],
            seed=int(seed),
            context_seed=int(context_seed),
            sample_seed=int(seed + 70000),
            stage=stage,
            outer=outer,
            options=opts,
            trees=b["trees"],
            calibrations=b["calibrations"],
            discriminators=stage in {"external", "controls"},
        )  # Tudo necessário à reprodução.

    def select(self, condition, indices, outer):
        data = load_iris_data()  # A seleção recebe apenas os índices do treino externo.
        splitter = StratifiedKFold(
            self.budget["inner_folds"], shuffle=True, random_state=12000 + outer
        )  # Subdivisões internas independentes.
        splits = [
            (indices[a], indices[b])
            for a, b in splitter.split(
                data["x_real"][indices], data["y_species"][indices]
            )
        ]  # Índices absolutos sem teste externo.
        screen = [
            self.spec(
                condition,
                variant,
                train,
                valid,
                seed,
                13000 + outer * 10 + fold,
                "screen",
                outer,
            )
            for fold, (train, valid) in enumerate(splits)
            for variant in candidate_catalog(condition)
            for seed in self.budget["screen_seeds"]
        ]  # Grade pequena predefinida.
        rows = self.batch(
            screen, f"{condition}: triagem fold {outer}"
        )  # Orçamento reduzido apenas para triagem.
        ranked = rank_candidates(rows)  # Não usa discriminador nem dados externos.
        finalists = [name for name in ranked if name != "reference"][
            :2
        ]  # No máximo duas variantes além da referência.
        confirm = [
            self.spec(
                condition,
                variant,
                train,
                valid,
                seed,
                13000 + outer * 10 + fold,
                "confirm",
                outer,
            )
            for fold, (train, valid) in enumerate(splits)
            for variant in ["reference", *finalists]
            for seed in self.budget["confirm_seeds"]
        ]  # Novas seeds e orçamento completo.
        confirmed = self.batch(
            confirm, f"{condition}: confirmação fold {outer}"
        )  # Mesmo orçamento em todos os finalistas.
        decision = choose_guarded(
            confirmed
        )  # Mantém referência se não houver melhora e guardas.
        base_variant = decision["selected"]  # Melhor opção elegível após confirmação.
        base_options = options_for(
            condition, base_variant, self.budget["generations"], self.budget["epochs"]
        )  # Mesmo orçamento da confirmação.
        additional = {}  # Refinamentos aplicados depois da seleção dos métodos.
        for name, weights in (
            ("novelty", dict(plausibility=0.4, novelty=0.35, coherence=0.25)),
            ("coherence", dict(plausibility=0.4, novelty=0.25, coherence=0.35)),
        ):
            additional[f"refined_{name}"] = dict(
                base_options, fitness_weights=weights
            )  # Só muda os pesos do fitness.
        if condition == "vae_eda":
            additional["refined_learning_rate"] = dict(
                base_options,
                model_options=dict(base_options["model_options"], learning_rate=3e-4),
            )  # Segunda etapa de treinamento.
            additional["refined_refit"] = dict(
                base_options, vae_epochs=2 * base_options["vae_epochs"], refit_every=2
            )  # Aproxima orçamento total de épocas em gerações pares.
        if condition not in ("SBX", "OB-Scan", "Híbrido 50/50"):
            chosen = [
                r for r in confirmed if r["spec"]["variant"] == base_variant
            ]  # Diagnóstico apenas no treino interno.
            clip_rate = float(
                np.mean(
                    [
                        np.mean([h.get("clipped", 0) / 150 for h in r["history"]])
                        for r in chosen
                    ]
                )
            )  # Frequência média de clipping.
            if clip_rate > config.REFINEMENT_CLIP_TRIGGER:
                additional["refined_resample"] = dict(
                    base_options, boundary_policy="resample"
                )  # Gatilho predefinido de 5%.
        refinement_specs = [
            self.spec(
                condition,
                name,
                train,
                valid,
                seed,
                13000 + outer * 10 + fold,
                "refine",
                outer,
                options=opts,
            )
            for fold, (train, valid) in enumerate(splits)
            for name, opts in additional.items()
            for seed in self.budget["confirm_seeds"]
        ]  # Mesmas subdivisões e seeds da confirmação.
        refined = self.batch(
            refinement_specs, f"{condition}: refinamentos fold {outer}"
        )  # Nenhum teste externo decide variantes.
        combined = confirmed + refined  # Referência continua no conjunto elegível.
        decision = choose_guarded(
            combined
        )  # Reaplica todas as guardas ao resultado conjunto.
        selected_options = additional.get(
            decision["selected"]
        )  # Opções de variante composta, se selecionada.
        if selected_options is None:
            selected_options = options_for(
                condition,
                decision["selected"],
                self.budget["generations"],
                self.budget["epochs"],
            )  # Variante da primeira etapa.
        decision.update(
            options=selected_options,
            condition=condition,
            outer=outer,
            finalists=finalists,
            screen_rank=ranked,
        )  # Decisão auditável.
        write_json(
            self.directory / f"selection_{outer}_{slug(condition)}.json", decision
        )  # Nenhum teste externo consultado.
        return decision, combined  # Resultados da seleção interna.


def slug(text):
    return "".join(c if c.isalnum() else "_" for c in text)  # Nome de arquivo seguro.


def rank_candidates(rows):
    means = {}  # Média balanceada por fold e seed.
    for row in rows:
        means.setdefault(row["spec"]["variant"], []).append(
            row["loss"]
        )  # Perda conjunta predefinida.
    return sorted(
        means, key=lambda name: (float(np.mean(means[name])), name != "reference", name)
    )  # Empate favorece referência.


def choose_guarded(rows):
    reference = [
        r for r in rows if r["spec"]["variant"] == "reference"
    ]  # Controle pareado.
    base_loss = float(np.mean([r["loss"] for r in reference]))  # Referência agregada.
    base = {
        key: float(np.mean([r["metrics"][key] for r in reference]))
        for key in QUALITY_KEYS
    }  # Métricas individuais não ocultadas.
    margin = {
        key: float(np.mean([r["scale"][key] for r in reference]))
        * config.REFINEMENT_MARGIN_FRACTION
        for key in QUALITY_KEYS
    }  # Um quarto da variabilidade típica real-real.
    margin.update(
        near_copy_rate=config.REFINEMENT_RATE_MARGIN,
        incoherence=config.REFINEMENT_RATE_MARGIN,
    )  # Guardas explícitas em pontos percentuais.
    accepted = []  # Nenhuma promoção obrigatória.
    checks = {}  # Motivos auditáveis de aceitação ou rejeição de cada variante.
    for variant in rank_candidates(rows):
        if variant == "reference":
            continue  # Controle não concorre consigo mesmo.
        sample = [
            r for r in rows if r["spec"]["variant"] == variant
        ]  # Mesmos folds/seeds.
        means = {
            key: float(np.mean([r["metrics"][key] for r in sample]))
            for key in QUALITY_KEYS
        }  # Perdas por dimensão.
        improved = (
            float(np.mean([r["loss"] for r in sample]))
            <= (1 - config.REFINEMENT_MIN_GAIN) * base_loss
        )  # Ganho mínimo interno predefinido de 5%.
        guarded = all(
            means[key] <= base[key] + margin[key] for key in QUALITY_KEYS
        )  # Nenhuma dimensão pode deteriorar além da margem.
        duplicates = np.mean(
            [r["metrics"]["exact_duplicate_rate"] for r in sample]
        ) <= np.mean(
            [r["metrics"]["exact_duplicate_rate"] for r in reference]
        )  # Não aceita aumento de cópias exatas.
        candidate_guard = True  # Controle inicial não possui candidatos.
        if all("candidate_metrics" in r for r in reference + sample):
            candidate_guard = all(
                np.mean([r["candidate_metrics"][key] for r in sample])
                <= np.mean([r["candidate_metrics"][key] for r in reference])
                + margin[key]
                for key in QUALITY_KEYS
            )  # Protege a qualidade de geração separadamente.
        task_guard = True  # Médias globais não devem esconder dano em uma espécie.
        for stage in ("by_task", "candidate_by_task"):
            if all(stage in r for r in reference + sample):
                for task in {item["task"] for r in reference for item in r[stage]}:
                    for key in QUALITY_KEYS:
                        old = np.mean(
                            [
                                item[key]
                                for r in reference
                                for item in r[stage]
                                if item["task"] == task
                            ]
                        )  # Referência da espécie.
                        new = np.mean(
                            [
                                item[key]
                                for r in sample
                                for item in r[stage]
                                if item["task"] == task
                            ]
                        )  # Variante na mesma espécie.
                        task_guard = (
                            task_guard and new <= old + margin[key]
                        )  # Mesma tolerância declarada.
        checks[variant] = dict(minimum_gain=bool(improved), survivor_metrics=bool(guarded),
                               exact_duplicates=bool(duplicates), candidate_metrics=bool(candidate_guard),
                               per_species_metrics=bool(task_guard))  # Não altera os limiares.
        if improved and guarded and duplicates and candidate_guard and task_guard:
            accepted.append(variant)  # Ranking já está ordenado por perda.
    return dict(
        selected=accepted[0] if accepted else "reference",
        accepted=accepted,
        checks=checks,  # Cada False identifica uma guarda não satisfeita.
        reference_loss=base_loss,
        margins=margin,
        rule="5% loss reduction; per-metric guards; no increase in exact duplicates",
    )  # Regra operacional, não significância estatística.


def summarize(directory):
    paths = sorted(
        (directory / "jobs").glob("*.json")
    )  # Somente checkpoints finalizados.
    rows = []  # Uma linha por execução, sem tratar folds como indivíduos.
    for path in paths:
        r = json.loads(
            path.read_text(encoding="utf-8")
        )  # Lê um job por vez para limitar memória.
        s = r["spec"]  # Identidade experimental.
        row = {
            key: s[key] for key in ("condition", "variant", "stage", "outer", "seed")
        }  # Chave legível.
        row.update(
            loss=r["loss"],
            survivor_loss=r.get("survivor_loss", r["loss"]),
            candidate_loss=r.get("candidate_loss"),
            seconds=r["seconds"],
            **r["metrics"],
            **r.get("discriminators", {}),
        )  # Métricas e custo.
        if r.get("history"):
            row.update(
                {
                    key: r["history"][-1].get(key)
                    for key in (
                        "founder_fraction",
                        "fitness_evaluations",
                        "elapsed_seconds",
                    )
                }
            )  # Não soma avaliações como amostras independentes.
        row.update(
            {
                f"candidate_{key}": value
                for key, value in r.get("candidate_metrics", {}).items()
            }
        )  # Colunas separadas dos sobreviventes.
        rows.append(row)  # Preserva valores por execução.
    frame = pd.DataFrame(rows)  # Não agrega variabilidade prematuramente.
    frame.to_csv(
        directory / "runs.csv", index=False
    )  # Pode conter NaN por métricas não aplicáveis.
    if not frame.empty:
        frame.groupby(["condition", "variant", "stage"], sort=True).agg(
            loss_mean=("loss", "mean"),
            loss_sd=("loss", "std"),
            seconds_mean=("seconds", "mean"),
            runs=("loss", "size"),
        ).to_csv(directory / "summary.csv")  # DP descritivo, não erro-padrão de folds.
    external = (
        frame[frame.stage == "external"] if not frame.empty else frame
    )  # Apenas avaliação final.
    if not external.empty:
        external.to_csv(directory / "external.csv", index=False)
        paired = []  # Comparação selecionado-referência por fold e seed.
        for (condition, outer), group in external.groupby(["condition", "outer"]):
            decision = json.loads(
                (directory / f"selection_{outer}_{slug(condition)}.json").read_text(
                    encoding="utf-8"
                )
            )  # Escolha interna já congelada.
            baseline = group[group.variant == "reference"].set_index(
                "seed"
            )  # Pareamento por seed, sem pseudo-replicar folds.
            selected = group[group.variant == decision["selected"]].set_index(
                "seed"
            )  # Pode coincidir com referência.
            for seed in baseline.index.intersection(selected.index):
                paired.append(
                    dict(
                        condition=condition,
                        outer=int(outer),
                        seed=int(seed),
                        selected=decision["selected"],
                        **{
                            f"delta_{key}": float(
                                selected.loc[seed, key] - baseline.loc[seed, key]
                            )
                            for key in ("loss", *QUALITY_KEYS, "rf_accuracy", "rf_auc")
                        },
                    )
                )  # Diferença negativa favorece variante nas perdas.
        deltas = pd.DataFrame(paired)  # Reúne decisões por fold.
        deltas.to_csv(
            directory / "paired_external.csv", index=False
        )  # Base da análise de estabilidade.
        summaries = []  # Intervalos apenas da aleatoriedade algorítmica condicional aos folds fixos.
        for condition, group in deltas.groupby("condition"):
            by_seed = (
                group.groupby("seed").delta_loss.mean().to_numpy()
            )  # Primeiro agrega folds dependentes.
            rng = np.random.default_rng(
                91001
            )  # Bootstrap reprodutível, independente dos geradores.
            draws = rng.choice(by_seed, size=(2000, len(by_seed)), replace=True).mean(
                1
            )  # Unidade de reamostragem é a seed.
            summaries.append(
                dict(
                    condition=condition,
                    delta_loss=float(by_seed.mean()),
                    conditional_ci_low=float(np.quantile(draws, 0.025)),
                    conditional_ci_high=float(np.quantile(draws, 0.975)),
                    seeds=len(by_seed),
                    interpretation="algorithmic uncertainty conditional on fixed data/splits; exploratory, not population CI",
                )
            )  # Cinco seeds limitam precisão.
        pd.DataFrame(summaries).to_csv(
            directory / "conditional_uncertainty.csv", index=False
        )  # Não declara significância usando folds independentes.  # Separação explícita da busca interna.
    return len(paths)  # Usado pelo marcador de conclusão.


def main():
    parser = argparse.ArgumentParser(description=__doc__)  # CLI portável.
    parser.add_argument(
        "--profile", choices=["quick", "full"], default="quick"
    )  # Nenhum full implícito no notebook.
    parser.add_argument(
        "--phase", choices=["controls", "search", "nested"], default="controls"
    )  # Fases com finalidade distinta.
    parser.add_argument(
        "--conditions", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )  # Subconjunto explícito para diagnóstico.
    parser.add_argument(
        "--output", type=Path, required=True
    )  # Impede sobrescrever execução anterior por padrão.
    parser.add_argument(
        "--workers", type=int, default=1
    )  # Paralelismo limitado pelo usuário.
    parser.add_argument(
        "--resume", action="store_true"
    )  # Retomada exige identidade completa.
    args = parser.parse_args()  # Opções efetivas.
    if args.workers < 1:
        parser.error("workers deve ser positivo")  # Falha antes de criar arquivos.
    args.output.mkdir(
        parents=True, exist_ok=True
    )  # Pasta de resultados do repositório.
    code = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
        for folder in ("iris_generativa", "scripts")
        for p in sorted((ROOT / folder).glob("*.py"))
    }  # Hash inclui configuração e executor.
    data = load_iris_data()  # Dataset identificado pelos bytes.
    manifest = dict(
        protocol="refinement-1",
        profile=args.profile,
        phase=args.phase,
        conditions=args.conditions,
        budget=budgets(args.profile),
        catalog={c: candidate_catalog(c) for c in args.conditions},
        options={c: options_for(c, "reference") for c in args.conditions},
        code=code,
        python=platform.python_version(),
        platform=platform.platform(),
        versions={
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "pygad")
        },
        dataset=sha256(
            data["x_real"].tobytes() + data["y_species"].tobytes()
        ).hexdigest(),
        criteria=dict(
            minimum_internal_improvement=0.05,
            tolerance_fraction=0.25,
            rate_tolerance=0.05,
            external_promotion=False,
        ),
    )  # Congelado antes de qualquer resultado.
    with exclusive_lock(args.output):
        path = args.output / "manifest.json"  # Identidade desta execução.
        if path.exists():
            if (
                not args.resume
                or json.loads(path.read_text(encoding="utf-8")) != manifest
            ):
                raise ValueError(
                    "Pasta já usada ou manifesto incompatível; use nova pasta."
                )  # Não mistura código ou ambientes.
        elif args.resume:
            raise ValueError(
                "Não existe manifesto para retomar."
            )  # Evita interpretação ambígua de resume.
        else:
            write_json(path, manifest)  # Declara protocolo antes dos resultados.
        runner = Runner(
            args.output, manifest["budget"], args.workers
        )  # Executor protegido pelo lock.
        x, y = data["x_real"], data["y_species"]  # Arrays originais.
        if args.phase == "nested":
            splits = StratifiedKFold(
                manifest["budget"]["outer_folds"], shuffle=True, random_state=20261
            ).split(x, y)  # Teste externo excluído da seleção.
            for outer, (train, holdout) in enumerate(splits):
                for condition in args.conditions:
                    decision, _ = runner.select(
                        condition, train, outer
                    )  # Busca refeita só dentro do treino externo.
                    variants = list(
                        dict.fromkeys(
                            ["reference", decision["selected"], "initialization"]
                        )
                    )  # Evita executar referência duplicada.
                    specs = [
                        runner.spec(
                            condition,
                            variant,
                            train,
                            holdout,
                            seed,
                            22000 + outer,
                            "external",
                            outer,
                            options=decision["options"]
                            if variant == decision["selected"]
                            else None,
                        )
                        for variant in variants
                        for seed in manifest["budget"]["final_seeds"]
                    ]  # População 50, avaliação amostrada.
                    runner.batch(
                        specs, f"{condition}: teste externo {outer}"
                    )  # Dados externos apenas depois da decisão.
                    summarize(
                        args.output
                    )  # Relatório parcial utilizável após interrupção.
        elif args.phase == "search":
            for condition in args.conditions:
                runner.select(
                    condition, np.arange(len(x)), 99
                )  # Exploração; não é evidência externa.
                summarize(args.output)  # Resultados internos claramente rotulados.
        else:
            train, holdout = next(
                StratifiedShuffleSplit(1, test_size=0.2, random_state=20261).split(x, y)
            )  # Controle diagnóstico, não seleção.
            specs = [
                runner.spec(
                    condition, variant, train, holdout, seed, 22000, "controls", 0
                )
                for condition in args.conditions
                for variant in ("reference", "initialization")
                for seed in manifest["budget"]["confirm_seeds"]
            ]  # Evolução versus população inicial.
            runner.batch(
                specs, "Controles de inicialização"
            )  # Avaliação inclui os dois discriminadores.
        count = summarize(
            args.output
        )  # Verifica leitura de todos os checkpoints concluídos.
        write_json(
            args.output / "COMPLETE.json",
            dict(
                phase=args.phase,
                profile=args.profile,
                completed_jobs=count,
                full_nested_completed=args.phase == "nested" and args.profile == "full",
                defaults_changed=False,
            ),
        )  # Conclusão apenas após todas as etapas da fase.
        print(
            json.dumps(
                dict(output=str(args.output), completed_jobs=count), ensure_ascii=False
            )
        )  # Resumo curto no terminal.


if __name__ == "__main__":
    main()  # Compatível com spawn do Windows.
