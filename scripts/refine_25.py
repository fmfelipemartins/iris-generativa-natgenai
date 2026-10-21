"""Iteração 2.5: busca limitada, hipóteses isoladas e confirmação pareada."""
import argparse  # CLI explícita e compatível com Windows.
import json  # Manifestos e decisões auditáveis.
from pathlib import Path  # Caminhos portáveis.
from hashlib import sha256  # Congela código e dados antes da execução.
from importlib.metadata import version  # Registra dependências efetivas.
import platform  # Identifica o ambiente.
import numpy as np  # Índices e estatísticas.
from sklearn.model_selection import StratifiedKFold  # Separação aninhada.
from refine_experiments import (ROOT, Runner, budgets, write_json, exclusive_lock,
    load_iris_data, options_for, rank_candidates, choose_guarded, summarize, slug)
from iris_generativa import config  # Hiperparâmetros experimentais documentados.


def catalog(condition, generations, epochs):
    """Cada opção muda um único mecanismo em relação à referência."""
    base = options_for(condition, "reference", generations, epochs)  # Defaults históricos.
    variants = {"reference": base}  # Referência obrigatória em todas as fases.
    if condition == "SBX":
        variants.update(rank_selection=dict(base, survivor_policy="rank", survivor_elite_fraction=config.REFINEMENT25_ELITE_FRACTION),
                        novelty_band=dict(base, novelty_policy="band"),
                        no_transfer=dict(base, rmp=0.0),
                        eta20=dict(base, sbx_eta=config.REFINEMENT25_SBX_ETA))  # Quatro contrastes isolados.
    else:
        variants["fixed_parents"] = dict(base, fixed_parents=True)  # Congela exatamente os pais iniciais.
        for name, option in (("latent2", dict(latent=config.REFINEMENT25_VAE_LATENT)), ("hidden32", dict(hidden=config.REFINEMENT25_VAE_HIDDEN)),
                             ("learning_rate", dict(learning_rate=config.REFINEMENT25_VAE_LEARNING_RATE))):
            variants[name] = dict(base, model_options=dict(base["model_options"], **option))  # Demais opções iguais.
    return variants  # Cinco configurações por família, sem combinações pós-hoc.


def protocol(profile):
    """Teto principal full: 300 triagens + 200 confirmações + 100 testes + 125 controles."""
    b = budgets(profile)  # Reaproveita orçamento computacional documentado.
    if profile == "full":
        b.update(screen_seeds=[0, 1, 2], confirm_seeds=[3, 4, 5, 6, 7], final_seeds=[20, 21, 22, 23, 24], outer_folds=5, inner_folds=2)
    b["max_jobs"] = (2 * b["outer_folds"] * b["inner_folds"] * (5 * len(b["screen_seeds"]) + 2 * len(b["confirm_seeds"]))
                     + 9 * b["outer_folds"] * len(b["final_seeds"]))  # Inclui referência e finalista mesmo se rejeitado.
    return b  # 725 no full; limite testado antes de executar.


class LimitedRunner(Runner):
    """Mantém cache e paralelismo existentes, limitando a busca científica."""
    def batch(self, specs, description):
        requested = {sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() for s in specs}  # IDs previstos.
        existing = {p.stem for p in self.jobs.glob("*.json")}  # Somente checkpoints completos.
        if len(requested | existing) > self.budget["max_jobs"]:
            raise ValueError("Orçamento congelado excedido.")  # Não inicia trabalho excedente.
        return super().batch(specs, description)  # Escrita atômica e progresso por job.

    def select(self, condition, indices, outer):
        data = load_iris_data()  # Apenas índices de treino externo chegam à seleção.
        split = StratifiedKFold(self.budget["inner_folds"], shuffle=True, random_state=12000 + outer)
        pairs = [(indices[a], indices[b]) for a, b in split.split(data["x_real"][indices], data["y_species"][indices])]
        short = catalog(condition, self.budget["screen_generations"], self.budget["epochs"])  # Triagem curta.
        full = catalog(condition, self.budget["generations"], self.budget["epochs"])  # Orçamento final comum.
        def specs(names, stage, seeds, options):
            return [self.spec(condition, name, train, valid, seed, 13000 + outer * 10 + fold,
                              stage, outer, options=options[name])
                    for fold, (train, valid) in enumerate(pairs) for name in names for seed in seeds]  # Pareamento completo.
        screen = self.batch(specs(short, "screen", self.budget["screen_seeds"], short), f"{condition}: triagem {outer}")
        ranking = rank_candidates(screen)  # Não consulta resultados externos.
        finalist = next(name for name in ranking if name != "reference")  # Um único finalista além do controle.
        confirmation = self.batch(specs(["reference", finalist], "confirm", self.budget["confirm_seeds"], full), f"{condition}: confirmação {outer}")
        decision = choose_guarded(confirmation)  # Todas as guardas 2.4 preservadas.
        decision.update(condition=condition, outer=outer, finalists=[finalist], screen_rank=ranking,
                        options=full[decision["selected"]], finalist_options=full[finalist])  # Decisão congelada antes do teste.
        decision["reason"] = ("Ganho mínimo e todas as guardas satisfeitos." if decision["accepted"]
                              else "Finalista não satisfez conjuntamente ganho mínimo e guardas; referência mantida.")
        decision["screen_losses"] = {name: float(np.mean([r["loss"] for r in screen if r["spec"]["variant"] == name])) for name in short}
        write_json(self.directory / f"selection_{outer}_{slug(condition)}.json", decision)  # Rejeição explícita.
        return decision  # Nenhuma alteração de default.


def main():
    parser = argparse.ArgumentParser(description=__doc__)  # Sem execução full implícita.
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")  # Quick valida encadeamento.
    parser.add_argument("--output", type=Path, required=True)  # Pasta própria para a iteração.
    parser.add_argument("--workers", type=int, default=1)  # Processos independentes.
    parser.add_argument("--resume", action="store_true")  # Exige identidade de código/ambiente.
    args = parser.parse_args()  # Valida interface.
    if args.workers < 1:
        parser.error("workers deve ser positivo")  # Não cria lock inválido.
    args.output.mkdir(parents=True, exist_ok=True)  # Não sobrescreve evidência antiga.
    b = protocol(args.profile)  # Orçamento imutável.
    data = load_iris_data()  # Dataset rastreável.
    manifest = dict(protocol="refinement-2.5", profile=args.profile, budget=b,
                    catalog={c: catalog(c, b["generations"], b["epochs"]) for c in ["SBX", "vae_eda"]},
                    code={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for d in ["iris_generativa", "scripts"] for p in sorted((ROOT / d).glob("*.py"))},
                    python=platform.python_version(), platform=platform.platform(),
                    versions={n: version(n) for n in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "pygad"]},
                    dataset=sha256(data["x_real"].tobytes() + data["y_species"].tobytes()).hexdigest(),
                    criteria=dict(minimum_internal_improvement=0.05, tolerance_fraction=0.25, rate_tolerance=0.05, external_promotion=False),
                    checkpoints=list(config.REFINEMENT25_CHECKPOINTS), stopping="Fixed 40; checkpoints descriptive only", queue="Deferred until fixed-parent contrast resolves input concentration")
    with exclusive_lock(args.output):
        path = args.output / "manifest.json"  # Confere identidade antes de usar cache.
        if path.exists():
            if not args.resume or json.loads(path.read_text()) != manifest:
                raise ValueError("Manifesto incompatível; use nova pasta.")
        elif args.resume:
            raise ValueError("Manifesto ausente.")
        else:
            write_json(path, manifest)  # Protocolo salvo antes de qualquer job.
        runner = LimitedRunner(args.output, b, args.workers)  # Herda limites e progresso.
        outer_splits = StratifiedKFold(b["outer_folds"], shuffle=True, random_state=20261)
        for outer, (train, holdout) in enumerate(outer_splits.split(data["x_real"], data["y_species"])):
            for condition in ["SBX", "vae_eda"]:
                decision = runner.select(condition, train, outer)  # Holdout inacessível à busca.
                names = ["reference", decision["finalists"][0]]  # Mede finalista rejeitado sem promovê-lo.
                variants = catalog(condition, b["generations"], b["epochs"])  # Opções já declaradas.
                specs = [runner.spec(condition, name, train, holdout, seed, 22000 + outer, "external", outer, options=variants[name]) for name in names for seed in b["final_seeds"]]
                for spec in specs:
                    spec["research_evaluation"] = True  # Utilidade/proximidade externas, secundárias.
                runner.batch(specs, f"{condition}: teste externo {outer}")  # Sem reescolha após visualizar o resultado.
            control_specs = []  # Cinco controles por fold e seed.
            for name, condition in [("initialization", "SBX"), ("direct_gaussian", "gaussian_mixture"), ("own_species", "gaussian_mixture"), ("reference", "gaussian_mixture"), ("reference", "gaussian_product")]:
                opts = options_for(condition, "reference", b["generations"], b["epochs"])  # Mesma duração evolutiva.
                if name == "own_species":
                    opts["model_options"]["own_weight"] = 1.0  # Elimina transferência entre espécies.
                for seed in b["final_seeds"]:
                    spec = runner.spec(condition, name, train, holdout, seed, 22000 + outer, "controls", outer, options=opts)
                    spec["research_evaluation"] = True  # Mesma avaliação externa.
                    control_specs.append(spec)  # Sem seleção de hiperparâmetros nos controles.
            runner.batch(control_specs, f"Controles: fold {outer}")  # Comparadores predefinidos.
            summarize(args.output)  # Relatório parcial retomável.
        count = summarize(args.output)  # Verifica todos os checkpoints legíveis.
        if count != b["max_jobs"]:
            raise ValueError("Quantidade de jobs diferente do protocolo congelado.")  # Sem conclusão parcial.
        write_json(args.output / "COMPLETE.json", dict(completed_jobs=count, expected_jobs=b["max_jobs"], profile=args.profile, defaults_changed=False))
        print(json.dumps(dict(output=str(args.output), completed_jobs=count)))  # Resumo final.


if __name__ == "__main__":
    main()  # Obrigatório para spawn no Windows.
