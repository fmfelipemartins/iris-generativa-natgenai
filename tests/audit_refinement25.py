"""Auditoria independente das divisões, pareamento e decisões da rodada 2.5."""
import json  # Lê apenas resultados completos.
import sys  # CLI e importação do protocolo de decisão.
from pathlib import Path  # Localiza evidências.
from collections import Counter  # Contagem independente por fase.
import numpy as np  # Índices absolutos dos folds.
from sklearn.datasets import load_iris  # Fonte independente do executor.
from sklearn.model_selection import StratifiedKFold  # Reconstitui divisões predefinidas.
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))  # Importa sem executar CLI.
from refine_experiments import choose_guarded, slug  # Mesma regra científica, reavaliada dos registros.


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())  # Orçamento declarado previamente.
    budget = manifest["budget"]  # Não infere desenho a partir dos resultados.
    records = [json.loads(p.read_text()) for p in (directory / "jobs").glob("*.json")]  # Unidades concluídas.
    x, y = load_iris(return_X_y=True)  # Dataset original, sem transformação.
    splits = list(StratifiedKFold(budget["outer_folds"], shuffle=True, random_state=20261).split(x, y))
    expected_stages = dict(screen=2 * budget["outer_folds"] * budget["inner_folds"] * 5 * len(budget["screen_seeds"]),
                           confirm=2 * budget["outer_folds"] * budget["inner_folds"] * 2 * len(budget["confirm_seeds"]),
                           external=4 * budget["outer_folds"] * len(budget["final_seeds"]),
                           controls=5 * budget["outer_folds"] * len(budget["final_seeds"]))
    if not (Counter(r["spec"]["stage"] for r in records) == expected_stages):  # Verificação ativa também com python -O.
        raise ValueError('Auditoria inválida: Não compensa falta de uma fase com excesso em outra.')
    for outer, (train, holdout) in enumerate(splits):
        inner = [(train[a].tolist(), train[b].tolist()) for a, b in StratifiedKFold(budget["inner_folds"], shuffle=True, random_state=12000 + outer).split(x[train], y[train])]
        for r in [r for r in records if r["spec"]["outer"] == outer]:
            s = r["spec"]  # Identidade dos dados efetivamente usados.
            if s["stage"] in ["screen", "confirm"]:
                if not ((s["train"], s["valid"]) in inner):  # Verificação ativa também com python -O.
                    raise ValueError('Auditoria inválida: Nenhum índice externo participa da seleção.')
                if not (s["seed"] in budget[s["stage"] + "_seeds"]):  # Verificação ativa também com python -O.
                    raise ValueError('Auditoria inválida: Seeds internas declaradas.')
            else:
                if not (s["train"] == train.tolist() and s["valid"] == holdout.tolist()):  # Verificação ativa também com python -O.
                    raise ValueError('Auditoria inválida: Mesmo teste por variante.')
                if not (s["seed"] in budget["final_seeds"]):  # Verificação ativa também com python -O.
                    raise ValueError('Auditoria inválida: Seeds finais independentes.')
        for condition in ["SBX", "vae_eda"]:
            decision = json.loads((directory / f"selection_{outer}_{slug(condition)}.json").read_text())  # Escolha registrada.
            confirmed = [r for r in records if r["spec"]["outer"] == outer and r["spec"]["condition"] == condition and r["spec"]["stage"] == "confirm"]
            recomputed = choose_guarded(confirmed)  # Reavalia usando somente confirmação interna.
            if not (decision["selected"] == recomputed["selected"] and decision["checks"] == recomputed["checks"]):  # Verificação ativa também com python -O.
                raise ValueError('Auditoria inválida: Sem promoção pelo teste.')
            paired = Counter((tuple(r["spec"]["train"]), tuple(r["spec"]["valid"]), r["spec"]["seed"]) for r in confirmed)
            if not (len(paired) == budget["inner_folds"] * len(budget["confirm_seeds"]) and set(paired.values()) == {2}):  # Verificação ativa também com python -O.
                raise ValueError('Auditoria inválida: Referência e finalista pareados.')
    return dict(jobs=len(records), stages=expected_stages, split_reconstruction=True, decision_reconstruction=True, paired_confirmation=True)


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2))  # Falha de contrato impede relatório positivo.
