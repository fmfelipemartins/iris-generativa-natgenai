"""Auditoria independente dos artefatos de uma avaliação aninhada concluída."""

import argparse  # Recebe uma pasta explícita de resultados.
import csv  # Lê tabelas sem depender do ambiente científico.
from hashlib import sha256  # Confere identidade dos checkpoints.
import json  # Manifestos, decisões e jobs.
import math  # Verificação de finitude.
from pathlib import Path  # Caminhos portáveis.


def audit(directory):
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )  # Protocolo declarado antes da execução.
    complete = json.loads(
        (directory / "COMPLETE.json").read_text(encoding="utf-8")
    )  # Não audita parcial como completo.
    assert manifest["phase"] == "nested"  # Este auditor verifica separação aninhada.
    expected_folds = set(
        range(manifest["budget"]["outer_folds"])
    )  # Numeração usada pelo executor.
    conditions = set(manifest["conditions"])  # Famílias solicitadas.
    final_seeds = set(
        manifest["budget"]["final_seeds"]
    )  # Replicações externas predefinidas.
    decisions = {}  # Seleção congelada antes de consultar o teste externo.
    for path in directory.glob("selection_*.json"):
        decision = json.loads(
            path.read_text(encoding="utf-8")
        )  # Metadados de cada fold/família.
        key = (
            decision["condition"],
            decision["outer"],
        )  # Chave científica, não nome do arquivo.
        assert key not in decisions  # Proíbe decisões duplicadas.
        decisions[key] = decision  # Guarda configuração escolhida.
    assert set(decisions) == {
        (c, f) for c in conditions for f in expected_folds
    }  # Todos os casos foram concluídos.
    external_splits = {}  # Divisões externas recuperadas dos próprios checkpoints.
    external_keys = set()  # Uma execução por condição/fold/seed/variante.
    internal_indices = []  # Índices internos compactos para checar exclusão externa.
    jobs = list((directory / "jobs").glob("*.json"))  # Ignora arquivos temporários.
    for path in jobs:
        record = json.loads(path.read_text(encoding="utf-8"))  # Lê um job por vez.
        spec = record["spec"]  # Configuração integral do job.
        digest = sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest()  # Mesmo identificador determinístico do executor.
        assert (
            path.stem == digest
        )  # Não aceita checkpoint renomeado ou de outra especificação.
        train, valid = (
            set(spec["train"]),
            set(spec["valid"]),
        )  # Índices absolutos na Iris.
        assert len(train) == len(spec["train"]) and len(valid) == len(
            spec["valid"]
        )  # Sem duplicação nos splits.
        assert train.isdisjoint(valid)  # Primeiro requisito de isolamento.
        evaluated = record[
            "evaluation_indices"
        ]  # Reais efetivamente usados nas métricas.
        assert set(evaluated).issubset(valid) and len(set(evaluated)) == len(
            evaluated
        )  # Amostra sem reposição do conjunto reservado.
        assert len(record["sample"]["x"]) == len(
            evaluated
        )  # Classes de origem balanceadas.
        for key in ("metrics", "candidate_metrics", "scale", "discriminators"):
            assert all(
                math.isfinite(value) for value in record.get(key, {}).values()
            )  # Nenhum NaN/inf nas métricas consolidadas.
        assert math.isfinite(record["loss"]) and math.isfinite(
            record["seconds"]
        )  # Critério e tempo válidos.
        if spec["variant"] != "initialization":
            assert (
                spec["options"]["pop_size"]
                == manifest["options"][spec["condition"]]["pop_size"]
            )  # Mesma população evolutiva entre fases.
            assert (
                "candidate_metrics" in record
            )  # Sem escolher variantes pelo sobrevivente apenas.
            assert (
                len(record["history"]) == spec["options"]["generations"]
            )  # Todas as gerações solicitadas concluídas.
        if spec["stage"] == "external":
            key = (
                spec["condition"],
                spec["outer"],
                spec["seed"],
                spec["variant"],
            )  # Identidade externa.
            assert key not in external_keys  # Sem inflação do número de replicações.
            external_keys.add(key)  # Conjunto efetivamente avaliado.
            split_key = (
                spec["condition"],
                spec["outer"],
            )  # Mesmo split entre seeds e variantes.
            if split_key in external_splits:
                assert external_splits[split_key] == (
                    train,
                    valid,
                )  # Pareamento verdadeiro dos índices.
            external_splits[split_key] = (
                train,
                valid,
            )  # Registra para verificar os folds internos.
            assert train | valid == set(range(150))  # Partição da Iris completa.
        else:
            internal_indices.append(
                (spec["condition"], spec["outer"], train | valid)
            )  # Apenas índices; evita carregar históricos na memória.
    for condition, outer, used in internal_indices:
        assert (
            used == external_splits[(condition, outer)][0]
        )  # Nenhum real externo foi usado para escolher a configuração.
    expected_external = {
        (condition, outer, seed, variant)
        for (condition, outer), decision in decisions.items()
        for seed in final_seeds
        for variant in {"reference", "initialization", decision["selected"]}
    }  # Referência deduplicada quando selecionada.
    assert (
        external_keys == expected_external
    )  # Nenhuma execução final faltante ou excedente.
    for condition in conditions:
        holdouts = [
            external_splits[(condition, fold)][1] for fold in expected_folds
        ]  # Conjuntos externos por fold.
        assert set.union(*holdouts) == set(range(150))  # Cobertura integral da base.
        assert (
            sum(map(len, holdouts)) == 150
        )  # Cada real fica no teste uma única vez nesta divisão.
    paired = list(
        csv.DictReader((directory / "paired_external.csv").open(encoding="utf-8"))
    )  # Diferenças selecionado-referência.
    keys = {
        (r["condition"], int(r["outer"]), int(r["seed"])) for r in paired
    }  # Identidade sem variante, pois a escolha varia por fold.
    assert (
        len(keys)
        == len(paired)
        == len(conditions) * len(expected_folds) * len(final_seeds)
    )  # Um par por unidade experimental.
    assert (
        len(jobs) == complete["completed_jobs"]
    )  # Marcador corresponde ao conjunto auditado.
    return dict(
        passed=True,
        jobs=len(jobs),
        decisions=len(decisions),
        external_runs=len(external_keys),
        paired_comparisons=len(paired),
        profile=manifest["profile"],
        folds=len(expected_folds),
        seeds_per_fold=len(final_seeds),
        note="Verificações estruturais e de finitude; não provam superioridade científica.",
    )  # Resultado explícito da auditoria.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)  # Interface curta de revisão.
    parser.add_argument("directory", type=Path)  # Pasta completa do protocolo aninhado.
    arguments = parser.parse_args()  # Entrada validada pela CLI.
    print(
        json.dumps(audit(arguments.directory), indent=2, ensure_ascii=False)
    )  # Saída pode ser preservada como evidência.
