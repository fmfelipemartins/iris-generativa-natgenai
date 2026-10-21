"""Audita e resume a iteração 2.5 sem treinar modelos ou escolher variantes."""
import argparse  # Relatório explícito após a execução.
import json  # Evidências brutas e decisões congeladas.
from pathlib import Path  # Arquivos portáveis.
from hashlib import sha256  # Verifica identidade dos checkpoints.
import numpy as np  # Comparações e agregações.
import pandas as pd  # Tabelas descritivas.
from refine_experiments import write_json, slug  # Escrita atômica e nomes existentes.


def report(directory, output):
    manifest = json.loads((directory / "manifest.json").read_text())  # Protocolo prévio.
    complete = json.loads((directory / "COMPLETE.json").read_text())  # Exige término declarado.
    b = manifest["budget"]  # Orçamento congelado, não inferido das saídas.
    paths = sorted((directory / "jobs").glob("*.json"))  # Jobs atômicos completos.
    if not (len(paths) == b["max_jobs"] == complete["completed_jobs"]):  # Verificação ativa também com python -O.
        raise ValueError('Auditoria inválida: Não aceita rodada incompleta.')
    output.mkdir(parents=True, exist_ok=True)  # Resumo separado dos jobs volumosos.
    diagnostics, latents, training, utility, controls, selected, checks = [], [], [], [], [], [], []
    rows = []  # Conserva identidade de cada unidade experimental.
    for path in paths:
        r = json.loads(path.read_text())  # Leitura de um arquivo por vez.
        s = r["spec"]  # Chave de execução.
        if not (path.stem == sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()):  # Verificação ativa também com python -O.
            raise ValueError('Auditoria inválida: Nome íntegro.')
        if not (not set(s["train"]) & set(s["valid"])):  # Verificação ativa também com python -O.
            raise ValueError('Auditoria inválida: Sem sobreposição de treino/validação.')
        if not (set(r["evaluation_indices"]) <= set(s["valid"])):  # Verificação ativa também com python -O.
            raise ValueError('Auditoria inválida: Amostra vem do holdout declarado.')
        if not (np.isfinite(r["loss"])):  # Verificação ativa também com python -O.
            raise ValueError('Auditoria inválida: Não aceita falhas como resultados.')
        key = {k: s[k] for k in ["condition", "variant", "stage", "outer", "seed"]}  # Pareamento explícito.
        row = dict(key, loss=r["loss"], survivor_loss=r.get("survivor_loss", r["loss"]), candidate_loss=r.get("candidate_loss"), seconds=r["seconds"], **r["metrics"], **r.get("discriminators", {}))
        row.update({"candidate_" + k: v for k, v in r.get("candidate_metrics", {}).items()})  # Candidatos separados.
        if r.get("history"):
            row["founder_fraction"] = r["history"][-1]["founder_fraction"]  # Retenção da inicialização.
        if r.get("training_history"):
            row["optimizer_steps"] = r["training_history"][-1]["optimizer_steps"]  # Orçamento observado.
        rows.append(row)  # Uma linha por job.
        if s["stage"] in ["external", "controls"]:
            for dest, name in [(diagnostics, "diagnostics"), (latents, "latent_diagnostics")]:
                dest.extend(dict(key, **d) for d in r.get(name, []) if d["generation"] in manifest["checkpoints"])  # Mesma trajetória.
            utility.append(dict(key, synthetic_utility=r["utility"]["synthetic"], real_utility=r["utility"]["real"], heldout_near_copy=float(np.mean([d["near_copy_rate"] for d in r["heldout_proximity"]]))))  # Secundárias.
            if s["stage"] == "controls":
                controls.append(row)  # Não mistura escore direto com composto no resumo de seleção.
            else:
                decision = json.loads((directory / f"selection_{s['outer']}_{slug(s['condition'])}.json").read_text())
                if s["variant"] == decision["selected"]:
                    selected.append(row)  # Procedimento escolhido somente com dados internos.
    frame = pd.DataFrame(rows)  # Tabela de todas as execuções.
    for name, values in [("runs", rows), ("selected_procedure", selected), ("controls", controls), ("checkpoints", diagnostics), ("latent_checkpoints", latents), ("utility", utility)]:
        pd.DataFrame(values).to_csv(output / f"{name}.csv", index=False)  # Resultados sem interpretação causal automática.
    for path in sorted(directory.glob("selection_*.json")):
        decision = json.loads(path.read_text())  # Histórico interno congelado.
        write_json(output / path.name, decision)  # Preserva todos os motivos.
        for variant, flags in decision["checks"].items():
            checks.append(dict(condition=decision["condition"], outer=decision["outer"], variant=variant, selected=decision["selected"], **flags))
    pd.DataFrame(checks).to_csv(output / "selection_checks.csv", index=False)  # Guardas satisfeitas e rejeitadas.
    for name in ["manifest.json", "COMPLETE.json", "paired_external.csv", "conditional_uncertainty.csv"]:
        (output / name).write_bytes((directory / name).read_bytes())  # Evidência compacta do protocolo.
    audit = dict(jobs=len(paths), spec_hashes_valid=True, train_valid_disjoint=True, finite_losses=True,
                 expected_jobs=b["max_jobs"], defaults_changed=False, limitation="Historical Iris; exploratory evidence")
    write_json(output / "audit.json", audit)  # Somente após todos os contratos verificados.
    print(json.dumps(audit))  # Confirmação curta e legível.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)  # Entrada explícita.
    parser.add_argument("directory", type=Path)  # Pasta da execução finalizada.
    parser.add_argument("output", type=Path)  # Destino do resumo versionável.
    args = parser.parse_args()  # Não inicia geração.
    report(args.directory, args.output)  # Audita antes de publicar resultados.
