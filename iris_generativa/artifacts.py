"""Exportação explícita para comparar execuções sem depender de tabelas truncadas."""

from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import pandas as pd
from .extended_experiments import run_metadata


def export_execution(root, runs, settings, tables):
    """Cria diretório exclusivo por execução; nunca sobrescreve o histórico."""
    root = Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = root / "results" / f"execution_{settings['profile']}_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    metadata = run_metadata(settings)
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        metadata["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        metadata["git_commit"] = None
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for number, (condition, result) in enumerate(runs.items()):
        destination = target / f"condition_{number}"
        destination.mkdir()
        (destination / "condition.txt").write_text(condition, encoding="utf-8")
        for key in (
            "history",
            "training_history",
            "offspring_events",
            "diagnostics",
            "latent_diagnostics",
        ):
            result[key].to_csv(destination / f"{key}.csv", index=False)
        pd.DataFrame(result["synthetic_x"]).assign(
            task=result["synthetic_species"]
        ).to_csv(destination / "synthetic.csv", index=False)
        for task, candidates in result["final_candidates"].items():
            for stage, values in candidates.items():
                pd.DataFrame(values).to_csv(
                    destination / f"candidates_task{task}_{stage}.csv", index=False
                )
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(target / f"{name}.csv", index=True)
    return target
