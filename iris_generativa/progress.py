"""Progresso opcional, sem interferência no estado aleatório."""

from tqdm.auto import tqdm

from .config import SHOW_PROGRESS


def progress(iterable, description, enabled=None, total=None, leave=True):
    """Conta unidades concluídas; usa widget quando disponível e texto no terminal."""
    enabled = SHOW_PROGRESS if enabled is None else enabled
    return tqdm(
        iterable,
        desc=description,
        total=total,
        disable=not enabled,
        leave=leave,
        dynamic_ncols=True,
    )
