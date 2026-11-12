"""Carga y union de los archivos CSV publicados en Zenodo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, RAW_FILES


def _read_raw_file(base_dir: Path, name: str) -> pd.DataFrame:
    path = base_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Ejecuta 'python scripts/download_data.py' primero."
        )
    return pd.read_csv(path)


def load_raw_tickets(raw_dir=None) -> pd.DataFrame:
    """Une X/y de train y test en un unico DataFrame de tickets.

    Devuelve columnas ``id``, ``text``, ``category_truth`` y ``split``.
    Las etiquetas se conservan solo como referencia de evaluacion.
    """
    base_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR

    frames = []
    for split in ("train", "test"):
        x = _read_raw_file(base_dir, f"X_{split}.csv")
        y = _read_raw_file(base_dir, f"y_{split}.csv")

        if len(x) != len(y) or not (x["id"].values == y["id"].values).all():
            raise ValueError(f"Los archivos X_{split} y y_{split} no estan alineados por 'id'.")

        merged = x.merge(y, on="id", how="left", validate="one_to_one")
        merged["split"] = split
        frames.append(merged)

    tickets = pd.concat(frames, ignore_index=True)
    tickets = tickets.dropna(subset=["text"]).drop_duplicates(subset=["id"]).reset_index(drop=True)
    return tickets


def load_processed_tickets(path) -> pd.DataFrame:
    """Carga el CSV procesado; falla con un mensaje claro si no existe."""
    path = pd.io.common.stringify_path(path)
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No se encontro {path}. Ejecuta 'python scripts/prepare_data.py' primero."
        ) from exc


__all__ = ["load_raw_tickets", "load_processed_tickets", "RAW_FILES"]
