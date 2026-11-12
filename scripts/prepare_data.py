"""Prepara el dataset: une train/test y agrega el texto normalizado.

Uso:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import PROCESSED_DATA_DIR, PROCESSED_TICKETS_PATH, RAW_DATA_DIR  # noqa: E402
from src.data_loader import load_raw_tickets  # noqa: E402
from src.text_cleaning import add_clean_column  # noqa: E402


def main() -> None:
    if not RAW_DATA_DIR.exists() or not any(RAW_DATA_DIR.glob("*.csv")):
        raise SystemExit(
            "No hay archivos crudos. Ejecuta primero 'python scripts/download_data.py'."
        )

    tickets = load_raw_tickets()
    tickets = add_clean_column(tickets)
    tickets = tickets[tickets["clean_text"].str.len() > 0].reset_index(drop=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tickets.to_csv(PROCESSED_TICKETS_PATH, index=False)

    print(f"Tickets procesados: {len(tickets)}")
    print(f"Duplicados de texto: {tickets['text'].duplicated().sum()}")
    print("Distribucion por split:")
    print(tickets["split"].value_counts().to_string())
    print("\nCategorias de referencia (no usadas para agrupar):")
    print(tickets["category_truth"].value_counts().to_string())
    print(f"\nGuardado en: {PROCESSED_TICKETS_PATH}")


if __name__ == "__main__":
    main()
