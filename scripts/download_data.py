"""Descarga los CSV del dataset de Zenodo y verifica su integridad.

Uso:
    python scripts/download_data.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, RAW_FILES, ZENODO_FILES_BASE_URL  # noqa: E402

CHECKSUMS = {
    "X_train.csv": "0a79f0c2f95fcdb407f24b2c0ad7f7d1",
    "X_test.csv": "02328c4e09a78a41acb22ce17e6e0312",
    "y_train.csv": "90876921486a26a67bc1976da87305d9",
    "y_test.csv": "2f054c0e3d8cbbbd76adf30d13a7710a",
}


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(name: str) -> Path:
    destination = RAW_DATA_DIR / name
    if destination.exists():
        print(f"[=] {name} ya existe, se omite la descarga.")
        return destination

    url = f"{ZENODO_FILES_BASE_URL}/{name}/content"
    print(f"[+] Descargando {name} ...")
    urllib.request.urlretrieve(url, destination)

    expected = CHECKSUMS.get(name)
    if expected and md5_of(destination) != expected:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"El MD5 de {name} no coincide con el esperado.")
    print(f"[ok] {name} ({destination.stat().st_size} bytes)")
    return destination


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        download_file(name)
    print("\nDataset disponible en:", RAW_DATA_DIR)


if __name__ == "__main__":
    main()
