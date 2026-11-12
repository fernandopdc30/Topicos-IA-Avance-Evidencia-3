"""Limpieza y normalizacion de texto para los tickets de soporte."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

import pandas as pd

_PII_TAGS = re.compile(
    r"\[\s*(?:ticket\s*id|name|company|location|email|phone|date|"
    r"ip|url|user|password|number|server|group|department|address|"
    r"hostname|domain|machine|system|path)\s*\]",
    flags=re.IGNORECASE,
)
_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_HTML = re.compile(r"<[^>]+>")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")

_BOILERPLATE = re.compile(
    r"\b(?:new\s+support\s+ticket\s+received|"
    r"a\s+support\s+ticket\s+was\s+forwarded\s+to\s+your\s+role|"
    r"support\s+ticket\s+received|"
    r"ticket\s+was\s+forwarded\s+to\s+your\s+role)\b",
    flags=re.IGNORECASE,
)

DEFAULT_MIN_TOKEN_LENGTH = 2


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def clean_text(text: object, min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH) -> str:
    """Normaliza un ticket a minusculas y palabras alfanumericas separadas por espacios.

    Se eliminan etiquetas de datos personales (por ejemplo ``[NAME]``),
    URLs, correos, HTML, acentos y signos de puntuacion. El texto resultante
    esta listo para ser vectorizado con TF-IDF.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    value = str(text).lower()
    value = _PII_TAGS.sub(" ", value)
    value = _BOILERPLATE.sub(" ", value)
    value = _URL.sub(" ", value)
    value = _EMAIL.sub(" ", value)
    value = _HTML.sub(" ", value)
    value = _strip_accents(value)
    value = _NON_ALNUM.sub(" ", value)
    value = _WHITESPACE.sub(" ", value).strip()

    if min_token_length > 1:
        tokens = [t for t in value.split(" ") if len(t) >= min_token_length]
        value = " ".join(tokens)

    return value


def clean_series(texts: Iterable, min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH) -> List[str]:
    """Aplica :func:`clean_text` a una coleccion de textos."""
    return [clean_text(text, min_token_length=min_token_length) for text in texts]


def add_clean_column(df: pd.DataFrame, source: str = "text", target: str = "clean_text") -> pd.DataFrame:
    """Devuelve una copia del DataFrame con la columna de texto limpio."""
    result = df.copy()
    result[target] = clean_series(result[source])
    return result
