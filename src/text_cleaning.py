"""Limpieza y normalizacion de texto para los tickets de soporte."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

import pandas as pd

# El dataset reemplaza informacion sensible con etiquetas entre corchetes. Se
# elimina cualquier etiqueta, no solo una lista cerrada, para cubrir variantes
# como [USERNAME], [AWS_ACCESS_KEY], [IP_ADDRESS], [IMAGE] y futuras etiquetas.
_PII_TAGS = re.compile(r"\[[^\[\]\r\n]{1,80}\]", flags=re.IGNORECASE)
_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_HTML = re.compile(r"<[^>]+>")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")

_BOILERPLATE = re.compile(
    r"\b(?:new\s+support\s+ticket\s+received|"
    r"a\s+support\s+ticket\s+was\s+forwarded\s+to\s+your\s+role|"
    r"a\s+support\s+ticket\s+was\s+forwarded\s+to\s+you|"
    r"support\s+ticket\s+received|"
    r"ticket\s+was\s+forwarded\s+to\s+your\s+role|"
    r"initiated\s+by\s+git|service\s+request|automated\s+ticket\s+creation|"
    r"additional\s+information|desired\s+date|requestor|comment)\b",
    flags=re.IGNORECASE,
)

_FORM_FIELDS = re.compile(
    r"\b(?:user\s*name|first\s*name|last\s*name|cost\s*center|"
    r"expiration\s+date|mailaddress|ailaddress|ip\s+address|access\s+type|"
    r"department|office|function)\b",
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
    if text is None:
        return ""
    try:
        if bool(pd.isna(text)):
            return ""
    except (TypeError, ValueError):
        pass

    value = str(text).lower()
    value = _PII_TAGS.sub(" ", value)
    value = _BOILERPLATE.sub(" ", value)
    value = _FORM_FIELDS.sub(" ", value)
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
