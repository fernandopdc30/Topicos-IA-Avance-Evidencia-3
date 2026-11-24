"""Interpretacion reproducible de clusters a partir de sus terminos clave."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TopicSuggestion:
    """Tipo funcional sugerido para un cluster y evidencia que lo sustenta."""

    name: str
    matched_keywords: tuple[str, ...]
    score: float
    subtype: str = ""
    confidence: float = 0.0
    review_required: bool = True


TOPIC_TAXONOMY: Mapping[str, Mapping[str, float]] = {
    "Buzones compartidos": {
        "shared mailbox": 5.0,
        "mailbox access": 4.0,
        "create shared": 3.0,
        "mailbox": 2.0,
        "buzon compartido": 5.0,
        "caixa compartilhada": 5.0,
    },
    "Fin de vida de equipos": {
        "end of life": 5.0,
        "nexthink": 4.0,
        "reservations": 3.0,
        "remove system": 3.0,
        "fin de vida": 5.0,
        "retiro de equipo": 4.0,
    },
    "Accesos y permisos": {
        "file share": 5.0,
        "share access": 4.0,
        "acesso pasta": 5.0,
        "pasta": 2.0,
        "fileserver": 2.0,
        "read write": 2.0,
        "carpeta compartida": 5.0,
        "archivo compartido": 5.0,
        "acceso a carpeta": 4.0,
        "pasta compartilhada": 5.0,
    },
    "Gestión de cuentas de usuario": {
        "basis user": 5.0,
        "light user": 5.0,
        "create windows account": 4.0,
        "user external": 3.0,
        "username": 1.0,
        "cuenta de usuario": 5.0,
        "crear usuario": 4.0,
        "usuario externo": 3.0,
    },
    "Correo y colaboración": {
        "email": 4.0,
        "mail": 3.0,
        "skype": 3.0,
        "outlook": 3.0,
        "o365": 4.0,
        "correo": 4.0,
        "correo electronico": 5.0,
    },
    "Directorio activo y permisos": {
        "active directory": 5.0,
        "directory": 2.0,
        "permission": 2.0,
        "group access": 3.0,
        "directorio activo": 5.0,
        "permisos de grupo": 3.0,
    },
    "Software e instalaciones": {
        "software": 4.0,
        "installation": 3.0,
        "install": 3.0,
        "update": 2.0,
        "instalar programa": 4.0,
        "instalacion": 3.0,
    },
    "Conectividad y acceso remoto": {
        "vpn": 4.0,
        "wifi": 4.0,
        "network": 3.0,
        "connection": 2.0,
        "sin internet": 4.0,
        "conexion remota": 3.0,
    },
    "Equipos e impresión": {
        "printer": 4.0,
        "impressora": 4.0,
        "computer": 3.0,
        "laptop": 3.0,
        "impresora": 4.0,
        "computadora": 3.0,
    },
}


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _suggest_subtype(topic: str, terms: Sequence[str]) -> str:
    combined = " ".join(terms)
    if topic == "Accesos y permisos":
        if any(token in combined for token in ("acesso", "pasta", "rede")):
            return "Accesos diversos en portugués"
        if any(token in combined for token in ("username", "following user", "read write")):
            return "Gestión detallada de permisos"
        return "Solicitudes breves de archivos compartidos"
    if topic == "Correo y colaboración":
        return "Incidencias generales de correo y mensajería"
    if topic == "Gestión de cuentas de usuario":
        return "Altas de usuarios internos y externos"
    if topic == "Buzones compartidos":
        return "Alta y acceso a buzones compartidos"
    if topic == "Fin de vida de equipos":
        return "Retiro de sistemas y reservas"
    return topic


def suggest_topic(terms: Sequence[str]) -> TopicSuggestion:
    """Sugiere un tipo funcional sin consultar las categorias humanas.

    Los terminos de mayor rango pesan mas. Si no hay evidencia suficiente, se
    devuelve ``Soporte general`` en lugar de forzar una categoria engañosa.
    """
    normalized_terms = [_normalize(term) for term in terms if str(term).strip()]
    scores: dict[str, float] = {}
    evidence: dict[str, set[str]] = {}
    total_terms = max(len(normalized_terms), 1)

    for topic, keywords in TOPIC_TAXONOMY.items():
        score = 0.0
        matched: set[str] = set()
        for rank, term in enumerate(normalized_terms):
            rank_weight = 1.0 + (total_terms - rank) / total_terms
            for keyword, keyword_weight in keywords.items():
                if _contains_phrase(term, keyword):
                    score += keyword_weight * rank_weight
                    matched.add(keyword)
        scores[topic] = score
        evidence[topic] = matched

    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]
    if best_score <= 0:
        return TopicSuggestion("Soporte general", (), 0.0, "Casos no especializados", 0.0, True)

    ordered_scores = sorted(scores.values(), reverse=True)
    second_score = ordered_scores[1] if len(ordered_scores) > 1 else 0.0
    confidence = best_score / (best_score + second_score) if second_score > 0 else 1.0
    matched_keywords = tuple(sorted(evidence[best_topic]))

    return TopicSuggestion(
        name=best_topic,
        matched_keywords=matched_keywords,
        score=float(best_score),
        subtype=_suggest_subtype(best_topic, normalized_terms),
        confidence=float(confidence),
        review_required=confidence < 0.65 or len(matched_keywords) < 2,
    )


def topics_for_clusters(terms: Mapping[int, Sequence[str]]) -> dict[int, TopicSuggestion]:
    """Interpreta todos los clusters conservando sus identificadores."""
    return {int(cluster_id): suggest_topic(cluster_terms) for cluster_id, cluster_terms in terms.items()}


__all__ = ["TOPIC_TAXONOMY", "TopicSuggestion", "suggest_topic", "topics_for_clusters"]
