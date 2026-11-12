"""Representacion TF-IDF y agrupamiento con K-Means."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import RANDOM_STATE, STOP_WORDS


@dataclass
class ClusterResult:
    """Artefactos producidos por el agrupamiento."""

    labels: np.ndarray
    model: KMeans
    vectorizer: TfidfVectorizer
    matrix: object
    terms: Dict[int, List[str]]


def build_tfidf(
    texts: Sequence[str],
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
    max_features: int | None = 5000,
    stop_words: Sequence[str] | None = STOP_WORDS,
) -> Tuple[TfidfVectorizer, object]:
    """Ajusta un vectorizador TF-IDF y transforma los textos."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
        stop_words=list(stop_words) if stop_words else None,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def run_kmeans(
    matrix,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
    n_init: int = 10,
) -> KMeans:
    """Ejecuta K-Means sobre la matriz TF-IDF."""
    if n_clusters < 1:
        raise ValueError("n_clusters debe ser mayor o igual a 1.")
    if matrix.shape[0] < n_clusters:
        raise ValueError("No hay suficientes documentos para el numero de clusters solicitado.")

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    model.fit(matrix)
    return model


def top_terms_per_cluster(
    vectorizer: TfidfVectorizer,
    model: KMeans,
    n_terms: int = 10,
) -> Dict[int, List[str]]:
    """Devuelve los terminos con mayor peso en el centroide de cada cluster."""
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    terms: Dict[int, List[str]] = {}
    for cluster_id, center in enumerate(model.cluster_centers_):
        top_indices = center.argsort()[::-1][:n_terms]
        terms[cluster_id] = feature_names[top_indices].tolist()
    return terms


def assign_to_nearest(matrix, model: KMeans) -> np.ndarray:
    """Asigna cada documento de ``matrix`` al cluster mas cercano."""
    return model.predict(matrix)


def cluster_texts(
    texts: Sequence[str],
    n_clusters: int,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
    max_features: int | None = 5000,
    n_terms: int = 10,
    random_state: int = RANDOM_STATE,
    stop_words: Sequence[str] | None = STOP_WORDS,
) -> ClusterResult:
    """Pipeline completo: TF-IDF + K-Means + terminos representativos."""
    vectorizer, matrix = build_tfidf(
        texts,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words=stop_words,
    )
    model = run_kmeans(matrix, n_clusters=n_clusters, random_state=random_state)
    terms = top_terms_per_cluster(vectorizer, model, n_terms=n_terms)
    return ClusterResult(
        labels=model.labels_,
        model=model,
        vectorizer=vectorizer,
        matrix=matrix,
        terms=terms,
    )
