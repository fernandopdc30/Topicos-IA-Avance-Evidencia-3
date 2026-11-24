"""Representacion TF-IDF y agrupamiento con K-Means."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
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
    sample_weights: np.ndarray


def duplicate_sample_weights(
    texts: Sequence[str], strategy: str = "balanced"
) -> np.ndarray:
    """Reduce el dominio de textos duplicados sin eliminarlos del dataset.

    ``balanced`` aplica una correccion moderada de cuarta raiz: un texto
    repetido ``n`` veces aporta en total ``n**0.75`` en vez de ``n``.
    """
    if strategy not in {"none", "balanced", "sqrt", "inverse"}:
        raise ValueError(
            "La estrategia debe ser 'none', 'balanced', 'sqrt' o 'inverse'."
        )
    counts = Counter(str(text) for text in texts)
    if strategy == "none":
        weights = np.ones(len(texts), dtype=float)
    elif strategy == "inverse":
        weights = np.asarray([1.0 / counts[str(text)] for text in texts], dtype=float)
    elif strategy == "balanced":
        weights = np.asarray(
            [counts[str(text)] ** -0.25 for text in texts], dtype=float
        )
    else:
        weights = np.asarray(
            [1.0 / np.sqrt(counts[str(text)]) for text in texts], dtype=float
        )
    if weights.size:
        weights /= weights.mean()
    return weights


def build_tfidf(
    texts: Sequence[str],
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
    max_features: int | None = 5000,
    stop_words: Sequence[str] | None = STOP_WORDS,
) -> Tuple[TfidfVectorizer, object]:
    """Ajusta un vectorizador TF-IDF y transforma los textos."""
    if not texts:
        raise ValueError("Se necesita al menos un ticket para construir TF-IDF.")
    if not any(str(text).strip() for text in texts):
        raise ValueError("Los tickets no contienen texto util para vectorizar.")

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
    sample_weight: Sequence[float] | None = None,
) -> KMeans:
    """Ejecuta K-Means sobre la matriz TF-IDF."""
    if n_clusters < 1:
        raise ValueError("n_clusters debe ser mayor o igual a 1.")
    if matrix.shape[0] < n_clusters:
        raise ValueError("No hay suficientes documentos para el numero de clusters solicitado.")
    if sample_weight is not None and len(sample_weight) != matrix.shape[0]:
        raise ValueError("Los pesos deben tener la misma cantidad de filas que la matriz.")

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    if sample_weight is not None and len(sample_weight) != matrix.shape[0]:
        raise ValueError("Los pesos y la matriz deben tener la misma cantidad de filas.")
    model.fit(matrix, sample_weight=sample_weight)
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


def representative_documents(
    matrix,
    model: KMeans,
    labels: Sequence[int],
    n_per_cluster: int = 5,
) -> Dict[int, List[Tuple[int, float]]]:
    """Devuelve los tickets mas cercanos al centroide de cada cluster."""
    labels_array = np.asarray(labels)
    if matrix.shape[0] != len(labels_array):
        raise ValueError("La matriz y las etiquetas deben tener la misma cantidad de filas.")
    if n_per_cluster < 1:
        raise ValueError("n_per_cluster debe ser mayor o igual a 1.")

    distances = model.transform(matrix)
    representatives: Dict[int, List[Tuple[int, float]]] = {}
    for cluster_id in range(model.n_clusters):
        indices = np.flatnonzero(labels_array == cluster_id)
        ordered = indices[np.argsort(distances[indices, cluster_id])]
        representatives[cluster_id] = [
            (int(index), float(distances[index, cluster_id]))
            for index in ordered[:n_per_cluster]
        ]
    return representatives


def project_to_2d(matrix, random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, float]:
    """Proyecta la matriz TF-IDF a dos dimensiones mediante Truncated SVD."""
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("No se puede proyectar una matriz vacia.")

    n_components = min(2, matrix.shape[1])
    reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
    coordinates = reducer.fit_transform(matrix)
    if n_components == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(matrix.shape[0])])
    explained_variance = float(reducer.explained_variance_ratio_.sum())
    return coordinates, explained_variance


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
    duplicate_weighting: str = "balanced",
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
    sample_weights = duplicate_sample_weights(texts, strategy=duplicate_weighting)
    model = run_kmeans(
        matrix,
        n_clusters=n_clusters,
        random_state=random_state,
        sample_weight=sample_weights,
    )
    terms = top_terms_per_cluster(vectorizer, model, n_terms=n_terms)
    return ClusterResult(
        labels=model.labels_,
        model=model,
        vectorizer=vectorizer,
        matrix=matrix,
        terms=terms,
        sample_weights=sample_weights,
    )


__all__ = [
    "ClusterResult",
    "assign_to_nearest",
    "build_tfidf",
    "cluster_texts",
    "duplicate_sample_weights",
    "project_to_2d",
    "representative_documents",
    "run_kmeans",
    "top_terms_per_cluster",
]
