"""Metricas internas y externas para evaluar el agrupamiento."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.clustering import run_kmeans
from src.config import RANDOM_STATE


def _unique_count(labels: Sequence[int]) -> int:
    return len(set(np.asarray(labels).tolist()))


def internal_metrics(matrix, labels: Sequence[int], sample_size: int | None = 2000) -> Dict[str, float]:
    """Calcula metricas internas; devuelve ``nan`` cuando no son aplicables."""
    labels = np.asarray(labels)
    if matrix.shape[0] != len(labels):
        raise ValueError("La matriz y las etiquetas deben tener la misma cantidad de filas.")
    n_labels = _unique_count(labels)
    results: Dict[str, float] = {
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }

    if n_labels < 2 or n_labels >= len(labels):
        return results

    metric_matrix = matrix
    if hasattr(matrix, "toarray"):
        max_components = min(50, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if max_components >= 2 and matrix.shape[1] > 100:
            metric_matrix = TruncatedSVD(
                n_components=max_components, random_state=RANDOM_STATE
            ).fit_transform(matrix)
        else:
            metric_matrix = matrix.toarray()
    results["davies_bouldin"] = float(davies_bouldin_score(metric_matrix, labels))
    results["calinski_harabasz"] = float(calinski_harabasz_score(metric_matrix, labels))

    try:
        results["silhouette"] = float(
            silhouette_score(
                matrix,
                labels,
                metric="cosine",
                sample_size=min(sample_size, len(labels)) if sample_size else None,
                random_state=RANDOM_STATE,
            )
        )
    except ValueError:
        results["silhouette"] = float("nan")

    return results


def external_metrics(labels: Sequence[int], truth: Sequence[str]) -> Dict[str, float]:
    """Contrasta la particion descubierta con las categorias humanas."""
    if len(labels) != len(truth):
        raise ValueError("Las etiquetas y las categorias deben tener igual longitud.")
    return {
        "adjusted_rand": float(adjusted_rand_score(truth, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(truth, labels)),
    }


def compare_cluster_counts(
    matrix,
    cluster_counts: Sequence[int],
    sample_size: int | None = 1500,
    random_state: int = RANDOM_STATE,
    sample_weight: Sequence[float] | None = None,
) -> list[dict[str, float | int]]:
    """Compara valores de k con Silhouette coseno, inercia y balance."""
    results: list[dict[str, float | int]] = []
    for n_clusters in sorted(set(int(value) for value in cluster_counts)):
        if n_clusters < 2 or n_clusters >= matrix.shape[0]:
            continue
        model = run_kmeans(
            matrix,
            n_clusters=n_clusters,
            random_state=random_state,
            sample_weight=sample_weight,
        )
        score = silhouette_score(
            matrix,
            model.labels_,
            metric="cosine",
            sample_size=min(sample_size, matrix.shape[0]) if sample_size else None,
            random_state=random_state,
        )
        counts = np.bincount(model.labels_, minlength=n_clusters)
        results.append(
            {
                "k": n_clusters,
                "silhouette": float(score),
                "inertia": float(model.inertia_),
                "smallest_cluster": int(counts.min()),
                "smallest_cluster_share": float(counts.min() / matrix.shape[0]),
            }
        )
    if not results:
        raise ValueError("No hay valores de k validos para comparar.")
    return results


__all__ = ["compare_cluster_counts", "external_metrics", "internal_metrics"]
