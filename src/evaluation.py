"""Metricas internas y externas para evaluar el agrupamiento."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def _unique_count(labels: Sequence[int]) -> int:
    return len(set(np.asarray(labels).tolist()))


def internal_metrics(matrix, labels: Sequence[int], sample_size: int | None = 2000) -> Dict[str, float]:
    """Calcula metricas internas; devuelve ``nan`` cuando no son aplicables."""
    labels = np.asarray(labels)
    n_labels = _unique_count(labels)
    results: Dict[str, float] = {
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }

    if n_labels < 2 or n_labels >= len(labels):
        return results

    results["davies_bouldin"] = float(davies_bouldin_score(matrix.toarray() if hasattr(matrix, "toarray") else matrix, labels))
    results["calinski_harabasz"] = float(calinski_harabasz_score(matrix.toarray() if hasattr(matrix, "toarray") else matrix, labels))

    try:
        results["silhouette"] = float(
            silhouette_score(
                matrix,
                labels,
                sample_size=min(sample_size, len(labels)) if sample_size else None,
                random_state=42,
            )
        )
    except ValueError:
        results["silhouette"] = float("nan")

    return results


def external_metrics(labels: Sequence[int], truth: Sequence[str]) -> Dict[str, float]:
    """Contrasta la particion descubierta con las categorias humanas."""
    return {
        "adjusted_rand": float(adjusted_rand_score(truth, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(truth, labels)),
    }
