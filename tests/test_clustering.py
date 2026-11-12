import numpy as np
import pytest

from src.clustering import build_tfidf, cluster_texts, run_kmeans, top_terms_per_cluster
from src.evaluation import external_metrics, internal_metrics

TEXTS = [
    "no puedo acceder a la red wifi",
    "error de conexion a internet en la oficina",
    "solicitud de instalacion de software ofimatico",
    "actualizar programa de contabilidad",
    "olvide mi contrasena del correo",
    "no puedo entrar a mi correo corporativo",
]


def test_build_tfidf_returns_matrix():
    vectorizer, matrix = build_tfidf(TEXTS, min_df=1, max_df=1.0)
    assert matrix.shape[0] == len(TEXTS)
    assert vectorizer.get_feature_names_out().size > 0


def test_cluster_texts_labels_and_terms():
    result = cluster_texts(TEXTS, n_clusters=2, min_df=1, max_df=1.0, n_terms=3)
    assert len(result.labels) == len(TEXTS)
    assert set(result.labels).issubset({0, 1})
    assert set(result.terms.keys()) == {0, 1}
    assert all(len(terms) > 0 for terms in result.terms.values())


def test_run_kmeans_rejects_more_clusters_than_documents():
    vectorizer, matrix = build_tfidf(TEXTS, min_df=1, max_df=1.0)
    with pytest.raises(ValueError):
        run_kmeans(matrix, n_clusters=100)


def test_internal_metrics_not_applicable_for_single_cluster():
    vectorizer, matrix = build_tfidf(TEXTS, min_df=1, max_df=1.0)
    metrics = internal_metrics(matrix, np.zeros(len(TEXTS), dtype=int))
    assert np.isnan(metrics["silhouette"])
    assert np.isnan(metrics["davies_bouldin"])


def test_internal_metrics_valid_partition():
    vectorizer, matrix = build_tfidf(TEXTS, min_df=1, max_df=1.0)
    labels = np.array([0, 0, 1, 1, 0, 0])
    metrics = internal_metrics(matrix, labels)
    assert np.isfinite(metrics["davies_bouldin"])
    assert np.isfinite(metrics["calinski_harabasz"])


def test_external_metrics_identical_partitions():
    labels = [0, 0, 1, 1]
    truth = ["a", "a", "b", "b"]
    metrics = external_metrics(labels, truth)
    assert metrics["adjusted_rand"] == pytest.approx(1.0)
    assert metrics["normalized_mutual_info"] == pytest.approx(1.0)
