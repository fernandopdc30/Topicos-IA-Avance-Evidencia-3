import numpy as np
import pytest

from src.clustering import (
    build_tfidf,
    cluster_texts,
    duplicate_sample_weights,
    project_to_2d,
    representative_documents,
    run_kmeans,
    top_terms_per_cluster,
)
from src.evaluation import compare_cluster_counts, external_metrics, internal_metrics

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


def test_run_kmeans_rejects_misaligned_weights():
    _, matrix = build_tfidf(TEXTS, min_df=1, max_df=1.0)
    with pytest.raises(ValueError, match="pesos"):
        run_kmeans(matrix, n_clusters=2, sample_weight=np.ones(2))


def test_build_tfidf_rejects_empty_corpus():
    with pytest.raises(ValueError, match="ticket"):
        build_tfidf([])


def test_duplicate_weights_reduce_template_dominance():
    weights = duplicate_sample_weights(["repetido", "repetido", "unico"])
    assert weights[0] == pytest.approx(weights[1])
    assert weights[0] < weights[2]
    assert weights.mean() == pytest.approx(1.0)


def test_projection_and_representatives_have_expected_shape():
    result = cluster_texts(TEXTS, n_clusters=2, min_df=1, max_df=1.0)
    coordinates, explained_variance = project_to_2d(result.matrix)
    representatives = representative_documents(
        result.matrix, result.model, result.labels, n_per_cluster=2
    )

    assert coordinates.shape == (len(TEXTS), 2)
    assert 0 <= explained_variance <= 1
    assert set(representatives) == {0, 1}
    assert all(1 <= len(rows) <= 2 for rows in representatives.values())
    assert all(distance >= 0 for rows in representatives.values() for _, distance in rows)


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


def test_compare_cluster_counts_returns_one_row_per_valid_k():
    _, matrix = build_tfidf(TEXTS, min_df=1, max_df=1.0)
    comparison = compare_cluster_counts(matrix, [2, 3], sample_size=None)

    assert [row["k"] for row in comparison] == [2, 3]
    assert all(np.isfinite(row["silhouette"]) for row in comparison)
    assert all(row["smallest_cluster"] >= 1 for row in comparison)
    assert all(0 < row["smallest_cluster_share"] <= 1 for row in comparison)
