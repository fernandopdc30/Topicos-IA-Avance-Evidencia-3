"""Dashboard Streamlit para explorar agrupamientos de tickets de soporte."""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.clustering import (  # noqa: E402
    build_tfidf,
    cluster_texts,
    duplicate_sample_weights,
    project_to_2d,
    representative_documents,
)
from src.config import PROCESSED_TICKETS_PATH  # noqa: E402
from src.data_loader import load_processed_tickets, load_raw_tickets  # noqa: E402
from src.evaluation import compare_cluster_counts, external_metrics, internal_metrics  # noqa: E402
from src.interpretation import suggest_topic, topics_for_clusters  # noqa: E402
from src.text_cleaning import add_clean_column, clean_text  # noqa: E402

st.set_page_config(
    page_title="TicketScope · IA para soporte",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="auto",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
        [data-testid="stSidebar"] {border-right: 1px solid #dbe4f0;}
        [data-testid="stMetric"] {
            background: rgba(255,255,255,.72); border: 1px solid #dbe4f0;
            padding: 1rem 1.1rem; border-radius: 16px;
        }
        .hero {
            padding: 1.5rem 1.7rem; border-radius: 22px; margin-bottom: 1.2rem;
            background: linear-gradient(120deg, #0f172a 0%, #173b69 55%, #2563eb 100%);
            color: white; box-shadow: 0 14px 34px rgba(15, 23, 42, .14);
        }
        .hero h1 {margin: 0 0 .35rem; font-size: 2.1rem; color: white;}
        .hero p {margin: 0; color: #dbeafe; max-width: 900px; font-size: 1.02rem;}
        .eyebrow {font-size: .75rem; letter-spacing: .14em; font-weight: 700; color: #93c5fd;}
        .cluster-chip {
            display: inline-block; padding: .28rem .62rem; margin: .18rem .22rem .18rem 0;
            border-radius: 999px; background: #dbeafe; color: #1e3a8a; font-size: .84rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if PROCESSED_TICKETS_PATH.exists():
        tickets = load_processed_tickets(PROCESSED_TICKETS_PATH)
    else:
        tickets = add_clean_column(load_raw_tickets())
    return tickets[tickets["clean_text"].fillna("").str.len() > 0].reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_analysis(
    texts: tuple[str, ...],
    n_clusters: int,
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    max_features: int,
    n_terms: int,
):
    result = cluster_texts(
        list(texts),
        n_clusters=n_clusters,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        n_terms=n_terms,
    )
    coordinates, explained_variance = project_to_2d(result.matrix)
    representatives = representative_documents(result.matrix, result.model, result.labels)
    return result, coordinates, explained_variance, representatives


@st.cache_data(show_spinner=False)
def analyze_cluster_counts(
    texts: tuple[str, ...],
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    max_features: int,
    min_k: int,
    max_k: int,
) -> pd.DataFrame:
    _, matrix = build_tfidf(
        list(texts),
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    weights = duplicate_sample_weights(texts, strategy="balanced")
    return pd.DataFrame(
        compare_cluster_counts(
            matrix,
            range(min_k, max_k + 1),
            sample_weight=weights,
        )
    )


def cluster_title(cluster_id: int, terms: dict[int, list[str]]) -> str:
    keywords = " · ".join(terms.get(cluster_id, [])[:3])
    return f"Cluster {cluster_id}" + (f" — {keywords}" if keywords else "")


def metric_text(value: float) -> str:
    return "N/D" if not math.isfinite(value) else f"{value:.3f}"


def metrics_table(internal: dict[str, float], external: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Silhouette (coseno)", internal["silhouette"], "Mayor es mejor"),
            ("Davies-Bouldin", internal["davies_bouldin"], "Menor es mejor"),
            ("Calinski-Harabasz", internal["calinski_harabasz"], "Mayor es mejor"),
            ("Adjusted Rand Index", external["adjusted_rand"], "Referencia humana"),
            ("Normalized Mutual Info", external["normalized_mutual_info"], "Referencia humana"),
        ],
        columns=["Métrica", "Valor", "Lectura"],
    ).set_index("Métrica")


def render_keyword_chips(terms: list[str]) -> None:
    chips = "".join(f'<span class="cluster-chip">{term}</span>' for term in terms)
    st.markdown(chips or "Sin términos", unsafe_allow_html=True)


def main() -> None:
    apply_styles()
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">PROCESAMIENTO DE LENGUAJE NATURAL · APRENDIZAJE NO SUPERVISADO</div>
          <h1>🧭 TicketScope</h1>
          <p>Descubre temas recurrentes en tickets de soporte mediante TF-IDF y K-Means.
          Las categorías humanas se reservan exclusivamente para evaluar los resultados.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        tickets = load_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"No fue posible cargar los datos: {exc}")
        st.code("python scripts/download_data.py\npython scripts/prepare_data.py", language="bash")
        st.stop()

    with st.sidebar:
        st.title("⚙️ Configuración")
        st.caption("Ajusta el modelo y observa cómo cambia la agrupación.")
        n_clusters = st.slider(
            "Número de clusters (k)", 2, min(15, len(tickets) - 1), 11,
            help="Cantidad de grupos temáticos que intentará descubrir K-Means.",
        )
        use_bigrams = st.toggle(
            "Incluir bigramas", value=True,
            help="Incluye expresiones como «active directory» además de palabras individuales.",
        )
        with st.expander("Parámetros avanzados"):
            min_df = st.slider("Frecuencia mínima", 1, 10, 2)
            max_df = st.slider("Proporción máxima", 0.50, 1.00, 0.90, 0.05)
            max_features = st.select_slider(
                "Máximo de características",
                options=[500, 1000, 2000, 5000, 10000],
                value=5000,
            )
            n_terms = st.slider("Términos por cluster", 5, 20, 10)

        st.divider()
        st.caption("CONJUNTO DE DATOS")
        st.metric("Tickets válidos", f"{len(tickets):,}")
        duplicate_rate = tickets["clean_text"].duplicated().mean()
        st.metric("Textos duplicados", f"{duplicate_rate:.1%}")
        if "category_truth" in tickets.columns:
            st.metric("Categorías de referencia", tickets["category_truth"].nunique())
        st.caption("Zenodo 7648117 · datos anonimizados")
        st.caption("Los duplicados se ponderan para que las plantillas no dominen K-Means.")

    ngram_range = (1, 2) if use_bigrams else (1, 1)
    try:
        with st.spinner("Preparando el espacio TF-IDF y descubriendo clusters..."):
            result, coordinates, explained_variance, representatives = build_analysis(
                tuple(tickets["clean_text"].tolist()),
                n_clusters,
                ngram_range,
                min_df,
                max_df,
                max_features,
                n_terms,
            )
    except ValueError as exc:
        st.error(f"No se pudo construir el modelo: {exc}")
        st.info("Prueba reduciendo la frecuencia mínima o aumentando la proporción máxima.")
        st.stop()

    tickets = tickets.assign(cluster=result.labels)
    topic_suggestions = topics_for_clusters(result.terms)
    tickets["suggested_type"] = tickets["cluster"].map(
        lambda value: topic_suggestions[int(value)].name
    )
    tickets["suggested_subtype"] = tickets["cluster"].map(
        lambda value: topic_suggestions[int(value)].subtype
    )
    tickets["type_confidence"] = tickets["cluster"].map(
        lambda value: topic_suggestions[int(value)].confidence
    )
    tickets["type_review_required"] = tickets["cluster"].map(
        lambda value: topic_suggestions[int(value)].review_required
    )
    tickets["cluster_name"] = tickets["cluster"].map(
        lambda value: cluster_title(int(value), result.terms)
    )
    reference_summary = None
    if "category_truth" in tickets.columns:
        reference_summary = tickets.groupby("cluster")["category_truth"].agg(
            reference_category=lambda values: values.value_counts().index[0],
            reference_share=lambda values: values.value_counts().iloc[0] / len(values),
        )
        tickets["reference_category"] = tickets["cluster"].map(
            reference_summary["reference_category"]
        )
    internal = internal_metrics(result.matrix, result.labels)
    external = (
        external_metrics(result.labels, tickets["category_truth"].tolist())
        if "category_truth" in tickets.columns
        else {"adjusted_rand": float("nan"), "normalized_mutual_info": float("nan")}
    )

    summary_tab, map_tab, explore_tab, quality_tab, classify_tab = st.tabs(
        ["📊 Resumen", "🗺️ Mapa 2D", "🔎 Explorar", "🧪 Calidad y k", "✨ Clasificar"]
    )

    with summary_tab:
        st.subheader("Panorama del modelo")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Clusters", n_clusters)
        col2.metric("Tipos sugeridos", tickets["suggested_type"].nunique())
        col3.metric("Silhouette", metric_text(internal["silhouette"]))
        col4.metric("Davies-Bouldin", metric_text(internal["davies_bouldin"]))
        col5.metric("NMI vs. categorías", metric_text(external["normalized_mutual_info"]))

        left, right = st.columns(2)
        with left:
            st.subheader("Tamaño de los grupos")
            sizes = tickets["cluster_name"].value_counts().sort_index()
            st.bar_chart(sizes, horizontal=True)
        with right:
            st.subheader("Tickets por tipo sugerido")
            type_sizes = tickets["suggested_type"].value_counts().sort_values()
            st.bar_chart(type_sizes, horizontal=True)

        st.subheader("Tipos sugeridos frente a categorías humanas")
        if "category_truth" in tickets.columns:
            composition = pd.crosstab(
                tickets["suggested_type"], tickets["category_truth"], normalize="index"
            ).mul(100)
            st.dataframe(composition.style.format("{:.1f}%"), width="stretch")
            st.caption(
                "Los tipos se infieren desde palabras clave; las categorías humanas se "
                "muestran después y nunca intervienen en K-Means."
            )
        else:
            st.info("El dataset no incluye categorías de referencia.")

        export_columns = [
            "id", "text", "clean_text", "cluster", "suggested_type",
            "suggested_subtype", "type_confidence", "type_review_required",
            "cluster_name", "split"
        ]
        if "category_truth" in tickets.columns:
            export_columns.insert(5, "category_truth")
            export_columns.insert(6, "reference_category")
        st.download_button(
            "⬇️ Descargar todas las asignaciones",
            data=tickets[[column for column in export_columns if column in tickets]].to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name=f"tickets_agrupados_k{n_clusters}.csv",
            mime="text/csv",
        )

    with map_tab:
        st.subheader("Mapa semántico de los tickets")
        st.write(
            "Cada punto representa un ticket. La cercanía indica similitud aproximada en la "
            "proyección; puedes acercar el gráfico y consultar cada caso."
        )
        projection = pd.DataFrame(
            {"SVD 1": coordinates[:, 0], "SVD 2": coordinates[:, 1]}, index=tickets.index
        ).join(tickets[["cluster_name", "suggested_type", "suggested_subtype", "text"]])
        tooltip = [
            alt.Tooltip("suggested_type:N", title="Tipo sugerido"),
            alt.Tooltip("suggested_subtype:N", title="Subtipo"),
            alt.Tooltip("cluster_name:N", title="Cluster"),
            alt.Tooltip("text:N", title="Ticket"),
        ]
        if "category_truth" in tickets.columns:
            projection = projection.join(tickets[["category_truth"]])
            tooltip.append(alt.Tooltip("category_truth:N", title="Categoría real"))
        chart = (
            alt.Chart(projection)
            .mark_circle(size=60, opacity=0.64)
            .encode(
                x=alt.X("SVD 1:Q", title="Componente SVD 1"),
                y=alt.Y("SVD 2:Q", title="Componente SVD 2"),
                color=alt.Color("cluster_name:N", title="Cluster"),
                tooltip=tooltip,
            )
            .properties(height=570)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            f"La proyección explica {explained_variance:.1%} de la varianza. "
            "K-Means utiliza el espacio TF-IDF completo, no este mapa 2D."
        )

    with explore_tab:
        options = {
            f"{topic_suggestions[cluster_id].name} · Cluster {cluster_id}": cluster_id
            for cluster_id in sorted(tickets["cluster"].unique())
        }
        selected_title = st.selectbox("Selecciona un cluster", options.keys())
        selected = options[selected_title]
        cluster_tickets = tickets[tickets["cluster"] == selected]

        st.markdown(f"### {topic_suggestions[selected].name}")
        st.markdown(f"**Subtipo:** {topic_suggestions[selected].subtype}")
        status = "Revisión recomendada" if topic_suggestions[selected].review_required else "Confianza alta"
        st.caption(
            f"Confianza de interpretación: {topic_suggestions[selected].confidence:.0%} · {status}"
        )
        evidence = topic_suggestions[selected].matched_keywords
        if evidence:
            st.caption("Tipo inferido por: " + ", ".join(evidence))

        term_col, count_col, reference_col = st.columns([3, 1, 1.4])
        with term_col:
            st.markdown("**Términos característicos**")
            render_keyword_chips(result.terms.get(selected, []))
        with count_col:
            st.metric("Tickets del cluster", f"{len(cluster_tickets):,}")
        with reference_col:
            if reference_summary is not None:
                st.metric(
                    "Categoría humana predominante",
                    reference_summary.loc[selected, "reference_category"],
                    f"{reference_summary.loc[selected, 'reference_share']:.0%} del cluster",
                )

        st.markdown("#### Casos más representativos")
        representative_pairs = representatives[selected]
        representative_rows = tickets.iloc[[index for index, _ in representative_pairs]].copy()
        representative_rows.insert(0, "posición", range(1, len(representative_rows) + 1))
        representative_rows["distancia"] = [distance for _, distance in representative_pairs]
        representative_columns = ["posición", "id", "text", "distancia"]
        if "category_truth" in tickets.columns:
            representative_columns.insert(3, "category_truth")
        st.dataframe(
            representative_rows[representative_columns],
            width="stretch",
            hide_index=True,
            column_config={"distancia": st.column_config.NumberColumn(format="%.3f")},
        )

        st.markdown("#### Todos los tickets")
        search = st.text_input("Buscar dentro del cluster", placeholder="correo, acceso, software...")
        view = cluster_tickets
        if search.strip():
            view = view[
                view["text"].str.contains(search.strip(), case=False, na=False, regex=False)
            ]
        view_columns = ["id", "text"]
        if "category_truth" in view.columns:
            view_columns.append("category_truth")
        st.dataframe(view[view_columns], width="stretch", hide_index=True, height=360)
        st.download_button(
            "⬇️ Descargar este cluster",
            data=cluster_tickets.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"cluster_{selected}.csv",
            mime="text/csv",
        )

    with quality_tab:
        st.subheader("Métricas del modelo actual")
        st.dataframe(
            metrics_table(internal, external).style.format({"Valor": "{:.4f}"}, na_rep="N/D"),
            width="stretch",
        )
        st.info(
            "No existe un único k correcto. La métrica ayuda a comparar, pero la decisión "
            "también debe considerar la interpretabilidad y el equilibrio de los grupos."
        )

        st.subheader("Interpretación de los clusters")
        interpretation_rows = []
        for cluster_id in sorted(result.terms):
            row = {
                "Cluster": cluster_id,
                "Tipo sugerido": topic_suggestions[cluster_id].name,
                "Subtipo": topic_suggestions[cluster_id].subtype,
                "Confianza": topic_suggestions[cluster_id].confidence,
                "Estado": (
                    "Revisar" if topic_suggestions[cluster_id].review_required else "Confiable"
                ),
                "Evidencia": ", ".join(topic_suggestions[cluster_id].matched_keywords),
                "Términos principales": ", ".join(result.terms[cluster_id][:5]),
            }
            if reference_summary is not None:
                row["Categoría humana predominante"] = reference_summary.loc[
                    cluster_id, "reference_category"
                ]
                row["Coincidencia interna"] = reference_summary.loc[
                    cluster_id, "reference_share"
                ]
            interpretation_rows.append(row)
        interpretation = pd.DataFrame(interpretation_rows)
        column_config = {}
        if "Coincidencia interna" in interpretation.columns:
            column_config["Coincidencia interna"] = st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="percent"
            )
        column_config["Confianza"] = st.column_config.ProgressColumn(
            min_value=0.0, max_value=1.0, format="percent"
        )
        st.dataframe(
            interpretation,
            width="stretch",
            hide_index=True,
            column_config=column_config,
        )

        st.subheader("Comparar números de clusters")
        max_candidate = min(12, len(tickets) - 1)
        candidate_range = st.slider("Rango de k", 2, max_candidate, (2, max_candidate))
        analysis_key = (ngram_range, min_df, max_df, max_features, candidate_range)
        if st.button("Ejecutar comparación", type="primary"):
            with st.spinner("Entrenando modelos candidatos..."):
                st.session_state["k_analysis"] = analyze_cluster_counts(
                    tuple(tickets["clean_text"].tolist()),
                    ngram_range,
                    min_df,
                    max_df,
                    max_features,
                    candidate_range[0],
                    candidate_range[1],
                )
                st.session_state["k_analysis_key"] = analysis_key

        if st.session_state.get("k_analysis_key") == analysis_key:
            comparison = st.session_state["k_analysis"]
            min_balanced_size = max(10, math.ceil(len(tickets) * 0.02))
            balanced = comparison[comparison["smallest_cluster"] >= min_balanced_size]
            candidates = balanced if not balanced.empty else comparison
            recommendation = int(candidates.loc[candidates["silhouette"].idxmax(), "k"])
            st.success(
                f"Recomendación equilibrada: k = {recommendation}. Se exige que el cluster "
                f"menor tenga al menos {min_balanced_size} tickets (2% del corpus)."
            )
            chart_left, chart_right = st.columns(2)
            chart_left.line_chart(comparison.set_index("k")[["silhouette"]])
            chart_right.line_chart(comparison.set_index("k")[["inertia"]])
            st.dataframe(
                comparison.style.format(
                    {
                        "silhouette": "{:.4f}",
                        "inertia": "{:.2f}",
                        "smallest_cluster_share": "{:.1%}",
                    }
                ).highlight_max(subset=["silhouette"]),
                width="stretch",
                hide_index=True,
            )

    with classify_tab:
        st.subheader("Clasificar un ticket nuevo")
        st.write("El ticket se asigna al centroide TF-IDF más cercano sin reentrenar el modelo.")
        new_text = st.text_area(
            "Descripción del incidente",
            placeholder="Ejemplo: no puedo acceder a mi correo corporativo desde esta mañana",
            height=140,
        )
        if st.button("Clasificar ticket", type="primary"):
            cleaned = clean_text(new_text)
            if not cleaned:
                st.warning("Escribe un texto con términos útiles para poder clasificarlo.")
            else:
                vector = result.vectorizer.transform([cleaned])
                if vector.nnz == 0:
                    st.warning(
                        "Los términos no aparecen en el vocabulario. Agrega más detalle técnico."
                    )
                else:
                    distances = result.model.transform(vector)[0]
                    ordered = distances.argsort()
                    cluster_id = int(ordered[0])
                    best = float(distances[cluster_id])
                    runner_up = float(distances[ordered[1]])
                    separation = max(0.0, (runner_up - best) / runner_up) if runner_up else 0.0
                    direct_topic = suggest_topic([cleaned])
                    cluster_topic = topic_suggestions[cluster_id]
                    final_topic = (
                        direct_topic if direct_topic.name != "Soporte general" else cluster_topic
                    )
                    st.success(
                        f"Tipo sugerido: {final_topic.name} · {final_topic.subtype} · "
                        f"Cluster {cluster_id}"
                    )
                    if final_topic.name != cluster_topic.name:
                        st.info(
                            f"El texto indica «{final_topic.name}», aunque por similitud global "
                            f"quedó en un cluster de «{cluster_topic.name}»."
                        )
                    distance_col, separation_col = st.columns(2)
                    distance_col.metric("Distancia al centroide", f"{best:.3f}")
                    separation_col.metric("Separación del 2.º cluster", f"{separation:.1%}")
                    st.markdown(f"**Texto normalizado:** `{cleaned}`")
                    render_keyword_chips(result.terms.get(cluster_id, []))
                    st.caption(
                        "La separación compara los dos centroides más cercanos; no es una "
                        "probabilidad de acierto."
                    )


if __name__ == "__main__":
    main()
