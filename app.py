"""Dashboard Streamlit: agrupador de tickets de soporte (demo funcional ~50%).

Ejecutar:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.clustering import cluster_texts  # noqa: E402
from src.config import PROCESSED_TICKETS_PATH, RAW_DATA_DIR  # noqa: E402
from src.data_loader import load_processed_tickets, load_raw_tickets  # noqa: E402
from src.evaluation import external_metrics, internal_metrics  # noqa: E402
from src.text_cleaning import add_clean_column, clean_text  # noqa: E402

st.set_page_config(page_title="Agrupador de tickets de soporte", page_icon="🗂️", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if PROCESSED_TICKETS_PATH.exists():
        tickets = load_processed_tickets(PROCESSED_TICKETS_PATH)
    else:
        tickets = add_clean_column(load_raw_tickets())
    return tickets[tickets["clean_text"].str.len() > 0].reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_clusters(
    texts: tuple[str, ...],
    n_clusters: int,
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    max_features: int,
    n_terms: int,
):
    return cluster_texts(
        list(texts),
        n_clusters=n_clusters,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        n_terms=n_terms,
    )


def metrics_table(internal: dict, external: dict) -> pd.DataFrame:
    rows = [
        ("Silhouette", internal["silhouette"], "Mas alto es mejor"),
        ("Davies-Bouldin", internal["davies_bouldin"], "Mas bajo es mejor"),
        ("Calinski-Harabasz", internal["calinski_harabasz"], "Mas alto es mejor"),
        ("Adjusted Rand Index", external["adjusted_rand"], "Referencia humana"),
        ("Normalized Mutual Info", external["normalized_mutual_info"], "Referencia humana"),
    ]
    return pd.DataFrame(rows, columns=["Metrica", "Valor", "Interpretacion"]).set_index("Metrica")


def main() -> None:
    st.title("🗂️ Agrupador de tickets de soporte")
    st.caption(
        "TF-IDF + K-Means sobre tickets reales anonimizados (Zenodo, CC BY 4.0). "
        "Las categorias humanas solo se usan como referencia de evaluacion."
    )

    try:
        tickets = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Ejecuta primero:\n\n"
            "```\npython scripts/download_data.py\npython scripts/prepare_data.py\n```"
        )
        st.stop()

    with st.sidebar:
        st.header("Configuracion del agrupamiento")
        n_clusters = st.slider("Numero de clusters (k)", 2, 12, 7)
        use_bigrams = st.checkbox("Incluir bigramas", value=True)
        min_df = st.slider("min_df (frecuencia minima)", 1, 5, 2)
        max_df = st.slider("max_df (proporcion maxima)", 0.5, 1.0, 0.9, 0.05)
        max_features = st.select_slider(
            "Maximo de caracteristicas", options=[500, 1000, 2000, 5000, 10000], value=5000
        )
        n_terms = st.slider("Palabras clave por cluster", 5, 20, 10)

        st.divider()
        st.metric("Tickets cargados", f"{len(tickets):,}")
        if "category_truth" in tickets.columns:
            st.metric("Categorias de referencia", tickets["category_truth"].nunique())

    ngram_range = (1, 2) if use_bigrams else (1, 1)

    with st.spinner("Vectorizando y agrupando..."):
        result = build_clusters(
            tuple(tickets["clean_text"].tolist()),
            n_clusters,
            ngram_range,
            min_df,
            max_df,
            max_features,
            n_terms,
        )

    tickets = tickets.assign(cluster=result.labels)

    internal = internal_metrics(result.matrix, result.labels)
    external = (
        external_metrics(result.labels, tickets["category_truth"].tolist())
        if "category_truth" in tickets.columns
        else {"adjusted_rand": float("nan"), "normalized_mutual_info": float("nan")}
    )

    st.subheader("Resumen del agrupamiento")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clusters", n_clusters)
    col2.metric("Silhouette", f"{internal['silhouette']:.3f}")
    col3.metric("Davies-Bouldin", f"{internal['davies_bouldin']:.3f}")
    col4.metric("ARI vs. categorias", f"{external['adjusted_rand']:.3f}")

    with st.expander("Ver todas las metricas"):
        st.dataframe(metrics_table(internal, external), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Tamano de cada cluster")
        sizes = tickets["cluster"].value_counts().sort_index()
        sizes.index = [f"Cluster {i}" for i in sizes.index]
        st.bar_chart(sizes)
    with right:
        st.subheader("Composicion por categoria de referencia")
        if "category_truth" in tickets.columns:
            composition = pd.crosstab(tickets["cluster"], tickets["category_truth"])
            st.dataframe(composition, use_container_width=True)
        else:
            st.info("El dataset no incluye categorias de referencia.")

    st.subheader("Explorar clusters")
    selected = st.selectbox("Selecciona un cluster", sorted(tickets["cluster"].unique()))
    cluster_tickets = tickets[tickets["cluster"] == selected]

    term_col, ticket_col = st.columns([1, 2])
    with term_col:
        st.markdown("**Palabras clave**")
        st.write(", ".join(result.terms.get(selected, [])) or "Sin terminos")
        st.markdown("**Tickets en el cluster**")
        st.write(f"{len(cluster_tickets):,}")

    with ticket_col:
        search = st.text_input("Filtrar tickets por texto", "")
        view = cluster_tickets
        if search:
            mask = view["text"].str.contains(search, case=False, na=False)
            view = view[mask]
        st.dataframe(
            view[["id", "text", "category_truth"]].head(50),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Probar un ticket nuevo")
    new_text = st.text_area(
        "Escribe un ticket y se asignara al cluster mas cercano",
        placeholder="Ejemplo: no puedo acceder a mi correo corporativo",
    )
    if st.button("Asignar ticket") and new_text.strip():
        cleaned = clean_text(new_text)
        if not cleaned:
            st.warning("El texto no contiene terminos utiles despues de la limpieza.")
        else:
            vector = result.vectorizer.transform([cleaned])
            cluster_id = int(result.model.predict(vector)[0])
            st.success(f"Asignado al cluster {cluster_id}")
            st.write("Palabras clave del cluster:", ", ".join(result.terms.get(cluster_id, [])))


if __name__ == "__main__":
    main()
