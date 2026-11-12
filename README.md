# Topicos-IA-Avance-Evidencia-3

# Agrupador de tickets de soporte

Prototipo academico que agrupa tickets reales de soporte tecnico segun la similitud
de su texto, usando Procesamiento de Lenguaje Natural (TF-IDF) y aprendizaje no
supervisado (K-Means).

**Universidad La Salle** — Ingenieria de Software
**Integrantes:** Enyelbert Anderson Panta Huaracha, Fernando Perez del Castillo
**Entregable:** Evidencia 3 — Repositorio, dataset/script de obtencion y demo funcional (~50%)

## Descripcion

Los equipos de soporte reciben incidencias descritas en lenguaje natural. Un mismo
problema puede reportarse con palabras distintas, lo que dificulta identificar temas
recurrentes. Este proyecto transforma las descripciones en vectores TF-IDF y descubre
grupos de incidencias similares **sin usar las categorias originales**, que se conservan
unicamente como referencia de evaluacion.

## Estado del entregable (Evidencia 3)

- [x] Repositorio del proyecto con estructura reproducible.
- [x] Dataset real descargado y script que lo obtiene desde Zenodo con verificacion MD5.
- [x] Demo funcional: dashboard Streamlit con agrupamiento, metricas y exploracion.
- [x] Pruebas automatizadas con `pytest` (limpieza, carga y agrupamiento).
- [ ] Ajuste fino del modelo y visualizaciones avanzadas (siguiente iteracion).

## Estructura del repositorio

```
.
├── app.py                      # Dashboard Streamlit (demo funcional)
├── conftest.py                 # Configuracion de rutas para pytest
├── requirements.txt
├── data/
│   ├── raw/                    # CSV originales de Zenodo (X/y train/test)
│   └── processed/tickets.csv   # Tickets unidos + texto normalizado
├── scripts/
│   ├── download_data.py        # Descarga y verifica el dataset (Zenodo)
│   └── prepare_data.py         # Une train/test y agrega texto limpio
├── src/
│   ├── config.py               # Rutas, constantes y stop words
│   ├── data_loader.py          # Carga y union de los CSV
│   ├── text_cleaning.py        # Normalizacion y limpieza de texto
│   ├── clustering.py           # TF-IDF + K-Means + terminos clave
│   └── evaluation.py           # Metricas internas y externas
└── tests/                      # Pruebas automatizadas
```

## Dataset

**Classification of IT Support Tickets** (version 2, 2022)
Leonardo Santiago Benitez Pereira — Zenodo.
DOI: https://doi.org/10.5281/zenodo.7648117

- 2 229 tickets reales anonimizados de una empresa de soporte en Florianopolis (Brasil).
- 7 categorias asignadas manualmente por profesionales de TI en 2020.
- Corpus multilingue (ingles, portugues, espanol y aleman).
- Licencia **CC BY 4.0**: permite uso y redistribucion con atribucion.

### Archivos usados

| Archivo | Uso en el proyecto |
| --- | --- |
| `X_train.csv`, `X_test.csv` | Texto de los tickets a limpiar y vectorizar. |
| `y_train.csv`, `y_test.csv` | Etiquetas de referencia (no se usan para agrupar). |

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Obtencion y preparacion de los datos

Los CSV ya estan incluidos en `data/raw/`. Para descargarlos de nuevo y verificar su
integridad (MD5):

```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

`prepare_data.py` genera `data/processed/tickets.csv` con el texto normalizado.

## Demo funcional

```bash
streamlit run app.py
```

El dashboard permite:

- Configurar `k`, bigramas, `min_df`, `max_df` y numero de caracteristicas.
- Ejecutar TF-IDF + K-Means sobre los tickets.
- Ver el tamano de cada cluster y su composicion frente a las categorias humanas.
- Consultar palabras clave y tickets representativos de cada cluster.
- Clasificar un ticket nuevo asignandolo al cluster mas cercano.
- Revisar metricas: Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI y NMI.

## Pruebas

```bash
python -m pytest -q
```

Cubren entradas vacias, limpieza de texto, eliminacion de etiquetas PII, integridad de
registros, alineacion de archivos y estructura de resultados del agrupamiento.

## Metodologia

1. **Preparacion:** union de train/test, deteccion de duplicados y valores faltantes.
2. **Limpieza:** minusculas, eliminacion de etiquetas PII, URLs, correos, HTML, acentos,
   puntuacion y frases plantilla; descarte de tokens muy cortos.
3. **TF-IDF:** unigramas y bigramas con `sublinear_tf`; stop words multilingues.
4. **K-Means:** `k` configurable (las 7 categorias son referencia, no obligacion).
5. **Interpretacion:** terminos de mayor peso por centroide y tickets representativos.

## Metricas

- **Internas:** Silhouette Score, Davies-Bouldin y Calinski-Harabasz.
- **Externas (contraste):** Adjusted Rand Index (ARI) y Normalized Mutual Information (NMI).

Los valores se calculan en cada ejecucion del dashboard; no se fijan como resultados
definitivos del proyecto.

## Limitaciones y trabajo futuro

- La diversidad de idiomas afecta la similitud textual; se evaluara agrupar subconjuntos
  por idioma.
- K-Means asume clusters esfericos y requiere fijar `k`; se comparara con otros valores.
- Pendiente: reduccion de dimensionalidad (SVD/PCA), visualizacion 2D de clusters,
  seleccion automatica de `k` y exportacion de resultados.

## Referencia

Benitez Pereira, L. S. (2022). *Classification of IT Support Tickets* (version 2)
[Conjunto de datos]. Zenodo. https://doi.org/10.5281/zenodo.7648117
