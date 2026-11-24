# TicketScope — Agrupador inteligente de tickets de soporte

Prototipo académico que agrupa tickets reales de soporte técnico según la similitud
de su texto, usando Procesamiento de Lenguaje Natural (TF-IDF) y aprendizaje no
supervisado (K-Means).

**Universidad La Salle** — Ingeniería de Software
**Integrantes:** Enyelbert Anderson Panta Huaracha, Fernando Perez del Castillo
**Entregable:** Evidencia 3 — Repositorio reproducible, dataset real y demo funcional

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
- [x] Visualizacion 2D interactiva mediante Truncated SVD.
- [x] Comparacion automatizada de valores de `k` con Silhouette e inercia.
- [x] Tickets representativos, clasificacion de casos nuevos y exportacion CSV.
- [x] Tipos y subtipos funcionales sugeridos a partir de palabras clave, sin etiquetas reales.

## Estructura del repositorio

```
.
├── app.py                      # Dashboard Streamlit interactivo
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
│   ├── clustering.py           # TF-IDF, K-Means, SVD y casos representativos
│   ├── evaluation.py           # Metricas y comparacion de valores de k
│   └── interpretation.py       # Traduce terminos clave a tipos funcionales
├── .streamlit/config.toml      # Tema visual del dashboard
├── .github/workflows/tests.yml # Pruebas automaticas en GitHub Actions
└── tests/                      # Pruebas automatizadas
```

## Dataset

**Classification of IT Support Tickets** (version 2, 2022)
Leonardo Santiago Benitez Pereira — Zenodo.
DOI: https://doi.org/10.5281/zenodo.7648117

- 2 229 tickets reales anonimizados de una empresa de soporte en Florianopolis (Brasil).
- 7 categorias asignadas manualmente por profesionales de TI en 2020.
- Corpus multilingue (ingles, portugues, espanol y aleman).
- **Licencia no especificada en la ficha de Zenodo.** El acceso es abierto, pero antes
  de redistribuir los CSV debe confirmarse la autorizacion con el autor o el repositorio.

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

El entorno validado usa **Python 3.12** y las versiones exactas declaradas en
`requirements.txt`. En Streamlit Community Cloud selecciona Python 3.12 en
**Advanced settings** al crear la aplicación.

En PowerShell también puedes ejecutar el entorno sin activarlo:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
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

- **Resumen:** tamaños, composición y descarga de todas las asignaciones.
- **Mapa 2D:** proyección SVD interactiva con información de cada ticket.
- **Explorar:** tipo, subtipo, palabras clave, casos representativos y descarga por cluster.
- **Calidad y k:** métricas internas/externas y comparación de modelos candidatos.
- **Clasificar:** asignación de tickets nuevos a un tipo funcional, con distancia y separación.

Los tipos sugeridos —por ejemplo, *Buzones compartidos*, *Acceso a archivos y carpetas*
o *Gestión de cuentas de usuario*— se infieren exclusivamente de los términos del
centroide. La categoría humana predominante se presenta por separado para comprobar e
interpretar el resultado, pero nunca se utiliza para entrenar K-Means.

Los controles laterales permiten configurar `k`, unigramas/bigramas, `min_df`, `max_df`
y el número máximo de características. Los cálculos costosos se mantienen en caché
mientras la configuración no cambie.

## Pruebas

```bash
python -m pytest -q
```

Cubren entradas vacías, limpieza de texto, etiquetas PII, validación de columnas,
alineación de archivos, agrupamiento, proyección 2D, casos representativos, comparación
de valores de `k` y arranque completo del dashboard. La misma suite se ejecuta en cada
`push` y `pull request` mediante GitHub Actions.

## Metodologia

1. **Preparacion:** union de train/test, deteccion de duplicados y valores faltantes.
2. **Limpieza:** minusculas, eliminacion de etiquetas PII, URLs, correos, HTML, acentos,
   puntuacion y frases plantilla; descarte de tokens muy cortos.
3. **TF-IDF:** unigramas y bigramas con `sublinear_tf`; stop words multilingues.
4. **Ponderacion de duplicados:** conserva todos los tickets, pero aplica una correccion
   moderada de cuarta raiz a plantillas exactamente repetidas durante el entrenamiento.
5. **K-Means:** `k` configurable (las 7 categorias son referencia, no obligacion).
6. **Seleccion de k:** contraste de Silhouette coseno, inercia y tamano del menor grupo;
   la recomendacion exige que cada grupo contenga al menos 2% del corpus.
7. **Interpretacion:** terminos de mayor peso y tickets mas cercanos a cada centroide.
8. **Tipado semantico:** taxonomia reproducible que convierte terminos clave en tipos legibles.
9. **Visualizacion:** Truncated SVD proyecta TF-IDF a dos dimensiones solo para explorar.

## Metricas

- **Internas:** Silhouette coseno sobre TF-IDF; Davies-Bouldin y Calinski-Harabasz sobre
  una proyeccion SVD de hasta 50 componentes para evitar matrices densas muy grandes.
- **Externas (contraste):** Adjusted Rand Index (ARI) y Normalized Mutual Information (NMI).

Los valores se calculan en cada ejecucion del dashboard; no se fijan como resultados
definitivos del proyecto.

### Resultado reproducible de referencia

Con la configuracion inicial (`k=11`, bigramas y ponderacion moderada de duplicados), la
validacion del 23 de noviembre de 2026 produjo 2 225 tickets utiles, Silhouette `0.326`,
Davies-Bouldin `1.351`, ARI `0.201` y NMI `0.465`. El comparador equilibrado tambien
recomendo `k=11` dentro del rango 2–12; esta recomendacion no reemplaza la revision
cualitativa.

## Limitaciones y trabajo futuro

- La diversidad de idiomas afecta la similitud textual; se evaluara agrupar subconjuntos
  por idioma.
- K-Means asume grupos aproximadamente esfericos y requiere elegir `k`; el comparador
  ayuda a decidir, pero no sustituye la interpretacion humana.
- La proyeccion SVD pierde informacion y se usa unicamente como mapa exploratorio.
- Como siguiente iteracion se recomienda comparar embeddings multilingues y algoritmos
  como HDBSCAN, ademas de incorporar evaluacion cualitativa por especialistas.

## Referencia

Benitez Pereira, L. S. (2022). *Classification of IT Support Tickets* (version 2)
[Conjunto de datos]. Zenodo. https://doi.org/10.5281/zenodo.7648117
