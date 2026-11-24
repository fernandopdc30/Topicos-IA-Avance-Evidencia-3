from streamlit.testing.v1 import AppTest


def test_dashboard_starts_without_exceptions():
    app = AppTest.from_file("app.py", default_timeout=60).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "📊 Resumen",
        "🗺️ Mapa 2D",
        "🔎 Explorar",
        "🧪 Calidad y k",
        "✨ Clasificar",
    ]
    assert any(metric.label == "Clusters" for metric in app.metric)
