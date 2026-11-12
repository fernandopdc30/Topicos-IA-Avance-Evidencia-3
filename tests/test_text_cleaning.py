import pandas as pd
import pytest

from src.text_cleaning import add_clean_column, clean_series, clean_text


def test_clean_text_empty_values():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("   ") == ""


def test_clean_text_normalizes_case_punctuation_and_accents():
    result = clean_text("No puedo ACCEDER al correo: ¡urgente!!")
    assert result == "no puedo acceder al correo urgente"


def test_clean_text_removes_pii_tags_and_urls():
    text = "Solicitud [TICKET ID] de [NAME] en [COMPANY] https://ejemplo.com"
    result = clean_text(text)
    assert "ticket" not in result
    assert "name" not in result
    assert "company" not in result
    assert "https" not in result


def test_clean_text_removes_server_tag():
    result = clean_text("Java Installation on [SERVER]")
    assert result == "java installation on"


def test_clean_text_removes_boilerplate():
    text = "[TICKET ID] New Support Ticket received - Reset de contrasena"
    result = clean_text(text)
    assert "support" not in result
    assert "received" not in result
    assert "reset de contrasena" in result


def test_clean_text_drops_short_tokens():
    assert clean_text("a b de la red", min_token_length=3) == "red"


def test_clean_series_preserves_length_and_order():
    texts = ["Hola Mundo", None, "Error 500"]
    cleaned = clean_series(texts)
    assert len(cleaned) == 3
    assert cleaned[0] == "hola mundo"
    assert cleaned[1] == ""


def test_add_clean_column_does_not_mutate_input():
    df = pd.DataFrame({"text": ["Abrir TICKET urgente"]})
    result = add_clean_column(df)
    assert "clean_text" in result.columns
    assert "clean_text" not in df.columns
    assert result.loc[0, "clean_text"] == "abrir ticket urgente"
