"""Rutas y constantes compartidas por el proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PROCESSED_TICKETS_PATH = PROCESSED_DATA_DIR / "tickets.csv"

ZENODO_RECORD_ID = "7648117"
ZENODO_FILES_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"

RAW_FILES = ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv")

RANDOM_STATE = 42

STOP_WORDS = tuple(
    sorted(
        {
            # Ingles
            "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
            "at", "by", "from", "as", "is", "are", "was", "were", "be", "been",
            "it", "this", "that", "these", "those", "i", "we", "you", "he", "she",
            "they", "my", "our", "your", "me", "us", "do", "does", "did", "have",
            "has", "had", "can", "will", "would", "should", "could", "please",
            # Espanol
            "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
            "al", "y", "o", "en", "por", "para", "con", "sin", "que", "se", "su",
            "sus", "mi", "mis", "es", "son", "esta", "este", "estos", "estas",
            "lo", "le", "les", "como", "mas", "pero", "muy", "ya",
            # Portugues
            "os", "as", "um", "uma", "do", "da", "dos", "das", "em", "no", "na",
            "nos", "nas", "sem", "seu", "sua", "ao", "aos", "ele", "ela", "isso",
            "isto", "muito", "ja",
            # Aleman
            "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
            "eines", "und", "oder", "von", "zu", "im", "in", "auf", "fur",
            "mit", "ist", "sind", "war", "wie", "ich", "wir", "sie", "nicht",
        }
    )
)
