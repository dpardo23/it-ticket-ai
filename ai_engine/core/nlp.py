"""
Pipeline de procesamiento de lenguaje natural para tickets IT.

Carga el modelo de spaCy en español (es_core_news_sm) al importar el módulo
y expone dos funciones puras: clean_text y preprocess_text.
Llamado por: core.engine, api.main.
"""

import spacy
import unicodedata
import re

nlp = spacy.load("es_core_news_sm")


def clean_text(text: str) -> str:
    """
    Normaliza un texto a ASCII lowercase sin caracteres especiales.

    Aplica descomposición NFD, elimina diacríticos, convierte a ASCII
    e iguala espacios múltiples a uno solo.
    Llamado por: preprocess_text.
    """
    text = unicodedata.normalize("NFD", text.lower())
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> tuple[str, list[str]]:
    """
    Aplica el pipeline NLP completo sobre un texto crudo.

    Limpia el texto con clean_text, lo procesa con spaCy y extrae lemas
    de tokens que no sean stopwords, puntuación ni secuencias de 1-2 caracteres.

    Retorna una tupla (texto_lematizado_unido, lista_de_lemas) lista para
    ser consumida por el vectorizador TF-IDF del motor de clasificación.
    Llamado por: api.main (predict, feedback, batch, batch_predict).
    """
    cleaned_base = clean_text(text)

    doc = nlp(cleaned_base)

    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]

    return " ".join(tokens), tokens
