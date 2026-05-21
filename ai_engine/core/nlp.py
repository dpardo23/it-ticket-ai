import spacy
import unicodedata
import re

# Cargar el modelo en español optimizado
nlp = spacy.load("es_core_news_sm")

def clean_text(text: str) -> str:
    """Aplica limpieza básica de caracteres (ASCII, minúsculas, espacios)."""
    text = unicodedata.normalize("NFD", text.lower())
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess_text(text: str) -> tuple[str, list[str]]:
    """
    Función pura de NLP.
    Retorna: (texto_limpio_lematizado, lista_de_tokens)
    """
    cleaned_base = clean_text(text)
    
    # Procesar con spaCy
    doc = nlp(cleaned_base)
    
    # Aplicar comprensión de listas (Programación Funcional)
    # Filtramos stopwords, puntuación, textos muy cortos y obtenemos la RAÍZ (lemma)
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]
    
    return " ".join(tokens), tokens