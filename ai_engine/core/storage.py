import pandas as pd
from core.db import supabase

def fetch_master_dataset() -> pd.DataFrame:
    """Extrae el dataset completo desde Supabase como un DataFrame."""
    response = supabase.table("dataset_master").select("text, department").execute()
    
    # Uso de operador ternario funcional para manejar datos vacíos
    return pd.DataFrame(response.data) if response.data else pd.DataFrame(columns=["text", "department"])

def fetch_feedback_count() -> int:
    """Retorna la cantidad de feedbacks registrados (para reentrenamiento batch)."""
    response = supabase.table("feedback_log").select("id", count="exact").execute()
    return response.count if response.count else 0

def insert_feedback(original_text: str, correct_department: str) -> None:
    """
    Registra el feedback humano en el log y lo consolida en el dataset maestro.
    """
    # Inserción paralela lógica
    supabase.table("feedback_log").insert({
        "original_text": original_text,
        "correct_department": correct_department
    }).execute()
    
    supabase.table("dataset_master").insert({
        "text": original_text,
        "department": correct_department
    }).execute()