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

def bulk_insert_dataset(records: list[dict]) -> None:
    """
    Inserta masivamente registros en el Data Lake (Supabase).
    Utiliza programación funcional para evitar la inserción de tickets duplicados.
    """
    if not records:
        return
        
    # 1. Obtener textos existentes en Supabase
    response = supabase.table("dataset_master").select("text").execute()
    
    # 2. Map: Transformar la respuesta en un Set (Conjunto) para búsquedas O(1)
    existing_texts = set(map(lambda r: r["text"], response.data)) if response.data else set()
    
    # 3. Filter: Retener únicamente los registros cuyo texto no exista en la base de datos
    new_records = list(filter(lambda r: r["text"] not in existing_texts, records))
    
    if not new_records:
        return
        
    # 4. Inserción por bloques (chunks) para no saturar la red de Supabase
    chunk_size = 1000
    for i in range(0, len(new_records), chunk_size):
        supabase.table("dataset_master").insert(new_records[i:i+chunk_size]).execute()