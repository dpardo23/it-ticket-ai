"""
Capa de acceso a datos sobre Supabase para el sistema de tickets IT.

Agrupa todas las operaciones de lectura y escritura contra las tablas
dataset_master, feedback_log y training_log. Ninguna función de este
módulo conoce el modelo de IA; solo maneja persistencia de datos.
Llamado por: api.main.
"""

import pandas as pd
from core.db import supabase


def fetch_master_dataset() -> pd.DataFrame:
    """
    Extrae el dataset completo de tickets desde Supabase como DataFrame.

    Retorna un DataFrame con columnas text y department. Si la tabla
    está vacía, retorna un DataFrame vacío con esas mismas columnas.
    Llamado por: api.main (startup, feedback, batch implícitamente).
    """
    response = supabase.table("dataset_master").select("text, department").execute()
    return pd.DataFrame(response.data) if response.data else pd.DataFrame(columns=["text", "department"])


def fetch_feedback_count() -> int:
    """
    Retorna el número total de registros en feedback_log.

    Usado por api.main.feedback para decidir si lanzar un reentrenamiento
    periódico cada 20 correcciones acumuladas.
    Llamado por: api.main.feedback.
    """
    response = supabase.table("feedback_log").select("id", count="exact").execute()
    return response.count if response.count else 0


def insert_feedback(original_text: str, correct_department: str) -> None:
    """
    Registra una corrección de perfil en feedback_log y en dataset_master.

    Escribe en paralelo lógico en ambas tablas: feedback_log para auditoría
    y dataset_master para que el ticket corregido forme parte del Data Lake
    de entrenamiento futuro.
    Llamado por: api.main.feedback.
    """
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
    Inserta registros en dataset_master evitando duplicados por texto.

    Deduplica primero dentro del propio lote entrante en O(n) con un set.
    Luego consulta Supabase en chunks de 200 usando el índice
    idx_dataset_master_text para identificar textos ya existentes sin
    saturar el filtro IN. Finalmente inserta solo los registros nuevos
    en bloques de 500 filas para no sobrecargar la red.
    Soporta campo opcional solution para tickets con resolución conocida.
    Llamado por: api.main.batch.
    """
    if not records:
        return

    seen: set = set()
    unique_incoming = list(filter(
        lambda r: not (r["text"] in seen or seen.add(r["text"])),  # type: ignore[func-returns-value]
        records
    ))

    incoming_texts = list(map(lambda r: r["text"], unique_incoming))
    existing_texts: set = set()
    check_chunk = 200
    for i in range(0, len(incoming_texts), check_chunk):
        chunk = incoming_texts[i:i + check_chunk]
        resp = supabase.table("dataset_master").select("text").in_("text", chunk).execute()
        if resp.data:
            existing_texts.update(map(lambda r: r["text"], resp.data))

    new_records = list(filter(lambda r: r["text"] not in existing_texts, unique_incoming))

    if not new_records:
        return

    insert_chunk = 500
    for i in range(0, len(new_records), insert_chunk):
        supabase.table("dataset_master").insert(new_records[i:i + insert_chunk]).execute()


def insert_training_log(
    trigger_type: str,
    record_count: int,
    department_count: int,
    f1_score: float,
    accuracy: float,
    avg_confidence: float | None,
) -> None:
    """
    Persiste una sesión de entrenamiento en training_log.

    Registra el tipo de disparador (csv_upload, feedback_retrain, startup),
    el volumen de datos y las métricas de calidad del modelo resultante.
    Llamado por: api.main (startup, batch, feedback).
    """
    supabase.table("training_log").insert({
        "trigger_type": trigger_type,
        "record_count": record_count,
        "department_count": department_count,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
    }).execute()


def fetch_training_history(limit: int = 50) -> list[dict]:
    """
    Retorna las sesiones de entrenamiento más recientes desde training_log.

    Ordena por created_at descendente y limita a `limit` filas.
    Llamado por: api.main.get_stats.
    """
    response = (
        supabase.table("training_log")
        .select("id, created_at, trigger_type, record_count, department_count, f1_score, accuracy, avg_confidence")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def upsert_solution_vote(department: str, solution: str, source: str) -> None:
    """
    Registra un voto para una solución o incrementa su contador si ya existe.

    Consulta solution_votes por (department, solution). Si el registro existe,
    incrementa vote_count en uno. Si no, inserta un nuevo registro con
    vote_count=1 y el source recibido. La unicidad está garantizada por la
    restricción UNIQUE(department, solution) de la tabla.
    Llamado por: api.main.vote_solution, api.main.submit_solution_feedback.
    """
    existing = (
        supabase.table("solution_votes")
        .select("id, vote_count")
        .eq("department", department)
        .eq("solution", solution)
        .execute()
    )
    if existing.data:
        supabase.table("solution_votes").update(
            {"vote_count": existing.data[0]["vote_count"] + 1}
        ).eq("id", existing.data[0]["id"]).execute()
    else:
        supabase.table("solution_votes").insert(
            {"department": department, "solution": solution, "vote_count": 1, "source": source}
        ).execute()


def fetch_voted_solutions(department: str) -> list[dict]:
    """
    Retorna las soluciones votadas para un departamento, ordenadas por relevancia.

    Ordena por vote_count descendente para que las soluciones más validadas
    por perfiles aparezcan primeras en la interfaz de conocimiento.
    Llamado por: api.main.get_solutions.
    """
    response = (
        supabase.table("solution_votes")
        .select("solution, vote_count, source")
        .eq("department", department)
        .order("vote_count", desc=True)
        .execute()
    )
    return response.data or []


def fetch_solutions_by_department(department: str) -> list[dict]:
    """
    Retorna las soluciones únicas registradas para un departamento dado.

    Consulta dataset_master filtrando por departamento y excluyendo filas
    sin solución, apoyándose en el índice parcial
    idx_dataset_master_solution_nn (department, solution WHERE NOT NULL)
    para evitar scans completos. Deduplica en memoria y descarta valores
    vacíos, "nan" y "None" antes de retornar.
    Llamado por: api.main.get_solutions.
    """
    response = (
        supabase.table("dataset_master")
        .select("solution")
        .eq("department", department)
        .not_.is_("solution", "null")
        .execute()
    )
    _BAD = frozenset(("", "nan", "None", "NaN"))
    seen: set = set()

    def _unique_sol(row: dict):
        sol = str(row.get("solution") or "").strip()
        if sol not in _BAD and sol not in seen:
            seen.add(sol)
            return {"solution": sol}
        return None

    return list(filter(None, map(_unique_sol, response.data or [])))
