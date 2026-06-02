"""
API REST de clasificación de tickets IT construida con FastAPI.

Expone endpoints para inferencia individual, inferencia masiva, entrenamiento
por lotes via CSV, retroalimentación de perfiles y consulta de estadísticas.
Gestiona el ciclo de vida del modelo mediante un singleton ITTicketModel
y aplica MLOps de recuperación automática en el evento de arranque.
"""

import time
import io
import pandas as pd
from functools import reduce
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.engine import ITTicketModel
from core.nlp import preprocess_text
from core.storage import (
    insert_feedback,
    fetch_master_dataset,
    fetch_feedback_count,
    bulk_insert_dataset,
    fetch_solutions_by_department,
    insert_training_log,
    fetch_training_history,
    upsert_solution_vote,
    fetch_voted_solutions,
)

app = FastAPI(title="IT Ticket AI - Functional API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ITTicketModel()


@app.on_event("startup")
def restore_brain():
    """
    Recupera el modelo desde Supabase si los artefactos .pkl no están en disco.

    Detecta amnesia temporal causada por reinicios de contenedor en entornos
    serverless (ej. Hugging Face Spaces). Si el clasificador o el vectorizador
    están ausentes, descarga el Data Lake completo y ejecuta train_batch para
    reconstruir el modelo. Registra la sesión en training_log con trigger_type
    "startup". Si el Data Lake está vacío o tiene menos de 5 registros, espera
    la primera carga manual de CSV.
    Llamado por: FastAPI en el evento startup.
    """
    if not engine.classifier or not engine.vectorizer:
        print("[MLOps] Amnesia temporal detectada. Restaurando cerebro desde Supabase...")
        df = fetch_master_dataset()
        if not df.empty and len(df) >= 5:
            stats = engine.train_batch(df)
            insert_training_log(
                trigger_type="startup",
                record_count=len(df),
                department_count=len(stats["labels"]),
                f1_score=stats["f1Score"],
                accuracy=stats["accuracy"],
                avg_confidence=stats.get("avgConfidence"),
            )
            print(f"[MLOps] Cerebro restaurado exitosamente con {len(df)} tickets históricos.")
        else:
            print("[MLOps] Data Lake vacío o insuficiente. Esperando carga inicial de CSV.")


class PredictRequest(BaseModel):
    title: str
    description: str


class FeedbackRequest(BaseModel):
    original_text: str
    correct_department: str


class BlindTicketRequest(BaseModel):
    text: str


class SolutionVoteRequest(BaseModel):
    department: str
    solution: str


class SolutionFeedbackRequest(BaseModel):
    department: str
    solution: str


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_ready": engine.classifier is not None,
        "message": "API Funcional Operativa"
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    Clasifica un ticket individual con escudo OOD (Out-of-Domain).

    Concatena título y descripción, aplica el pipeline NLP y rechaza el ticket
    si tiene menos de 4 tokens o si menos del 15% de sus tokens existen en el
    vocabulario TF-IDF entrenado (detección vectorial de anomalías). De pasar
    ambos filtros, delega la inferencia a engine.predict_single y calcula el
    nivel de criticidad heurístico basado en palabras clave de urgencia.

    Retorna winner, probabilities, tokens, latency, level, originalText,
    cleanText, topTfidf e is_garbage. Si el texto es rechazado retorna solo
    is_garbage y message.
    Llamado por: frontend playground / integración externa.
    """
    start = time.time()
    original_text = f"{req.title}\n{req.description}"

    clean_text, tokens = preprocess_text(original_text)

    if len(tokens) < 4:
        return {"is_garbage": True, "message": "Texto insuficiente. Describe el problema con mayor detalle técnico."}

    try:
        vocab = engine.vectorizer.vocabulary_

        known_tokens = list(filter(lambda t: t in vocab, tokens))
        known_ratio = len(known_tokens) / len(tokens)

        if known_ratio < 0.15:
            return {
                "is_garbage": True,
                "message": f"Anomalía detectada. El texto carece de contexto IT válido (Coincidencia semántica: {known_ratio*100:.1f}%). Por favor, ingresa un ticket técnico."
            }

        result = engine.predict_single(clean_text, tokens)
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Modelo no entrenado. Ve a la pestaña Batch y sube un CSV primero.")

    latency = int((time.time() - start) * 1000)

    t_lower = original_text.lower()
    score = sum([3 for w in ["urgente", "caído", "caida", "crítico", "producción"] if w in t_lower]) + \
            sum([1 for w in ["lento", "error", "falla", "problema"] if w in t_lower])

    level = "Nivel 3 (Crítico)" if score >= 4 else "Nivel 2 (Especializado)" if score >= 2 else "Nivel 1 (Triaje Directo)"

    return {
        "winner": result["winner"],
        "probabilities": result["probabilities"],
        "tokens": result["tokens"],
        "latency": latency,
        "level": level,
        "originalText": original_text,
        "cleanText": result["cleanText"],
        "topTfidf": result["topTfidf"],
        "is_garbage": False
    }


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """
    Registra la corrección de un perfil y actualiza el modelo de forma adaptativa.

    Persiste el feedback en Supabase via insert_feedback, preprocesa el texto
    corregido y llama a engine.learn_online. Si el resultado es "NEEDS_RETRAIN"
    (departamento nuevo), descarga el Data Lake completo y ejecuta un
    reentrenamiento batch inmediato. Si el resultado es "LEARNED", revisa si
    el total de feedbacks acumulados es múltiplo de 20 y, de serlo, lanza
    también un reentrenamiento periódico. En ambos casos de reentrenamiento
    registra la sesión en training_log con trigger_type "feedback_retrain".

    Retorna success, learnedImmediately, retrainedBatch y message.
    Llamado por: frontend (corrección manual de predicción).
    """
    insert_feedback(req.original_text, req.correct_department)

    clean_txt, _ = preprocess_text(req.original_text)
    learning_status = engine.learn_online(clean_txt, req.correct_department)

    retrained = False

    is_safe_to_train = lambda data_frame: not data_frame.empty and len(data_frame) >= 5

    if learning_status == "NEEDS_RETRAIN":
        df = fetch_master_dataset()
        if is_safe_to_train(df):
            stats = engine.train_batch(df)
            insert_training_log(
                trigger_type="feedback_retrain",
                record_count=len(df),
                department_count=len(stats["labels"]),
                f1_score=stats["f1Score"],
                accuracy=stats["accuracy"],
                avg_confidence=stats.get("avgConfidence"),
            )
            retrained = True
    else:
        count = fetch_feedback_count()
        if count > 0 and count % 20 == 0:
            df = fetch_master_dataset()
            if is_safe_to_train(df):
                stats = engine.train_batch(df)
                insert_training_log(
                    trigger_type="feedback_retrain",
                    record_count=len(df),
                    department_count=len(stats["labels"]),
                    f1_score=stats["f1Score"],
                    accuracy=stats["accuracy"],
                    avg_confidence=stats.get("avgConfidence"),
                )
                retrained = True

    return {
        "success": True,
        "learnedImmediately": learning_status == "LEARNED",
        "retrainedBatch": retrained,
        "message": "Feedback consolidado. La IA se ha adaptado orgánicamente."
    }


@app.post("/api/batch")
async def batch(file: UploadFile = File(...)):
    """
    Entrena el modelo desde un archivo CSV subido por el perfil.

    Lee el CSV, acepta columnas text/department o titulo/descripcion/departamento,
    normaliza la columna solution opcional (solucion → solution, NaN → None).
    Aplica un pipeline Map/Filter: primero descarta registros con menos de 4
    tokens o sin departamento, luego formatea los válidos. Persiste los registros
    limpios en Supabase via bulk_insert_dataset, entrena el modelo con
    engine.train_batch y registra la sesión en training_log con trigger_type
    "csv_upload".

    Retorna totalTickets, processedCount, rejectedCount, métricas del modelo
    (f1Score, accuracy, avgConfidence, confusionMatrix, labels),
    departmentDistribution, globalTfidf y speed en milisegundos.
    Llamado por: frontend pestaña Batch / carga inicial de Data Lake.
    """
    start = time.time()

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido.")

    if "text" not in df.columns or "department" not in df.columns:
        if "titulo" in df.columns and "descripcion" in df.columns and "departamento" in df.columns:
            df["text"] = df["titulo"].astype(str) + " " + df["descripcion"].astype(str)
            df["department"] = df["departamento"]
        else:
            raise HTTPException(status_code=400, detail="El CSV debe tener las columnas 'text' y 'department'.")

    if "solucion" in df.columns and "solution" not in df.columns:
        df["solution"] = df["solucion"]
    if "solution" not in df.columns:
        df["solution"] = None

    df["text"] = df["text"].fillna("")
    records = df.to_dict('records')

    for r in records:
        raw = r.get("solution")
        if raw is None:
            continue
        sol = str(raw).strip()
        r["solution"] = sol if sol and sol not in ("nan", "None", "NaN") else None

    preprocess_mapper = lambda r: {
        "raw_dept": r.get("department"),
        "nlp": preprocess_text(str(r.get("text", ""))),
        "raw_solution": r.get("solution"),
    }
    processed_records = list(map(preprocess_mapper, records))

    is_valid = lambda item: len(item["nlp"][1]) >= 4 and pd.notna(item["raw_dept"])
    valid_records = list(filter(is_valid, processed_records))
    rejected = len(records) - len(valid_records)

    if not valid_records:
        raise HTTPException(status_code=400, detail="El CSV no contenía tickets válidos tras la limpieza NLP.")

    def _format_record(item: dict) -> dict:
        record = {"text": item["nlp"][0], "department": item["raw_dept"]}
        sol = item.get("raw_solution")
        if sol and isinstance(sol, str) and sol.strip():
            record["solution"] = sol.strip()
        return record

    clean_data = list(map(_format_record, valid_records))

    bulk_insert_dataset(clean_data)

    clean_df = pd.DataFrame(clean_data)
    train_stats = engine.train_batch(clean_df)

    insert_training_log(
        trigger_type="csv_upload",
        record_count=len(clean_df),
        department_count=len(train_stats["labels"]),
        f1_score=train_stats["f1Score"],
        accuracy=train_stats["accuracy"],
        avg_confidence=train_stats.get("avgConfidence"),
    )

    speed = int((time.time() - start) * 1000)
    dist = clean_df["department"].value_counts().to_dict()

    return {
        "totalTickets": len(df),
        "processedCount": len(clean_df),
        "rejectedCount": rejected,
        "f1Score": train_stats["f1Score"],
        "accuracy": train_stats["accuracy"],
        "avgConfidence": train_stats.get("avgConfidence"),
        "bestModelName": train_stats["bestModelName"],
        "optimalAlpha": "N/A (Log Loss)",
        "confusionMatrix": train_stats["confusionMatrix"],
        "labels": train_stats["labels"],
        "departmentDistribution": dist,
        "globalTfidf": train_stats["globalTfidf"],
        "speed": speed
    }


@app.post("/api/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    """
    Clasifica masivamente un CSV de tickets sin etiquetas conocidas.

    Lee el CSV (columnas text o titulo/descripcion), aplica el pipeline NLP
    y la detección OOD a cada ticket via la función interna infer_ticket.
    Usa map para proyectar la inferencia sobre toda la colección y reduce
    para acumular la distribución de frecuencias por departamento predicho
    sin mutar variables externas.

    Retorna totalTickets, speed en milisegundos, departmentDistribution
    y la lista completa de predictions con id, text_original,
    predicted_department y confidence.
    Llamado por: frontend pestaña de inferencia masiva ciega.
    """
    start_time = time.time()

    if not engine.classifier or not engine.vectorizer:
        raise HTTPException(status_code=400, detail="La IA no ha sido entrenada. Sube primero un dataset en Lotes.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido.")

    if "text" not in df.columns:
        if "titulo" in df.columns and "descripcion" in df.columns:
            df["text"] = df["titulo"].astype(str) + " " + df["descripcion"].astype(str)
        else:
            raise HTTPException(status_code=400, detail="El CSV debe contener 'text' o 'titulo' y 'descripcion'.")

    df["text"] = df["text"].fillna("")
    records = df.to_dict('records')
    vocab = engine.vectorizer.vocabulary_

    def infer_ticket(idx: int, raw_text: str) -> dict:
        """
        Clasifica un ticket individual dentro del pipeline de inferencia masiva.

        Aplica dos filtros de rechazo: cantidad mínima de tokens (< 4) y
        detección OOD por ratio de tokens conocidos (< 15%). De pasar ambos,
        vectoriza con el TF-IDF del motor y retorna la clase con mayor
        probabilidad junto a su score formateado como porcentaje.
        Llamado por: batch_predict (vía map).
        """
        clean_txt, tokens = preprocess_text(raw_text)
        text_preview = raw_text[:120] + "..." if len(raw_text) > 120 else raw_text

        if len(tokens) < 4:
            return {
                "id": idx + 1,
                "text_original": text_preview,
                "predicted_department": "Rechazado (Texto Insuficiente)",
                "confidence": "0.0%"
            }

        known_tokens = list(filter(lambda t: t in vocab, tokens))
        if len(known_tokens) / len(tokens) < 0.15:
            return {
                "id": idx + 1,
                "text_original": text_preview,
                "predicted_department": "Rechazado (Anomalía OOD)",
                "confidence": "0.0%"
            }

        X_vec = engine.vectorizer.transform([clean_txt])
        probas = engine.classifier.predict_proba(X_vec)[0]
        max_idx = probas.argmax()

        return {
            "id": idx + 1,
            "text_original": text_preview,
            "predicted_department": engine.classifier.classes_[max_idx],
            "confidence": f"{round(float(probas[max_idx]) * 100, 1)}%"
        }

    predictions = list(map(lambda item: infer_ticket(item[0], str(item[1].get("text", ""))), enumerate(records)))

    distribution = reduce(
        lambda acc, p: {**acc, p["predicted_department"]: acc.get(p["predicted_department"], 0) + 1},
        predictions,
        {}
    )

    speed = int((time.time() - start_time) * 1000)

    return {
        "totalTickets": len(df),
        "speed": speed,
        "departmentDistribution": distribution,
        "predictions": predictions
    }


@app.get("/api/stats")
def get_stats():
    """
    Retorna el historial de sesiones de entrenamiento del modelo.

    Consulta training_log via fetch_training_history y extrae las métricas
    de la sesión más reciente (f1_score, accuracy, avg_confidence, record_count)
    como campos de acceso rápido junto al historial completo.

    Retorna history, totalSessions, latestF1, latestAccuracy,
    latestConfidence y totalRecords.
    Llamado por: frontend curva de aprendizaje / dashboard MLOps.
    """
    history = fetch_training_history()
    if not history:
        return {
            "history": [],
            "totalSessions": 0,
            "latestF1": None,
            "latestAccuracy": None,
            "latestConfidence": None,
            "totalRecords": 0,
        }
    latest = history[0]
    return {
        "history": history,
        "totalSessions": len(history),
        "latestF1": latest["f1_score"],
        "latestAccuracy": latest["accuracy"],
        "latestConfidence": latest["avg_confidence"],
        "totalRecords": latest["record_count"],
    }


@app.get("/api/solutions")
def get_solutions(department: str):
    """
    Retorna las soluciones conocidas para un departamento, ordenadas por relevancia.

    Fusiona dos fuentes: solution_votes (soluciones votadas y aportadas por perfiles,
    ordenadas por vote_count DESC) y dataset_master (soluciones del Data Lake de
    entrenamiento, sin duplicar). Las soluciones con votos aparecen primeras para
    que el conocimiento validado por la comunidad sea más visible.

    Retorna department, solutions (lista de objetos con solution, voteCount y source)
    y count.
    Llamado por: frontend modal de soluciones sugeridas.
    """
    dept = department.strip()
    if not dept:
        raise HTTPException(status_code=400, detail="Parámetro 'department' requerido.")

    voted = fetch_voted_solutions(dept)
    voted_texts = {item["solution"] for item in voted}

    dataset_sols = fetch_solutions_by_department(dept)
    dataset_unique = [s for s in dataset_sols if s["solution"] not in voted_texts]

    merged = [
        {"solution": s["solution"], "voteCount": s["vote_count"], "source": s["source"]}
        for s in voted
    ] + [
        {"solution": s["solution"], "voteCount": 0, "source": "dataset"}
        for s in dataset_unique
    ]

    return {"department": dept, "solutions": merged, "count": len(merged)}


@app.post("/api/solutions/vote")
def vote_solution(req: SolutionVoteRequest):
    """
    Registra que una solución existente funcionó para el perfil que vota.

    Delega a upsert_solution_vote con source="vote". Si la solución ya tiene
    votos, incrementa su vote_count; si no existe en solution_votes, la crea
    con vote_count=1. La operación es idempotente para el mismo (dept, solution).

    Retorna success y message.
    Llamado por: frontend botón "Funcionó" del modal de soluciones.
    """
    dept = req.department.strip()
    sol = req.solution.strip()
    if not dept or not sol:
        raise HTTPException(status_code=400, detail="department y solution son requeridos.")
    upsert_solution_vote(dept, sol, source="vote")
    return {"success": True, "message": "Voto registrado correctamente."}


@app.post("/api/solutions/feedback")
def submit_solution_feedback(req: SolutionFeedbackRequest):
    """
    Registra una solución aportada por un perfil que no encontró respuesta en la base.

    Persiste en solution_votes con source="user_feedback" y vote_count=1.
    La solución pasa a ser visible inmediatamente para otros perfiles que
    enfrenten el mismo tipo de ticket en el mismo departamento.

    Retorna success y message.
    Llamado por: frontend formulario de aporte del modal de soluciones.
    """
    dept = req.department.strip()
    sol = req.solution.strip()
    if not dept or not sol:
        raise HTTPException(status_code=400, detail="department y solution son requeridos.")
    upsert_solution_vote(dept, sol, source="user_feedback")
    return {"success": True, "message": "Solución de la comunidad registrada. Será visible para otros perfiles."}
